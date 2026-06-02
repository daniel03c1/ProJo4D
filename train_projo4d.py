import json
import numpy as np
import os
import random
import subprocess
import taichi as ti
import torch
import torch.nn as nn
import torchvision
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from new_trajectory import load_pcd_file, read_estimation_result, gen_xyz_list
from scene import Scene, DeformModel
from simulator import MPMSimulator, Estimator, Simulator
from train_dynamic import prepare_gt
from train_gs import training
from train_gs_fixed_pcd import assign_gs_to_pcd
from utils.system_utils import check_gs_model
from utils.data_utils import load_gt_pcds, write_dict_to_json
from utils.eval import export_result, evaluate, eval_mat_est_acc
from utils.general_utils import safe_state
from utils.loss_utils import psnr, ssim, l1_loss
from utils.lr_scheduler import CosineAnnealingWithWarmup


image_scale = 1.0


def execute_stage_code(estimator, lr_schedulers, code):
    if "S" in code:  # state
        estimator.state_optimizer.step()
        if lr_schedulers is not None:
            lr_schedulers[0].step()
    if "M" in code:  # material
        estimator.material_optimizer.step()
        if lr_schedulers is not None:
            lr_schedulers[1].step()
    if "G" in code:  # gaussians
        estimator.scene.gaussians.optimizer.step()
        if lr_schedulers is not None:
            lr_schedulers[2].step()
        estimator.scene.gaussians.x_optimizer.step()
        if lr_schedulers is not None:
            lr_schedulers[3].step()
    if "A" in code:  # appearance (color + opacity + scale + rotation)
        estimator.scene.gaussians.optimizer.step()
        if lr_schedulers is not None:
            lr_schedulers[2].step()
    if "X" in code:  # position
        estimator.scene.gaussians.x_optimizer.step()
        if lr_schedulers is not None:
            lr_schedulers[3].step()


def iter_train(
    estimator: Estimator,
    phys_args,
    stage_codes,
    max_f=None,
    n_iters=None,
    n_chunk_steps=1,
    test_views=None,
    gt_positions=None,
    gt_params=None,
    vel_iter_cnt=0,
    postfix="",
):
    assert n_chunk_steps > 0, "n_chunk_steps must be positive"
    assert len(stage_codes) > 0, "stage_codes must be non-empty"

    losses = []

    if max_f is not None:
        estimator.max_f = max_f

    if n_iters is None:
        n_iters = len(stage_codes)

    sub_iters = int(n_iters / len(stage_codes))

    kwargs = {"eta_min": phys_args.eta_min}
    joined_stage_codes = "".join(stage_codes)

    lr_schedulers = [
        CosineAnnealingWithWarmup(
            estimator.state_optimizer,
            T_max=int(sub_iters * joined_stage_codes.count("S")),
            **kwargs,
        ),
        CosineAnnealingWithWarmup(
            estimator.material_optimizer,
            T_max=int(sub_iters * joined_stage_codes.count("M")),
            **kwargs,
        ),
        CosineAnnealingWithWarmup(
            estimator.scene.gaussians.optimizer,
            T_max=int(
                sub_iters
                * (joined_stage_codes.count("G") + joined_stage_codes.count("A"))
            ),
            **kwargs,
        ),
        CosineAnnealingWithWarmup(
            estimator.scene.gaussians.x_optimizer,
            T_max=int(
                sub_iters
                * (joined_stage_codes.count("G") + joined_stage_codes.count("X"))
            ),
            **kwargs,
        ),
    ]

    n_frames = min(phys_args.n_frames, len(estimator.views))

    for i in tqdm(range(n_iters)):
        code = stage_codes[(i // n_chunk_steps) % len(stage_codes)]

        if i < vel_iter_cnt:
            max_f = phys_args.vel_estimation_frames
        else:
            max_f = n_frames

        forward(estimator, max_f=max_f)
        losses.append(estimator.loss[None] + estimator.image_loss)
        backward(estimator, max_f=max_f)

        # Capture losses pre-step (they came from this iter's forward),
        # but defer printing until after execute_stage_code so the logged
        # material params reflect the just-applied update.
        loss_3d = estimator.loss[None]
        loss_2d = estimator.image_loss.item()

        execute_stage_code(estimator, lr_schedulers, code)

        message = {"3D": loss_3d, "2D": loss_2d}

        if gt_params is not None:
            message.update(
                eval_mat_est_acc(
                    estimator, gt_params, prefix="", use_default_value=False
                )
            )

        print(", ".join(f"{k}: {v:.4f}" for k, v in message.items()))

        # Zero every optimizer each iter so params whose optimizer wasn't
        # stepped don't accumulate stale grads (e.g. gaussian params get
        # grads from the 2D image-loss backward every iter regardless of
        # stage code).
        estimator.state_optimizer.zero_grad()
        estimator.material_optimizer.zero_grad()
        estimator.scene.gaussians.optimizer.zero_grad()
        estimator.scene.gaussians.x_optimizer.zero_grad()

    return losses


def forward(estimator: Estimator, max_f=None, return_positions=False):
    dt = estimator.sim.dt_ori[None]

    if max_f is None:
        max_f = estimator.max_f
    while True:
        assert round(estimator.sim.frame_dt / dt) <= 800

        estimator.initialize(optimize_pos=True)
        estimator.sim.set_dt(dt)

        positions = []

        for idx in range(max_f):
            x = estimator.forward(idx)
            positions.append(x)

        if not estimator.succeed():
            dt /= 2
            print(
                f"cfl condition dissatisfy, shrink dt {dt}, "
                f"step cnt {estimator.sim.n_substeps[None] * 2}"
            )
        else:
            break

    if return_positions:
        return positions
    return


def backward(estimator: Estimator, max_f=None):
    if max_f is None:
        max_f = estimator.max_f

    estimator.loss.grad[None] = 1
    estimator.clear_grads()

    for i in reversed(range(max_f)):
        estimator.backward(i)


@torch.no_grad()
def extract_grads(estimator):
    def get_grad(param, avg_norm=False):
        if not hasattr(param, "grad") or param.grad is None:
            return 0.0
        if avg_norm:
            return param.grad.norm(dim=-1).mean().item()
        return param.grad.detach().cpu().numpy().tolist()

    grads = {}

    # physical states
    grads["v_grad"] = get_grad(estimator.init_vel)

    # material parameters
    grads["E_grad"] = get_grad(estimator.E)
    grads["nu_grad"] = get_grad(estimator.nu)
    grads["mu_grad"] = get_grad(estimator.global_mu)
    grads["kappa_grad"] = get_grad(estimator.global_kappa)
    grads["yield_stress_grad"] = get_grad(estimator.yield_stress)
    grads["plastic_viscosity_grad"] = get_grad(estimator.plastic_viscosity)
    grads["friction_alpha_grad"] = get_grad(estimator.friction_alpha)

    # gaussians
    _gaussians = estimator.scene.gaussians
    grads["avg_x_grad"] = get_grad(_gaussians._xyz, True)
    grads["avg_scale_grad"] = get_grad(_gaussians._scaling, True)
    grads["avg_opacity_grad"] = get_grad(_gaussians._opacity, True)
    grads["avg_feature_dc_grad"] = get_grad(_gaussians._features_dc, True)
    grads["avg_feature_rest_grad"] = get_grad(_gaussians._features_rest, True)

    return grads


def gaussian_color_init(scene, dataset, batch_size=1024):
    gaussians = GaussianModel(scene.gaussians.max_sh_degree)

    point_cloud_path = os.path.join(
        dataset.model_path,
        "point_cloud",
        f"iteration_{gs_args.iteration}",
        "point_cloud.ply",
    )
    print(f"Loading point cloud from: {point_cloud_path}")
    gaussians.load_ply(point_cloud_path)

    deform = DeformModel(dataset)
    deform.load_weights(dataset.model_path)

    t_token = torch.zeros((1,)).expand(1, -1).to(gaussians.get_xyz)
    xyz = gaussians.get_xyz + deform.step(gaussians.get_xyz, t_token)[0]

    cur_xyz = scene.gaussians.get_xyz

    with torch.no_grad():
        for i in tqdm(range(0, len(cur_xyz), batch_size)):
            distances = torch.cdist(cur_xyz[i : i + batch_size], xyz)
            nearest_gaussian_indices = distances.argmin(dim=1)

            scene.gaussians._features_dc[i : i + batch_size] = gaussians._features_dc[
                nearest_gaussian_indices
            ]
            scene.gaussians._features_rest[i : i + batch_size] = (
                gaussians._features_rest[nearest_gaussian_indices]
            )

    return gaussians


@torch.no_grad()
def inference(
    estimator,
    gaussians,
    x_list,
    pipe_args,
    n_train_frames,
    views,
    target_positions=None,
    target_params=None,
    background=None,
    render_sim_path=None,
    render_gt_path=None,
    eval_pc_seed=None,
    eval_pc_until=None,
    n_samples=2048,
):
    performances = {}

    # 1.3. PATH
    if render_sim_path is not None:
        os.makedirs(render_sim_path, exist_ok=True)
    if render_gt_path is not None:
        os.makedirs(render_gt_path, exist_ok=True)

    # 1.4. Others
    if background is None:
        background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    views = sorted(views, key=lambda x: x.fid)
    frames = torch.unique(torch.cat([v.fid for v in views]))

    scene = estimator.scene

    # 2D metrics
    train_psnr_list = []
    train_ssim_list = []
    test_psnr_list = []
    test_ssim_list = []

    for f in range(len(frames)):
        curr_views = scene.getTestCameras(scale=1.0, exact_fid=frames[f], cam_idxs=None)

        xyz = x_list[f]
        d_xyz = xyz - scene.gaussians.get_xyz

        for view in curr_views:
            results = render(
                view, scene.gaussians, pipe_args, background, d_xyz, 0.0, 0.0, False
            )
            image = results["render"]
            image_with_bg = image + (1 - results["alpha"]) * view.background.cuda()
            gt_image = view.original_image.cuda()

            if render_sim_path is not None:
                torchvision.utils.save_image(
                    image, os.path.join(render_sim_path, f"{view.uid}_{f:05d}_wobg.png")
                )
                torchvision.utils.save_image(
                    image_with_bg,
                    os.path.join(render_sim_path, f"{view.uid}_{f:05d}_wbg.png"),
                )
            if render_gt_path is not None:
                torchvision.utils.save_image(
                    gt_image,
                    os.path.join(render_gt_path, f"{view.uid}_{f:05d}_wobg.png"),
                )
                torchvision.utils.save_image(
                    view.gt_image.cuda(),
                    os.path.join(render_gt_path, f"{view.uid}_{f:05d}_wbg.png"),
                )

            if f < n_train_frames:
                train_psnr_list.append(psnr(image, gt_image))
                train_ssim_list.append(ssim(image, gt_image))
            else:
                test_psnr_list.append(psnr(image, gt_image))
                test_ssim_list.append(ssim(image, gt_image))

    psnr_list = train_psnr_list + test_psnr_list
    ssim_list = train_ssim_list + test_ssim_list
    if len(train_psnr_list) > 0:
        train_psnr = torch.mean(torch.stack(train_psnr_list)).item()
        train_ssim = torch.mean(torch.stack(train_ssim_list)).item()
    else:
        train_psnr = 0.0
        train_ssim = 0.0

    if len(test_psnr_list) > 0:
        test_psnr = torch.mean(torch.stack(test_psnr_list)).item()
        test_ssim = torch.mean(torch.stack(test_ssim_list)).item()
    else:
        test_psnr = 0.0
        test_ssim = 0.0

    # 3D metrics
    if target_positions is not None:
        train_cd, test_cd, cd_list = evaluate(
            x_list,
            target_positions,
            n_train_frames,
            "CD",
            seed=eval_pc_seed,
            until=eval_pc_until,
            n_samples=n_samples,
        )
        train_emd, test_emd, emd_list = evaluate(
            x_list,
            target_positions,
            n_train_frames,
            "EMD",
            seed=eval_pc_seed,
            until=eval_pc_until,
            n_samples=n_samples,
        )
    else:
        train_cd = 0.0
        test_cd = 0.0
        cd_list = []
        train_emd = 0.0
        test_emd = 0.0
        emd_list = []

    # State and Material
    if target_params is not None:
        performances.update(eval_mat_est_acc(estimator, target_params))

    cwd = os.getcwd()

    camera_ids = list(set(view.uid for view in views))
    try:
        for path in [render_sim_path, render_gt_path]:
            if path is None:
                continue

            os.chdir(path)
            for cam_id in camera_ids:
                for ext in ["gif", "mp4"]:
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-framerate",
                        "30",
                        "-pattern_type",
                        "glob",
                        "-i",
                        f"{cam_id}_*_wobg.png",
                        "-pix_fmt",
                        "yuv420p",
                        f"_cam{cam_id}_wobg.{ext}",
                    ]
                    subprocess.run(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )

                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-framerate",
                        "30",
                        "-pattern_type",
                        "glob",
                        "-i",
                        f"{cam_id}_*_wbg.png",
                        "-pix_fmt",
                        "yuv420p",
                        f"_cam{cam_id}_wbg.{ext}",
                    ]
                    subprocess.run(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
    finally:
        os.chdir(cwd)

    performances["train_psnr"] = train_psnr
    performances["train_ssim"] = train_ssim
    performances["test_psnr"] = test_psnr
    performances["test_ssim"] = test_ssim
    performances["train_cd"] = train_cd
    performances["test_cd"] = test_cd
    performances["train_emd"] = train_emd
    performances["test_emd"] = test_emd

    for i, _psnr in enumerate(psnr_list):
        performances[f"psnr{i:02d}"] = _psnr.mean().item()
    for i, _ssim in enumerate(ssim_list):
        performances[f"ssim{i:02d}"] = _ssim.item()
    for i, _cd in enumerate(cd_list):
        performances[f"cd{i:02d}"] = _cd
    for i, _emd in enumerate(emd_list):
        performances[f"emd{i:02d}"] = _emd

    return performances


def load_gt_params(path):
    # Normalize so a trailing slash doesn't shift os.path.split below
    # (e.g. "data/pacnerf/elastic/0/" -> "data/pacnerf/elastic/0").
    path = os.path.normpath(path)
    # Spring-Gaus Synthetic
    if os.path.exists(os.path.join(path, "physical.json")):
        with open(os.path.join(path, "physical.json"), "r") as f:
            params = json.load(f)
            gt_params = {
                "v": params[1]["INIT_VELOCITY"],
                "E": params[0]["E"],
                "nu": params[0]["NU"],
            }
    # GSO
    elif os.path.exists(os.path.join(path, "gt_phys_params.yaml")):
        with open(os.path.join(path, "gt_phys_params.yaml"), "r") as f:
            lines = f.readlines()
        gt_params = {
            "E": float(lines[0].split()[1]),
            "nu": float(lines[1].split()[1]),
            "v": [0.0, 0.0, 0],
        }
    # PAC-NeRF
    else:
        try:
            folder, index = os.path.split(path)
            with open(f"{folder}.json", "r") as f:
                gt_params = json.load(f)[index]
        except:
            gt_params = {}
    return gt_params


if __name__ == "__main__":
    parser = ArgumentParser(description="Physical parameter estimation")

    # Progressive joint optimization schedule. See README for full semantics.
    # `stage_codes` is a comma-separated list of stage tags (e.g. "SG,SG,SMG").
    # Each tag is a set of single-letter codes selecting which optimizers step
    # during that stage: S=state, M=material, G=all Gaussian attrs (A+X),
    # A=appearance bundle (color SH + opacity + scale + rotation), X=position.
    parser.add_argument("--stage_codes", default=None, type=str)
    parser.add_argument("--n_chunk_steps", default=None, type=int)
    parser.add_argument("--n_repeat", default=None, type=int)

    parser.add_argument("--mpm_iter_cnt", default=100, type=int)

    parser.add_argument("--eta_min", default=0.1, type=float)

    parser.add_argument("--postfix", default="", type=str)
    parser.add_argument("--cam_idxs", nargs="+")

    model_parser = ModelParams(parser)
    pipe_parser = PipelineParams(parser)
    opt_parser = OptimizationParams(parser)

    gs_args, phys_args = get_combined_args(parser)

    if gs_args.stage_codes is None:
        raise ValueError(
            "stage_codes must be specified via --stage_codes on CLI "
            "or 'stage_codes' in the config JSON 'gs' section."
        )

    model_args = model_parser.extract(gs_args)
    pipe_args = pipe_parser.extract(gs_args)
    opt_args = opt_parser.extract(gs_args)

    config_id = phys_args.id

    phys_args.mpm_iter_cnt = gs_args.mpm_iter_cnt

    phys_args.eta_min = gs_args.eta_min

    gs_args.stage_codes = gs_args.stage_codes.split(",")

    # Validate stage_codes: known letters only, and no overlapping codes
    # within a single tag (G overlaps both A and X, which would double-step).
    _VALID_LETTERS = {"S", "M", "G", "A", "X"}
    for tag in gs_args.stage_codes:
        unknown = set(tag) - _VALID_LETTERS
        if unknown:
            raise ValueError(
                f"stage_codes tag {tag!r} contains unknown letter(s) "
                f"{sorted(unknown)}. Valid letters: {sorted(_VALID_LETTERS)}."
            )
        if "G" in tag and "A" in tag:
            raise ValueError(
                f"stage_codes tag {tag!r} contains both 'G' and 'A'. "
                "'G' already includes appearance ('A'); this would step "
                "gaussians.optimizer twice. Use 'G' alone instead."
            )
        if "G" in tag and "X" in tag:
            raise ValueError(
                f"stage_codes tag {tag!r} contains both 'G' and 'X'. "
                "'G' already includes position ('X'); this would step "
                "x_optimizer twice. Use 'G' alone instead."
            )

    gs_args.n_iters = (
        len(gs_args.stage_codes) * gs_args.n_repeat * gs_args.n_chunk_steps
    )

    # Normalize --postfix: if non-empty and doesn't start with '_', prepend one
    # so output files always look like `projo4d_pred_<tag>.json`.
    if gs_args.postfix and not gs_args.postfix.startswith("_"):
        gs_args.postfix = "_" + gs_args.postfix

    # for sparse view
    if not hasattr(gs_args, "cam_idxs") or gs_args.cam_idxs is None:
        gs_args.cam_idxs = None
    else:
        gs_args.cam_idxs = list(map(int, gs_args.cam_idxs))

    # initialization
    safe_state(gs_args.quiet, None)

    # 1. train def gs: prepare scene BEFORE ti.init for pre-init render test
    model_args.model_path = os.path.abspath(model_args.model_path)
    model_args.cam_idxs = gs_args.cam_idxs

    gs_args.save_iterations.append(gs_args.iterations)
    gs_args.test_iterations.append(gs_args.iterations)
    if not check_gs_model(model_args.model_path, gs_args.save_iterations):
        training(
            model_args,
            opt_args,
            pipe_args,
            gs_args.test_iterations + list(range(10000, 40001, 1000)),
            gs_args.save_iterations,
        )
        torch.cuda.empty_cache()

    background = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    background = torch.tensor(background, dtype=torch.float32, device="cuda")

    # prepare Gaussians (outputs are on CPU to conserve GPU memory)
    gts, vol, init_opacities, _, surface_index, cam_info = prepare_gt(
        model_args, gs_args.iteration, pipe_args, phys_args
    )

    # Move back to GPU for pre-ti.init render test
    gts = [g.cuda() for g in gts]
    surface_index = surface_index.cuda()
    vol = vol.cuda()
    init_opacities = init_opacities.cuda()

    scene = assign_gs_to_pcd(
        vol,
        init_opacities,
        model_args,
        opt_args,
        pipe_args,
        None,
        phys_args.density_grid_size,
    )
    scene.gaussians.active_sh_degree = scene.gaussians.max_sh_degree

    # Now init Taichi
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)

    test_views = scene.getTestCameras(scale=model_args.res_scale, cam_idxs=None)
    print("N_CAMS: ", len(test_views))

    gt_positions = load_gt_pcds(gs_args.source_path)
    gt_params = load_gt_params(gs_args.source_path)

    gaussian_path = os.path.join(
        model_args.model_path, f"projo4d_gaussian{gs_args.postfix}.ply"
    )

    volumes = torch.full(
        (len(scene.gaussians._xyz),),
        0.125 * phys_args.voxel_size**3,
        device=scene.gaussians._xyz.device,
        dtype=torch.float32,
    )

    if os.path.exists(gaussian_path) and os.path.getsize(gaussian_path) > 0:
        print("Loading trained Gaussians.")
        scene.gaussians.load_ply(gaussian_path)

        estimator = Estimator(
            phys_args,
            "float32",
            gts=gts,
            init_vol=scene.gaussians._xyz,
            volumes=volumes,
            surface_index=surface_index,
            dynamic_scene=scene,
            image_scale=image_scale,
            pipeline=pipe_args,
            image_op=opt_args,
            cam_idxs=gs_args.cam_idxs,
            background=background,
        )
    else:
        # Train
        # Gaussian color initialization
        gaussian_color_init(scene, model_args)

        scene.gaussians.training_setup(opt_args, fix_pcd=True)

        if hasattr(phys_args, "n_frames"):
            n_frames = phys_args.n_frames
        else:
            n_frames = len(gts)
            phys_args.n_frames = n_frames
        assert n_frames > 0

        estimator = Estimator(
            phys_args,
            "float32",
            gts=gts,
            init_vol=scene.gaussians._xyz,
            volumes=volumes,
            surface_index=surface_index,
            dynamic_scene=scene,
            image_scale=image_scale,
            pipeline=pipe_args,
            image_op=opt_args,
            cam_idxs=gs_args.cam_idxs,
            background=background,
        )

        # optimize everything
        losses = iter_train(
            estimator,
            phys_args,
            gs_args.stage_codes,
            n_frames,
            n_iters=gs_args.n_iters,
            n_chunk_steps=gs_args.n_chunk_steps,
            test_views=test_views,
            gt_positions=gt_positions,
            gt_params=gt_params,
            postfix=gs_args.postfix,
            vel_iter_cnt=phys_args.vel_iter_cnt,
        )

        export_result(
            model_args,
            phys_args,
            estimator,
            losses,
            config_id,
            postfix=gs_args.postfix,
        )

    pred_file = os.path.join(
        model_args.model_path, f"projo4d_pred{gs_args.postfix}.json"
    )
    estimation_params = Namespace(
        **read_estimation_result(model_args, phys_args, pred_file=pred_file)
    )

    if not hasattr(phys_args, "n_frames"):
        phys_args.n_frames = len(gts)
    assert phys_args.n_frames > 0

    simulator = Simulator(estimation_params, scene.gaussians.get_xyz, volumes)

    model_path = os.path.abspath(model_args.model_path)
    gt_path = os.path.join(model_path, "gt_renders")
    img_path = os.path.join(model_path, f"projo4d_renders{gs_args.postfix}")

    total_frames = len(torch.unique(torch.cat([v.fid for v in test_views])))
    n_gt_positions = 0 if gt_positions is None else len(gt_positions)
    x_list = gen_xyz_list(simulator, max(total_frames, n_gt_positions), diff=False)

    performances = inference(
        estimator,
        scene.gaussians,
        x_list,
        pipe_args,
        phys_args.n_frames,
        test_views,
        gt_positions,
        gt_params,
        background=background,
        render_sim_path=img_path,
        render_gt_path=gt_path,
    )

    write_dict_to_json(
        performances, os.path.join(model_path, f"projo4d_perf{gs_args.postfix}.json")
    )
