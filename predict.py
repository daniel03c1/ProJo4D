import json
import numpy as np
import os
import subprocess
import taichi as ti
import torch
import torchvision
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import render
from new_trajectory import load_pcd_file, read_estimation_result, gen_xyz_list
from simulator import Simulator
from train_gs_fixed_pcd import train_gs_with_fixed_pcd
from utils.data_utils import load_gt_pcds, write_dict_to_json
from utils.eval import evaluate
from train_projo4d import load_gt_params
from utils.general_utils import safe_state
from utils.loss_utils import psnr, ssim
from utils.system_utils import check_gs_model


if __name__ == "__main__":
    parser = ArgumentParser(description="Prediction")
    parser.add_argument("--predict_frames", default=30, type=int)
    parser.add_argument("-cid", "--config_id", type=int, default=0)
    parser.add_argument("--cam_idxs", nargs="+")

    model = ModelParams(parser)  # , sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    gs_args, phys_args = get_combined_args(parser)
    setattr(phys_args, "config_id", gs_args.config_id)

    if not hasattr(gs_args, "cam_idxs") or gs_args.cam_idxs is None:
        gs_args.cam_idxs = None
    else:
        gs_args.cam_idxs = list(map(int, gs_args.cam_idxs))
    print(gs_args.cam_idxs)

    # 2. Load Gt
    gts = load_gt_pcds(gs_args.source_path)
    gt_params = load_gt_params(gs_args.source_path)

    safe_state(gs_args.quiet, None)
    dataset = model.extract(gs_args)
    dataset.cam_idxs = gs_args.cam_idxs
    opt = op.extract(gs_args)
    pipe = pipeline.extract(gs_args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    ti.init(arch=ti.cuda, debug=False, fast_math=True, device_memory_fraction=0.6)

    model_path = Path(os.path.abspath(dataset.model_path))
    (model_path / "gic_renders").mkdir(exist_ok=True)
    (model_path / "gt_renders").mkdir(exist_ok=True)

    # 0. Load trained pcd
    vol = load_pcd_file(dataset.model_path, gs_args.iteration)

    estimation_params = Namespace(**read_estimation_result(dataset, phys_args))
    saving_iters = gs_args.save_iterations

    if check_gs_model(dataset.model_path, saving_iters, fix_pcd=True):
        # Refined model exists; skip first simulator + gen_xyz_list + training
        d_xyz_list = None
    else:
        simulator = Simulator(estimation_params, vol)
        d_xyz_list = gen_xyz_list(
            simulator,
            max(gs_args.predict_frames, phys_args.n_frames),
            diff=True,
            save_ply=False,
            path=dataset.model_path,
        )

    scene = train_gs_with_fixed_pcd(
        vol,
        dataset,
        opt,
        pipe,
        gs_args.test_iterations + list(range(10000, 40001, 1000)),
        saving_iters,
        d_xyz_list,
        estimation_params.fps,
        force_train=False,
        grid_size=estimation_params.density_grid_size,
    )

    # For training: use filtered cameras
    train_views = scene.getTrainCameras(
        scale=dataset.res_scale, cam_idxs=gs_args.cam_idxs
    ).copy()
    eval_views = scene.getTestCameras(scale=dataset.res_scale, cam_idxs=None).copy()

    gs_args.predict_frames = len(np.unique([v.fid.item() for v in eval_views]))

    simulator = Simulator(estimation_params, vol)
    train_psnr_list = []
    test_psnr_list = []
    train_ssim_list = []
    test_ssim_list = []

    if d_xyz_list is not None:
        del d_xyz_list
        torch.cuda.empty_cache()

    # 1. Predicting trajectory
    n_gts = 0 if gts is None else len(gts)
    n_fids = len(np.unique([v.fid.item() for v in eval_views]))
    seq = gen_xyz_list(
        simulator,
        max(n_gts, n_fids),
        diff=False,
        save_ply=False,
        path=dataset.model_path,
    )

    max_f = gs_args.predict_frames
    with torch.no_grad():
        for f in range(max_f):
            xyz = seq[f]
            curr_views = [
                view
                for view in eval_views
                if torch.abs(view.fid - f / estimation_params.fps) < 1e-4
            ]
            d_xyz = xyz - scene.gaussians.get_xyz.detach()

            for view in curr_views:
                results = render(
                    view, scene.gaussians, pipeline, background, d_xyz, 0.0, 0.0, False
                )
                image = results["render"]
                image_with_bg = image + (1 - results["alpha"]) * view.background.cuda()
                gt_image = view.original_image.cuda()

                torchvision.utils.save_image(
                    image,
                    model_path / "gic_renders" / f"{view.uid}_{f:05d}_wobg.png",
                )
                torchvision.utils.save_image(
                    image_with_bg,
                    model_path / "gic_renders" / f"{view.uid}_{f:05d}_wbg.png",
                )
                torchvision.utils.save_image(
                    gt_image, model_path / "gt_renders" / f"{f:05d}_wobg.png"
                )
                torchvision.utils.save_image(
                    view.gt_image.cuda(),
                    model_path / "gt_renders" / f"{f:05d}_wbg.png",
                )

                if f < phys_args.n_frames:
                    train_psnr_list.append(psnr(image, gt_image))
                    train_ssim_list.append(ssim(image, gt_image))
                else:
                    test_psnr_list.append(psnr(image, gt_image))
                    test_ssim_list.append(ssim(image, gt_image))

    psnr_list = train_psnr_list + test_psnr_list
    ssim_list = train_ssim_list + test_ssim_list
    train_psnr = torch.mean(torch.stack(train_psnr_list)).item()
    train_ssim = torch.mean(torch.stack(train_ssim_list)).item()

    if len(test_psnr_list) > 0:
        test_psnr = torch.mean(torch.stack(test_psnr_list)).item()
        test_ssim = torch.mean(torch.stack(test_ssim_list)).item()
    else:
        test_psnr = 0.0
        test_ssim = 0.0

    # 3.evaluate CD loss & EMD loss
    if gts is not None and len(gts) > 0:
        train_cd, test_cd, cd_list = evaluate(seq, gts, phys_args.n_frames, "CD")
        train_emd, test_emd, emd_list = evaluate(seq, gts, phys_args.n_frames, "EMD")
    else:
        train_cd, test_cd, cd_list = 0.0, 0.0, []
        train_emd, test_emd, emd_list = 0.0, 0.0, []

    print(f"average psnr: {test_psnr}")
    print(f"average ssim: {test_ssim}")

    camera_ids = list(set(view.uid for view in eval_views))

    render_abs_path = (model_path / "gic_renders").resolve()
    gt_abs_path = (model_path / "gt_renders").resolve()

    os.chdir(render_abs_path)
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
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.chdir(gt_abs_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"30",
        "-i",
        "%05d.png",
        "-pix_fmt",
        "yuv420p",
        "gt.gif",
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"30",
        "-i",
        "%05d.png",
        "-pix_fmt",
        "yuv420p",
        "gt.mp4",
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    performances = {
        "train_psnr": train_psnr,
        "train_ssim": train_ssim,
        "test_psnr": test_psnr,
        "test_ssim": test_ssim,
        "train_cd": train_cd,
        "test_cd": test_cd,
        "train_emd": train_emd,
        "test_emd": test_emd,
    }

    if gt_params:
        mat = estimation_params.mat_params
        if "v" in gt_params:
            performances["MAE v"] = float(
                np.mean(
                    np.abs(np.array(estimation_params.vel) - np.array(gt_params["v"]))
                )
            )
        if "E" in gt_params and "E" in mat:
            performances["MAE log E"] = abs(
                np.log10(mat["E"]) - np.log10(gt_params["E"])
            )
        if "nu" in gt_params and "nu" in mat:
            performances["MAE nu"] = abs(mat["nu"] - gt_params["nu"])
        if "mu" in gt_params and "mu" in mat:
            performances["MAE log mu"] = abs(
                np.log10(mat["mu"]) - np.log10(gt_params["mu"])
            )
        if "kappa" in gt_params and "kappa" in mat:
            performances["MAE log kappa"] = abs(
                np.log10(mat["kappa"]) - np.log10(gt_params["kappa"])
            )
        if "ys" in gt_params and "yield_stress" in mat:
            performances["MAE log yield stress"] = abs(
                np.log10(mat["yield_stress"]) - np.log10(gt_params["ys"])
            )
        if "eta" in gt_params and "plastic_viscosity" in mat:
            performances["MAE log eta"] = abs(
                np.log10(mat["plastic_viscosity"]) - np.log10(gt_params["eta"])
            )
        if "fa" in gt_params and "friction_alpha" in mat:
            performances["MAE friction_alpha"] = abs(
                mat["friction_alpha"] - gt_params["fa"]
            )

    for i, _psnr in enumerate(psnr_list):
        performances[f"psnr{i:02d}"] = _psnr.mean().item()
    for i, _ssim in enumerate(ssim_list):
        performances[f"ssim{i:02d}"] = _ssim.item()
    for i, _cd in enumerate(cd_list):
        performances[f"cd{i:02d}"] = _cd
    for i, _emd in enumerate(emd_list):
        performances[f"emd{i:02d}"] = _emd

    write_dict_to_json(performances, os.path.join(model_path, "gic_perf.json"))
