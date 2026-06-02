import json
import copy
import os
import taichi as ti
import time
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene, DeformModel
from simulator import MPMSimulator, Estimator
from train_gs import training
from train_gs_fixed_pcd import train_gs_with_fixed_pcd, assign_gs_to_pcd
from utils.general_utils import safe_state
from utils.system_utils import check_gs_model, draw_curve, write_particles


image_scale = 1.0


@torch.no_grad()
def prepare_gt(
    dataset: ModelParams, iteration: int, pipeline: PipelineParams, phys_args
):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=iteration,
        shuffle=False,
        resolution_scales=[image_scale],
    )
    deform = DeformModel(dataset)
    deform.load_weights(dataset.model_path)

    gts = []

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras(scale=image_scale, cam_idxs=None)  # dataset.cam_idxs)
    fids = torch.unique(torch.stack([view.fid for view in views]))

    xyz_canonical = gaussians.get_xyz.detach()
    opacity = gaussians.get_opacity.squeeze()

    grid_size = phys_args.density_grid_size
    density_min_th = phys_args.density_min_th
    density_max_th = phys_args.density_max_th
    num_iter = 4 if phys_args.random_sample else 5
    filling_grid_size = grid_size / 2**5
    opacity_threshold = phys_args.opacity_threshold

    for idx, fid in enumerate(tqdm(fids, desc="Filling progress")):
        if getattr(phys_args, "n_frames", None) and idx >= phys_args.n_frames:
            break

        time_input = fid.unsqueeze(0).expand(1, -1)

        d_xyz = deform.step(xyz_canonical, time_input)[0]
        xyzt = xyz_canonical + d_xyz
        xyzt = xyzt[opacity > opacity_threshold]

        bbox_mins = xyzt.amin(dim=0) - grid_size
        bbox_maxs = xyzt.amax(dim=0) + grid_size
        if getattr(phys_args, "xyz_min", None) is not None:
            scene_min = torch.tensor(
                phys_args.xyz_min, device=bbox_mins.device, dtype=bbox_mins.dtype
            )
            scene_max = torch.tensor(
                phys_args.xyz_max, device=bbox_maxs.device, dtype=bbox_maxs.dtype
            )
            center = (scene_min + scene_max) / 2
            half_extent = (scene_max - scene_min) / 2
            scene_min = center - half_extent * 2
            scene_max = center + half_extent * 2
            bbox_mins = torch.max(bbox_mins, scene_min)
            bbox_maxs = torch.min(bbox_maxs, scene_max)
        bbox_bounds = bbox_maxs - bbox_mins

        volume_size = torch.round(bbox_bounds / filling_grid_size).to(torch.int64) + 1
        grid_ids = [torch.arange(size) for size in volume_size]
        grid_coords = (
            torch.stack(torch.meshgrid(*grid_ids, indexing="ij"), dim=-1).reshape(-1, 3)
            * filling_grid_size
        )
        grid_coords = grid_coords.to(xyzt)
        init_inner_points = grid_coords + bbox_mins.reshape(1, 3)

        curr_views = scene.getTrainCameras(
            scale=image_scale, exact_fid=fid, cam_idxs=dataset.cam_idxs
        )

        for viewpoint_cam in tqdm(curr_views, desc="Rendering progress"):
            results = render(
                viewpoint_cam, gaussians, pipeline, background, d_xyz, 0.0, 0.0, False
            )
            depth = results["depth"][0]

            # TODO: why logical_and?
            render_mask = torch.logical_and(
                results["alpha"][0] > 1 / 255,
                viewpoint_cam.gt_alpha_mask[0] > 0,
            )

            # Filter init_inner_points in chunks to avoid OOM
            chunk_size = 1 << 20  # ~1M points per chunk
            survivors = []
            for ci in range(0, init_inner_points.shape[0], chunk_size):
                chunk = init_inner_points[ci : ci + chunk_size]
                pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(chunk)

                # remove points outside image space
                in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
                chunk = chunk[in_mask]
                pix_w = pix_w[in_mask]
                pix_h = pix_h[in_mask]
                pix_d = pix_d[in_mask]

                # remove points outside object mask
                pix_mask = render_mask[pix_h, pix_w]
                chunk = chunk[pix_mask]
                pix_h = pix_h[pix_mask]
                pix_w = pix_w[pix_mask]
                pix_d = pix_d[pix_mask]

                # remove points behind depth map
                render_pix_d = depth[pix_h, pix_w]
                depth_mask = render_pix_d < pix_d
                survivors.append(chunk[depth_mask])

            init_inner_points = (
                torch.cat(survivors) if survivors else init_inner_points[:0]
            )

            # remove outliers in xyzt
            render_mask = results["alpha"][0] > 1 / 255
            pix_w, pix_h, pix_d = viewpoint_cam.pw2pix(xyzt)
            in_mask = viewpoint_cam.is_in_view(pix_w, pix_h)
            pix_w, pix_h, pix_d = pix_w[in_mask], pix_h[in_mask], pix_d[in_mask]
            xyzt = xyzt[in_mask]
            pix_mask = render_mask[pix_h, pix_w]
            xyzt = xyzt[pix_mask]

        curr_grid_size = grid_size / 2
        volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
        bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
        bbox_bounds = bbox_maxs - bbox_mins
        density_volume = torch.zeros(volume_size.cpu().numpy().tolist()).to(
            init_inner_points
        )
        ids = torch.round(
            (init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size
        ).to(torch.int64)
        valid_mask = (ids >= 0).all(dim=1) & (
            ids < torch.tensor(density_volume.shape).to(ids.device)
        ).all(dim=1)
        if valid_mask.any():
            valid_ids = ids[valid_mask]
            density_volume[valid_ids.T[0], valid_ids.T[1], valid_ids.T[2]] = 1.0
        ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(
            torch.int64
        )
        valid_mask = (ids >= 0).all(dim=1) & (
            ids < torch.tensor(density_volume.shape).to(ids.device)
        ).all(dim=1)
        if valid_mask.any():
            valid_ids = ids[valid_mask]
            density_volume[valid_ids.T[0], valid_ids.T[1], valid_ids.T[2]] = 1.0
        weight = torch.ones((1, 1, 3, 3, 3)).to(xyzt)
        weight = weight / weight.sum()

        for i in range(2, num_iter):
            curr_grid_size = grid_size / 2**i
            volume_size = torch.round(bbox_bounds / curr_grid_size).to(torch.int64) + 1
            bbox_maxs = bbox_mins + (volume_size - 1) * curr_grid_size
            grid_xyz = (
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, volume_size[0] - 1, volume_size[0]),
                        torch.linspace(0, volume_size[1] - 1, volume_size[1]),
                        torch.linspace(0, volume_size[2] - 1, volume_size[2]),
                    ),
                    dim=-1,
                ).to(bbox_mins)
                * curr_grid_size
                + bbox_mins[None, None, None]
            )
            ids_norm = (grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[
                None, None, None
            ] * 2 - 1
            ids_norm = ids_norm[None].flip((-1,))
            density_volume = torch.nn.functional.grid_sample(
                density_volume[None, None],
                ids_norm,
                mode="bilinear",
                align_corners=True,
            )
            density_volume = torch.nn.functional.conv3d(
                density_volume, weight=weight, padding="same"
            )[0, 0]
            density_volume[density_volume < 0.5] = 0.0
            ids = torch.round(
                (init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size
            ).to(torch.int64)
            valid_mask = (ids >= 0).all(dim=1) & (
                ids < torch.tensor(density_volume.shape).to(ids.device)
            ).all(dim=1)
            if valid_mask.any():
                valid_ids = ids[valid_mask]
                density_volume[valid_ids.T[0], valid_ids.T[1], valid_ids.T[2]] = 1.0
            ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(
                torch.int64
            )
            valid_mask = (ids >= 0).all(dim=1) & (
                ids < torch.tensor(density_volume.shape).to(ids.device)
            ).all(dim=1)
            if valid_mask.any():
                valid_ids = ids[valid_mask]
                density_volume[valid_ids.T[0], valid_ids.T[1], valid_ids.T[2]] = 1.0
            bbox_bounds = bbox_maxs - bbox_mins
        for i in range(20):
            density_volume = torch.nn.functional.conv3d(
                density_volume[None, None], weight=weight, padding="same"
            )[0, 0]
            density_volume[density_volume < 0.5] = 0.0
            ids = torch.round(
                (init_inner_points - bbox_mins.reshape(1, 3)) / curr_grid_size
            ).to(torch.int64)
            valid_mask = (ids >= 0).all(dim=1) & (
                ids < torch.tensor(density_volume.shape).to(ids.device)
            ).all(dim=1)
            if valid_mask.any():
                valid_ids = ids[valid_mask]
                density_volume[valid_ids.T[0], valid_ids.T[1], valid_ids.T[2]] = 1.0
            ids = torch.round((xyzt - bbox_mins.reshape(1, 3)) / curr_grid_size).to(
                torch.int64
            )
            valid_mask = (ids >= 0).all(dim=1) & (
                ids < torch.tensor(density_volume.shape).to(ids.device)
            ).all(dim=1)
            if valid_mask.any():
                valid_ids = ids[valid_mask]
                density_volume[valid_ids.T[0], valid_ids.T[1], valid_ids.T[2]] = 1.0
        if phys_args.random_sample:
            density_volume = torch.nn.functional.conv3d(
                density_volume[None, None], weight=weight, padding="same"
            )[0, 0]
            half_grid_xyz = (
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(
                            0, volume_size[0] - 0.5, 2 * (volume_size[0] - 1)
                        ),
                        torch.linspace(
                            0, volume_size[1] - 0.5, 2 * (volume_size[1] - 1)
                        ),
                        torch.linspace(
                            0, volume_size[2] - 0.5, 2 * (volume_size[2] - 1)
                        ),
                    ),
                    -1,
                ).to(bbox_mins)
                * curr_grid_size
                + bbox_mins[None, None, None]
            )
            ids_norm = (half_grid_xyz - bbox_mins[None, None, None]) / bbox_bounds[
                None, None, None
            ] * 2 - 1
            ids_norm = ids_norm[None].flip((-1,))
            density_half_grid_xyz = torch.nn.functional.grid_sample(
                density_volume[None, None],
                ids_norm,
                mode="bilinear",
                align_corners=True,
            )[0, 0]
            half_grid_xyz = half_grid_xyz[density_half_grid_xyz > 0.5]
            delta = (torch.rand_like(half_grid_xyz) * curr_grid_size * 0.5).to(xyzt)
            particles = half_grid_xyz + delta
            ids_norm = (
                particles[None, None] - bbox_mins[None, None, None]
            ) / bbox_bounds[None, None, None] * 2 - 1
            ids_norm = ids_norm[None].flip((-1,))
            density_particles = torch.nn.functional.grid_sample(
                density_volume[None, None],
                ids_norm,
                mode="bilinear",
                align_corners=True,
            )[0, 0, 0, 0]
            sampled_pts = particles[density_particles > density_min_th]
            surface_pts = particles[
                (density_particles > density_min_th)
                * (density_particles < density_max_th)
            ]
            gts.append(surface_pts)
            curr_grid_size = curr_grid_size / 2
            if idx == 0:
                vol = sampled_pts
                vol_densities = density_particles[density_particles > density_min_th]
                vol_surface_mask = (
                    density_particles[density_particles > density_min_th]
                    < density_max_th
                )
                vol_surface = (
                    torch.arange(vol_surface_mask.shape[0])
                    .to(vol.device)
                    .to(torch.int64)[vol_surface_mask]
                )
                write_particles(vol, 0, dataset.model_path, "static")
        else:
            density_volume = torch.nn.functional.conv3d(
                density_volume[None, None], weight=weight, padding="same"
            )[0, 0]
            internal_mask = density_volume >= density_min_th
            sampled_pts = torch.stack(
                torch.where(internal_mask), dim=-1
            ) * curr_grid_size + bbox_mins.reshape(1, 3)
            density_volume_smoothed = torch.nn.functional.conv3d(
                density_volume[None, None], weight=weight, padding="same"
            )[0, 0]
            surface_mask = (
                (density_volume_smoothed > 0)
                * (density_volume_smoothed < density_max_th)
                * internal_mask
            )
            surface_pts = torch.stack(
                torch.where(surface_mask == 1), dim=-1
            ) * curr_grid_size + bbox_mins.reshape(1, 3)
            gts.append(surface_pts)
            if idx == 0:
                vol = sampled_pts
                vol_densities = density_volume[internal_mask]
                vol_surface_mask = surface_mask[internal_mask]
                vol_surface = (
                    torch.arange(vol_surface_mask.shape[0])
                    .to(vol.device)
                    .to(torch.int64)[vol_surface_mask]
                )
                write_particles(vol, 0, dataset.model_path, "static")

    train_cams, test_cams, cameras_extent = scene.overwrite_alphas(
        pipeline, dataset, deform
    )
    cam_info = {
        "train_cams": train_cams,
        "test_cams": test_cams,
        "cameras_extent": cameras_extent,
    }

    # Move outputs to CPU to free GPU memory before ti.init
    gts = [g.cpu() for g in gts]
    vol = vol.cpu()
    vol_densities = vol_densities.cpu()
    vol_surface = vol_surface.cpu()

    return (
        gts,
        vol,
        vol_densities,
        torch.tensor([curr_grid_size]),
        vol_surface,
        cam_info,
    )


def forward(estimator: Estimator, img_backward=True):
    dt = estimator.sim.dt_ori[None]
    while True:
        for idx in range(estimator.max_f):
            if idx == 0:
                estimator.initialize()
                estimator.sim.set_dt(dt)
            x = estimator.forward(idx, img_backward)
        if not estimator.succeed():
            dt /= 2
            print(
                "cfl condition dissatisfy, shrink dt {}, step cnt {}".format(
                    dt, estimator.sim.n_substeps[None] * 2
                )
            )
        else:
            break


def backward(estimator: Estimator):
    print(
        "Geometry loss {}, image loss {}, step {}".format(
            estimator.loss[None], estimator.image_loss, estimator.sim.n_substeps[None]
        )
    )
    max_f = estimator.max_f
    pbar = trange(max_f)
    pbar.set_description(f"[Backward]")

    estimator.loss.grad[None] = 1
    estimator.clear_grads()

    for ri in pbar:
        i = max_f - 1 - ri
        estimator.backward(i)


def train(estimator: Estimator, phys_args, max_f=None):
    losses = []
    estimated_params = []
    if estimator.stage[None] == Estimator.velocity_stage:
        iter_cnt = phys_args.vel_iter_cnt
    elif estimator.stage[None] == Estimator.physical_params_stage:
        iter_cnt = phys_args.iter_cnt

    if max_f is not None:
        estimator.max_f = max_f

    for stage, train_param in enumerate(zip([max_f], [iter_cnt])):
        max_f, iter_cnt = train_param
        if max_f is not None:
            estimator.max_f = max_f
        for i in range(iter_cnt):
            # 1. record current params
            d = {}
            param_groups = estimator.get_optimizer().param_groups
            report_msg = ""
            report_msg += f"iter {i}"
            report_msg += f"\nvelocity: {estimator.init_vel.cpu().detach().tolist()}"
            for params in param_groups:
                name = params["name"]
                p = params["params"][0].detach().cpu()

                if name == "Poisson ratio":
                    p = estimator.get_nu().detach().cpu()
                    report_msg += f"\n{name}: {p}"
                elif name in [
                    "Youngs modulus",
                    "Yield stress",
                    "plastic viscosity",
                    "shear modulus",
                    "bulk modulus",
                ]:
                    p = 10**p
                    report_msg += f"\n{name}: {p}"

                if name in ["velocity", "gravity"]:
                    d.update({name: p})
                else:
                    d.update({name: p.item()})
            print(report_msg)
            estimated_params.append(d)

            # 2. forward, backward, and update
            estimator.zero_grad()
            estimator.loss[None] = 0.0
            forward(estimator)
            losses.append(estimator.loss[None] + estimator.image_loss)
            backward(estimator)

            estimator.step(i)

            # 3. record loss and save best params
            min_idx = losses.index(min(losses))
            best_params = estimated_params[min_idx]
            print("Best params: ", best_params, "in {} iteration".format(min_idx))
            print("Min loss: {}".format(losses[min_idx]))

    if estimator.stage[None] == Estimator.velocity_stage and len(losses) > 0:
        min_idx = losses.index(min(losses))
        best_params = estimated_params[min_idx]
        estimator.init_vel = nn.Parameter(best_params["velocity"].to(estimator.device))

    return losses, estimated_params


def export_result(
    dataset, phys_args, estimator: Estimator, losses, estimated_params, config_id
):
    save_attr = ["mpm_iter_cnt", "rho", "voxel_size", "bc", "fps", "density_grid_size"]
    pred = dict()
    pred["config_id"] = config_id
    for attr in save_attr:
        pred[attr] = getattr(phys_args, attr)

    pred["vel"] = estimator.init_vel.detach().cpu().numpy().tolist()
    pred["gravity"] = estimator.gravity.detach().cpu().numpy().tolist()
    min_idx = losses.index(min(losses))
    best_params = estimated_params[min_idx]

    mat_params = dict()
    m = phys_args.material
    mat_params["material"] = m

    if (
        m == MPMSimulator.von_mises and estimator.sim.non_newtonian == 1
    ) or m == MPMSimulator.viscous_fluid:
        # non_newtonian & newtonian
        mu = best_params["shear modulus"]
        kappa = best_params["bulk modulus"]
        mat_params["mu"] = mu
        mat_params["kappa"] = kappa
    else:
        # elasticity, drucker_prager, plasticine
        if "Youngs modulus" in best_params and "Poisson ratio" in best_params:
            E = best_params["Youngs modulus"]
            nu = best_params["Poisson ratio"]
        else:
            E = float((10**estimator.E).detach().cpu().numpy())
            nu = float((estimator.get_nu()).detach().cpu().numpy())
        mat_params["E"] = E
        mat_params["nu"] = nu

    if m == MPMSimulator.drucker_prager:
        mat_params["friction_alpha"] = best_params["friction angle"]

    if m == MPMSimulator.von_mises:
        ys = best_params["Yield stress"]
        mat_params["yield_stress"] = ys
        if estimator.sim.non_newtonian == 1:
            eta = best_params["plastic viscosity"]
            mat_params["plastic_viscosity"] = eta

    pred["mat_params"] = mat_params

    with open(os.path.join(dataset.model_path, "gic_pred.json"), "w") as f:
        json.dump(pred, f, indent=4)


if __name__ == "__main__":
    # Set up command line argument parser
    start_time = time.time()

    parser = ArgumentParser(description="Physical parameter estimation")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--config_file", default="config/torus.json", type=str)
    parser.add_argument("--cam_idxs", nargs="+")

    gs_args, phys_args = get_combined_args(parser)

    config_id = phys_args.id
    print(phys_args)
    safe_state(gs_args.quiet, None)

    if not hasattr(gs_args, "cam_idxs") or gs_args.cam_idxs is None:
        gs_args.cam_idxs = None
    else:
        gs_args.cam_idxs = list(map(int, gs_args.cam_idxs))
    print(gs_args.cam_idxs)

    gs_args.save_iterations.append(gs_args.iterations)
    gs_args.test_iterations.append(gs_args.iterations)

    # 1. train def gs
    dataset = model.extract(gs_args)
    dataset.cam_idxs = gs_args.cam_idxs
    if not check_gs_model(dataset.model_path, gs_args.save_iterations):
        training(
            dataset,
            op.extract(gs_args),
            pipeline.extract(gs_args),
            gs_args.test_iterations + list(range(10000, 40001, 1000)),
            gs_args.save_iterations,
        )
    torch.cuda.empty_cache()

    # 2. estimate velocity
    gts, vol, vol_densities, grid_size, volume_surface, cam_info = prepare_gt(
        dataset, gs_args.iteration, pipeline.extract(gs_args), phys_args
    )

    torch.cuda.empty_cache()
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)

    # Move tensors back to GPU after ti.init has reserved its memory
    vol = vol.cuda()
    vol_densities = vol_densities.cuda()
    gts = [g.cuda() for g in gts]
    volume_surface = volume_surface.cuda()

    scene = assign_gs_to_pcd(
        vol,
        vol_densities,
        dataset,
        op.extract(gs_args),
        pipeline.extract(gs_args),
        cam_info,
        phys_args.density_grid_size,
    )

    volumes = torch.full(
        (len(vol),),
        0.125 * phys_args.voxel_size**3,
        device=vol.device,
        dtype=torch.float32,
    )

    estimator = Estimator(
        phys_args,
        "float32",
        gts,
        surface_index=volume_surface,
        init_vol=vol,
        volumes=volumes,
        dynamic_scene=scene,
        image_scale=image_scale,
        pipeline=pipeline.extract(gs_args),
        image_op=op.extract(gs_args),
        cam_idxs=dataset.cam_idxs,
    )
    estimator.set_stage(Estimator.velocity_stage)
    estimator.img_loss = False

    print("gt point count: {}".format(gts[0].shape[0]))

    losses, e_s = train(estimator, phys_args, phys_args.vel_estimation_frames)
    torch.cuda.empty_cache()

    # 3. estimate physical parameters
    max_f = len(gts)
    estimator.set_stage(Estimator.physical_params_stage)
    estimator.img_loss = True
    losses, e_s = train(estimator, phys_args, max_f)

    print(phys_args)
    print(estimator.init_vel)
    print(config_id)

    export_result(dataset, phys_args, estimator, losses, e_s, config_id)
    print("consume time {}".format(time.time() - start_time))
