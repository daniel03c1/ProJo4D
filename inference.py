import lpips
import numpy as np
import os
import random
import torch
import torchvision
import scipy
import tqdm
from pytorch3d.loss import chamfer_distance

from gaussian_renderer import render
from simulator import Simulator
from utils.image_utils import psnr
from utils.loss_utils import ssim, earth_moving_distance
from utils.time_utils import DeformNetwork
from new_trajectory import gen_xyz_list


LPIPS_FN = lpips.LPIPS(net="alex").cuda()


def inference(
    estimation_params_or_simulator,
    gaussians,
    pipeline_params,
    views,
    n_train_frames,
    target_positions=None,
    render_cam_id=None,
    background=None,
    render_sim_path=None,
    render_gt_path=None,
    return_positions=False,
    deform=None,
):
    # 1. Preparation
    # 1.1. Simulator
    simulator = estimation_params_or_simulator

    # 1.2. Metrics
    metrics = {
        "train_psnr": 0.0,
        "train_ssim": 0.0,
        "train_lpips": 0.0,
        "train_cd": 0.0,
        "train_emd": 0.0,
        "train_frames": [],
        "test_psnr": 0.0,
        "test_ssim": 0.0,
        "test_lpips": 0.0,
        "test_cd": 0.0,
        "test_emd": 0.0,
        "test_frames": [],
    }

    # 1.3. PATH
    if render_sim_path is not None:
        os.makedirs(render_sim_path, exist_ok=True)
    if render_gt_path is not None:
        os.makedirs(render_gt_path, exist_ok=True)

    # 1.4. Others
    positions = []
    if background is None:
        background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    # 4. Simulation
    simulator.initialize()

    if deform is not None:
        zero_token = torch.zeros(1, 1).to(gaussians.get_xyz)

    views = sorted(views, key=lambda x: x.fid)
    frames = torch.unique(torch.cat([v.fid for v in views]))

    # assume there are no missing frames
    f = -1
    xyz = None
    d_xyz = None
    point_losses = {}

    dx = gen_xyz_list(simulator, len(frames), diff=True)

    for view in tqdm.tqdm(views):
        new_fid = torch.searchsorted(frames, view.fid).item()

        if new_fid > f:
            f = new_fid  # this is index
            t = frames[new_fid]

            if deform is None:
                d_xyz = dx[f]
                d_xyz = gaussians.get_xyz + d_xyz
            else:
                d_xyz = deform.step(gaussians.get_xyz, t + zero_token)[0]
                xyz = gaussians.get_xyz + d_xyz

            if return_positions:
                positions.append(xyz.cuda())

            if target_positions is not None:
                # Chamfer distance and Earth Moving Distance
                n_sample = min(8192, xyz.shape[0], target_positions[f].shape[0])
                pcd0 = xyz[random.sample(range(xyz.shape[0]), n_sample), :]
                pcd1 = target_positions[f][
                    random.sample(range(target_positions[f].shape[0]), n_sample), :
                ]

                curr_cd = (chamfer_distance(pcd0[None], pcd1[None])[0] * 1e3).item()

                n_sample = min(2048, xyz.shape[0], target_positions[f].shape[0])
                pcd0 = xyz[random.sample(range(xyz.shape[0]), n_sample), :]
                pcd1 = target_positions[f][
                    random.sample(range(target_positions[f].shape[0]), n_sample), :
                ]
                curr_emd = earth_moving_distance(pcd0, pcd1).item()

                point_losses = {"cd": float(curr_cd), "emd": float(curr_emd)}

        results = render(
            view, gaussians, pipeline_params, background, d_xyz, 0.0, 0.0, False
        )
        image = results["render"]

        # save images
        if render_cam_id is None:
            img_name = f"{view.uid}_{f:05d}.png"
        elif view.uid == render_cam_id:
            img_name = f"{f:05d}.png"
        else:
            img_name = None

        if render_sim_path and img_name:
            torchvision.utils.save_image(image, os.path.join(render_sim_path, img_name))

        results = {"frame": f}

        if hasattr(view, "original_image") and view.original_image is not None:
            gt_image = view.original_image.cuda()

            # saving image
            if render_gt_path and img_name:
                torchvision.utils.save_image(
                    gt_image, os.path.join(render_gt_path, img_name)
                )

            # evaluation
            curr_psnr = psnr(image, gt_image).mean().item()
            curr_ssim = ssim(image, gt_image).mean().item()
            curr_lpips = LPIPS_FN(image * 2 - 1, gt_image * 2 - 1).mean().item()

            results.update(
                {
                    "psnr": float(curr_psnr),
                    "ssim": float(curr_ssim),
                    "lpips": float(curr_lpips),
                }
            )

        results.update(point_losses)

        if f < n_train_frames:
            key = "train_frames"
        else:
            key = "test_frames"
        metrics[key].append(results)

    # Calculate mean metrics
    if metrics["train_frames"]:
        metrics["train_psnr"] = np.mean([f["psnr"] for f in metrics["train_frames"]])
        metrics["train_ssim"] = np.mean([f["ssim"] for f in metrics["train_frames"]])
        metrics["train_lpips"] = np.mean([f["lpips"] for f in metrics["train_frames"]])
        metrics["train_cd"] = np.mean([f["cd"] for f in metrics["train_frames"]])
        metrics["train_emd"] = np.mean([f["emd"] for f in metrics["train_frames"]])
    if metrics["test_frames"]:
        metrics["test_psnr"] = np.mean([f["psnr"] for f in metrics["test_frames"]])
        metrics["test_ssim"] = np.mean([f["ssim"] for f in metrics["test_frames"]])
        metrics["test_lpips"] = np.mean([f["lpips"] for f in metrics["test_frames"]])
        metrics["test_cd"] = np.mean([f["cd"] for f in metrics["test_frames"]])
        metrics["test_emd"] = np.mean([f["emd"] for f in metrics["test_frames"]])

    simulator.sim.set_dt(simulator.sim.dt_ori[None])

    if return_positions:
        return metrics, positions
    return metrics
