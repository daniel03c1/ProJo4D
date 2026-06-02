import json
import numpy as np
import os
import random
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm

from simulator import Estimator, MPMSimulator
from utils.loss_utils import earth_moving_distance


def export_result(
    dataset, phys_args, estimator: Estimator, losses, config_id, postfix=""
):
    save_attr = ["mpm_iter_cnt", "rho", "voxel_size", "bc", "fps", "density_grid_size"]
    pred = {}
    pred["config_id"] = config_id
    for attr in save_attr:
        pred[attr] = getattr(phys_args, attr)

    pred["vel"] = estimator.init_vel.detach().cpu().numpy().tolist()
    pred["gravity"] = estimator.gravity.detach().cpu().numpy().tolist()

    mat_params = dict()
    m = phys_args.material
    mat_params["material"] = m

    estimator.eval()

    if m in [MPMSimulator.elasticity, MPMSimulator.neo_hookean]:
        # elastic
        mat_params["E"] = estimator.get_E().item()
        mat_params["nu"] = estimator.get_nu().item()
    elif m == MPMSimulator.viscous_fluid:
        # newtonian fluid
        mat_params["mu"] = estimator.get_mu().item()
        mat_params["kappa"] = estimator.get_kappa().item()
    elif m == MPMSimulator.von_mises and estimator.sim.non_newtonian:
        # non Newtonian fluid
        mat_params["mu"] = estimator.get_mu().item()
        mat_params["kappa"] = estimator.get_kappa().item()
        mat_params["yield_stress"] = estimator.get_yield_stress().item()
        mat_params["plastic_viscosity"] = estimator.get_plastic_viscosity().item()
    elif m == MPMSimulator.von_mises and not estimator.sim.non_newtonian:
        # plasticine
        mat_params["E"] = estimator.get_E().item()
        mat_params["nu"] = estimator.get_nu().item()
        mat_params["yield_stress"] = estimator.get_yield_stress().item()
    elif m == MPMSimulator.drucker_prager:
        # sand
        mat_params["E"] = estimator.get_E().item()
        mat_params["nu"] = estimator.get_nu().item()
        mat_params["friction_alpha"] = estimator.get_friction_alpha().item()
    else:
        raise ValueError(f"Invalid material: {m}")

    pred["mat_params"] = mat_params

    estimator.scene.gaussians.save_ply(
        os.path.join(dataset.model_path, f"projo4d_gaussian{postfix}.ply")
    )

    with open(
        os.path.join(dataset.model_path, f"projo4d_pred{postfix}.json"), "w"
    ) as f:
        json.dump(pred, f, indent=4)


def evaluate(
    preds,
    gts,
    train_frames,
    loss_type="CD",
    seed=0,
    until=None,
    n_samples=2048,
    identical=False,
):
    print(f"Prediction sequence {len(preds)}, gts sequence {len(gts)}")

    if len(preds) != len(gts):
        print("[Error]: The prediction sequence is not align " "with the gt sequence.")
        return

    print(
        f"Prediction pcd particles cnt {preds[0].shape[0]}, "
        f"gt pcd particles cnt {gts[0].shape[0]}"
    )
    max_f = len(preds)
    fit_loss = 0.0
    predict_loss = 0.0

    losses = []

    # Evaluation only runs after training, so reseeding the global RNG here is
    # safe and keeps CD/EMD reproducible regardless of the training-time seed.
    random.seed(0 if seed is None else seed)

    for f in tqdm(range(max_f), desc=f"Evaluate {loss_type} Loss"):
        if until is not None and f > until:
            loss = 0.0
        else:
            pcd0 = preds[f]
            pcd1 = gts[f]

            # cd align with https://zlicheng.com/spring_gaus/
            n_sample = n_samples if loss_type == "EMD" else 4 * n_samples
            n_sample = min(n_sample, pcd0.shape[0], pcd1.shape[0])

            if identical:
                indices = random.sample(range(pcd0.shape[0]), n_samples)
                pcd0 = pcd0[indices, :]
                pcd1 = pcd1[indices, :]
            else:
                pcd0 = pcd0[random.sample(range(pcd0.shape[0]), n_sample), :]
                pcd1 = pcd1[random.sample(range(pcd1.shape[0]), n_sample), :]

            if loss_type == "CD":
                loss = (chamfer_distance(pcd0[None], pcd1[None])[0] * 1e3).item()
            elif loss_type == "EMD":
                loss = earth_moving_distance(pcd0, pcd1).item()
            else:
                print("[Error]: undefined error type.")

        if f < train_frames:
            fit_loss += loss
        else:
            predict_loss += loss

        losses.append(loss)

    fit_loss /= train_frames
    if max_f - train_frames > 0.0:
        predict_loss /= max_f - train_frames
    print(
        f"{loss_type} loss train: {fit_loss}, "
        f"{loss_type} loss predict: {predict_loss}"
    )

    return fit_loss, predict_loss, losses


def eval_mat_est_acc(estimator, target_params, prefix="MAE ", use_default_value=True):
    results = {}

    if "v" in target_params:
        results[f"{prefix}v"] = np.mean(
            np.abs(
                estimator.init_vel.detach().cpu().numpy() - np.array(target_params["v"])
            )
        )
    elif use_default_value:
        results[f"{prefix}v"] = 0.0
    if "E" in target_params:
        results[f"{prefix}log E"] = abs(
            np.log10(estimator.get_E().item()) - np.log10(target_params["E"])
        )
    elif use_default_value:
        results[f"{prefix}log E"] = 0.0
    if "nu" in target_params:
        results[f"{prefix}nu"] = abs(estimator.get_nu().item() - target_params["nu"])
    elif use_default_value:
        results[f"{prefix}nu"] = 0.0
    if "mu" in target_params:
        results[f"{prefix}log mu"] = abs(
            np.log10(estimator.get_mu().item()) - np.log10(target_params["mu"])
        )
    elif use_default_value:
        results[f"{prefix}log mu"] = 0.0
    if "kappa" in target_params:
        results[f"{prefix}log kappa"] = abs(
            np.log10(estimator.get_kappa().item()) - np.log10(target_params["kappa"])
        )
    elif use_default_value:
        results[f"{prefix}log kappa"] = 0.0
    if "ys" in target_params:
        results[f"{prefix}log yield stress"] = abs(
            np.log10(estimator.get_yield_stress().item())
            - np.log10(target_params["ys"])
        )
    elif use_default_value:
        results[f"{prefix}log yield stress"] = 0.0
    if "eta" in target_params:
        results[f"{prefix}log eta"] = abs(
            np.log10(estimator.get_plastic_viscosity().item())
            - np.log10(target_params["eta"])
        )
    elif use_default_value:
        results[f"{prefix}log eta"] = 0.0
    if "fa" in target_params:
        results[f"{prefix}friction_alpha"] = abs(
            estimator.get_friction_alpha().item() - target_params["fa"]
        )
    elif use_default_value:
        results[f"{prefix}friction_alpha"] = 0.0

    return results
