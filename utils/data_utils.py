import json
import numpy as np
import os
import torch
import trimesh as tm


def load_gt_pcds(path):
    if os.path.exists(path.replace("render", "simulation")) and path != path.replace(
        "render", "simulation"
    ):
        # spring gauss (synthetic)
        print("Loading synthetic Spring-Gauss Point Clouds")
        path = path.replace("render", "simulation")
    elif "gso" in path.lower():
        # GSO
        print("GSO dataset does not provide GT point clouds")
        return None
    elif os.path.exists(os.path.join(path, "all_data.json")):
        # pac-nerf (sys identification)
        print("Loading synthetic PAC-NeRF Point Clouds")
        path = path.replace("pacnerf", "simulation_data")
    else:
        return None

    gts = []
    indices = [ply.split(".")[0] for ply in os.listdir(path)]
    indices.sort()
    n_digits = len(indices[0])
    indices = [int(idx) for idx in indices]
    indices.sort()

    for idx in range(len(indices)):
        # tm.load (not tm.load_mesh): vertex-only PLYs become PointCloud, which
        # exposes all vertices; load_mesh would build a Trimesh and silently
        # drop them because no faces are present.
        pcd = tm.load(os.path.join(path, f"{idx:0{n_digits}d}.ply"))
        np_pcd = np.asarray(pcd.vertices)
        if np_pcd.size == 0:
            print(f"[load_gt_pcds] warning: {idx:0{n_digits}d}.ply has 0 vertices")
        gts.append(torch.tensor(np_pcd, dtype=torch.float32, device="cuda"))
    return gts


def write_dict_to_json(data: dict, filename: str):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
