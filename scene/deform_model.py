import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general_utils import get_expon_lr_func
from utils.time_utils import DeformNetwork, Deformable3DGaussians


class DeformModel:
    def __init__(self, args):
        self.deform = DeformNetwork(
            args.x_multires,
            args.t_multires,
            args.timenet,
            args.time_out,
            args.num_basis,
            args.num_attribute,
            args.num_coeff_set_per_basis,
            args.channel_cn,
            args.depth_cn,
            args.channel_mb,
            args.depth_mb,
            args.softmax,
        ).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time):
        return self.deform(xyz, time)

    def train_setting(self, training_args):
        l = [
            {
                "params": list(self.deform.parameters()),
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "deform",
            }
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps,
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform", f"iteration_{iteration}")

        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(
            self.deform.state_dict(), os.path.join(out_weights_path, "deform.pth")
        )

    def load_weights(self, model_path, iteration=-1):
        from utils.system_utils import searchForMaxIteration

        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(model_path, "deform"))
        loaded_iter = iteration

        weights_path = os.path.join(
            model_path, f"deform/iteration_{loaded_iter}/deform.pth"
        )

        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
