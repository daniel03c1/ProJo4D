import math
import numpy as np
import os
import random
import taichi as ti
import torch
import torch.nn as nn

from gaussian_renderer import render
from simulator import MPMSimulator
from utils.general_utils import get_expon_lr_func
from utils.loss_utils import l1_loss, ssim


def constraint(x, bound):
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return y_scale * torch.tanh(x_scale * x) + (bound[0] + y_scale)


def constraint_inv(y, bound):
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return torch.arctanh((y - (bound[0] + y_scale)) / y_scale) / x_scale


@ti.func
def point_distance(x: ti.f32, y: ti.f32):
    return (x - y).norm(eps=1e-6)


def friction_alpha_activation(friction_alpha):
    sin_phi = torch.sin(friction_alpha / 180 * np.pi)
    return np.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)


@ti.data_oriented
class Estimator(torch.nn.Module):
    velocity_stage = 0
    physical_params_stage = 1
    joint_stage = 2

    def __init__(
        self,
        phys_args,
        dtype,
        gts: list,
        init_vol,
        volumes,
        surface_index=None,
        cuda_chunk_size=100,
        dynamic_scene=None,
        image_scale=1.0,
        pipeline=None,
        image_op=None,
        cam_idxs=None,
        background=None,
    ):
        super(Estimator, self).__init__()
        self.scene = dynamic_scene
        self.cam_idxs = cam_idxs

        self.args = phys_args
        self.pipeline = pipeline
        self.image_op = image_op

        self.image_scale = image_scale
        self.init_vol = init_vol
        self.volumes = volumes
        self.gts = gts

        self.config_id = phys_args.id
        self.device = init_vol.device
        self.dtype = ti.f64 if dtype == "float64" else ti.f32
        self.frame_dt = 1.0 / phys_args.fps
        self.material = phys_args.material
        self.max_f = len(gts)

        self._setup_views()

        if background is None:
            self.background = torch.tensor(
                [0, 0, 0], dtype=torch.float32, device="cuda"
            )
        else:
            assert isinstance(background, torch.Tensor)
            self.background = background

        self.stage = ti.field(ti.int32, shape=())
        self.stage[None] = -1

        self.n_particles = ti.field(ti.i32, shape=())
        self.n_particles[None] = len(init_vol)

        # loss
        self.loss = ti.field(self.dtype, shape=(), needs_grad=True)
        self.image_loss = 0.0
        self.pos_grad_seq = []

        self.img_loss = getattr(phys_args, "img_loss", True)
        self.geo_loss = getattr(phys_args, "geo_loss", True)
        assert self.img_loss or self.geo_loss
        if self.geo_loss:
            assert len(gts) > 0

        self.w_img = torch.tensor(
            getattr(phys_args, "w_img", 0.0), device=init_vol.device
        )
        self.w_alp = torch.tensor(
            getattr(phys_args, "w_alp", 0.0), device=init_vol.device
        )
        self.w_geo = ti.field(ti.f32, shape=(), needs_grad=True)
        self.w_geo[None] = getattr(phys_args, "w_geo", 1.0)

        self.gamma = ti.field(ti.f32, shape=())
        self.g_weight = ti.field(ti.f32, shape=())
        self.set_gamma(getattr(phys_args, "gamma", 1.0))

        self._setup_3d_loss(gts, surface_index)

        # params
        self._setup_material_params(phys_args)
        self._setup_state_params(phys_args)

        self.lr_schedulers = {}

        for param_name, info in phys_args.params.items():
            if info.get("lr_decay", False):
                lr_init = info.get("init_lr", 0.1)
                lr_final = info.get("final_lr", 0.01)
                max_steps = info.get("max_steps", 60)
                self.lr_schedulers[param_name] = get_expon_lr_func(
                    lr_init=lr_init,
                    lr_final=lr_final,
                    max_steps=max_steps,
                    lr_delay_mult=0.01,
                )

        # simulation
        self.particle_rho = ti.field(dtype=self.dtype)

        particle_chunk_size = 2**14
        self.particle = ti.root.dynamic(ti.i, 2**30, particle_chunk_size)
        self.particle.place(self.particle_rho)

        self.sim = MPMSimulator(
            dtype=self.dtype,
            dt=self.frame_dt / phys_args.mpm_iter_cnt,
            frame_dt=self.frame_dt,
            n_particles=self.n_particles,
            material=phys_args.material,
            dx=phys_args.voxel_size,
            particle_layout=self.particle,
            args=phys_args,
            gravity=phys_args.gravity,
            cuda_chunk_size=cuda_chunk_size,
        )

        self._x_cache = None
        self._v_cache = None
        self._rho_cache = None
        self._vol_cache = None

        self.init_yield_stress = None
        self.init_plastic_viscosity = None
        self.init_friction_alpha = None
        self.init_cohesion = None

        self.training = True

    """    C O R E    A P I S    """

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def initialize(self, optimize_pos=False):
        torch.cuda.synchronize()
        self.sim.cached_states.clear()
        ti.sync()

        self.pos_grad_seq.clear()
        self.image_loss = 0.0
        self.loss[None] = 0.0
        cnt = self.init_vol.shape[0]

        self.sim.reset_dt()

        self.clear_grads()
        if self.material in [MPMSimulator.elasticity, MPMSimulator.neo_hookean]:
            E = self.get_E()
            nu = self.get_nu()
            self.init_mu = E / (2.0 * (1.0 + nu)).repeat(cnt)
            self.init_lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)).repeat(cnt)
        elif self.material == MPMSimulator.von_mises:
            if getattr(self.args, "init_E", None) and getattr(
                self.args, "init_nu", None
            ):
                nu = self.get_nu()
                E = self.get_E()
                mu = E / (2.0 * (1.0 + nu))
                lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                self.init_mu = mu.repeat(cnt)
                self.init_lam = lam.repeat(cnt)
            elif getattr(self.args, "kappa", None) and getattr(self.args, "mu", None):
                self.init_mu = self.get_mu().repeat(cnt)
                self.init_lam = self.get_kappa() - 2.0 / 3.0 * self.init_mu
        elif self.material == MPMSimulator.viscous_fluid:
            self.init_mu = self.get_mu().repeat(cnt)
            self.init_lam = self.get_kappa() - 2.0 / 3.0 * self.init_mu
        elif self.material == MPMSimulator.drucker_prager:
            E = self.get_E()
            nu = self.get_nu()
            self.init_mu = (E / (2.0 * (1.0 + nu))).repeat(cnt)
            self.init_lam = (E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))).repeat(cnt)

        yield_stress = self.get_yield_stress()
        eta = self.get_plastic_viscosity()
        friction_alpha = friction_alpha_activation(self.get_friction_alpha())
        cohesion = self.cohesion

        if optimize_pos:
            self._x_cache = self.init_vol * 1.0
        else:
            self._x_cache = self.init_vol.clone().requires_grad_(True)
        self._v_cache = self.init_vel.repeat(cnt).reshape(cnt, -1)

        self._rho_cache = self.global_rho.repeat(self.n_particles[None])
        # Refactor: volume is now an external per-particle tensor (computed in
        # train_dynamic.py / train_projo4d.py), not a learnable sigmoid param.
        self._vol_cache = self.volumes

        self.from_torch(
            self._x_cache,
            self._v_cache,
            self._rho_cache,
            self._vol_cache,
            self.init_mu,
            self.init_lam,
        )

        self.sim.gravity[None] = self.gravity.tolist()

        self.compute_particle_mass()

        self.sim.yield_stress[None] = yield_stress.item()
        self.sim.plastic_viscosity[None] = eta.item()
        self.sim.friction_alpha[None] = friction_alpha.item()
        self.sim.cohesion[None] = cohesion.item()
        self.sim.cfl_satisfy[None] = True

        torch.cuda.empty_cache()

    def forward(self, f, img_backward=True, random_img=False):
        xyz = torch.zeros(
            [self.n_particles[None], 3],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        if f > 0:
            self.sim.advance(f - 1)

        if not self.succeed():
            return xyz

        if self.training and self.geo_loss:
            local_index = (f * self.sim.n_substeps[None]) % self.sim.cuda_chunk_size

            self.update_match_indices_gt2sim(f, local_index)
            self.compute_loss_gt2sim(f, local_index)
            self.update_match_indices_sim2gt(f, local_index)
            self.compute_loss_sim2gt(f, local_index)

        self.sim.get_x(f, xyz)

        if self.training and self.img_loss:
            self._compute_2d_loss(f, xyz, img_backward, random_img)

        return xyz

    def backward(self, f):
        local_index = (f * self.sim.n_substeps[None]) % self.sim.cuda_chunk_size

        if self.img_loss and (self.w_img > 0.0 or self.w_alp > 0.0):
            self.set_pos_grad(f, self.pos_grad_seq[f])

        if self.geo_loss:
            self.compute_loss_gt2sim.grad(f, local_index)
            self.compute_loss_sim2gt.grad(f, local_index)

        if f > 0:
            self.sim.advance_grad(f - 1)
        else:
            self.compute_particle_mass.grad()

            kwargs = {"dtype": torch.float32, "device": self.device}

            x_grad = torch.empty([self.n_particles[None], 3], **kwargs)
            v_grad = torch.empty([self.n_particles[None], 3], **kwargs)

            mu_grad = torch.empty([self.n_particles[None]], **kwargs)
            lam_grad = torch.empty([self.n_particles[None]], **kwargs)

            yield_stress_grad = torch.empty([1], **kwargs)
            viscosity_grad = torch.empty([1], **kwargs)
            friction_alpha_grad = torch.empty([1], **kwargs)
            cohesion_grad = torch.empty([1], **kwargs)

            # Refactor: vol is no longer a learnable leaf, so no vol_grad pull.
            self.get_input_grad(x_grad, v_grad, mu_grad, lam_grad)

            assert not torch.any(torch.isnan(x_grad))
            assert not torch.any(torch.isnan(v_grad))
            assert not torch.any(torch.isnan(mu_grad))
            assert not torch.any(torch.isnan(lam_grad))

            yield_stress_grad[0] = self.sim.yield_stress.grad[None]
            friction_alpha_grad[0] = self.sim.friction_alpha.grad[None]
            viscosity_grad[0] = self.sim.plastic_viscosity.grad[None]
            cohesion_grad[0] = self.sim.cohesion.grad[None]

            assert not torch.any(torch.isnan(yield_stress_grad))
            assert not torch.any(torch.isnan(viscosity_grad))
            assert not torch.any(torch.isnan(friction_alpha_grad))
            assert not torch.any(torch.isnan(cohesion_grad))

            self._x_cache.backward(retain_graph=True, gradient=x_grad)
            self._v_cache.backward(retain_graph=True, gradient=v_grad)

            self.init_mu.backward(retain_graph=True, gradient=mu_grad)
            self.init_lam.backward(retain_graph=True, gradient=lam_grad)

            self.yield_stress.backward(retain_graph=True, gradient=yield_stress_grad)
            self.plastic_viscosity.backward(retain_graph=True, gradient=viscosity_grad)
            self.friction_alpha.backward(
                retain_graph=True, gradient=friction_alpha_grad
            )
            self.cohesion.backward(retain_graph=True, gradient=cohesion_grad)

            gravity_grad = torch.empty([3], **kwargs)
            for i in range(3):
                gravity_grad[i] = self.sim.gravity.grad[None][i]
            self.gravity.backward(gradient=gravity_grad)

    def succeed(self):
        return self.sim.cfl_satisfy[None]

    def step(self, i):
        if self.stage[None] == self.velocity_stage:
            self.state_optimizer.step()
        elif self.stage[None] == self.physical_params_stage:
            self.material_optimizer.step()
            self.update_learning_rate(i)
        elif self.stage[None] == self.joint_stage:
            self.state_optimizer.step()
            self.material_optimizer.step()
        else:
            raise ValueError()

    def zero_grad(self):
        self.state_optimizer.zero_grad()
        self.material_optimizer.zero_grad()

    def clear_grads(self):
        self.sim.clear_grads()

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.material_optimizer.param_groups:
            if param_group["name"] in self.lr_schedulers:
                f = self.lr_schedulers[param_group["name"]]
                lr = f(iteration)
                param_group["lr"] = lr

    # setters and getters
    def set_scene(self, scene):
        self.scene = scene

    def set_stage(self, stage):
        self.stage[None] = stage

    def set_gamma(self, gamma):
        self.gamma[None] = gamma
        if gamma == 1.0:
            self.g_weight[None] = 1.0
        else:
            g_sum = (1 - self.gamma[None] ** self.args.n_frames) / (
                1 - self.gamma[None]
            )
            self.g_weight[None] = self.args.n_frames / g_sum

    def get_optimizer(self):
        if self.stage[None] == self.velocity_stage:
            return self.state_optimizer
        elif self.stage[None] == self.physical_params_stage:
            return self.material_optimizer

    def get_E(self):
        return 10**self.E

    def get_nu(self):
        return constraint(self.nu, self.nu_bound)

    def get_mu(self):
        return 10**self.global_mu

    def get_kappa(self):
        return 10**self.global_kappa

    def get_yield_stress(self):
        return 10**self.yield_stress

    def get_plastic_viscosity(self):
        return 10**self.plastic_viscosity

    def get_friction_alpha(self):
        return self.friction_alpha

    def get_state_params(self):
        return {"v0": self.init_vel}

    def get_material_params(self):
        return {
            "E": self.get_E().item(),
            "nu": self.get_nu().item(),
            "mu": self.get_mu().item(),
            "kappa": self.get_kappa().item(),
            "yield_stress": self.get_yield_stress().item(),
            "plastic_viscosity": self.get_plastic_viscosity().item(),
            "friction_alpha": self.get_friction_alpha().item(),
        }

    """    I N T E R N A L    F U N C T I O N S    """
    # These shouldn't be called outside the Estimator

    def _setup_views(self):
        views = self.scene.getTrainCameras(
            scale=self.image_scale, cam_idxs=self.cam_idxs
        )
        t_ls = torch.unique(torch.stack([view.fid for view in views if view.fid >= 0]))
        t_ls, _ = torch.sort(t_ls.cpu())
        all_views = []
        for t in t_ls:
            views_by_t = [v for v in views if torch.abs(v.fid.cpu() - t) < 1e-7]
            all_views.append(views_by_t)

        self.views = all_views

    def _setup_material_params(self, phys_args):
        # material parameters
        kwargs = {"device": self.device, "dtype": torch.float32}
        self.global_rho = nn.Parameter(
            torch.tensor(phys_args.rho, **kwargs), requires_grad=False
        )

        self.E = nn.Parameter(torch.tensor(getattr(phys_args, "init_E", 0.0), **kwargs))
        self.yield_stress = nn.Parameter(
            torch.tensor([getattr(phys_args, "init_yield_stress", 0.0)], **kwargs)
        )
        self.plastic_viscosity = nn.Parameter(
            torch.tensor([getattr(phys_args, "init_plastic_viscosity", -1e6)], **kwargs)
        )
        self.friction_alpha = nn.Parameter(
            torch.tensor([getattr(phys_args, "init_friction_alpha", 0.0)], **kwargs)
        )
        self.cohesion = nn.Parameter(
            torch.tensor([getattr(phys_args, "init_cohesion", 0.0)], **kwargs)
        )

        self.global_mu = nn.Parameter(
            torch.tensor([np.log10(getattr(phys_args, "mu", 1.0))], **kwargs)
        )
        self.global_kappa = nn.Parameter(
            torch.tensor([np.log10(getattr(phys_args, "kappa", 1.0))], **kwargs)
        )

        self.nu_bound = getattr(phys_args, "nu_bound", [-0.99, 0.49])
        self.nu = nn.Parameter(
            constraint_inv(
                torch.tensor(getattr(phys_args, "init_nu", 0.0), **kwargs),
                self.nu_bound,
            )
        )

        params = []
        for param_name, info in phys_args.params.items():
            if param_name == "Youngs modulus":
                params.append(
                    {
                        "params": self.E,
                        "lr": info.get("init_lr", 0.1),
                        "name": param_name,
                    }
                )
            elif param_name == "Poisson ratio":
                params.append(
                    {
                        "params": self.nu,
                        "lr": info.get("init_lr", 0.025),
                        "name": param_name,
                    }
                )
            elif param_name == "bulk modulus":
                params.append(
                    {
                        "params": self.global_kappa,
                        "lr": info.get("init_lr", 0.1),
                        "name": param_name,
                    }
                )
            elif param_name == "shear modulus":
                params.append(
                    {
                        "params": self.global_mu,
                        "lr": info.get("init_lr", 0.1),
                        "name": param_name,
                    }
                )
            elif param_name == "Yield stress":
                params.append(
                    {
                        "params": self.yield_stress,
                        "lr": info.get("init_lr", 0.1),
                        "name": param_name,
                    }
                )
            elif param_name == "plastic viscosity":
                params.append(
                    {
                        "params": self.plastic_viscosity,
                        "lr": info.get("init_lr", 0.05),
                        "name": param_name,
                    }
                )
            elif param_name == "friction angle":
                params.append(
                    {
                        "params": self.friction_alpha,
                        "lr": info.get("init_lr", 1.0),
                        "name": param_name,
                    }
                )

        self.material_optimizer = torch.optim.Adam([*params])

    def _setup_state_params(self, phys_args):
        self.init_vel = nn.Parameter(
            torch.tensor(phys_args.init_vel, device=self.device)
        )
        self.gravity = nn.Parameter(torch.tensor(phys_args.gravity, device=self.device))

        self.state_optimizer = torch.optim.Adam(
            [
                {"params": self.init_vel, "lr": phys_args.vel_lr, "name": "velocity"},
                {
                    "params": self.gravity,
                    "lr": getattr(phys_args, "gravity_lr", 0.0),
                    "name": "gravity",
                },
            ],
        )

    def _setup_3d_loss(self, gts, surface_index):
        max_f = max(self.max_f, 1)
        surface_count = max(self._get_n_particles(gts), 1)
        surface_particles_cnt = self._get_n_particles(surface_index)

        self.n_particles_surface = ti.field(ti.i32, shape=(max_f))
        self.sim_surface_cnt = ti.field(dtype=ti.i32, shape=max_f)
        self.sim_surface_index = ti.field(
            dtype=ti.i32, shape=(max_f, surface_particles_cnt)
        )

        self.gt = ti.Vector.field(
            n=3, dtype=self.dtype, shape=(max_f, surface_count), needs_grad=True
        )

        self.match_indices_gt2sim = ti.field(ti.i32, shape=(max_f, surface_count))
        self.match_indices_sim2gt = ti.field(
            ti.i32, shape=(max_f, surface_particles_cnt)
        )

        self._load_gts(gts)
        self._load_all_sim_surfaces(surface_index, max_f)

    def _get_n_particles(self, particles):
        if particles is None:
            return self.n_particles[None]
        elif isinstance(particles, list):
            if len(particles) == 0:
                return 0
            return max([len(indices) for indices in particles])
        else:
            return len(particles)

    def _load_gts(self, gts):
        for idx in range(len(gts)):
            surface_count, _ = gts[idx].shape
            self.n_particles_surface[idx] = surface_count
            self.load_gt(idx, gts[idx], surface_count)

    def _load_all_sim_surfaces(self, surfaces, max_f):
        if surfaces is None:
            surfaces = torch.arange(self.n_particles[None], dtype=torch.int32)

        cnt = max_f
        if isinstance(surfaces, list):
            cnt = len(surfaces)

        for idx in range(cnt):
            surface = surfaces[idx] if isinstance(surfaces, list) else surfaces
            surface = surface.to(torch.int32)
            self.sim_surface_cnt[idx] = surface.shape[0]
            self.load_sim_surface(surface, surface.shape[0], idx)

    def _compute_2d_loss(self, f, xyz, backward=True, random_img=False):
        gaussians = self.scene.gaussians
        views = self.views[f]
        d_xyz = xyz - gaussians.get_xyz

        loss_img = torch.tensor(0.0, device=self.device)
        loss_alp = torch.tensor(0.0, device=self.device)

        if random_img:
            views = [random.choice(views)]

        for view in views:
            gt_image = view.original_image.cuda()
            gt_alpha_mask = view.gt_alpha_mask

            results = render(
                view, gaussians, self.pipeline, self.background, d_xyz, 0.0, 0.0, False
            )
            image = results["render"]
            alpha = results["alpha"]

            # crop image loss
            mask = torch.logical_or(gt_alpha_mask[0] > 0, alpha[0] > 0)
            ids = torch.where(mask)

            h_min, h_max = ids[0].aminmax()
            w_min, w_max = ids[1].aminmax()
            h_min, h_max = max(h_min - 50, 0), min(h_max + 50, image.shape[1])
            w_min, w_max = max(w_min - 50, 0), min(w_max + 50, image.shape[2])

            image = image[:, h_min:h_max, w_min:w_max]
            gt_image = gt_image[:, h_min:h_max, w_min:w_max]
            alpha = alpha[:, h_min:h_max, w_min:w_max]
            gt_alpha_mask = gt_alpha_mask[:, h_min:h_max, w_min:w_max]

            if self.w_img > 0.0:
                Ll1 = l1_loss(image, gt_image)
                loss_img = (
                    loss_img
                    + (1.0 - self.image_op.lambda_dssim) * Ll1
                    + self.image_op.lambda_dssim * (1.0 - ssim(image, gt_image))
                )

            if self.w_alp > 0.0:
                loss_alp = loss_alp + l1_loss(alpha, gt_alpha_mask)

        loss = (self.w_img * loss_img + self.w_alp * loss_alp) / len(views)

        loss = loss * (self.gamma[None] ** f) * self.g_weight[None]

        if backward:
            loss.backward()

        with torch.no_grad():
            self.image_loss += loss.detach().cpu()

        if self.w_img > 0.0 or self.w_alp > 0.0:
            assert not torch.any(torch.isnan(xyz.grad.clone()))
            self.pos_grad_seq.append(xyz.grad.clone())

    """    T A I C H I    K E R N E L S    """

    @ti.kernel
    def load_gt(self, f: ti.i32, gt: ti.types.ndarray(), count: ti.i32):
        for p in range(count):
            for d in ti.static(range(3)):
                self.gt[f, p][d] = gt[p, d]

    @ti.kernel
    def load_sim_surface(
        self, surface_index: ti.types.ndarray(), cnt: ti.int32, f: ti.int32
    ):
        for i in range(cnt):
            self.sim_surface_index[f, i] = surface_index[i]

    @ti.kernel
    def from_torch(
        self,
        x: ti.types.ndarray(),
        v: ti.types.ndarray(),
        rho: ti.types.ndarray(),
        vol: ti.types.ndarray(),
        mu: ti.types.ndarray(),
        lam: ti.types.ndarray(),
    ):
        for p in range(self.n_particles[None]):
            for d in ti.static(range(3)):
                self.sim.x[p, 0][d] = x[p, d]
                self.sim.v[p, 0][d] = v[p, d]
            self.sim.C[p, 0] = ti.Matrix.zero(self.dtype, 3, 3)
            self.sim.F[p, 0] = ti.Matrix.identity(self.dtype, 3)

            self.particle_rho[p] = rho[p]
            self.sim.vol[p] = vol[p]

            self.sim.mu[p] = mu[p]
            self.sim.lam[p] = lam[p]

    @ti.kernel
    def compute_particle_mass(self):
        for p in range(self.n_particles[None]):
            self.sim.mass[p] = self.particle_rho[p] * self.sim.vol[p]

    @ti.kernel
    def update_match_indices_gt2sim(self, f: ti.i32, local_index: ti.i32):
        for i in range(self.n_particles_surface[f]):
            min_value = ti.math.inf
            min_index = 0

            for j in range(self.sim_surface_cnt[f]):
                index_ = self.sim_surface_index[f, j]
                d = point_distance(self.sim.x[index_, local_index], self.gt[f, i])
                old_min_value = ti.atomic_min(min_value, d)
                if min_value != old_min_value:
                    min_index = index_

            self.match_indices_gt2sim[f, i] = min_index

    @ti.kernel
    def compute_loss_gt2sim(self, f: ti.i32, local_index: ti.i32):
        for i in range(self.n_particles_surface[f]):
            d = point_distance(
                self.sim.x[self.match_indices_gt2sim[f, i], local_index], self.gt[f, i]
            )

            self.loss[None] += (
                self.w_geo[None]
                * d
                / self.n_particles_surface[f]
                * (self.gamma[None] ** f)
                * self.g_weight[None]
            )

    @ti.kernel
    def update_match_indices_sim2gt(self, f: ti.i32, local_index: ti.i32):
        for i in range(self.sim_surface_cnt[f]):
            min_value = ti.math.inf
            min_index = 0

            for j in range(self.n_particles_surface[f]):
                d = point_distance(
                    self.sim.x[self.sim_surface_index[f, i], local_index], self.gt[f, j]
                )
                old_min_value = ti.atomic_min(min_value, d)
                if min_value != old_min_value:
                    min_index = j
            self.match_indices_sim2gt[f, i] = min_index

    @ti.kernel
    def compute_loss_sim2gt(self, f: ti.i32, local_index: ti.i32):
        for i in range(self.sim_surface_cnt[f]):
            x = self.sim.x[self.sim_surface_index[f, i], local_index]
            idx = self.match_indices_sim2gt[f, i]
            y = self.gt[f, idx]
            d = point_distance(x, y)

            self.loss[None] += (
                self.w_geo[None]
                * d
                / self.sim_surface_cnt[f]
                * (self.gamma[None] ** f)
                * self.g_weight[None]
            )

    @ti.kernel
    def set_pos_grad(self, f: ti.i32, dLdpo: ti.types.ndarray()):
        s = (f * self.sim.n_substeps[None]) % self.sim.cuda_chunk_size
        for p in range(self.n_particles[None]):
            for d in ti.static(range(3)):
                self.sim.x.grad[p, s][d] += dLdpo[p, d]

    @ti.kernel
    def get_input_grad(
        self,
        position_grad: ti.types.ndarray(),
        velocity_grad: ti.types.ndarray(),
        mu_grad: ti.types.ndarray(),
        lam_grad: ti.types.ndarray(),
    ):
        for p in range(self.n_particles[None]):
            for d in ti.static(range(3)):
                velocity_grad[p, d] = self.sim.v.grad[p, 0][d]
                position_grad[p, d] = self.sim.x.grad[p, 0][d]

            mu_grad[p] = self.sim.mu.grad[p]
            lam_grad[p] = self.sim.lam.grad[p]
