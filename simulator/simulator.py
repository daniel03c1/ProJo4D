import numpy as np
import taichi as ti
import torch
from argparse import Namespace
from simulator import MPMSimulator


@ti.data_oriented
class Simulator:
    material_attr_names = [
        "E",
        "nu",
        "yield_stress",
        "plastic_viscosity",
        "mu",
        "kappa",
        "friction_alpha",
    ]

    def __init__(self, phys_args, vol, volumes=None, device="cuda"):
        self.device = device
        self.num_particles = ti.field(ti.i32, shape=())
        self.num_particles[None] = vol.shape[0]

        self.particle_rho = ti.field(dtype=ti.f32)

        particle = ti.root.dynamic(ti.i, 2**30, 2**14)
        particle.place(self.particle_rho)

        frame_dt = 1.0 / phys_args.fps
        dt = frame_dt / phys_args.mpm_iter_cnt

        self.sim = MPMSimulator(
            dtype=ti.f32,
            dt=dt,
            frame_dt=frame_dt,
            particle_layout=particle,
            dx=phys_args.voxel_size,
            n_particles=self.num_particles,
            args=phys_args,
            gravity=phys_args.gravity,
            material=phys_args.mat_params["material"],
            cuda_chunk_size=100,
        )
        self.vol = vol
        if volumes is None:
            volumes = torch.full_like(vol[:, 0], self.sim.max_vol[None] * 0.25)
        self.volumes = volumes
        self.phys_args = phys_args
        self.mat = phys_args.mat_params

        self.vel = torch.tensor(phys_args.vel, device=self.device)
        self.gravity = torch.tensor(phys_args.gravity, device=self.device)
        self.sim.gravity[None] = self.gravity.tolist()

        self.omega = torch.tensor(
            getattr(phys_args, "omega", [0.0, 0.0, 0.0]), device=self.device
        )
        self.rho = torch.tensor([phys_args.rho], device=self.device)

    def forward(self, f):
        xyz = torch.zeros(
            [self.sim.n_particles[None], 3],
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        if f > 0:
            self.sim.advance(f - 1)

        if not self.succeed():
            return xyz

        self.sim.get_x(f, xyz)
        return xyz

    def succeed(self):
        return self.sim.cfl_satisfy[None]

    @ti.kernel
    def from_torch(
        self,
        particles: ti.types.ndarray(),
        velocities: ti.types.ndarray(),
        particle_rho: ti.types.ndarray(),
        volumes: ti.types.ndarray(),
        particle_mu: ti.types.ndarray(),
        particle_lam: ti.types.ndarray(),
    ):
        for p in range(self.num_particles[None]):
            for d in ti.static(range(3)):
                self.sim.x[p, 0][d] = particles[p, d]
                self.sim.v[p, 0][d] = velocities[p, d]
            self.sim.C[p, 0] = ti.Matrix.zero(ti.f32, 3, 3)
            self.sim.F[p, 0] = ti.Matrix.identity(ti.f32, 3)

            self.particle_rho[p] = particle_rho[p]
            self.sim.vol[p] = volumes[p]

            self.sim.mu[p] = particle_mu[p]
            self.sim.lam[p] = particle_lam[p]

    @ti.kernel
    def compute_particle_mass(self):
        for p in range(self.num_particles[None]):
            self.sim.mass[p] = self.particle_rho[p] * self.sim.vol[p]

    def reload(self, phys_args=None):
        if phys_args:
            self.phys_args = phys_args
            self.mat = phys_args.mat_params
            # TODO load phys_args to self.sim

        self._load_particles_params()
        self._load_material_params()
        self.sim.set_colliders(phys_args)

    def _load_particles_params(self):
        self.vel = torch.tensor(self.phys_args.vel, device=self.device)
        self.rho = torch.tensor([self.phys_args.rho], device=self.device)

    def _load_material_params(self):
        for attr_name in self.material_attr_names:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

        report_msg = ""
        for attr_name, value in self.mat.items():
            report_msg += f"{attr_name}: {value} "
            # TODO: Consider mpm simulator non-newtonian material
            if attr_name == "material":
                self.material = value
                self.sim.material = value
            else:
                setattr(self, attr_name, torch.tensor([value], device=self.device))
        print("Material info: " + report_msg)

    def compute_velocities(self):
        cnt = self.vol.shape[0]
        velocities = self.vel.repeat(cnt).reshape(cnt, -1)
        centroid = self.vol.sum(dim=0) / self.vol.shape[0]
        omega = self.omega.repeat(cnt).reshape(cnt, -1)
        velocities += torch.cross(omega, self.vol - centroid)
        return velocities

    def initialize(self, phys_args=None):
        if phys_args is None:
            phys_args = self.phys_args
        self.reload(phys_args)
        cnt = self.vol.shape[0]
        velocities = self.vel.repeat(cnt).reshape(cnt, -1)

        if getattr(self, "E", None) and getattr(self, "nu", None):
            mu = self.E / (2.0 * (1.0 + self.nu))
            lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        elif getattr(self, "kappa", None) and getattr(self, "mu", None):
            mu = self.mu
            lam = self.kappa - 2.0 / 3.0 * self.mu
        else:
            print("Error: material undefined! ")
        mu = mu.repeat(cnt)
        lam = lam.repeat(cnt)
        rho = self.rho.repeat(cnt)
        self.from_torch(self.vol, velocities, rho, self.volumes, mu, lam)
        self.sim.gravity[None] = self.gravity.tolist()

        self.compute_particle_mass()

        if getattr(self, "friction_alpha", None):
            sin_phi = torch.sin(self.friction_alpha / 180 * np.pi)
            friction_alpha = np.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
            self.sim.friction_alpha[None] = friction_alpha.item()

        if getattr(self, "yield_stress", None):
            self.sim.yield_stress[None] = self.yield_stress.item()

        if getattr(self, "plastic_viscosity", None):
            self.sim.plastic_viscosity[None] = self.plastic_viscosity.item()

        self.sim.cfl_satisfy[None] = True
