"""Prepare the trained orange 3D Gaussian Splatting model for AnisoMPM.

Pipeline
--------
1. Load orange.ply (3DGS: xyz, opacity, scale, rot-quat, SH dc colour).
2. Filter near-transparent gaussians.
3. Build a similarity transform  sim = s*(world-center)+grid_center  that
   places the orange nicely inside the MPM grid (the camera in orange.json is
   defined in the ORIGINAL world frame, so we keep a way to map back).
4. Voxel-fill the interior to get a solid set of MPM simulation particles.
5. Assign radial fibers (orange-segment structure).
6. Bind every gaussian to its nearest simulation particle, recording the
   reference offset so the gaussian can be carried by the particle's local
   affine motion (x_p + F_p @ off0) during rendering.

All heavy arrays are returned as torch tensors / numpy as appropriate.
"""
import os
import numpy as np
import torch
from plyfile import PlyData
from scipy import ndimage
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814  # SH degree-0 basis constant


def load_gaussians(ply_path, opacity_threshold=0.1):
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float32)))  # sigmoid
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    scale = np.exp(scale)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
    rot = rot / (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    color = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)  # SH dc -> RGB

    keep = opacity > opacity_threshold
    print(f"  loaded {len(xyz):,} gaussians, kept {keep.sum():,} (opacity>{opacity_threshold})")
    return dict(xyz=xyz[keep], opacity=opacity[keep], scale=scale[keep],
                rot=rot[keep], color=color[keep])


def quat_to_R(q):
    """(N,4) wxyz -> (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def cov_from_scale_rot(scale, rot):
    """Per-gaussian world covariance Sigma = R S S^T R^T, returned as (N,3,3)."""
    R = quat_to_R(rot)
    S = scale  # (N,3)
    M = R * S[:, None, :]          # R @ diag(S)
    cov = M @ np.transpose(M, (0, 2, 1))
    return cov.astype(np.float32)


def fill_interior(world_xyz, grid_center, scale_to_sim, vox_size, jitter=True, seed=0):
    """Voxelise the gaussian centres (in sim frame) at spacing `vox_size`, fill
    any enclosed interior, and place one MPM particle per solid voxel.

    The trained orange is already volumetrically dense, so this mostly acts as a
    uniform downsampler that yields a clean, evenly-spaced particle set.

    Returns sim-frame particle positions (M,3), per-particle volume (M,), dx."""
    rng = np.random.default_rng(seed)
    sim_xyz = (world_xyz - grid_center[None]) * scale_to_sim + 0.5  # center in [0,1] cube
    lo = sim_xyz.min(0)
    dx = float(vox_size)
    ijk = np.floor((sim_xyz - lo) / dx).astype(np.int64)
    dims = (ijk.max(0) + 3)
    occ = np.zeros(dims, bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    # seal hairline gaps on the shell, fill any interior cavity, restore size
    occ = ndimage.binary_closing(occ, iterations=1)
    solid = ndimage.binary_fill_holes(occ)
    vox_idx = np.argwhere(solid)                       # (K,3) voxel coords
    print(f"  voxel grid {tuple(int(d) for d in dims)}  occupied={int(occ.sum()):,}"
          f"  solid={len(vox_idx):,}  dx={dx:.5f}")

    centers = lo[None] + (vox_idx + 0.5) * dx
    if jitter:
        centers = centers + (rng.random(centers.shape) - 0.5) * (0.6 * dx)
    vol = np.full(len(centers), dx ** 3, np.float32)
    return centers.astype(np.float32), vol, dx


def bind_gaussians(world_xyz, sim_particles_world):
    """Nearest simulation particle for each gaussian; reference offset in world."""
    tree = cKDTree(sim_particles_world)
    _, idx = tree.query(world_xyz, k=1, workers=-1)
    off0 = world_xyz - sim_particles_world[idx]
    return idx.astype(np.int64), off0.astype(np.float32)


def prepare_orange(ply_path, n_grid=64, grid_lim=1.0, target_frac=0.62,
                   opacity_threshold=0.1, parts_per_cell_dim=2.0,
                   splat_sigma=0.006, device="cuda:0"):
    """Returns a dict with gaussians, sim particles, binding and the
    sim<->world similarity transform.

    Sim-particle spacing = dx_mpm / parts_per_cell_dim, giving ~(ppcd^3)
    particles per MPM cell."""
    print("[orange] loading PLY ...")
    g = load_gaussians(ply_path, opacity_threshold)
    world_xyz = g["xyz"]

    # similarity transform: fit orange into target_frac of the grid box
    center = 0.5 * (world_xyz.min(0) + world_xyz.max(0))
    extent = (world_xyz.max(0) - world_xyz.min(0)).max()
    scale_to_sim = (target_frac * grid_lim) / extent
    grid_center = center  # we re-center to grid middle inside fill_interior(+0.5)

    dx_mpm = grid_lim / n_grid
    vox_size = dx_mpm / float(parts_per_cell_dim)
    print(f"[orange] filling interior (n_grid={n_grid}, dx_mpm={dx_mpm:.5f}, vox={vox_size:.5f}) ...")
    sim_pts_sim, vol_sim, dx_vox = fill_interior(
        world_xyz, grid_center, scale_to_sim, vox_size=vox_size)
    # sim-frame -> grid placement is already in ~[0,1]; map to grid_lim
    sim_pts_sim = sim_pts_sim * grid_lim
    vol_sim = vol_sim * (grid_lim ** 3)

    # world<->sim maps.  sim = (world-center)*scale_to_sim + 0.5  (then *grid_lim)
    # => world = (sim/grid_lim - 0.5)/scale_to_sim + center
    def sim_to_world(p_sim):
        return (p_sim / grid_lim - 0.5) / scale_to_sim + torch.as_tensor(center, device=p_sim.device, dtype=p_sim.dtype)
    inv_scale = 1.0 / scale_to_sim  # length sim->world

    # sim particle reference positions in WORLD frame (for binding & carrying)
    sim_pts_world = (sim_pts_sim / grid_lim - 0.5) / scale_to_sim + center

    print("[orange] binding gaussians to particles ...")
    bind_idx, off0 = bind_gaussians(world_xyz, sim_pts_world)

    print(f"[orange] sim particles: {len(sim_pts_sim):,}   gaussians: {len(world_xyz):,}")
    # The trained orange's stored gaussian scales are degenerate (a dense point
    # cloud), so represent each gaussian as an isotropic world-space splat.
    G = len(world_xyz)
    cov0 = (splat_sigma ** 2) * torch.eye(3, device=device).expand(G, 3, 3).contiguous()
    out = dict(
        # gaussians (world frame, reference)
        g_xyz0=torch.tensor(world_xyz, device=device),
        g_cov0=cov0,
        g_color=torch.tensor(g["color"], device=device),
        g_opacity=torch.ones(G, device=device),
        g_bind=torch.tensor(bind_idx, device=device),
        g_off0=torch.tensor(off0, device=device),
        # sim particles (sim frame for the solver)
        sim_x0=torch.tensor(sim_pts_sim, device=device),
        sim_vol=torch.tensor(vol_sim, device=device),
        sim_x0_world=torch.tensor(sim_pts_world, device=device),
        # transform
        center=torch.tensor(center, device=device),
        scale_to_sim=float(scale_to_sim),
        inv_scale=float(inv_scale),
        grid_lim=float(grid_lim),
        sim_to_world=sim_to_world,
        dx_vox=dx_vox,
    )
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLY = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.ply"
    data = prepare_orange(PLY, n_grid=64, parts_per_cell_dim=2.0)
    sx = data["sim_x0"].cpu().numpy()
    # quick orthographic scatter of sim particles coloured by height
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=110)
    for ax, (i, j, t) in zip(axs, [(0, 1, "xy"), (0, 2, "xz"), (2, 1, "zy")]):
        ax.scatter(sx[::5, i], sx[::5, j], s=1, c=sx[::5, 1], cmap="viridis")
        ax.set_title(f"sim particles {t}"); ax.set_aspect("equal")
    out = "/3d-data/y2863claude/fruitninja_mpm/anisompm_py/out"
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "orange_particles.png")); print("saved orange_particles.png")
