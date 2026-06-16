"""A self-contained GPU Gaussian-splatting renderer in pure PyTorch.

No CUDA extension is required (there is no nvcc on this machine), so we
implement EWA splatting (Zwicker et al. 2001) directly:

  * project each 3D gaussian to a 2D screen-space gaussian (mean + 2x2 cov)
  * for opaque scenes, composite with a two-pass *soft z-buffer*: per pixel we
    find the nearest splat depth, then blend all splats with a weight that
    decays with (depth - depth_min).  This gives a crisp front surface with
    smooth anti-aliased edges and avoids an explicit per-fragment sort.

Footprints are enumerated as ragged per-gaussian pixel windows (bounded radius)
and accumulated with scatter operations, so the whole frame is a handful of
vectorised kernels.
"""
import json
import math
import numpy as np
import torch


class Camera:
    def __init__(self, json_path=None, R=None, T=None, fx=None, fy=None,
                 width=None, height=None, device="cuda:0"):
        if json_path is not None:
            with open(json_path) as f:
                c = json.load(f)
            R = np.array(c["R"], np.float32); T = np.array(c["T"], np.float32)
            fx, fy = float(c["fx"]), float(c["fy"])
            width, height = int(c["width"]), int(c["height"])
        self.device = torch.device(device)
        self.R = torch.tensor(np.asarray(R, np.float32), device=device)   # GSCamera R
        self.T = torch.tensor(np.asarray(T, np.float32), device=device)
        self.fx, self.fy = fx, fy
        self.W, self.H = width, height
        self.cx, self.cy = width / 2.0, height / 2.0

    def world_to_cam(self, X):
        # 3DGS convention: X_cam = R^T X_world + T  ==  X_world @ R + T
        return X @ self.R + self.T


def _quat_cov_unused():
    pass


def _footprint(u0, v0, rad, u, v, ci00, ci01, ci11, z, W, H, dtype, dev):
    """Enumerate the ragged pixel footprints of a set of gaussians.
    Returns (pix linear-id, galpha, zf, gid local index)."""
    Ng = u0.shape[0]
    side = 2 * rad + 1
    counts = side * side
    total = int(counts.sum().item())
    offs = torch.zeros(Ng + 1, device=dev, dtype=torch.long)
    offs[1:] = torch.cumsum(counts, 0)
    gid = torch.repeat_interleave(torch.arange(Ng, device=dev), counts)
    k = torch.arange(total, device=dev) - offs[gid]
    sd = side[gid]
    px = u0[gid] + (k % sd) - rad[gid]
    py = v0[gid] + (k // sd) - rad[gid]
    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py, gid = px[inb], py[inb], gid[inb]
    dx = px.to(dtype) + 0.5 - u[gid]
    dy = py.to(dtype) + 0.5 - v[gid]
    power = -0.5 * (ci00[gid] * dx * dx + ci11[gid] * dy * dy) - ci01[gid] * dx * dy
    galpha = (power.exp_() if False else torch.exp(torch.clamp(power, max=0.0)))
    keep = galpha > (1.0 / 255.0)
    return px[keep] + py[keep] * W, galpha[keep], z[gid[keep]], gid[keep]


@torch.no_grad()
def render_gaussians(cam: Camera, xyz, cov3d, color, opacity,
                     bg=(1.0, 1.0, 1.0), scale_modifier=1.0, r_max=10,
                     tau=None, near=0.05, downsample=None, seed=0,
                     chunk=500_000, max_raw_radius=70):
    """Render gaussians to an (H,W,3) image in [0,1] with bounded memory.

    Footprints are processed in gaussian chunks (two passes: nearest-depth, then
    soft-z-buffer blend) so peak memory is independent of the total splat area.
    Gaussians whose un-clamped screen radius exceeds `max_raw_radius` (e.g. from
    a numerically blown-up particle) are dropped rather than painting the frame.
    """
    dev = cam.device
    dtype = torch.float32
    H, W = cam.H, cam.W
    bg_t = torch.tensor(bg, device=dev, dtype=dtype)

    if downsample is not None and downsample < xyz.shape[0]:
        g = torch.Generator(device="cpu").manual_seed(seed)
        sel = torch.randperm(xyz.shape[0], generator=g)[:downsample].to(dev)
        xyz, cov3d, color, opacity = xyz[sel], cov3d[sel], color[sel], opacity[sel]

    Xc = cam.world_to_cam(xyz)
    z = Xc[:, 2]
    front = z > near
    if not torch.any(front):
        return bg_t.expand(H, W, 3).clone()
    Xc = Xc[front]; cov3d = cov3d[front]; color = color[front]; opacity = opacity[front]
    z = Xc[:, 2]
    u = cam.fx * Xc[:, 0] / z + cam.cx
    v = cam.fy * Xc[:, 1] / z + cam.cy

    invz = 1.0 / z
    J = torch.zeros(Xc.shape[0], 2, 3, device=dev, dtype=dtype)
    J[:, 0, 0] = cam.fx * invz
    J[:, 0, 2] = -cam.fx * Xc[:, 0] * invz * invz
    J[:, 1, 1] = cam.fy * invz
    J[:, 1, 2] = -cam.fy * Xc[:, 1] * invz * invz
    Rt = cam.R.transpose(0, 1)
    cov_cam = (Rt @ cov3d @ Rt.transpose(0, 1)) * (scale_modifier ** 2)
    cov2d = J @ cov_cam @ J.transpose(1, 2)
    a = cov2d[:, 0, 0] + 0.3; b = cov2d[:, 0, 1]; d = cov2d[:, 1, 1] + 0.3
    det = a * d - b * b

    mid = 0.5 * (a + d)
    lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.0))
    raw_rad = 3.0 * torch.sqrt(torch.clamp(lam, min=1e-6))

    u0 = torch.floor(u).to(torch.long); v0 = torch.floor(v).to(torch.long)
    radc = torch.ceil(raw_rad).to(torch.long).clamp(1, r_max)
    valid = (det > 1e-9) & (raw_rad < max_raw_radius)
    valid &= (u0 + radc >= 0) & (u0 - radc < W) & (v0 + radc >= 0) & (v0 - radc < H)

    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return bg_t.expand(H, W, 3).clone()
    inv_det = 1.0 / det[idx]
    ci00 = d[idx] * inv_det; ci01 = -b[idx] * inv_det; ci11 = a[idx] * inv_det
    u, v, z = u[idx], v[idx], z[idx]
    u0, v0, radc = u0[idx], v0[idx], radc[idx]
    color, opacity = color[idx], opacity[idx]
    Ng = idx.numel()
    if tau is None:
        tau = 0.02 * float(z.median())
    inv_tau = 1.0 / max(tau, 1e-6)

    # ---- pass 1: per-pixel nearest splat depth ----
    zmin = torch.full((H * W,), float("inf"), device=dev, dtype=dtype)
    for s in range(0, Ng, chunk):
        e = min(s + chunk, Ng)
        pix, _, zf, _ = _footprint(u0[s:e], v0[s:e], radc[s:e], u[s:e], v[s:e],
                                   ci00[s:e], ci01[s:e], ci11[s:e], z[s:e], W, H, dtype, dev)
        zmin.scatter_reduce_(0, pix, zf, reduce="amin", include_self=True)

    # ---- pass 2: soft-z-buffer weighted blend ----
    Csum = torch.zeros(H * W, 3, device=dev, dtype=dtype)
    Wsum = torch.zeros(H * W, device=dev, dtype=dtype)
    logrev = torch.zeros(H * W, device=dev, dtype=dtype)
    for s in range(0, Ng, chunk):
        e = min(s + chunk, Ng)
        pix, galpha, zf, gid = _footprint(u0[s:e], v0[s:e], radc[s:e], u[s:e], v[s:e],
                                          ci00[s:e], ci01[s:e], ci11[s:e], z[s:e], W, H, dtype, dev)
        w = galpha * torch.exp(-(zf - zmin[pix]) * inv_tau)
        Csum.index_add_(0, pix, color[s:e][gid] * w.unsqueeze(1))
        Wsum.index_add_(0, pix, w)
        logrev.index_add_(0, pix, torch.log(torch.clamp(1.0 - galpha, min=1e-6)))

    cover = 1.0 - torch.exp(logrev)
    rgb = torch.where(Wsum.unsqueeze(1) > 1e-8,
                      Csum / Wsum.unsqueeze(1).clamp(min=1e-8),
                      bg_t.expand(H * W, 3))
    out = rgb * cover.unsqueeze(1) + bg_t.expand(H * W, 3) * (1.0 - cover).unsqueeze(1)
    return out.view(H, W, 3).clamp(0, 1)


def to_uint8(img):
    return (img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
