"""Drive the trained orange 3DGS with the PyTorch AnisoMPM solver and render
video.  The Gaussians ride the simulation particles' local affine motion
(position x_p + F_p @ off0, covariance F_p Sigma0 F_p^T), so the 3DGS texture
deforms and tears exactly as the material fractures.

Scenarios
---------
crush : a plate descends and squashes the orange against the ground; radial
        fibers (alpha=-1) make it split into orange-segment wedges.
split : two grippers seize opposite sides and pull the orange apart.

Use this module's `run` for a single condition and `deform_and_render` to make
frames; `sweep.py` orchestrates side-by-side parameter-control comparisons.
"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from anisompm import AnisoMPM, halfspace_collider, sphere_collider
from render import Camera, render_gaussians, to_uint8

GRID_LIM = 1.0


# --------------------------------------------------------------------------- sim
def build_sim(data, p, scenario, device="cuda:0"):
    n_grid = p.get("n_grid", 64)
    dt = p.get("dt", 5e-4)
    sim = AnisoMPM(n_grid=n_grid, grid_lim=GRID_LIM, dt=dt,
                   gravity=(0, p.get("gravity", -4.0), 0),
                   grid_damp=p.get("grid_damp", 0.999),
                   f_clamp=p.get("f_clamp", (0.35, 2.8)),
                   damage_every=p.get("damage_every", 2), device=device)
    sim.add_object(
        data["sim_x0"], data["sim_vol"],
        rho=p.get("rho", 500), E=p["E"], nu=p.get("nu", 0.4),
        fibers=None, alpha=0.0,
        percentage=p["percentage"], l0_scale=p.get("l0_scale", 0.5),
        eta=p["eta"], zeta=1.0, residual=p.get("residual", 0.005),
        allow_damage=p.get("allow_damage", True),
    )
    if p.get("anisotropic", True):
        sim.set_radial_fibers(center=[0.5 * GRID_LIM] * 3, zero_dim=1,
                              alpha=p.get("alpha", -1.0))

    lo = data["sim_x0"].min(0).values.cpu().numpy()
    hi = data["sim_x0"].max(0).values.cpu().numpy()
    setup_scenario(sim, scenario, p, lo, hi)
    return sim, (lo, hi)


def setup_scenario(sim, scenario, p, lo, hi):
    ybot, ytop = float(lo[1]), float(hi[1])
    cx, cz = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[2] + hi[2])
    if scenario == "pull":
        # grab two caps and pull apart along x (prescribed velocity, no contact).
        # Direct tension -> reliable AnisoMPM fracture; the crack path reveals the
        # fiber anisotropy.  This is the orange analogue of the validated block test.
        W = hi[0] - lo[0]
        capf = p.get("cap_frac", 0.18)
        x = sim.x[:, 0]
        left = x < lo[0] + capf * W
        right = x > hi[0] - capf * W
        sim.allow[left] = False
        sim.allow[right] = False
        speed = p.get("pull_speed", 0.14)
        ramp = p.get("ramp", 0.35)

        def grip(s, t):
            vmag = speed * min(t / ramp, 1.0)
            s.v[left, 0] = -vmag; s.v[left, 1] = 0; s.v[left, 2] = 0
            s.v[right, 0] = vmag; s.v[right, 1] = 0; s.v[right, 2] = 0
        sim.particle_bc.append(grip)
        return
    if scenario == "squeeze":
        # compress between two clamped caps (prescribed velocity, no contact) ->
        # equatorial bulge -> hoop tension -> radial cracks into segments.
        H = ytop - ybot
        capf = p.get("cap_frac", 0.16)
        y = sim.x[:, 1]
        top = y > ytop - capf * H
        bot = y < ybot + capf * H
        sim.allow[top] = False
        sim.allow[bot] = False
        speed = p.get("squeeze_speed", 0.16)
        ramp = p.get("ramp", 0.25)

        def grip(s, t):
            vmag = speed * min(t / ramp, 1.0)
            s.v[bot] = 0.0
            s.v[top, 0] = 0.0; s.v[top, 2] = 0.0
            s.v[top, 1] = -vmag
        sim.particle_bc.append(grip)
        return
    if scenario == "crush":
        # ground (slip+friction) just below the orange
        sim.colliders.append(halfspace_collider(
            lambda t: [0, ybot + 0.5 * sim.dx, 0], [0, 1, 0], mode="slip", friction=0.6))
        speed = p.get("crush_speed", 0.18)
        h0 = ytop + 2 * sim.dx
        hmin = ybot + 0.42 * (ytop - ybot)
        sim.colliders.append(halfspace_collider(
            lambda t: [0, max(h0 - speed * t, hmin), 0], [0, -1, 0], mode="slip", friction=0.4))
    elif scenario == "split":
        sim.colliders.append(halfspace_collider(
            lambda t: [0, ybot - 3 * sim.dx, 0], [0, 1, 0], mode="slip", friction=0.4))
        speed = p.get("pull_speed", 0.12)
        r = 0.10 * (hi[0] - lo[0]) + 1.5 * sim.dx
        yc = 0.5 * (ybot + ytop)
        sim.colliders.append(sphere_collider(
            lambda t: [lo[0] + 0.10 * (hi[0] - lo[0]) - speed * t, yc, cz], r, mode="sticky"))
        sim.colliders.append(sphere_collider(
            lambda t: [hi[0] - 0.10 * (hi[0] - lo[0]) + speed * t, yc, cz], r, mode="sticky"))


# --------------------------------------------------------------- gaussian motion
def deform_and_render(sim, data, cam, p, color_override=None, opacity=None,
                      crack_tint=0.0, bg=(1, 1, 1)):
    """Carry every gaussian on its bound particle's affine motion and render."""
    idx = data["g_bind"]
    F = sim.F[idx]                                   # (G,3,3)
    xp_world = data["sim_to_world"](sim.x)[idx]      # (G,3) particle pos in world
    g_pos = xp_world + torch.bmm(F, data["g_off0"].unsqueeze(-1)).squeeze(-1)
    g_cov = F @ data["g_cov0"] @ F.transpose(1, 2)
    color = data["g_color"] if color_override is None else color_override
    if crack_tint > 0.0:                             # darken heavily-damaged gaussians
        dmg = sim.d[idx].unsqueeze(1)
        color = color * (1.0 - crack_tint * dmg)
    op = data["g_opacity"] if opacity is None else opacity
    img = render_gaussians(cam, g_pos, g_cov, color, op, bg=bg,
                           scale_modifier=1.0, r_max=p.get("r_max", 12),
                           tau=p.get("tau", None))
    del F, xp_world, g_pos, g_cov
    return img


def make_camera(json_path, device, zoom=1.5, dy=0.0):
    cam = Camera(json_path, device=device)
    cam.fx *= zoom; cam.fy *= zoom
    cam.cy += dy
    return cam


# ----------------------------------------------------------------------- runner
def run(data, cam, p, scenario, frames=48, fps=24, out_prefix=None, device="cuda:0",
        verbose=True, return_frames=True):
    sim, (lo, hi) = build_sim(data, p, scenario, device)
    n_sub = max(1, int((1.0 / fps) / sim.dt))
    imgs = []
    t0 = time.time()
    for f in range(frames):
        sim.run_frame(n_sub, f / fps)
        img = deform_and_render(sim, data, cam, p, crack_tint=p.get("crack_tint", 0.0))
        if return_frames:
            imgs.append(to_uint8(img))
        if out_prefix is not None:
            import imageio.v2 as imageio
            imageio.imwrite(f"{out_prefix}_{f:03d}.png", to_uint8(img))
        if verbose and (f % 8 == 0 or f == frames - 1):
            d = sim.d
            print(f"    frame {f:2d}/{frames}  d[mean,max]={float(d.mean()):.3f},{float(d.max()):.3f}"
                  f"  frac>0.5={float((d > 0.5).float().mean()):.3f}"
                  f"  nan={bool(torch.isnan(sim.x).any())}  ({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    return imgs
