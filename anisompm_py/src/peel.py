#!/usr/bin/env python3
"""peel.py -- watch two bonded layers DELAMINATE: top lifts, bottom drops, a gap opens.

    [====]            [====]
    [====]    ->
                       [====]

Grip the top and bottom of a slab and pull them apart in y.  The bond plane in
the middle gives way and a real GAP opens.  Output is an ANIMATED GIF (+ a frame
strip) so you can actually watch the motion over time -- not one frozen frame.

    python3 src/peel.py                 # default
    python3 src/peel.py --frames 60 --gap 0.25
"""
import os, sys, math, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from anisompm import AnisoMPM

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "slab")


def build(cfg, dev):
    c = np.array([0.5, 0.5, 0.5])
    dx = 1.0 / cfg.grid
    s = 0.5 * dx
    nx, ny, nz = (max(2, round(L / s)) for L in (cfg.Lx, cfg.Ly, cfg.Lz))
    axes = [(np.arange(n) + 0.5) * s for n in (nx, ny, nz)]
    axes = [a - a.mean() for a in axes]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    x0 = (np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) + c[None]).astype(np.float32)

    dt = 0.2 * dx / math.sqrt(cfg.E / cfg.rho)
    sim = AnisoMPM(n_grid=cfg.grid, grid_lim=1.0, dt=dt, gravity=(0, 0, 0),
                   grid_damp=0.999, f_clamp=None, damage_every=1, device=dev)
    # fibers run along x (in-plane with the layers); pulling in y is across the
    # grain -> the weak direction, so the bond parts cleanly.
    a = np.tile(np.array([1, 0, 0], np.float32), (len(x0), 1))
    sim.add_object(torch.tensor(x0, device=dev), torch.full((len(x0),), s ** 3, device=dev),
                   rho=cfg.rho, E=cfg.E, nu=0.3, fibers=torch.tensor(a, device=dev), alpha=-1.0,
                   eta=0.01, residual=0.005, allow_damage=True)
    sim.sigma_c[:] = cfg.sigc_frac * cfg.E

    rel = sim.x[:, 1] - c[1]
    # the bond: a thin plane of pre-broken (d=1) particles at mid-height
    bond = rel.abs() < 0.75 * s
    sim.d[bond] = 1.0; sim.allow[bond] = False
    # grips: pull the top band up, the bottom band down
    top = rel > cfg.Ly / 2 - 2 * s
    bot = rel < -cfg.Ly / 2 + 2 * s
    sim.allow[top] = False; sim.allow[bot] = False
    v = cfg.gap / (cfg.frames / cfg.fps) / 2.0          # each side speed -> total gap ~cfg.gap

    def bc(s_, t):
        r = min(t / cfg.ramp, 1.0)
        s_.v[top] = torch.tensor([0.0, +v * r, 0.0], device=dev, dtype=s_.dtype)
        s_.v[bot] = torch.tensor([0.0, -v * r, 0.0], device=dev, dtype=s_.dtype)
    sim.particle_bc.append(bc)
    layer = (sim.x[:, 1] > c[1]).detach().cpu().numpy()  # True = top half (for coloring)
    n_sub = max(1, int((1.0 / cfg.fps) / dt))
    return sim, layer, n_sub


def run(cfg, dev):
    sim, layer, n_sub = build(cfg, dev)
    frames = [sim.x.detach().cpu().numpy().copy()]
    for f in range(cfg.frames):
        sim.run_frame(n_sub, f / cfg.fps)
        frames.append(sim.x.detach().cpu().numpy().copy())
        if torch.isnan(sim.x).any():
            print("  NaN -- aborting"); break
    return frames, layer


def _render_frame(ax, x, layer, lims):
    ax.clear()
    sl = np.abs(x[:, 2] - 0.5) < 0.5                     # all (thin slab)
    ax.scatter(x[sl & layer, 0], x[sl & layer, 1], s=14, c="#1f77b4")        # top layer
    ax.scatter(x[sl & ~layer, 0], x[sl & ~layer, 1], s=14, c="#ff7f0e")      # bottom layer
    ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def make_gif(frames, layer, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    allf = np.concatenate(frames)
    lims = (allf[:, 0].min() - 0.03, allf[:, 0].max() + 0.03,
            allf[:, 1].min() - 0.03, allf[:, 1].max() + 0.03)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    imgs = []
    for x in frames:
        _render_frame(ax, x, layer, lims)
        ax.set_title("delamination: two bonded layers pulled apart", fontsize=11)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3]
        imgs.append(Image.fromarray(buf.copy()))
    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=60, loop=0)
    print(f"[peel] {path}  ({len(imgs)} frames)")


def make_strip(frames, layer, path, n=6):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    allf = np.concatenate(frames)
    lims = (allf[:, 0].min() - 0.03, allf[:, 0].max() + 0.03,
            allf[:, 1].min() - 0.03, allf[:, 1].max() + 0.03)
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    fig, axs = plt.subplots(1, n, figsize=(3.0 * n, 3.2), dpi=120)
    for ax, i in zip(axs, idx):
        _render_frame(ax, frames[i], layer, lims)
        ax.set_title(f"frame {i}", fontsize=10)
    fig.suptitle("delamination: top layer (blue) lifts, bottom (orange) drops — a gap opens", fontsize=12)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); print(f"[peel] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--Lx", type=float, default=0.50); ap.add_argument("--Ly", type=float, default=0.12)
    ap.add_argument("--Lz", type=float, default=0.05)
    ap.add_argument("--E", type=float, default=1e4); ap.add_argument("--rho", type=float, default=1000.0)
    ap.add_argument("--sigc-frac", dest="sigc_frac", type=float, default=0.40)  # strong layers; only the bond parts
    ap.add_argument("--frames", type=int, default=50); ap.add_argument("--fps", type=int, default=48)
    ap.add_argument("--gap", type=float, default=0.22); ap.add_argument("--ramp", type=float, default=0.15)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = "cuda:0" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    os.makedirs(OUT, exist_ok=True)
    frames, layer = run(a, dev)
    make_strip(frames, layer, os.path.join(OUT, "delam_strip.png"))
    make_gif(frames, layer, os.path.join(OUT, "delam.gif"))


if __name__ == "__main__":
    main()
