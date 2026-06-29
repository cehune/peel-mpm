#!/usr/bin/env python3
"""peel.py -- watch anisotropic delamination, as an animation.

Two modes:
  split : two bonded layers, bond PRE-broken, pulled apart -> a gap opens (kinematic).
  peel  : the Fig.4 analog. A homogeneous FIBERED slab (grain along x). Anchor the
          bulk, start a small notch at one end of an interface plane, grip the flap
          above it and lift.  The bond fails on its OWN across the grain (weak),
          while the flap is pulled along the grain (strong) so it stays coherent --
          the strip peels off and the crack front travels.  No pre-broken plane.

    python3 src/peel.py --mode peel
    python3 src/peel.py --mode split

Output: out/slab/<mode>_strip.png (frames) + out/slab/<mode>.gif (animation).
Defaults derived: spacing=dx/2, dt=0.2 dx/sqrt(E/rho).
"""
import os, sys, math, argparse, subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from anisompm import AnisoMPM

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "slab")


def _slab(cfg, dev):
    c = np.array([0.5, 0.5, 0.5]); dx = 1.0 / cfg.grid; s = 0.5 * dx
    nx, ny, nz = (max(2, round(L / s)) for L in (cfg.Lx, cfg.Ly, cfg.Lz))
    axes = [(np.arange(n) + 0.5) * s for n in (nx, ny, nz)]
    axes = [a - a.mean() for a in axes]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    x0 = (np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) + c[None]).astype(np.float32)
    dt = 0.2 * dx / math.sqrt(cfg.E / cfg.rho)
    sim = AnisoMPM(n_grid=cfg.grid, grid_lim=1.0, dt=dt, gravity=(0, 0, 0),
                   grid_damp=0.999, f_clamp=None, damage_every=1, device=dev)
    iso = getattr(cfg, "iso", False)
    fib = None if iso else torch.tensor(np.tile(np.array([1, 0, 0], np.float32), (len(x0), 1)), device=dev)
    sim.add_object(torch.tensor(x0, device=dev), torch.full((len(x0),), s ** 3, device=dev),
                   rho=cfg.rho, E=cfg.E, nu=0.3, fibers=fib, alpha=(0.0 if iso else -1.0),
                   eta=0.01, residual=0.005, allow_damage=True)
    sim.sigma_c[:] = cfg.sigc_frac * cfg.E
    n_sub = max(1, int((1.0 / cfg.fps) / dt))
    return sim, c, s, n_sub


def build_split(cfg, dev):
    sim, c, s, n_sub = _slab(cfg, dev)
    sim.sigma_c[:] = 0.40 * cfg.E                      # strong layers; only the bond parts
    rel = sim.x[:, 1] - c[1]
    bond = rel.abs() < 0.75 * s
    sim.d[bond] = 1.0; sim.allow[bond] = False         # pre-broken bond
    top = rel > cfg.Ly / 2 - 2 * s
    bot = rel < -cfg.Ly / 2 + 2 * s
    sim.allow[top] = False; sim.allow[bot] = False
    v = cfg.gap / (cfg.frames / cfg.fps) / 2.0

    def bc(s_, t):
        r = min(t / cfg.ramp, 1.0)
        s_.v[top] = torch.tensor([0.0, +v * r, 0.0], device=sim.device, dtype=s_.dtype)
        s_.v[bot] = torch.tensor([0.0, -v * r, 0.0], device=sim.device, dtype=s_.dtype)
    sim.particle_bc.append(bc)
    return sim, (sim.x[:, 1] > c[1]).detach().cpu().numpy(), n_sub


def build_peel(cfg, dev):
    sim, c, s, n_sub = _slab(cfg, dev)
    rx = sim.x[:, 0] - c[0]; ry = sim.x[:, 1] - c[1]
    y_if = cfg.strip_off                                # interface plane (above center)
    xhi = cfg.Lx / 2
    if getattr(cfg, "bond_layer", False):
        # Make the interface the ONLY weak path: strong layers + a thin weak bond at y_if,
        # so the crack runs along the interface instead of fanning into the arm/bulk.
        # Decouples "stress that breaks the bond" from "stress that breaks the arm".
        bond = (ry - y_if).abs() < cfg.bond_width * s
        sim.sigma_c[:]    = cfg.sigc_bulk * cfg.E       # arm + bulk: survive bending
        sim.sigma_c[bond] = cfg.sigc_bond * cfg.E       # bond: the crack's only path
    notch = (ry - y_if).abs() < 0.8 * s
    notch &= rx > xhi - cfg.notch_len                  # starter crack at the right end
    sim.d[notch] = 1.0; sim.allow[notch] = False
    flap = (ry > y_if) & (rx > xhi - cfg.grip_len)     # the strip's right end
    clamp = ry < -cfg.Ly / 2 + 2 * s                   # anchor the bulk bottom
    sim.allow[flap] = False; sim.allow[clamp] = False
    v = cfg.lift

    def bc(s_, t):
        r = min(t / cfg.ramp, 1.0)
        s_.v[clamp] = torch.zeros(3, device=sim.device, dtype=s_.dtype)
        s_.v[flap] = torch.tensor([0.0, +v * r, 0.0], device=sim.device, dtype=s_.dtype)
    sim.particle_bc.append(bc)
    strip = (ry > y_if).detach().cpu().numpy()         # color: the strip that peels
    return sim, strip, n_sub


def run(cfg, dev, build_fn):
    sim, color, n_sub = build_fn(cfg, dev)
    d0 = (sim.d.detach().cpu().numpy().copy() > 0.5)   # seeded breaks (notch/bond) BEFORE peel
    xs = [sim.x.detach().cpu().numpy().copy()]
    ds = [sim.d.detach().cpu().numpy().copy()]
    for f in range(cfg.frames):
        sim.run_frame(n_sub, f / cfg.fps)
        xs.append(sim.x.detach().cpu().numpy().copy())
        ds.append(sim.d.detach().cpu().numpy().copy())
        if torch.isnan(sim.x).any():
            print("  NaN -- aborting"); break
    return xs, ds, color, d0          # <- was: return xs, ds, color


def _lims(xs):
    a = np.concatenate(xs)
    return (a[:, 0].min() - 0.03, a[:, 0].max() + 0.03, a[:, 1].min() - 0.03, a[:, 1].max() + 0.03)


def _draw(ax, x, d, color, lims):
    ax.clear()
    sl = np.abs(x[:, 2] - 0.5) < 0.5
    ax.scatter(x[sl & ~color, 0], x[sl & ~color, 1], s=12, c="#bbbbbb")     # bulk
    ax.scatter(x[sl & color, 0], x[sl & color, 1], s=12, c="#1f77b4")       # strip / top
    br = sl & (d > 0.5)
    ax.scatter(x[br, 0], x[br, 1], s=10, c="#d62728")                       # crack (broken)
    ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def make_gif(xs, ds, color, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    lims = _lims(xs); fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=110); imgs = []
    for x, d in zip(xs, ds):
        _draw(ax, x, d, color, lims); ax.set_title(title, fontsize=11)
        fig.canvas.draw(); w, h = fig.canvas.get_width_height()
        imgs.append(Image.fromarray(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()))
    plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=60, loop=0)
    print(f"[peel] {path}  ({len(imgs)} frames)")


def make_strip(xs, ds, color, path, title, n=6):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lims = _lims(xs); idx = np.linspace(0, len(xs) - 1, n).astype(int)
    fig, axs = plt.subplots(1, n, figsize=(3.0 * n, 3.2), dpi=120)
    for ax, i in zip(axs, idx):
        _draw(ax, xs[i], ds[i], color, lims); ax.set_title(f"frame {i}", fontsize=10)
    fig.suptitle(title, fontsize=12); fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"[peel] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["split", "peel"], default="peel")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--Lx", type=float, default=0.64); ap.add_argument("--Ly", type=float, default=0.18)
    ap.add_argument("--Lz", type=float, default=0.05)
    ap.add_argument("--E", type=float, default=1e4); ap.add_argument("--rho", type=float, default=1000.0)
    ap.add_argument("--sigc-frac", dest="sigc_frac", type=float, default=0.06)
    ap.add_argument("--frames", type=int, default=60); ap.add_argument("--fps", type=int, default=48)
    ap.add_argument("--ramp", type=float, default=0.15); ap.add_argument("--device", default="auto")
    ap.add_argument("--iso", action="store_true", help="isotropic (no grain) -> Fig.4-right 'small piece'")
    # split:
    ap.add_argument("--gap", type=float, default=0.22)
    # peel:
    ap.add_argument("--strip-off", dest="strip_off", type=float, default=0.03)   # interface above center
    ap.add_argument("--notch-len", dest="notch_len", type=float, default=0.14)
    ap.add_argument("--grip-len", dest="grip_len", type=float, default=0.08)
    ap.add_argument("--lift", type=float, default=0.18)
    ap.add_argument("--bond-layer", dest="bond_layer", action="store_true",
                    help="weak interface bond between strong layers (decouples bond vs arm failure)")
    ap.add_argument("--sigc-bond", dest="sigc_bond", type=float, default=0.02)
    ap.add_argument("--sigc-bulk", dest="sigc_bulk", type=float, default=0.30)
    ap.add_argument("--bond-width", dest="bond_width", type=float, default=1.0,
                    help="bond half-width in units of particle spacing s")
    ap.add_argument("--dump", action="store_true",
                    help="save per-frame npz (positions+damage+d0) for exposed-surface analysis")
    ap.add_argument("--verify", action="store_true",
                    help="run verify_exposed.py on the dump right after (implies --dump)")
    a = ap.parse_args()
    dev = "cuda:0" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    os.makedirs(OUT, exist_ok=True)
    if a.mode == "split":
        xs, ds, color, d0 = run(a, dev, build_split)
        ttl = "delamination (split): two layers pulled apart"
    else:
        xs, ds, color, d0 = run(a, dev, build_peel)
        ttl = ("peel ISOTROPIC (Fig.4 right): o grain -> tears off a small piece" if a.iso
               else "peel transverse-iso (Fig.4 middle): grain || x, a strip peels off")
    suf = "_iso" if a.iso else ""
    btag = f"_bond{a.sigc_bond}-bulk{a.sigc_bulk}" if a.bond_layer else ""
    tag = f"{a.mode}{suf}_lift{a.lift}{btag}"           # one stamp shared by every output of this run
    if a.verify:
        a.dump = True                                   # verify needs the npz
    dpath = None
    if a.dump:
        dpath = os.path.join(OUT, f"{tag}_dump.npz")
        np.savez_compressed(dpath,
            x=np.stack(xs), d=np.stack(ds), d0=d0, color=color,
            mode=a.mode, grid=a.grid, Lx=a.Lx, Ly=a.Ly, Lz=a.Lz,
            strip_off=a.strip_off, notch_len=a.notch_len, grip_len=a.grip_len,
            bond_layer=a.bond_layer, sigc_bond=a.sigc_bond, sigc_bulk=a.sigc_bulk)
        print(f"[peel] dump -> {dpath}  (x {np.stack(xs).shape})")
    make_strip(xs, ds, color, os.path.join(OUT, f"{tag}_strip.png"), ttl)
    make_gif(xs, ds, color, os.path.join(OUT, f"{tag}.gif"), ttl)
    if a.verify and dpath:
        vp = os.path.join(os.path.dirname(__file__), "verify_exposed.py")
        print(f"[peel] --verify -> verify_exposed.py {os.path.basename(dpath)}")
        subprocess.run([sys.executable, vp, dpath], check=False)


if __name__ == "__main__":
    main()
