#!/usr/bin/env python3
"""slab_test.py -- paper-faithful anisotropy-driven fracture on a flat slab.

This is the clean-room reproduction of AnisoMPM's CORE claim (Wolper et al. 2020,
Fig. 2 Tube Pull / Fig. 4 pork): anisotropy lives ONLY in the damage *driving
force* via a structural tensor built from a fiber field, degradation is the
standard isotropic g(d), and a crack is started with a d=1 Dirichlet seed (Eq. 3).
NO interface layer, NO directional stress split (Option B), NO toughness contrast
-- those are extensions, validated separately by the (shelved) ball gate.

Mechanism (reused from anisompm.py, no solver change):
  * driver  Phi = (A sigma+ : sigma+ A)/sigma_c^2,  A0 = I + alpha a0(x) a0(x)
  * degrade standard g(d) (interface OFF -> isotropic degradation, anisompm.py:388)
  * seed    d=1 pinned (allow=False) on a notch -> the non-local term propagates it

Built MODULAR so curvature / true-peel machinery drops in later: geometry, fiber
field, crack seed, and loading are each a pluggable function selected by config.
Add a new GEOMS/FIBERS/SEEDS/LOADS entry; nothing else changes.

    python3 src/slab_test.py run  --tag aniso --fiber uniform --fiber-deg 45
    python3 src/slab_test.py run  --tag iso   --fiber iso
    python3 src/slab_test.py plot
"""
import os, sys, math, time, argparse
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from anisompm import AnisoMPM

GRID_LIM = 1.0


# ----------------------------------------------------------------- pluggable --
def geom_slab(cfg, c):
    """Flat slab (box), jittered lattice. Returns (x0, geom-dict)."""
    s = cfg.spacing
    nx, ny, nz = (max(2, int(round(L / s))) for L in (cfg.Lx, cfg.Ly, cfg.Lz))
    axes = []
    for n in (nx, ny, nz):
        g = (np.arange(n) + 0.5) * s
        axes.append(g - g.mean())
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    rng = np.random.default_rng(cfg.seed_rng)
    pts += (rng.random(pts.shape) - 0.5) * (0.3 * s)
    return (pts + c[None]).astype(np.float32), dict(kind="slab", Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz)


def fiber_uniform(x0, cfg):
    """Uniform in-plane fiber at cfg.fiber_deg (x-y plane). Transverse-iso (alpha<0)."""
    th = math.radians(cfg.fiber_deg)
    a = np.array([math.cos(th), math.sin(th), 0.0], np.float32)
    return np.tile(a, (len(x0), 1)), cfg.alpha


def fiber_iso(x0, cfg):
    return None, 0.0


def seed_center(x0, c, geom, cfg):
    """A d=1 notch at the mid-x bottom edge -> the crack climbs in y."""
    rel = x0 - c[None]
    ymin = -geom["Ly"] / 2.0
    return (rel[:, 1] < ymin + cfg.seed_len) & (np.abs(rel[:, 0]) < cfg.seed_wid)


def seed_slit(x0, c, geom, cfg):
    """A VERTICAL pre-crack: a thin x-narrow slit rising from the bottom edge to
    cfg.seed_frac of the height. Templates a mode-I crack normal to an x-pull."""
    rel = x0 - c[None]
    ymin = -geom["Ly"] / 2.0
    return (np.abs(rel[:, 0]) < cfg.seed_wid) & (rel[:, 1] < ymin + cfg.seed_frac * geom["Ly"])


def load_pull(sim, c, geom, cfg, dev):
    """Distributed pull-apart: grip the two x-end faces, move them apart along x."""
    rel = sim.x - torch.tensor(c, device=dev, dtype=sim.dtype)
    g = cfg.grip_frac * geom["Lx"]
    lo = rel[:, 0] < -geom["Lx"] / 2 + g
    hi = rel[:, 0] > geom["Lx"] / 2 - g
    sim.allow[lo] = False; sim.allow[hi] = False
    v = float(cfg.speed)

    def bc(s, t):
        m = min(t / cfg.ramp, 1.0) * v
        s.v[lo] = torch.tensor([-m, 0.0, 0.0], device=dev, dtype=s.dtype)
        s.v[hi] = torch.tensor([+m, 0.0, 0.0], device=dev, dtype=s.dtype)
    sim.particle_bc.append(bc)
    return dict(lo=lo.cpu().numpy(), hi=hi.cpu().numpy())


GEOMS = {"slab": geom_slab}
FIBERS = {"uniform": fiber_uniform, "iso": fiber_iso}
SEEDS = {"center": seed_center, "slit": seed_slit}
LOADS = {"pull": load_pull}


# ----------------------------------------------------------------- build/run --
def pick_device(spec):
    if spec and spec != "auto":
        return spec
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build(cfg, dev):
    c = np.array([0.5, 0.5, 0.5]) * GRID_LIM
    x0, geom = GEOMS[cfg.geom](cfg, c)
    fclamp = None if str(cfg.f_clamp).lower() == "none" else tuple(float(v) for v in str(cfg.f_clamp).split(","))
    sim = AnisoMPM(n_grid=cfg.ngrid, grid_lim=GRID_LIM, dt=cfg.dt, gravity=(0, 0, 0),
                   grid_damp=0.999, f_clamp=fclamp, damage_every=1, device=dev)
    a0, alpha = FIBERS[cfg.fiber](x0, cfg)
    fibers = None if a0 is None else torch.tensor(a0, device=dev)
    x = torch.tensor(x0, device=dev)
    vol = torch.full((len(x0),), cfg.spacing ** 3, device=dev)
    sim.add_object(x, vol, rho=500, E=cfg.E, nu=0.4, fibers=fibers, alpha=alpha,
                   eta=cfg.eta, zeta=1.0, residual=0.005, allow_damage=True)
    sim.sigma_c[:] = cfg.sigc_frac * cfg.E
    seed = SEEDS[cfg.seed](x0, c, geom, cfg)
    seed_t = torch.tensor(seed, device=dev)
    sim.d[seed_t] = 1.0; sim.allow[seed_t] = False            # d=1 Dirichlet crack seed (Eq.3)
    grips = LOADS[cfg.load](sim, c, geom, cfg, dev)
    a0v = a0[0] if a0 is not None else np.array([0.0, 0.0, 0.0], np.float32)
    print(f"[slab] N={sim.n:,}  slab={geom['Lx']}x{geom['Ly']}x{geom['Lz']}  "
          f"spacing={cfg.spacing:.4f}  fiber={cfg.fiber}@{cfg.fiber_deg:g}deg(alpha={alpha})  "
          f"sigma_c={cfg.sigc_frac*cfg.E:.0f}  seed={int(seed.sum())}  "
          f"grip={int(grips['lo'].sum())+int(grips['hi'].sum())}")
    return sim, dict(c=c, geom=geom, a0=a0v, seed=seed, **grips)


def run_capture(cfg, npz_path):
    dev = pick_device(cfg.device)
    sim, info = build(cfg, dev)
    held = torch.tensor(info["seed"] | info["lo"] | info["hi"], device=dev)   # grip+seed: exclude from "crack"
    free = ~held
    n_sub = max(1, int((1.0 / cfg.fps) / cfg.dt))
    t0 = time.time()
    hist = []
    grab = lambda: (sim.x.detach().cpu().numpy().copy(), sim.d.detach().cpu().numpy().copy())
    snaps = {"start": grab()}
    onset_f = -1
    for f in range(cfg.frames):
        sim.run_frame(n_sub, f / cfg.fps)
        d = sim.d
        frac = float((d[free] > 0.5).float().mean())
        nanf = bool(torch.isnan(sim.x).any())
        hist.append((f, float(d[free].max()), frac, nanf))
        if onset_f < 0 and frac > cfg.onset_frac:           # crack just spanned -> grade HERE
            onset_f = f; snaps["onset"] = grab()
        if f % max(1, cfg.frames // 6) == 0 or f == cfg.frames - 1:
            print(f"  f{f:3d} d_free[max]={d[free].max():.3f} cracked_frac={frac:.3f} "
                  f"nan={nanf} ({time.time()-t0:.0f}s)", flush=True)
        if nanf:
            print("  ABORT non-finite -- lower dt/speed."); break
    snaps["final"] = grab()
    if "onset" not in snaps:                                # never spanned -> use final
        snaps["onset"] = snaps["final"]
    print(f"[slab] crack-span onset at frame {onset_f} (cracked_frac>{cfg.onset_frac})")
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(
        npz_path,
        meta=np.array([cfg.fiber, cfg.fiber_deg, cfg.alpha, cfg.frames, onset_f], dtype=object),
        center=info["c"], a0=info["a0"], seed=info["seed"], lo=info["lo"], hi=info["hi"],
        Lx=info["geom"]["Lx"], Ly=info["geom"]["Ly"], Lz=info["geom"]["Lz"], spacing=cfg.spacing,
        hist=np.array(hist, dtype=object),
        x_start=snaps["start"][0], d_start=snaps["start"][1],
        x_onset=snaps["onset"][0], d_onset=snaps["onset"][1],
        x_final=snaps["final"][0], d_final=snaps["final"][1])
    print(f"[slab] saved {npz_path}  ({time.time()-t0:.0f}s)")


# ----------------------------------------------------------------- plotting ---
def _crack_angle(x, d, c, held):
    """PCA orientation (deg from +y) of the damaged free particles -> crack line angle."""
    m = (d > 0.5) & (~held)
    if m.sum() < 8:
        return float("nan"), 0
    p = x[m][:, :2] - c[:2]
    p = p - p.mean(0)
    _, _, Vt = np.linalg.svd(p, full_matrices=False)
    v = Vt[0]                                              # principal crack direction
    return float(math.degrees(math.atan2(v[0], v[1]))), int(m.sum())   # angle from +y axis


def fig_compare(runs, figpath):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(runs)
    fig, axs = plt.subplots(1, n, figsize=(6.2 * n, 6), dpi=130, squeeze=False)
    axs = axs[0]
    for ax, R in zip(axs, runs):
        c = R["center"]; sp = float(R["spacing"]); held = R["seed"] | R["lo"] | R["hi"]
        x = R["x_onset"]; d = R["d_onset"]            # grade at crack-span onset, not over-pull
        ang, ncr = _crack_angle(x, d, c, held)
        ax.scatter(x[:, 0], x[:, 1], s=10, c="#dddddd", zorder=1)              # slab
        dm = d > 0.15
        sc = ax.scatter(x[dm, 0], x[dm, 1], s=16, c=d[dm], cmap="plasma", vmin=0, vmax=1, zorder=3)
        a0 = R["a0"]
        if np.linalg.norm(a0) > 0:                                            # fiber direction arrow
            L = 0.18
            ax.annotate("", xy=(c[0] + L * a0[0], c[1] + L * a0[1]),
                        xytext=(c[0] - L * a0[0], c[1] - L * a0[1]),
                        arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=2.5))
            ax.text(c[0] + L * a0[0], c[1] + L * a0[1], " fiber", color="#1f77b4", fontsize=10)
        fiber = str(R["meta"][0]); fdeg = R["meta"][1]
        ttl = (f"isotropic (alpha=0)" if fiber == "iso"
               else f"anisotropic  fiber@{fdeg:g}°")
        crack = "no clear crack" if ncr == 0 else f"crack ≈ {ang:+.0f}° from +y (n={ncr})"
        ax.set_title(f"{ttl}\n{crack}", fontsize=12)
        ax.set_aspect("equal"); ax.set_xlim(0.05, 0.95); ax.set_ylim(0.2, 0.8)
        ax.set_xlabel("x  (pull axis →)"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axs[-1], label="damage d", fraction=0.046)
    fig.suptitle("Slab pull-apart: does the crack follow the grain?  "
                 "(anisotropy only in the damage driver — paper's core claim)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(figpath, bbox_inches="tight")
    print(f"[slab] {figpath}  (crack angles printed above)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    DEF = dict(device="cpu", geom="slab", fiber="uniform", seed="slit", load="pull",
               ngrid=32, ppcd=1.5, Lx=0.44, Ly=0.28, Lz=0.10, E=2e4, sigc_frac=0.07,
               eta=0.01, dt=4e-4, fps=48, frames=28, speed=0.10, ramp=0.20, onset_frac=0.12,
               fiber_deg=45.0, alpha=-1.0, grip_frac=0.12, seed_rng=0, seed_frac=0.35,
               f_clamp="0.35,2.8", tag="run", simdir=os.path.join(os.path.dirname(__file__), "..", "out", "slab"))
    r.add_argument("--tag", default="run")
    for k, v in DEF.items():
        if k == "tag":
            continue
        r.add_argument(f"--{k.replace('_','-')}", dest=k, type=type(v), default=v)
    r.add_argument("--seed-len", dest="seed_len", type=float, default=None)
    r.add_argument("--seed-wid", dest="seed_wid", type=float, default=None)
    p = sub.add_parser("plot")
    p.add_argument("--simdir", default=os.path.join(os.path.dirname(__file__), "..", "out", "slab"))
    p.add_argument("--tags", nargs="+", default=["aniso", "iso"])
    a = ap.parse_args()

    if a.mode == "run":
        cfg = SimpleNamespace(**{k: getattr(a, k) for k in vars(a) if k not in ("mode",)})
        cfg.spacing = (GRID_LIM / cfg.ngrid) / cfg.ppcd
        if cfg.seed_len is None:
            cfg.seed_len = 2.5 * cfg.spacing
        if cfg.seed_wid is None:
            cfg.seed_wid = 1.0 * cfg.spacing            # narrow -> a thin vertical pre-crack
        run_capture(cfg, os.path.join(cfg.simdir, f"run_{a.tag}.npz"))
    else:
        runs = []
        for t in a.tags:
            fp = os.path.join(a.simdir, f"run_{t}.npz")
            if os.path.exists(fp):
                runs.append(dict(np.load(fp, allow_pickle=True)))
        if runs:
            fig_compare(runs, os.path.join(a.simdir, "crack_compare.png"))


if __name__ == "__main__":
    main()
