#!/usr/bin/env python3
"""slab.py -- the simplest valid test of AnisoMPM's core claim.

Anisotropic damage makes a material DIRECTIONALLY TOUGH: with the fiber
structural tensor A0 = I + alpha a(x)a(x) (alpha<0), the damage driver is
suppressed for tension along the fiber.  So a bar is STRONG pulled along the
grain and WEAK across it.  This is a uniaxial tension test:

    python3 src/slab.py one  --deg 0                 # one stress-strain curve
    python3 src/slab.py sweep                         # x, two-ended (default)
    python3 src/slab.py sweep --axis y --one-ended    # vertical, pulled from one end
    python3 src/slab.py render --mode y1              # particle visuals

Loading modes (the bar is always long along the PULL axis; fiber angle is
measured FROM the pull axis, so deg 0 = along grain in every mode):
    --axis x|y        pull/long axis
    --one-ended       clamp the low end, pull only the high end (else pull apart)
The mode tag is <axis><1|2>, e.g. x2 (x, two-ended) or y1 (y, one-ended).

Every default is DERIVED, not tuned:
    spacing = dx/2                       2 particles/cell (standard MLS-MPM)
    dt      = 0.2 * dx / sqrt(E/rho)     CFL with ~5x margin
    sigma_c = 0.06 * E                   => isotropic failure near ~6% strain
    rate    = strain rate; pulled to ~16% strain over `frames`
"""
import os, sys, math, argparse
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from anisompm import AnisoMPM

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "slab")
RESID = 0.005


def make_cfg(**over):
    c = SimpleNamespace(
        device="cpu", grid=28, L=0.50, W=0.10, E=1.0e4, nu=0.3, rho=1000.0,
        sigc_frac=0.06, alpha=-1.0, eta=0.01, fps=48, frames=20, rate=0.5, ramp=0.10,
        deg=0.0, iso=False, axis="x", one_ended=False)
    for k, v in over.items():
        setattr(c, k, v)
    c.dx = 1.0 / c.grid
    c.spacing = 0.5 * c.dx
    c.dt = 0.2 * c.dx / math.sqrt(c.E / c.rho)
    return c


def mode_tag(cfg):
    return f"{cfg.axis}{'1' if cfg.one_ended else '2'}"


def build(cfg, dev):
    c = np.array([0.5, 0.5, 0.5])
    s = cfg.spacing
    axis = 0 if cfg.axis == "x" else 1          # pull/long axis
    trans = 1 - axis                            # in-plane transverse (fiber tilt + crack opens here)
    sizes = [cfg.W, cfg.W, cfg.W]; sizes[axis] = cfg.L
    ns = [max(2, round(sz / s)) for sz in sizes]
    axes = [(np.arange(n) + 0.5) * s for n in ns]
    axes = [a - a.mean() for a in axes]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    x0 = (np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) + c[None]).astype(np.float32)

    sim = AnisoMPM(n_grid=cfg.grid, grid_lim=1.0, dt=cfg.dt, gravity=(0, 0, 0),
                   grid_damp=0.999, f_clamp=None, damage_every=1, device=dev)
    if cfg.iso:
        fibers, alpha, a0 = None, 0.0, np.zeros(3, np.float32)
    else:
        th = math.radians(cfg.deg)
        a0 = np.zeros(3, np.float32); a0[axis] = math.cos(th); a0[trans] = math.sin(th)
        fibers, alpha = torch.tensor(np.tile(a0, (len(x0), 1)), device=dev), cfg.alpha
    x = torch.tensor(x0, device=dev)
    vol = torch.full((len(x0),), s ** 3, device=dev)
    sim.add_object(x, vol, rho=cfg.rho, E=cfg.E, nu=cfg.nu, fibers=fibers, alpha=alpha,
                   eta=cfg.eta, residual=RESID, allow_damage=True)
    sim.sigma_c[:] = cfg.sigc_frac * cfg.E

    rel = sim.x[:, axis] - c[axis]
    gw = 1.5 * cfg.dx
    lo = rel < -cfg.L / 2 + gw
    hi = rel > cfg.L / 2 - gw
    sim.allow[lo] = False; sim.allow[hi] = False
    L0 = float(sim.x[hi, axis].mean() - sim.x[lo, axis].mean())
    if cfg.one_ended:
        vlo, vhi = 0.0, cfg.rate * L0           # low end clamped, high end pulls
    else:
        vlo, vhi = -cfg.rate * L0 / 2, cfg.rate * L0 / 2
    ehat = torch.zeros(3, device=dev, dtype=sim.dtype); ehat[axis] = 1.0

    def bc(s_, t):
        r = min(t / cfg.ramp, 1.0)
        s_.v[lo] = ehat * (vlo * r); s_.v[hi] = ehat * (vhi * r)
    sim.particle_bc.append(bc)
    return sim, ~(lo | hi), lo, hi, L0, axis, a0


def carried_stress(sim, gauge, axis):
    """Mean degraded Cauchy stress along the pull axis over the gauge section."""
    F = sim.F[gauge]; mu = sim.mu[gauge]; lam = sim.lam[gauge]; d = sim.d[gauge]
    U, S, Vh = torch.linalg.svd(F)
    R = U @ Vh
    J = torch.linalg.det(F)
    I3 = torch.eye(3, device=F.device, dtype=F.dtype).expand_as(F)
    tau = 2 * mu[:, None, None] * (F - R) @ F.transpose(1, 2) + (lam * (J - 1) * J)[:, None, None] * I3
    g = (1 - d) ** 2 * (1 - RESID) + RESID
    return float((g * tau[:, axis, axis] / J).mean())


def run_one(cfg, dev):
    sim, gauge, lo, hi, L0, axis, a0 = build(cfg, dev)
    n_sub = max(1, int((1.0 / cfg.fps) / cfg.dt))
    grip = (lo | hi).detach().cpu().numpy()
    x0 = sim.x.detach().cpu().numpy().copy()
    strain, stress, cracked, xs, ds = [], [], [], [], []
    for f in range(cfg.frames):
        sim.run_frame(n_sub, f / cfg.fps)
        strain.append(float(sim.x[hi, axis].mean() - sim.x[lo, axis].mean()) / L0 - 1.0)
        stress.append(carried_stress(sim, gauge, axis))
        cracked.append(float((sim.d[gauge] > 0.5).float().mean()))
        xs.append(sim.x.detach().cpu().numpy().copy())
        ds.append(sim.d.detach().cpu().numpy().copy())
        if torch.isnan(sim.x).any():
            print("  NaN -- aborting"); break
    strain, stress, crk = np.array(strain), np.array(stress), np.array(cracked)
    i = int(np.argmax(stress))
    return dict(strain=strain, stress=stress, cracked=crk,
                strength=float(stress[i]), fail_strain=float(strain[i]),
                deg=float(cfg.deg), iso=bool(cfg.iso), axis=cfg.axis, one_ended=bool(cfg.one_ended),
                a0=a0.astype(np.float32), grip=grip, x0=x0,
                x_peak=xs[i], d_peak=ds[i], x_final=xs[-1], d_final=ds[-1])


def save(res, tag):
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"curve_{tag}.npz"), **res)


def _tag(cfg):
    return f"{mode_tag(cfg)}_" + ("iso" if cfg.iso else f"deg{int(cfg.deg):02d}")


def _dev(spec):
    return "cuda:0" if (spec == "auto" and torch.cuda.is_available()) else ("cpu" if spec == "auto" else spec)


def cmd_one(a):
    cfg = make_cfg(deg=a.deg, iso=a.iso, frames=a.frames, grid=a.grid, axis=a.axis, one_ended=a.one_ended)
    r = run_one(cfg, _dev(a.device))
    tag = _tag(cfg); save(r, tag)
    cf = float(r["cracked"][-1]); ms = float(r["strain"][-1])
    fs = next((e for e, c in zip(r["strain"], r["cracked"]) if c > 0.05), None)
    fstr = f"{fs*100:.1f}%" if fs is not None else f">{ms*100:.0f}% (no fail)"
    print(f"[slab] {tag}: cracked@{ms*100:.0f}%-strain={cf:.2f}  fail_strain={fstr}  peak_stress={r['strength']:.0f}")


def cmd_sweep(a):
    for deg in a.degs:
        cmd_one(SimpleNamespace(deg=float(deg), iso=False, frames=a.frames, grid=a.grid,
                                device=a.device, axis=a.axis, one_ended=a.one_ended))
    cmd_one(SimpleNamespace(deg=0.0, iso=True, frames=a.frames, grid=a.grid,
                            device=a.device, axis=a.axis, one_ended=a.one_ended))
    cmd_plot(SimpleNamespace(axis=a.axis, one_ended=a.one_ended))
    cmd_render(SimpleNamespace(axis=a.axis, one_ended=a.one_ended, state="final"))


def _load_mode(mode):
    aniso, iso = {}, None
    if not os.path.isdir(OUT):
        return aniso, iso
    for f in sorted(os.listdir(OUT)):
        if f.startswith(f"curve_{mode}_") and f.endswith(".npz"):
            r = dict(np.load(os.path.join(OUT, f)))
            if bool(r["iso"]):
                iso = r
            else:
                aniso[float(r["deg"])] = r
    return aniso, iso


def cmd_plot(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mode = mode_tag(a)
    aniso, iso = _load_mode(mode)
    if not aniso:
        print(f"[slab] no curves for mode {mode} -- run the matching `sweep` first"); return
    degs = sorted(aniso)
    ms = float(np.mean([aniso[d]["strain"][-1] for d in degs])) * 100
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), dpi=130)
    axs[0].plot(degs, [float(aniso[d]["cracked"][-1]) for d in degs], "-o", c="#d62728", lw=2, label="anisotropic")
    if iso is not None:
        axs[0].axhline(float(iso["cracked"][-1]), ls="--", c="#1f77b4", lw=2, label="isotropic (flat)")
    axs[0].set_xlabel("fiber angle vs pull axis  (0° = along grain, 90° = across)")
    axs[0].set_ylabel(f"damage fraction at ~{ms:.0f}% applied strain")
    axs[0].set_title("Directional toughness: intact along the grain, shatters across")
    axs[0].set_ylim(-0.03, None); axs[0].legend(); axs[0].grid(alpha=0.3)
    for d in degs:
        axs[1].plot(np.array(aniso[d]["strain"]) * 100, aniso[d]["stress"], label=f"{int(d)}°")
    axs[1].set_xlabel("strain  (%)"); axs[1].set_ylabel("carried stress")
    axs[1].set_title("Stress–strain by fiber angle"); axs[1].legend(title="fiber", fontsize=8); axs[1].grid(alpha=0.3)
    fig.suptitle(f"AnisoMPM core claim ({mode}): pull a fibered bar, rotate the grain", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"strength_{mode}.png"), bbox_inches="tight")
    print(f"[slab] {os.path.join(OUT, f'strength_{mode}.png')}")


def cmd_render(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mode = mode_tag(a)
    aniso, iso = _load_mode(mode)
    cols = [aniso[d] for d in sorted(aniso)] + ([iso] if iso is not None else [])
    if not cols:
        print(f"[slab] no particle data for mode {mode} -- run `sweep` (same --axis/--one-ended) first"); return
    state = getattr(a, "state", "final")
    allxy = np.concatenate([r["x_final"][:, :2] for r in cols])
    (x0, y0), (x1, y1) = allxy.min(0) - 0.03, allxy.max(0) + 0.03
    n = len(cols)
    fig, axs = plt.subplots(2, n, figsize=(3.1 * n, 6.4), dpi=130, squeeze=False)
    sc = None
    for j, r in enumerate(cols):
        a0 = r["a0"]; grip = r["grip"].astype(bool); free = ~grip
        cx, cy = float(r["x0"][:, 0].mean()), float(r["x0"][:, 1].mean())
        axs[0, j].scatter(r["x0"][:, 0], r["x0"][:, 1], s=6, c="#cccccc")
        if float(np.linalg.norm(a0)) > 0:
            L = 0.07
            axs[0, j].annotate("", xy=(cx + L * a0[0], cy + L * a0[1]), xytext=(cx - L * a0[0], cy - L * a0[1]),
                               arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=2.2))
        axs[0, j].set_title("isotropic" if bool(r["iso"]) else f"fiber @ {int(r['deg'])}°", fontsize=11)
        x = r[f"x_{state}"]; d = r[f"d_{state}"]
        axs[1, j].scatter(x[grip, 0], x[grip, 1], s=6, c="#cccccc")
        sc = axs[1, j].scatter(x[free, 0], x[free, 1], s=11, c=d[free], cmap="plasma", vmin=0, vmax=1)
        axs[1, j].set_title(f"{float(r['cracked'][-1])*100:.0f}% cracked", fontsize=10)
    for ax in axs.ravel():
        ax.set_aspect("equal"); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
    axs[0, 0].set_ylabel("undeformed\n(grain = arrow)", fontsize=10)
    one = bool(cols[0]["one_ended"]); ax_l = str(cols[0]["axis"])
    axs[1, 0].set_ylabel(f"pulled along {ax_l}\n({'one end' if one else 'both ends'})", fontsize=10)
    if sc is not None:
        fig.colorbar(sc, ax=axs[1, :].tolist(), label="damage d", fraction=0.015, pad=0.01)
    fig.suptitle(f"Fibered bar, mode {mode}: intact along the grain, cracks across it", fontsize=13)
    fig.savefig(os.path.join(OUT, f"render_{mode}.png"), bbox_inches="tight")
    print(f"[slab] {os.path.join(OUT, f'render_{mode}.png')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("one", "sweep"):
        q = sub.add_parser(name)
        q.add_argument("--frames", type=int, default=20); q.add_argument("--grid", type=int, default=28)
        q.add_argument("--device", default="auto"); q.add_argument("--axis", choices=["x", "y"], default="x")
        q.add_argument("--one-ended", dest="one_ended", action="store_true")
        if name == "one":
            q.add_argument("--deg", type=float, default=0.0); q.add_argument("--iso", action="store_true")
            q.set_defaults(fn=cmd_one)
        else:
            q.add_argument("--degs", nargs="+", type=float, default=[0, 15, 30, 45, 60, 75, 90])
            q.set_defaults(fn=cmd_sweep)
    for name, fn in (("plot", cmd_plot), ("render", cmd_render)):
        q = sub.add_parser(name)
        q.add_argument("--axis", choices=["x", "y"], default="x")
        q.add_argument("--one-ended", dest="one_ended", action="store_true")
        if name == "render":
            q.add_argument("--state", choices=["final", "peak"], default="final")
        q.set_defaults(fn=fn)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
