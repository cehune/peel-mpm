#!/usr/bin/env python3
"""make_figs.py -- real-data visuals of the peel machinery (for visual validation).

Two modes, so the (slow) runs are cached and the (fast) plotting is re-runnable:

    python3 src/make_figs.py run  --tag correct --aniso correct --frames 10
    python3 src/make_figs.py run  --tag wrong   --aniso wrong   --frames 10
    python3 src/make_figs.py plot --figdir out/figs

`run` drives the ACTUAL solver (build_sim from peel_test), captures per-frame
phi/damage history plus deformed (x,d) snapshots at start / delamination-onset /
final, and writes one npz per tag.  `plot` reads those npz and renders:

  1 delam_slices    z-slice cross-sections: regions, then damage d on the
                    DEFORMED particles at start / onset / final (the peel lifting)
  2 angular_law     log-log phi_int/phi_shear vs cot^2(theta): correct ~ +1 slope
                    (normal driver), wrong < 0 (shear driver) -- the initiation law
  3 routing         phi_int vs phi_bystander and broken_int/d_bystander vs frame:
                    the interface leads, bystanders stay quiet until over-pull
  4 bystander_hist  damage histogram OUTSIDE the interface at onset vs final:
                    selectivity is read at onset (quiet), not the over-pulled tail
"""
import os, sys, math, argparse
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from peel_test import build_sim, pick_device

DEF = dict(device="cpu", ngrid=24, ppcd=1.5, R=0.28, frames=10, fps=24, dt=5e-4,
           f_clamp="0.35,2.8", E_flesh=2e4, E_int=4e4, E_peel=1e5, sigc_frac=0.05,
           rho=0.1, damage_every=1, eta=0.02, speed=0.4, ramp=0.30, cap_deg=30.0,
           grip="cap", pull_deg=0.0, notch_deg=0.0, aniso="correct", directional="on",
           equal_E=False, seed=0, plot_every=0, log_every=99)


def _cap(sim):
    return sim.x.detach().cpu().numpy().copy(), sim.d.detach().cpu().numpy().copy()


def run_capture(cfg, npz_path):
    dev = pick_device(cfg.device)
    sim, geom = build_sim(cfg, dev)
    interface = geom["interface"]; flesh = geom["flesh"]; peel = geom["peel"]
    n_sub = max(1, int((1.0 / cfg.fps) / cfg.dt))
    phi_pp = torch.zeros(sim.n, device=dev, dtype=sim.dtype)
    psh_pp = torch.zeros(sim.n, device=dev, dtype=sim.dtype)
    hist = []
    onset = -1
    x_start, d_start = _cap(sim)
    snaps = {"start": (x_start, d_start)}
    for f in range(cfg.frames):
        sim.run_frame(n_sub, f / cfg.fps)
        ph = getattr(sim, "phi", None); psh = getattr(sim, "phi_shear", None)
        if ph is not None: phi_pp = torch.maximum(phi_pp, ph)
        if psh is not None: psh_pp = torch.maximum(psh_pp, psh)
        di = sim.d[interface]
        broken = float((di > 0.5).float().mean())
        hist.append((f,
                     float(ph[interface].max()) if ph is not None else np.nan,
                     float(ph[flesh].max()) if ph is not None else np.nan,
                     float(ph[peel].max()) if ph is not None else np.nan,
                     float(psh[interface].max()) if psh is not None else np.nan,
                     broken,
                     max(float(sim.d[flesh].max()), float(sim.d[peel].max())),
                     float(di.max())))
        if onset < 0 and broken > 0.05:
            onset = f; snaps["onset"] = _cap(sim)
    snaps["final"] = _cap(sim)
    if "onset" not in snaps:                       # never delaminated -> use final
        snaps["onset"] = snaps["final"]
    region = np.zeros(sim.n, np.int8)
    region[interface.cpu().numpy()] = 1; region[peel.cpu().numpy()] = 2
    cth = (geom["r_hat"] @ geom["pull_dir"]).clamp(-1, 1).cpu().numpy()
    np.savez_compressed(
        npz_path,
        meta=np.array([cfg.aniso, cfg.directional, cfg.rho, cfg.pull_deg, cfg.equal_E,
                       onset, cfg.frames], dtype=object),
        hist=np.array(hist, dtype=np.float64),
        region=region, cth=cth,
        phi_pp=phi_pp.cpu().numpy(), psh_pp=psh_pp.cpu().numpy(),
        center=np.array(geom["c"]), spacing=float(geom["spacing"]),
        break_thresh=float(geom["break_thresh"]),
        x_start=snaps["start"][0], d_start=snaps["start"][1],
        x_onset=snaps["onset"][0], d_onset=snaps["onset"][1],
        x_final=snaps["final"][0], d_final=snaps["final"][1])
    print(f"[figs] saved {npz_path}  (onset frame={onset}, N={sim.n})")


# ---------------------------------------------------------------- plotting ----
def _load(p):
    return dict(np.load(p, allow_pickle=True))


def fig_slices(R, figpath):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    c = R["center"]; sp = float(R["spacing"]); reg = R["region"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.4), dpi=130)
    # panel 0: regions (undeformed), so the viewer can locate the bond
    x0 = R["x_start"]; sl0 = np.abs(x0[:, 2] - c[2]) < 1.5 * sp
    cols = ["#c9c9c9", "#ff7f0e", "#2ca02c"]               # flesh / interface / peel
    for k, name in [(0, "flesh (bulk)"), (1, "interface = the bond"), (2, "peel = top layer")]:
        m = sl0 & (reg == k)
        axs[0].scatter(x0[m, 0], x0[m, 1], s=11, c=cols[k], label=name)
    axs[0].legend(markerscale=2, loc="lower center", fontsize=9, framealpha=0.9)
    axs[0].set_title("regions (undeformed mid-slice)", fontsize=12)
    # panels 1-3: light-gray ball + bright damage overlay on the DEFORMED particles
    for ax, key, ttl in [(axs[1], "start", "start (d≈0)"), (axs[2], "onset", "delamination ONSET"),
                         (axs[3], "final", "final (over-pulled)")]:
        x = R[f"x_{key}"]; d = R[f"d_{key}"]
        sl = np.abs(x[:, 2] - c[2]) < 1.5 * sp
        ax.scatter(x[sl, 0], x[sl, 1], s=9, c="#dddddd", zorder=1)          # intact ball
        i_sl = sl & (reg == 1) & (d <= 0.08)
        ax.scatter(x[i_sl, 0], x[i_sl, 1], s=9, c="#ffd9a8", zorder=2)      # faint bond marker
        dm = sl & (d > 0.08)
        sc = ax.scatter(x[dm, 0], x[dm, 1], s=22, c=d[dm], cmap="plasma", vmin=0, vmax=1, zorder=3)
        ax.set_title(f"damage — {ttl}", fontsize=12)
    fig.colorbar(sc, ax=axs[3], label="damage d  (bond breaking →)", fraction=0.046)
    for ax in axs:
        ax.set_aspect("equal"); ax.set_xlim(c[0] - 1.25 * R_, c[0] + 1.25 * R_)
        ax.set_ylim(c[1] - 1.25 * R_, c[1] + 1.7 * R_); ax.set_xticks([]); ax.set_yticks([])
    meta = R["meta"]
    fig.suptitle(f"Peel cross-sections  (aniso={meta[0]}, ρ={meta[2]}, pulled ↑ along +y, "
                 f"onset frame {meta[5]}):  damage (yellow) nucleates along the bond at the pulled pole",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(figpath, bbox_inches="tight")
    print(f"[figs] {figpath}")


def fig_angular(Rc, Rw, figpath):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    def pts(R):
        reg = R["region"]; cth = R["cth"]; phi = R["phi_pp"]; psh = R["psh_pp"]
        m = (reg == 1) & (cth > 0) & (phi > 1e-6) & (psh > 1e-9)
        cos2 = cth[m] ** 2; sin2 = np.clip(1 - cos2, 1e-6, None)
        xpred = cos2 / sin2; yobs = phi[m] / (psh[m] + 1e-12)
        g = (yobs > 1e-6) & (xpred > 1e-3) & (xpred < 1e3)
        return xpred[g], yobs[g]

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=130)
    for R, col, name in [(Rc, "#1f77b4", "correct (A0 = n⊗n, normal driver)"),
                         (Rw, "#d62728", "wrong (A0 = I−n⊗n, shear driver)")]:
        if R is None: continue
        X, Y = pts(R)
        lx, ly = np.log(X), np.log(Y)
        b = np.polyfit(lx, ly, 1)[0]
        ax.scatter(X, Y, s=10, c=col, alpha=0.35, label=f"{name}: slope={b:+.2f}")
        xs = np.array([X.min(), X.max()])
        ax.plot(xs, np.exp(np.polyval(np.polyfit(lx, ly, 1), np.log(xs))), c=col, lw=2.5)
    xr = np.array([1e-2, 1e2])
    ax.plot(xr, xr, "k--", lw=1.2, label="slope = +1 (theory: φ_int/φ_shear = cot²θ)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cot²θ   (geometry: normal² / in-plane² of the pull)", fontsize=11)
    ax.set_ylabel("φ_int / φ_shear   (measured driver ratio)", fontsize=11)
    ax.set_title("Angular initiation law (magnitude-free)\n"
                 "normal-driven 'correct' tracks cot²θ (+1); shear-driven 'wrong' inverts (<0)",
                 fontsize=12)
    ax.legend(fontsize=10, loc="upper left"); ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout(); fig.savefig(figpath, bbox_inches="tight"); print(f"[figs] {figpath}")


def fig_routing(R, figpath):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    h = R["hist"]; f = h[:, 0]; onset = int(R["meta"][5])
    phi_int = h[:, 1]; phi_by = np.maximum(h[:, 2], h[:, 3]); broken = h[:, 5]; d_by = h[:, 6]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), dpi=130, sharex=True)
    a1.semilogy(f, np.clip(phi_int, 1e-3, None), "-o", c="#ff7f0e", label="φ_int (the bond)")
    a1.semilogy(f, np.clip(phi_by, 1e-3, None), "-s", c="#777", label="φ_bystander (flesh/peel)")
    a1.set_ylabel("phase-field drive φ  (>1 ⇒ cracks)"); a1.legend(fontsize=10)
    a1.set_title("Routing: the interface drive runs far ahead of the bystanders", fontsize=12)
    a1.axhline(1.0, color="k", lw=0.8, ls=":")
    a2.plot(f, broken, "-o", c="#ff7f0e", label="broken_int fraction (d>0.5)")
    a2.plot(f, d_by, "-s", c="#d62728", label="max bystander damage")
    a2.axhline(float(R["break_thresh"]), color="#d62728", ls="--", lw=1,
               label=f"break threshold ({float(R['break_thresh']):.2f})")
    a2.set_xlabel("frame"); a2.set_ylabel("damage"); a2.legend(fontsize=10); a2.set_ylim(-0.03, 1.03)
    for a in (a1, a2):
        a.axvline(onset, color="#2ca02c", lw=1.5, alpha=0.7)
        a.text(onset, a.get_ylim()[1], " onset", color="#2ca02c", va="top", fontsize=9)
    fig.tight_layout(); fig.savefig(figpath, bbox_inches="tight"); print(f"[figs] {figpath}")


def fig_hist(R, figpath):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    reg = R["region"]; out = reg != 1
    d_on = R["d_onset"][out]; d_fi = R["d_final"][out]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=130)
    bins = np.linspace(0, 1, 41)
    ax.hist(d_on, bins=bins, color="#2ca02c", alpha=0.7, label="at delamination onset (graded here)")
    ax.hist(d_fi, bins=bins, color="#d62728", alpha=0.45, label="at final frame (over-pulled)")
    ax.axvline(float(R["break_thresh"]), color="k", ls="--", lw=1.2,
               label=f"break threshold ({float(R['break_thresh']):.2f})")
    ax.set_yscale("log"); ax.set_xlabel("damage d of NON-interface (bystander) particles")
    ax.set_ylabel("particle count (log)")
    ax.set_title("Selectivity: bystanders are quiet at onset; damage only grows under over-pull\n"
                 "(why the gate grades at onset, not the final frame)", fontsize=12)
    ax.legend(fontsize=10); fig.tight_layout(); fig.savefig(figpath, bbox_inches="tight")
    print(f"[figs] {figpath}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--simdir", default=os.path.join(os.path.dirname(__file__), "..", "out", "figs"))
    for k, v in DEF.items():
        if isinstance(v, bool):
            r.add_argument(f"--{k.replace('_','-')}", dest=k, action="store_true")
        else:
            r.add_argument(f"--{k.replace('_','-')}", dest=k, type=type(v), default=v)
    p = sub.add_parser("plot")
    p.add_argument("--figdir", default=os.path.join(os.path.dirname(__file__), "..", "out", "figs"))
    a = ap.parse_args()

    if a.mode == "run":
        cfg = SimpleNamespace(**{k: getattr(a, k) for k in DEF})
        os.makedirs(a.simdir, exist_ok=True)
        run_capture(cfg, os.path.join(a.simdir, f"run_{a.tag}.npz"))
    else:
        global R_
        d = a.figdir; os.makedirs(d, exist_ok=True)
        Rc = _load(os.path.join(d, "run_correct.npz")) if os.path.exists(os.path.join(d, "run_correct.npz")) else None
        Rw = _load(os.path.join(d, "run_wrong.npz")) if os.path.exists(os.path.join(d, "run_wrong.npz")) else None
        base = Rc or Rw
        R_ = float(base["meta"][3]) * 0 + 0.28      # R for axis limits
        if base is not None:
            fig_slices(base, os.path.join(d, "1_delam_slices.png"))
            fig_routing(base, os.path.join(d, "3_routing.png"))
            fig_hist(base, os.path.join(d, "4_bystander_hist.png"))
        if Rc is not None or Rw is not None:
            fig_angular(Rc, Rw, os.path.join(d, "2_angular_law.png"))


R_ = 0.28
if __name__ == "__main__":
    main()
