#!/usr/bin/env python3
"""cohesive.py -- the directional-damage EXTENSION test, posed cleanly.

A stiff top layer bonded to a soft bulk by a thin WEAK bond band.  The bond's
damage driver is the rank-1 normal filter A0 = n n^T (so it fails under normal
traction, the delamination mode).  We then degrade the damaged bond two ways and
A/B them with the damage routing held FIXED:

  directional ON  : Option B (directional_kirchhoff) -- release normal + shear,
                    KEEP the in-plane membrane.  -> the bond debonds but the
                    layers stay coherent in-plane (the extension's claim).
  directional OFF : plain isotropic g(d) -- release everything.  -> the bond
                    goes fully soft.

This is the thing the ball gate kept failing to test; here the loading is a clean
flap peel, so the only difference between the two runs is the stress model.

    python3 src/cohesive.py --directional on
    python3 src/cohesive.py --directional off

Reports the Gate-3 discriminant measured IN-SIM from tau_last/J_last:
  normal_release = 1 - |t_n|(broken)/|t_n|(intact)   (both ~1: the bond opens)
  inplane_keep   = |sigma_inplane|(broken)/(intact)  (ON ~1, OFF ~0)
"""
import os, sys, math, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from anisompm import AnisoMPM

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "cohesive")
NY = None


def lame(E, nu=0.3):
    return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


def build(cfg, dev):
    c = np.array([0.5, 0.5, 0.5]); dx = 1.0 / cfg.grid; s = 0.5 * dx
    nx, ny, nz = (max(2, round(L / s)) for L in (cfg.Lx, cfg.Ly, cfg.Lz))
    axes = [(np.arange(n) + 0.5) * s for n in (nx, ny, nz)]
    axes = [a - a.mean() for a in axes]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    x0 = (np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) + c[None]).astype(np.float32)
    dt = 0.2 * dx / math.sqrt(cfg.E_top / cfg.rho)
    sim = AnisoMPM(n_grid=cfg.grid, grid_lim=1.0, dt=dt, gravity=(0, 0, 0),
                   grid_damp=0.999, f_clamp=None, damage_every=1, device=dev)
    sim.add_object(torch.tensor(x0, device=dev), torch.full((len(x0),), s ** 3, device=dev),
                   rho=cfg.rho, E=cfg.E_bulk, nu=0.3, fibers=None, alpha=0.0,
                   eta=0.01, residual=0.005, allow_damage=True)

    ry = sim.x[:, 1] - c[1]; rx = sim.x[:, 0] - c[0]
    y_if = cfg.strip_off
    bond = (ry - y_if).abs() < 0.9 * s
    top = ry > y_if + 0.9 * s
    bulk = ry < y_if - 0.9 * s
    mu_t, lam_t = lame(cfg.E_top); sim.mu[top] = mu_t; sim.lam[top] = lam_t   # stiff top layer
    sim.sigma_c[:] = 0.40 * cfg.E_bulk
    sim.sigma_c[top] = 0.40 * cfg.E_top
    sim.sigma_c[bond] = cfg.bond_sigc                                         # weak bond
    nn = torch.zeros(3, 3, device=dev, dtype=sim.dtype); nn[1, 1] = 1.0       # A0 = n n^T (n = y)
    sim.set_structural_tensor(bond, nn)                                       # driver held fixed (both A/B)
    if cfg.directional:
        nvec = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=sim.dtype).expand(sim.n, 3)
        sim.set_interface_normals(bond, nvec)                                   # -> Option B stress

    xhi = cfg.Lx / 2
    notch = bond & (rx > xhi - cfg.notch_len)
    sim.d[notch] = 1.0; sim.allow[notch] = False                             # starter crack
    flap = top & (rx > xhi - cfg.grip_len)
    clamp = bulk & (ry < -cfg.Ly / 2 + 2 * s)
    sim.allow[flap] = False; sim.allow[clamp] = False
    v = cfg.lift

    def bc(s_, t):
        r = min(t / cfg.ramp, 1.0)
        s_.v[clamp] = torch.zeros(3, device=dev, dtype=s_.dtype)
        s_.v[flap] = torch.tensor([0.0, +v * r, 0.0], device=dev, dtype=s_.dtype)
    sim.particle_bc.append(bc)
    n_sub = max(1, int((1.0 / cfg.fps) / dt))
    return sim, dict(top=top, bulk=bulk, bond=bond), n_sub


def discriminant(sim, bond):
    """Gate-3 release/keep at the bond, measured from the degraded stress."""
    if getattr(sim, "tau_last", None) is None:
        return float("nan"), float("nan"), 0
    sig = sim.tau_last / sim.J_last.view(-1, 1, 1)
    nrm = torch.tensor([0.0, 1.0, 0.0], device=sim.device, dtype=sim.dtype)
    Sn = torch.einsum('pij,j->pi', sig, nrm)
    tn = (Sn * nrm).sum(1).abs()
    P = torch.outer(nrm, nrm); Q = torch.eye(3, device=sim.device, dtype=sim.dtype) - P
    QsQ = torch.einsum('ij,pjk,kl->pil', Q, sig, Q)
    sip = QsQ.reshape(sim.n, -1).norm(dim=1)
    intact = bond & (sim.d < 0.1)
    broken = bond & (sim.d > 0.5)
    if int(intact.sum()) < 3 or int(broken.sum()) < 3:
        return float("nan"), float("nan"), int(broken.sum())
    nr = 1.0 - float(tn[broken].mean() / (tn[intact].mean() + 1e-9))
    ik = float(sip[broken].mean() / (sip[intact].mean() + 1e-9))
    return nr, ik, int(broken.sum())


def run(cfg, dev):
    sim, m, n_sub = build(cfg, dev)
    xs = [sim.x.detach().cpu().numpy().copy()]
    ds = [sim.d.detach().cpu().numpy().copy()]
    best = (float("nan"), float("nan"), 0)
    for f in range(cfg.frames):
        sim.run_frame(n_sub, f / cfg.fps)
        xs.append(sim.x.detach().cpu().numpy().copy())
        ds.append(sim.d.detach().cpu().numpy().copy())
        nr, ik, nb = discriminant(sim, m["bond"])
        if nb > best[2] and not math.isnan(ik):                 # record at most-debonded valid frame
            best = (nr, ik, nb)
        if torch.isnan(sim.x).any():
            print("  NaN -- aborting"); break
    return xs, ds, {k: v.detach().cpu().numpy() for k, v in m.items()}, best


def _draw(ax, x, d, m, lims):
    ax.clear()
    sl = np.abs(x[:, 2] - 0.5) < 0.5
    ax.scatter(x[sl & m["bulk"], 0], x[sl & m["bulk"], 1], s=12, c="#bbbbbb")    # soft bulk
    ax.scatter(x[sl & m["top"], 0], x[sl & m["top"], 1], s=12, c="#1f77b4")      # stiff top
    b = sl & m["bond"]
    ax.scatter(x[b, 0], x[b, 1], s=16, c=d[b], cmap="plasma", vmin=0, vmax=1)    # bond, by damage
    ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def _lims(xs):
    a = np.concatenate(xs)
    return (a[:, 0].min() - 0.03, a[:, 0].max() + 0.03, a[:, 1].min() - 0.03, a[:, 1].max() + 0.03)


def make_outputs(xs, ds, m, tag, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    os.makedirs(OUT, exist_ok=True)
    lims = _lims(xs)
    idx = np.linspace(0, len(xs) - 1, 6).astype(int)
    fig, axs = plt.subplots(1, 6, figsize=(18, 3.2), dpi=120)
    for ax, i in zip(axs, idx):
        _draw(ax, xs[i], ds[i], m, lims); ax.set_title(f"frame {i}", fontsize=10)
    fig.suptitle(title, fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{tag}_strip.png"), bbox_inches="tight")
    print(f"[cohesive] {OUT}/{tag}_strip.png")
    fig2, ax = plt.subplots(figsize=(6.2, 4.2), dpi=110); imgs = []
    for x, d in zip(xs, ds):
        _draw(ax, x, d, m, lims); ax.set_title(title, fontsize=10)
        fig2.canvas.draw(); w, h = fig2.canvas.get_width_height()
        imgs.append(Image.fromarray(np.frombuffer(fig2.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()))
    plt.close(fig2); plt.close(fig)
    imgs[0].save(os.path.join(OUT, f"{tag}.gif"), save_all=True, append_images=imgs[1:], duration=60, loop=0)
    print(f"[cohesive] {OUT}/{tag}.gif  ({len(imgs)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directional", choices=["on", "off"], default="on")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--Lx", type=float, default=0.64); ap.add_argument("--Ly", type=float, default=0.18)
    ap.add_argument("--Lz", type=float, default=0.05)
    ap.add_argument("--E-top", dest="E_top", type=float, default=2e4)
    ap.add_argument("--E-bulk", dest="E_bulk", type=float, default=1e4)
    ap.add_argument("--rho", type=float, default=1000.0)
    ap.add_argument("--bond-sigc", dest="bond_sigc", type=float, default=400.0)
    ap.add_argument("--frames", type=int, default=60); ap.add_argument("--fps", type=int, default=48)
    ap.add_argument("--ramp", type=float, default=0.15)
    ap.add_argument("--strip-off", dest="strip_off", type=float, default=0.04)
    ap.add_argument("--notch-len", dest="notch_len", type=float, default=0.14)
    ap.add_argument("--grip-len", dest="grip_len", type=float, default=0.08)
    ap.add_argument("--lift", type=float, default=0.18)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    a.directional = (a.directional == "on")
    dev = "cuda:0" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    xs, ds, m, (nr, ik, nb) = run(a, dev)
    tag = "on" if a.directional else "off"
    ttl = (f"directional {tag.upper()}: " +
           ("Option B — bond opens, in-plane KEPT" if a.directional else "isotropic g(d) — bond fully soft"))
    make_outputs(xs, ds, m, tag, ttl)
    print(f"[cohesive] directional={tag}  normal_release={nr:.2f}  inplane_keep={ik:.2f}  (broken bond pts={nb})")


if __name__ == "__main__":
    main()
