#!/usr/bin/env python3
"""expose.py -- STEP 1 of the rendering harness: pull the freshly-exposed crack
face out of an AnisoMPM peel, per frame, with stable (non-flickering) labels.

This is the one genuinely new bit of code and everything downstream (baseline
render, the A/B, differentiable fitting) consumes its output: per frame, the set
of particles that are NEW surface created by the tear, plus their outward normals.

Definition of "newly exposed", per particle:
  surface_now   : neighborhood is one-sided -> |mean offset to neighbors| is large.
                  That same vector, negated and normalized, is the outward normal.
  not_original  : it was NOT surface at t=0 (so we exclude the slab's outer faces).
  in_crack      : local damage > thr (so generic deformation-surface doesn't count).
  exposed = surface_now & not_original & in_crack,  then LATCHED (monotone) over
  time so a face, once revealed, stays labeled -> no frame-to-frame flicker.

    python3 src/expose.py            # run peel, extract, visualize, save npz

Output: out/render/exposed.npz  (x per frame, exposed mask per frame, normals,
        damage, spacing) and out/render/expose_strip.png (sanity visualization).
"""
import os, sys, math
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import peel

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "render")


def main():
    cfg = SimpleNamespace(device="cpu", grid=32, Lx=0.64, Ly=0.18, Lz=0.05,
                          E=1e4, rho=1000.0, sigc_frac=0.06, fps=48, ramp=0.15,
                          strip_off=0.04, notch_len=0.14, grip_len=0.08, lift=0.18,
                          frames=60, iso=False)
    dev = "cpu"
    sim, _, n_sub = peel.build_peel(cfg, dev)
    dx = 1.0 / cfg.grid; spacing = 0.5 * dx
    h = 1.6 * spacing                                     # t=0 bonded-neighbor radius
    sep = 2.5 * spacing                                   # a bonded neighbor this far = torn

    # t=0 bonded adjacency + reference offsets (for the outward normal)
    x0 = sim.x.detach().clone()
    D0 = torch.cdist(x0, x0)
    bonded0 = (D0 < h) & (D0 > 1e-9)                      # who was a neighbor at rest
    dir0 = x0.unsqueeze(0) - x0.unsqueeze(1)              # dir0[i,j] = x0[j]-x0[i]

    xs, ds, exposed, normals = [], [], [], []
    latched = torch.zeros(sim.n, dtype=torch.bool)
    for f in range(cfg.frames):
        sim.run_frame(n_sub, f / cfg.fps)
        x = sim.x.detach(); d = sim.d.detach()
        D = torch.cdist(x, x)
        torn = bonded0 & (D > sep)                        # bonded neighbors now pulled apart
        newly = torn.any(1)                               # a particle that lost a bond = crack face
        latched = latched | newly                         # monotone latch -> no flicker
        # outward normal = mean t=0 direction to the torn-away neighbors (points into the gap)
        nd = (dir0 * torn.unsqueeze(-1)).sum(1)
        nrm = nd / (nd.norm(dim=1, keepdim=True) + 1e-9)
        xs.append(x.cpu().numpy().copy()); ds.append(d.cpu().numpy().copy())
        exposed.append(latched.cpu().numpy().copy()); normals.append(nrm.cpu().numpy().copy())
        if f % 10 == 0 or f == cfg.frames - 1:
            print(f"  f{f:3d} exposed={int(latched.sum()):4d}  (newly {int(newly.sum())})", flush=True)
    original_surf = torch.zeros(sim.n, dtype=torch.bool)   # (kept for npz compat; unused now)

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "exposed.npz"),
                        x=np.array(xs), d=np.array(ds), exposed=np.array(exposed),
                        normals=np.array(normals), original_surf=original_surf.cpu().numpy(),
                        spacing=spacing)
    print(f"[expose] saved {OUT}/exposed.npz  ({cfg.frames} frames, N={sim.n})")

    # ---- sanity visualization: interior / original-exterior / exposed crack face
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    allx = np.concatenate(xs)
    lims = (allx[:, 0].min() - .03, allx[:, 0].max() + .03, allx[:, 1].min() - .03, allx[:, 1].max() + .03)
    idx = np.linspace(0, cfg.frames - 1, 6).astype(int)
    fig, axs = plt.subplots(1, 6, figsize=(18, 3.2), dpi=120)
    for ax, i in zip(axs, idx):
        x = xs[i]; ex = exposed[i]; sl = np.abs(x[:, 2] - 0.5) < 0.5
        ax.scatter(x[sl & ~ex, 0], x[sl & ~ex, 1], s=8, c="#dddddd")                 # material
        ax.scatter(x[sl & ex, 0], x[sl & ex, 1], s=16, c="#d62728")                  # exposed crack face
        ax.set_title(f"frame {i}  (exposed={int(ex.sum())})", fontsize=10)
        ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Step 1: newly-exposed crack face (red) = particles that lost a bonded neighbor", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "expose_strip.png"), bbox_inches="tight")
    print(f"[expose] {OUT}/expose_strip.png")


if __name__ == "__main__":
    main()
