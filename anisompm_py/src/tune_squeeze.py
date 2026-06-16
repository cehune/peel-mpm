"""Tune the contact-free squeeze scenario: prepare once, try parameter sets."""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from orange_setup import prepare_orange
from run_orange import build_sim

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
PLY = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.ply"
data = prepare_orange(PLY, n_grid=64, parts_per_cell_dim=2.0, splat_sigma=0.006, device=DEV)

BASE = dict(n_grid=64, dt=3e-4, nu=0.4, rho=500, l0_scale=0.5,
            anisotropic=True, alpha=-1.0, gravity=-1.0, grid_damp=0.999,
            cap_frac=0.16, ramp=0.25)

TRIALS = {
    "elastic": dict(E=2e4, squeeze_speed=0.15, percentage=0.5, eta=0.1,
                    residual=0.01, allow_damage=False),
    "A":       dict(E=2e4, squeeze_speed=0.15, percentage=0.30, eta=0.06, residual=0.01),
    "B":       dict(E=2e4, squeeze_speed=0.16, percentage=0.18, eta=0.04, residual=0.008),
}

for name in (sys.argv[2].split(",") if len(sys.argv) > 2 else TRIALS):
    p = dict(BASE); p.update(TRIALS[name])
    sim, _ = build_sim(data, p, "squeeze", DEV)
    fps = 24; n_sub = max(1, int((1 / fps) / sim.dt))
    print(f"\n[{name}] {TRIALS[name]}  n_sub={n_sub}", flush=True)
    for f in range(40):
        sim.run_frame(n_sub, f / fps)
        if f % 5 == 0 or f == 39:
            S = torch.linalg.svdvals(sim.F)
            print(f"  f{f:2d} d[mean,max]={float(sim.d.mean()):.3f},{float(sim.d.max()):.3f}"
                  f" frac>.5={float((sim.d>0.5).float().mean()):.3f}"
                  f" Smax={float(S.max()):.2f} Smin={float(S.min()):.3f}"
                  f" yspan={float(sim.x[:,1].max()-sim.x[:,1].min()):.3f}"
                  f" nan={bool(torch.isnan(sim.x).any())}", flush=True)
print("\ndone", flush=True)
