"""Fast parameter tuning for the orange crush: prepare once, try several
parameter sets (sim-only, no render), report damage growth and stability."""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from orange_setup import prepare_orange
from run_orange import build_sim

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
PLY = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.ply"
data = prepare_orange(PLY, n_grid=64, parts_per_cell_dim=2.0, splat_sigma=0.006, device=DEV)

BASE = dict(n_grid=64, dt=4e-4, nu=0.4, rho=500, l0_scale=0.5,
            anisotropic=True, alpha=-1.0)

TRIALS = {
    "elastic":  dict(E=3e4, gravity=-2.0, grid_damp=0.999, percentage=0.5, eta=0.1,
                     residual=0.005, crush_speed=0.15, allow_damage=False),
    "A_tough":  dict(E=3e4, gravity=-2.0, grid_damp=0.999, percentage=0.6, eta=0.12,
                     residual=0.005, crush_speed=0.15),
    "B_mid":    dict(E=2e4, gravity=-2.0, grid_damp=0.998, percentage=0.45, eta=0.08,
                     residual=0.005, crush_speed=0.18),
    "C_brittle":dict(E=2e4, gravity=-2.5, grid_damp=0.998, percentage=0.30, eta=0.05,
                     residual=0.005, crush_speed=0.18),
}

for name in (sys.argv[2].split(",") if len(sys.argv) > 2 else TRIALS):
    p = dict(BASE); p.update(TRIALS[name])
    sim, _ = build_sim(data, p, "crush", DEV)
    fps = 24; n_sub = max(1, int((1 / fps) / sim.dt))
    print(f"\n[{name}] {TRIALS[name]}", flush=True)
    for f in range(28):
        sim.run_frame(n_sub, f / fps)
        if f % 4 == 0 or f == 27:
            S = torch.linalg.svdvals(sim.F)
            print(f"  f{f:2d} d[mean,max]={float(sim.d.mean()):.3f},{float(sim.d.max()):.3f}"
                  f" frac>.5={float((sim.d>0.5).float().mean()):.3f}"
                  f" Smax={float(S.max()):.2f} Smin={float(S.min()):.3f}"
                  f" ymin={float(sim.x[:,1].min()):.3f} nan={bool(torch.isnan(sim.x).any())}", flush=True)
print("\ndone", flush=True)
