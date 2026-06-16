import os, sys, time
import numpy as np, torch, imageio.v2 as imageio
sys.path.insert(0, os.path.dirname(__file__))
from orange_setup import prepare_orange
from run_orange import make_camera, build_sim, deform_and_render
from render import to_uint8

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
PLY = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.ply"
JSON = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.json"
OUT = "/3d-data/y2863claude/fruitninja_mpm/anisompm_py/out/squeeze"
os.makedirs(OUT, exist_ok=True)

data = prepare_orange(PLY, n_grid=64, parts_per_cell_dim=2.0, splat_sigma=0.006, device=DEV)
cam = make_camera(JSON, DEV, zoom=1.55, dy=8)

p = dict(n_grid=64, dt=5e-4, gravity=0.0, grid_damp=0.9995, nu=0.4, rho=500,
         E=2.0e4, percentage=0.12, eta=0.04, l0_scale=0.6, residual=0.01,
         anisotropic=True, alpha=-1.0, pull_speed=0.16, ramp=0.3, cap_frac=0.18,
         r_max=12, crack_tint=0.2, damage_every=2)

sim, _ = build_sim(data, p, "pull", DEV)
print(f"sim particles {sim.n:,}  f_clamp={sim.f_clamp}", flush=True)
fps = 24; n_sub = max(1, int((1 / fps) / sim.dt))
nframes = 48
save = {0, 8, 14, 20, 26, 32, 40, 47}
t0 = time.time()
for f in range(nframes):
    sim.run_frame(n_sub, f / fps)
    img = deform_and_render(sim, data, cam, p, crack_tint=p["crack_tint"])
    if f in save:
        imageio.imwrite(os.path.join(OUT, f"sq_{f:03d}.png"), to_uint8(img))
    S = torch.linalg.svdvals(sim.F)
    print(f"  f{f:2d} d[mean,max]={float(sim.d.mean()):.3f},{float(sim.d.max()):.3f}"
          f" frac>.5={float((sim.d>0.5).float().mean()):.3f} Smax={float(S.max()):.2f}"
          f" nan={bool(torch.isnan(sim.x).any())} ({time.time()-t0:.0f}s)", flush=True)
    torch.cuda.empty_cache()
print("done", flush=True)
