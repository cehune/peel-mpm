import os, sys, time
import numpy as np, torch, imageio.v2 as imageio
sys.path.insert(0, os.path.dirname(__file__))
from orange_setup import prepare_orange
from run_orange import make_camera, build_sim, deform_and_render
from render import to_uint8

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
PLY = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.ply"
JSON = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.json"
OUT = "/3d-data/y2863claude/fruitninja_mpm/anisompm_py/out/smoke"
os.makedirs(OUT, exist_ok=True)

data = prepare_orange(PLY, n_grid=64, parts_per_cell_dim=2.0, splat_sigma=0.006, device=DEV)
cam = make_camera(JSON, DEV, zoom=1.6, dy=10)

p = dict(n_grid=64, dt=5e-4, gravity=-4.0, grid_damp=0.999,
         E=1.0e4, nu=0.4, rho=500, percentage=0.15, eta=0.01, l0_scale=0.5,
         residual=0.001, anisotropic=True, alpha=-1.0, crush_speed=0.20,
         r_max=12, crack_tint=0.0)

sim, _ = build_sim(data, p, "crush", DEV)
print(f"sim particles {sim.n:,}", flush=True)
fps = 24; n_sub = max(1, int((1/fps)/sim.dt))
for f in range(15):
    t0 = time.time()
    sim.run_frame(n_sub, f / fps)
    img = deform_and_render(sim, data, cam, p)
    torch.cuda.synchronize(torch.device(DEV))
    if f in (0, 4, 8, 11, 14):
        imageio.imwrite(os.path.join(OUT, f"crush_{f:03d}.png"), to_uint8(img))
    print(f"  f{f:2d} d[mean,max]={float(sim.d.mean()):.3f},{float(sim.d.max()):.3f}"
          f" frac>.5={float((sim.d>0.5).float().mean()):.3f}"
          f" ymin={float(sim.x[:,1].min()):.3f} nan={bool(torch.isnan(sim.x).any())}"
          f" ({time.time()-t0:.1f}s)", flush=True)
print("done", flush=True)
