"""Render the static orange 3DGS with our PyTorch splatter to validate the
camera convention and image quality before wiring in the simulation."""
import os, sys, time
import numpy as np, torch, imageio.v2 as imageio
sys.path.insert(0, os.path.dirname(__file__))
from render import Camera, render_gaussians, to_uint8
from orange_setup import load_gaussians, cov_from_scale_rot

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
PLY = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.ply"
JSON = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.json"
OUT = "/3d-data/y2863claude/fruitninja_mpm/anisompm_py/out"
os.makedirs(OUT, exist_ok=True)

print("loading gaussians ...", flush=True)
g = load_gaussians(PLY, opacity_threshold=0.02)
xyz = torch.tensor(g["xyz"], device=DEV)
color = torch.tensor(g["color"], device=DEV)
# the trained orange has degenerate stored scales (point cloud); render each
# point as an isotropic splat of a fixed world radius so they fuse to a surface
opac = torch.ones(len(xyz), device=DEV)
I3 = torch.eye(3, device=DEV)
cam = Camera(JSON, device=DEV)
print(f"camera {cam.W}x{cam.H} fx={cam.fx:.1f}  N={len(xyz):,}", flush=True)

for sm, sigma in [(1.0, 0.006), (1.0, 0.010)]:
    cov = (sigma ** 2) * I3.expand(len(xyz), 3, 3)
    t0 = time.time()
    img = render_gaussians(cam, xyz, cov, color, opac, bg=(1, 1, 1),
                           scale_modifier=sm, r_max=24, downsample=None)
    torch.cuda.synchronize(torch.device(DEV))
    dt = time.time() - t0
    p = os.path.join(OUT, f"static_orange_sig{sigma}.png")
    imageio.imwrite(p, to_uint8(img))
    print(f"  sigma={sigma}: rendered in {dt:.2f}s -> {p}  "
          f"mean_rgb={float(img.mean()):.3f}", flush=True)
print("done", flush=True)
