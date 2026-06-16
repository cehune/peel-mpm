"""Parameter-control study + video generation for the orange.

Runs the AnisoMPM crush scenario under several parameter settings and writes:
  * one MP4 per condition
  * side-by-side comparison MP4s that isolate each control knob

Conditions are cached by name so comparisons reuse already-simulated frames.
Launch with a device and a comma-separated list of condition names to split the
work across the two GPUs.
"""
import os, sys, time, argparse
import numpy as np, torch, imageio.v2 as imageio
sys.path.insert(0, os.path.dirname(__file__))
from orange_setup import prepare_orange
from run_orange import make_camera, build_sim, deform_and_render
from render import to_uint8

PLY = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.ply"
JSON = "/3d-data/y2863claude/fruitninja_mpm/FruitNinja3DInterior/trained_gs/orange.json"
OUT = "/3d-data/y2863claude/fruitninja_mpm/anisompm_py/out"
FRAMEDIR = os.path.join(OUT, "frames")
os.makedirs(FRAMEDIR, exist_ok=True)

SCENARIO = "pull"

# base orange.  AnisoMPM phase-field damage with radial fibers (alpha=-1),
# pulled apart by two clamped caps (contact-free, direct tension -> clean tear).
BASE = dict(n_grid=64, dt=5e-4, gravity=0.0, grid_damp=0.9995, damage_every=2,
            E=2.0e4, nu=0.4, rho=500, percentage=0.12, eta=0.04, l0_scale=0.6,
            residual=0.01, anisotropic=True, alpha=-1.0,
            pull_speed=0.16, ramp=0.3, cap_frac=0.18,
            r_max=12, crack_tint=0.2)

# name -> (param overrides, caption)
CONDITIONS = {
    "aniso":     ({}, "Anisotropic (radial fibers)"),
    "iso":       (dict(anisotropic=False), "Isotropic (no fibers)"),
    "brittle":   (dict(percentage=0.06), "Brittle  (low sigma_c)"),
    "tough":     (dict(percentage=0.20), "Tough  (high sigma_c)"),
    "eta_fast":  (dict(eta=0.015), "Fast/brittle  eta=0.015"),
    "eta_slow":  (dict(eta=0.12), "Slow/gummy  eta=0.12"),
    "soft":      (dict(E=8.0e3), "Soft  E=8e3"),
    "stiff":     (dict(E=4.0e4), "Stiff  E=4e4"),
}

COMPARISONS = [
    ("compare_anisotropy", ["iso", "aniso"]),
    ("compare_brittleness", ["tough", "aniso", "brittle"]),
    ("compare_eta", ["eta_slow", "aniso", "eta_fast"]),
    ("compare_stiffness", ["soft", "aniso", "stiff"]),
]


def label(img, text):
    """Draw a caption bar at the bottom of an (H,W,3) uint8 frame."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        except Exception:
            font = ImageFont.load_default()
        W = img.shape[1]
        d.rectangle([0, img.shape[0] - 40, W, img.shape[0]], fill=(20, 20, 20))
        d.text((12, img.shape[0] - 34), text, fill=(255, 255, 255), font=font)
        return np.asarray(im)
    except Exception:
        return img


def run_condition(name, data, cam, frames, fps, device):
    cache = os.path.join(FRAMEDIR, f"{name}.npy")
    if os.path.exists(cache):
        print(f"[{name}] cached", flush=True)
        return
    ov, _cap = CONDITIONS[name]
    p = dict(BASE); p.update(ov)
    print(f"[{name}] simulating: {ov}", flush=True)
    sim, _ = build_sim(data, p, SCENARIO, device)
    n_sub = max(1, int((1.0 / fps) / sim.dt))
    out = np.empty((frames, cam.H, cam.W, 3), np.uint8)
    t0 = time.time()
    for f in range(frames):
        sim.run_frame(n_sub, f / fps)
        img = deform_and_render(sim, data, cam, p, crack_tint=p["crack_tint"])
        out[f] = to_uint8(img)
        if f % 10 == 0 or f == frames - 1:
            print(f"  [{name}] f{f:2d}/{frames} d_max={float(sim.d.max()):.3f}"
                  f" frac>.5={float((sim.d>0.5).float().mean()):.3f} ({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    np.save(cache, out)
    imageio.mimwrite(os.path.join(OUT, f"orange_{name}.mp4"), list(out), fps=fps, quality=8)
    print(f"[{name}] saved orange_{name}.mp4", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--frames", type=int, default=44)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--compose", action="store_true", help="only build comparison videos from cache")
    args = ap.parse_args()

    if args.compose:
        compose(args.fps)
        return

    data = prepare_orange(PLY, n_grid=BASE["n_grid"], parts_per_cell_dim=2.0,
                          splat_sigma=0.006, device=args.device)
    cam = make_camera(JSON, args.device, zoom=1.6, dy=10)
    names = list(CONDITIONS) if args.conditions == "all" else args.conditions.split(",")
    for nm in names:
        run_condition(nm, data, cam, args.frames, args.fps, args.device)


def compose(fps):
    """Build side-by-side comparison videos from cached frames."""
    def load(nm):
        return np.load(os.path.join(FRAMEDIR, f"{nm}.npy"))
    for out_name, names in COMPARISONS:
        if not all(os.path.exists(os.path.join(FRAMEDIR, f"{n}.npy")) for n in names):
            print(f"[{out_name}] missing cache, skip", flush=True)
            continue
        clips = [load(n) for n in names]
        caps = [CONDITIONS[n][1] for n in names]
        T = min(c.shape[0] for c in clips)
        frames = []
        for t in range(T):
            tiles = [label(clips[i][t].copy(), caps[i]) for i in range(len(clips))]
            frames.append(np.concatenate(tiles, axis=1))
        imageio.mimwrite(os.path.join(OUT, f"{out_name}.mp4"), frames, fps=fps, quality=8)
        print(f"[{out_name}] saved ({len(names)} panels, {T} frames)", flush=True)


if __name__ == "__main__":
    main()
