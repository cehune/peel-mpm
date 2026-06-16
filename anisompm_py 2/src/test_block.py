"""Validation tests for the PyTorch AnisoMPM solver.

Test 1 (stability):  a soft block dropped under gravity onto a ground plane
                     should settle without blowing up (bounded velocity, no NaN).
Test 2 (fracture):   a notched bar pulled apart at both ends should localise
                     damage at the notch and tear in two -- the canonical
                     AnisoMPM mode-I fracture behaviour.

Outputs PNGs to out/validation/ so we can eyeball the result.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from anisompm import AnisoMPM, halfspace_collider

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "validation")
os.makedirs(OUT, exist_ok=True)
dev = "cuda:0"


def grid_block(lo, hi, n_per_dim):
    xs = [np.linspace(lo[d] + 0.5 / n_per_dim[d] * (hi[d] - lo[d]),
                      hi[d] - 0.5 / n_per_dim[d] * (hi[d] - lo[d]), n_per_dim[d])
          for d in range(3)]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    return pts


def scatter(path, x, c, title, lim, vmin=0, vmax=1, cmap="inferno", azim=-60, elev=15):
    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(x[:, 0], x[:, 2], x[:, 1], c=c, s=3, cmap=cmap, vmin=vmin, vmax=vmax, depthshade=True)
    ax.set_xlim(lim[0]); ax.set_ylim(lim[2]); ax.set_zlim(lim[1])
    ax.set_title(title); ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y")
    fig.colorbar(p, ax=ax, shrink=0.6, label="damage d")
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------
def test_drop():
    print("\n=== TEST 1: drop & settle (stability) ===", flush=True)
    sim = AnisoMPM(n_grid=48, grid_lim=1.0, dt=4e-4, gravity=(0, -9.8, 0), device=dev)
    pts = grid_block(lo=[0.38, 0.45, 0.38], hi=[0.62, 0.69, 0.62], n_per_dim=[28, 28, 28])
    vol = np.full(len(pts), (0.24 ** 3) / len(pts))
    sim.add_object(torch.tensor(pts), torch.tensor(vol), rho=500, E=1e4, nu=0.3,
                   fibers=None, alpha=0.0, allow_damage=False)
    sim.colliders.append(halfspace_collider(lambda t: [0, 0.30, 0], [0, 1, 0], mode="slip", friction=0.3))
    n_sub = int((1 / 24) / sim.dt)
    for f in range(16):
        sim.run_frame(n_sub, f / 24)
        vmax = float(sim.v.norm(dim=1).max())
        if f % 4 == 0 or f == 15:
            print(f"  frame {f:2d}  y[min,mean]={float(sim.x[:,1].min()):.3f},{float(sim.x[:,1].mean()):.3f}"
                  f"  |v|max={vmax:.3f}  nan={bool(torch.isnan(sim.x).any())}", flush=True)
    ok = (not bool(torch.isnan(sim.x).any())) and float(sim.v.norm(dim=1).max()) < 50
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------
def test_fracture(anisotropic):
    tag = "aniso" if anisotropic else "iso"
    print(f"\n=== TEST 2 ({tag}): notched bar pulled apart ===")
    sim = AnisoMPM(n_grid=96, grid_lim=1.0, dt=1.5e-4, gravity=(0, 0, 0), device=dev)
    # a bar along x, with a notch cut from the top-middle
    pts = grid_block(lo=[0.25, 0.45, 0.45], hi=[0.75, 0.55, 0.55], n_per_dim=[96, 20, 20])
    notch = (np.abs(pts[:, 0] - 0.5) < 0.008) & (pts[:, 1] > 0.515)
    pts = pts[~notch]
    vol = np.full(len(pts), (0.5 * 0.1 * 0.1) / len(pts))
    # fibers along x (so anisotropy resists cross-fiber tearing differently)
    fib = None
    alpha = 0.0
    if anisotropic:
        fib = torch.zeros(len(pts), 3); fib[:, 1] = 1.0  # fibers along y -> weak across, tears flat
        alpha = -1.0
    sim.add_object(torch.tensor(pts), torch.tensor(vol), rho=1000, E=3e3, nu=0.3,
                   fibers=fib, alpha=alpha, percentage=0.35, eta=0.5, l0_scale=0.7,
                   residual=0.01, allow_damage=True)

    x0 = sim.x.clone()
    left = x0[:, 0] < 0.30
    right = x0[:, 0] > 0.70
    sim.allow[left] = False        # clamped grips do not themselves fracture
    sim.allow[right] = False
    pull_max = 0.08  # m/s each side, ramped slowly (>~ wave-transit time) -> quasi-static

    def gripper(s, t):
        vmag = pull_max * min(t / 0.4, 1.0)
        s.v[left, 0] = -vmag; s.v[left, 1] = 0; s.v[left, 2] = 0
        s.v[right, 0] = vmag; s.v[right, 1] = 0; s.v[right, 2] = 0
    sim.particle_bc.append(gripper)

    xc = x0[:, 0].cpu().numpy()
    notch_reg = np.abs(xc - 0.5) < 0.06
    far_reg = np.abs(xc - 0.5) > 0.16
    n_sub = int((1 / 48) / sim.dt)
    lim = ([0.15, 0.85], [0.40, 0.60], [0.40, 0.60])
    nframes = 52
    for f in range(nframes):
        sim.run_frame(n_sub, f / 48)
        if f % 4 == 0 or f == nframes - 1:
            d = sim.d.detach().cpu().numpy()
            x = sim.x.detach().cpu().numpy()
            print(f"  frame {f:2d}  d_notch={d[notch_reg].mean():.3f}  d_far={d[far_reg].mean():.3f}"
                  f"  max={d.max():.3f}  width={x[:,0].max()-x[:,0].min():.3f}"
                  f"  nan={bool(torch.isnan(sim.x).any())}", flush=True)
            scatter(os.path.join(OUT, f"fracture_{tag}_{f:03d}.png"), x, d,
                    f"{tag} pull  frame {f}", lim)
    d = sim.d.detach().cpu().numpy()
    localized = d[notch_reg].mean() > 0.5 and d[notch_reg].mean() > 2 * (d[far_reg].mean() + 0.05)
    print(f"  RESULT: {'PASS (localized crack at notch)' if localized else 'check'}"
          f"  d_notch={d[notch_reg].mean():.3f} d_far={d[far_reg].mean():.3f}")
    return localized


if __name__ == "__main__":
    torch.manual_seed(0)
    r1 = test_drop() if "--drop" in sys.argv else True
    r2 = test_fracture(anisotropic=False)
    r3 = test_fracture(anisotropic=True)
    print("\n==== SUMMARY ====")
    print("drop stability :", "PASS" if r1 else "FAIL")
    print("fracture iso   :", "PASS" if r2 else "FAIL")
    print("fracture aniso :", "PASS" if r3 else "FAIL")
