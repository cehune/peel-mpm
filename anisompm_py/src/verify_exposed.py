#!/usr/bin/env python3
"""verify_exposed.py -- check that the 'freshly exposed' crack points trace the
peel front, not the seeded notch and not the clamp.

  python3 src/verify_exposed.py out/slab/peel_dump.npz

Exposed set := (d_now > 0.5) & ~d0, where d0 = everything broken at t=0 (the
starter notch). That is the surface the peel ACTUALLY created -- the thing whose
appearance is your contribution -- as opposed to the boundary condition you typed.

Outputs (next to the dump): <name>_exposed_strip.png and <name>_exposed.gif.
Prints a per-frame table + automated PASS/FAIL gates so you can confirm:
frame-0 exposed ~ 0, count grows, and (peel) the front marches high-x -> low-x.
"""
import os, argparse
import numpy as np


def exposed_mask(d_now, d0, thr=0.5):
    """Points broken NOW but not broken at t=0 -> created by the peel."""
    return (d_now > thr) & (~d0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--zslab", type=float, default=0.5)
    ap.add_argument("--no-anim", action="store_true")
    a = ap.parse_args()

    z = np.load(a.dump, allow_pickle=True)
    X = z["x"]              # (F, N, 3)
    D = z["d"]              # (F, N)
    d0 = z["d0"].astype(bool)   # (N,)
    mode = str(z["mode"])
    Ly = float(z["Ly"]); grid = int(z["grid"])
    s = 0.5 * (1.0 / grid)
    cY = 0.5
    y_clamp = -Ly / 2 + 2 * s     # relative-y below which is the anchored bulk

    F, N, _ = X.shape
    print(f"loaded {a.dump}: {F} frames, {N} pts, mode={mode}")
    print(f"seeded-broken (d0): {int(d0.sum())} pts  [EXCLUDED from exposed]\n")

    hdr = f"{'frame':>5} {'n_exp':>7} {'n_new':>6} {'x_min':>8} {'x_mean':>8} {'x_max':>8} {'y_rel_min':>10}"
    print(hdr)
    counts, xmeans, masks = [], [], []
    for f in range(F):
        m = exposed_mask(D[f], d0)
        masks.append(m)
        new = m & ~masks[f - 1] if f > 0 else m
        n = int(m.sum()); counts.append(n)
        if n:
            xe = X[f][m, 0]; ye = X[f][m, 1] - cY
            xmeans.append(float(xe.mean()))
            print(f"{f:>5} {n:>7} {int(new.sum()):>6} {xe.min():>8.3f} "
                  f"{xe.mean():>8.3f} {xe.max():>8.3f} {ye.min():>10.3f}")
        else:
            xmeans.append(np.nan)
            print(f"{f:>5} {n:>7} {0:>6} {'-':>8} {'-':>8} {'-':>8} {'-':>10}")

    print("\n--- checks ---")
    f0 = counts[0]
    ok0 = f0 <= max(3, int(0.001 * N))
    print(f"[{'PASS' if ok0 else 'FAIL'}] frame-0 exposed ~= 0  (got {f0})"
          f"{'' if ok0 else '   <- d0 subtraction wrong: you are rendering the seeded notch'}")

    grew = counts[-1] > counts[0]
    print(f"[{'PASS' if grew else 'FAIL'}] exposed count grows  ({counts[0]} -> {counts[-1]})")

    if mode == "peel":
        valid = [x for x in xmeans if not np.isnan(x)]
        if len(valid) >= 2:
            marched = valid[-1] < valid[0] - s
            print(f"[{'PASS' if marched else 'CHECK'}] front marches high-x -> low-x  "
                  f"(x_mean {valid[0]:.3f} -> {valid[-1]:.3f})")
        else:
            print("[CHECK] too few exposed frames to judge front motion")
        clamp_hit = any((X[f][masks[f], 1] - cY < y_clamp).any()
                        for f in range(F) if counts[f])
        print(f"[{'PASS' if not clamp_hit else 'FAIL'}] no exposed pts in clamp zone "
              f"(y_rel < {y_clamp:.3f})"
              f"{'' if not clamp_hit else '   <- anchor is being labeled as crack'}")

    if a.no_anim:
        return

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    base = os.path.splitext(a.dump)[0]
    allx = X.reshape(-1, 3)
    lims = (allx[:, 0].min() - 0.03, allx[:, 0].max() + 0.03,
            allx[:, 1].min() - 0.03, allx[:, 1].max() + 0.03)

    def draw(ax, f):
        ax.clear()
        x = X[f]; m = masks[f]
        prev = masks[f - 1] if f > 0 else np.zeros_like(m)
        old = m & prev; new = m & ~prev
        zsl = np.abs(x[:, 2] - a.zslab) < 0.5
        ax.scatter(x[zsl & ~m, 0], x[zsl & ~m, 1], s=8, c="#dddddd")          # everything else
        if old.any():
            ax.scatter(x[zsl & old, 0], x[zsl & old, 1], s=12, c="#d62728")   # already exposed
        if new.any():
            ax.scatter(x[zsl & new, 0], x[zsl & new, 1], s=24, c="#ff7f0e",
                       edgecolors="k", linewidths=0.3)                        # FRONT this frame
        ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"exposed=created by peel  frame {f}  n={int(m.sum())}", fontsize=10)

    n = 6; idx = np.linspace(0, F - 1, n).astype(int)
    fig, axs = plt.subplots(1, n, figsize=(3.0 * n, 3.2), dpi=120)
    for ax, i in zip(axs, idx):
        draw(ax, i)
    fig.tight_layout(); fig.savefig(base + "_exposed_strip.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {base}_exposed_strip.png")

    fig2, ax2 = plt.subplots(figsize=(6.2, 4.2), dpi=110); imgs = []
    for f in range(F):
        draw(ax2, f); fig2.canvas.draw()
        w, h = fig2.canvas.get_width_height()
        imgs.append(Image.fromarray(np.frombuffer(
            fig2.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()))
    plt.close(fig2)
    imgs[0].save(base + "_exposed.gif", save_all=True,
                 append_images=imgs[1:], duration=60, loop=0)
    print(f"wrote {base}_exposed.gif")


if __name__ == "__main__":
    main()
    