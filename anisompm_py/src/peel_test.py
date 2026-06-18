"""Hand-tuned peel test on an ANALYTIC ball -- no PLY / assets needed.

Implements the four steps:
1. three concentric regions by SDF threshold phi(X) = |X - c| - R
     peel       -t_p <= phi
     interface  -(t_p+t_i) <= phi < -t_p
     flesh      phi < -(t_p+t_i)
   Thicknesses are auto-widened to >= 2.5 particle spacings (the plan's
   0.03R interface is ~1.2 spacings at default resolution -- unresolvable).
2. per-particle materials. sigma_c(interface) = sigc_frac * E_flesh; the
   bystanders get sigma_c(int)/rho (default rho=0.1: a real toughness
   competition). Pass --rho 0 for unbreakable bystanders (existence mode,
   the original rigged test).
3. structural tensors via the A0 refactor in anisompm.py:
     flesh           A0 = I - r r^T  (radial fiber, alpha=-1: orange recipe)
     peel/interface  A0 = r r^T      (rank-1 -> phi = (sigma_nn^+)^2 / sigma_c^2,
                                      tangent plane fully protected; no
                                      hairy-ball singularity to dodge)
   Peel and interface share A0 and differ only in sigma_c / E.
4. clamp the bottom cap (v=0), pull a small top cap of PEEL particles at
   45 degrees ( (x+y)/sqrt2 ), ramped prescribed velocity via particle_bc.

Run (GPU box):
    python3 src/peel_test.py --device cuda:0
Mac / Metal (uses the analytic Jacobi LA path automatically on mps):
    python3 src/anisompm.py            # LA self-test first
    python3 src/peel_test.py --device mps --ngrid 48 --ppcd 1.5 --frames 24
Local CPU smoke (small, a few frames):
    python3 src/peel_test.py --device cpu --ngrid 32 --ppcd 1.5 --frames 6 --plot-every 1
"""
import os, sys, math, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from anisompm import AnisoMPM

GRID_LIM = 1.0


def pick_device(spec):
    """Resolve --device. 'auto' -> cuda, else Apple MPS, else cpu."""
    if spec and spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto",
                    help="auto (cuda->mps->cpu), or cuda:0 / mps / cpu")
    ap.add_argument("--ngrid", type=int, default=64)
    ap.add_argument("--ppcd", type=float, default=2.0, help="particles per cell per dim")
    ap.add_argument("--R", type=float, default=0.28)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dt", type=float, default=3e-4)
    ap.add_argument("--f-clamp", default="0.35,2.8", dest="f_clamp",
                    help="F singular-value clamp 'smin,smax' (stability) or 'none'. "
                         "The analytic polar is float32 and the clamp rebuild can "
                         "NaN on aggressive configs (pre-existing; see LA self-test). "
                         "The gate uses 'none', which is stable and still develops "
                         "damage.")
    ap.add_argument("--E-flesh", type=float, default=2e4, dest="E_flesh")
    ap.add_argument("--E-int", type=float, default=4e4, dest="E_int")
    ap.add_argument("--E-peel", type=float, default=1e5, dest="E_peel")
    ap.add_argument("--sigc-frac", type=float, default=0.05, dest="sigc_frac",
                    help="sigma_c(interface) = sigc_frac * E_flesh")
    ap.add_argument("--rho", type=float, default=0.1,
                    help="toughness ratio: sigma_c(flesh,peel) = sigma_c(int)/rho; "
                         "0 = unbreakable bystanders (existence mode)")
    ap.add_argument("--damage-every", type=int, default=1, dest="damage_every",
                    help="update the damage field every k substeps (speed)")
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--speed", type=float, default=0.12, help="peel pull speed")
    ap.add_argument("--ramp", type=float, default=0.30)
    ap.add_argument("--cap-deg", type=float, default=30.0, dest="cap_deg",
                    help="half-angle of the pulled top peel cap")
    ap.add_argument("--pull-deg", type=float, default=45.0, dest="pull_deg",
                    help="pull angle off the pole normal (+y). 0=pure normal "
                         "(mode-I tension), 90=tangential (pure shear). "
                         "Pull dir = (sin, cos, 0) in the x-y plane.")
    ap.add_argument("--aniso", choices=["correct", "iso", "wrong"], default="correct",
                    help="interface (peel|interface) damage-DRIVER tensor A0: "
                         "correct=n(x)n (normal isolation), iso=I (isotropic, no "
                         "directional selection), wrong=I-n(x)n (tangential "
                         "isolation, protects the normal). Flesh stays I-r(x)r.")
    ap.add_argument("--directional", choices=["on", "off"], default="on",
                    help="interface STRESS model: on=directional split (release "
                         "normal+shear, keep in-plane) via directional_kirchhoff; "
                         "off=isotropic g(d) (bonded baseline). Decoupled from "
                         "--aniso so you can A/B the stress model with the driver "
                         "held fixed.")
    ap.add_argument("--equal-E", action="store_true", dest="equal_E",
                    help="force E_int=E_peel=E_flesh: kills modulus contrast so "
                         "correct/iso/wrong differ ONLY in the structural tensor.")
    ap.add_argument("--seed", type=int, default=0, help="particle-lattice jitter seed")
    ap.add_argument("--plot-every", type=int, default=4, dest="plot_every",
                    help="save a cross-section png every k frames (0=off)")
    ap.add_argument("--log-every", type=int, default=4, dest="log_every",
                    help="print a metrics line every k frames (fewer device syncs)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "out", "peel"))
    ap.add_argument("--summary-json", default="", dest="summary_json",
                    help="append a one-line JSON result record to this path "
                         "(used by sweep_peel.py)")
    return ap.parse_args()


def ball_particles(center, R, spacing, jitter=0.3, seed=0):
    """Jittered lattice filling a ball. Returns (N,3) float32 positions."""
    rng = np.random.default_rng(seed)
    n = int(np.ceil(2.0 * R / spacing)) + 2
    g = (np.arange(n) + 0.5) * spacing
    g -= g.mean()
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    pts = pts[np.linalg.norm(pts, axis=1) < R]
    pts += (rng.random(pts.shape) - 0.5) * (jitter * spacing)
    return (pts + np.asarray(center)[None]).astype(np.float32)


def save_slice(sim, masks, center, spacing, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = sim.x.detach().cpu().numpy()
    d = sim.d.detach().cpu().numpy()
    sl = np.abs(x[:, 2] - center[2]) < 1.5 * spacing
    fig, axs = plt.subplots(1, 2, figsize=(11, 5), dpi=110)
    colors = {"flesh": "#bbbbbb", "interface": "#ff7f0e", "peel": "#2ca02c"}
    for name, m in masks.items():
        mm = sl & m.cpu().numpy()
        axs[0].scatter(x[mm, 0], x[mm, 1], s=2, c=colors[name], label=name)
    axs[0].legend(markerscale=4); axs[0].set_title("regions")
    sc = axs[1].scatter(x[sl, 0], x[sl, 1], s=2, c=d[sl], cmap="inferno", vmin=0, vmax=1)
    fig.colorbar(sc, ax=axs[1]); axs[1].set_title("damage d")
    for ax in axs:
        ax.set_aspect("equal"); ax.set_xlim(0.1, 0.9); ax.set_ylim(0.1, 0.9)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    args = parse_args()
    dev = pick_device(args.device)
    print(f"[peel] device={dev}")
    c = np.array([0.5, 0.5, 0.5]) * GRID_LIM
    R = args.R
    dx = GRID_LIM / args.ngrid
    spacing = dx / args.ppcd
    l0 = 0.5 * dx  # solver default (l0_scale=0.5)

    # ---- Step 1: geometry. Enforce the resolvability rule explicitly. ------
    t_p = max(0.08 * R, 2.5 * spacing)
    t_i = max(0.07 * R, 2.5 * spacing)  # plan said 0.03R; see module docstring
    assert t_p + t_i < 0.7 * R, "shells eat the flesh core; raise ngrid/ppcd or R"

    pts = ball_particles(c, R, spacing, seed=args.seed)
    x = torch.tensor(pts, device=dev)
    vol = torch.full((len(pts),), spacing ** 3, device=dev)

    fclamp = (None if str(args.f_clamp).lower() == "none"
              else tuple(float(v) for v in str(args.f_clamp).split(",")))
    sim = AnisoMPM(n_grid=args.ngrid, grid_lim=GRID_LIM, dt=args.dt,
                   gravity=(0.0, 0.0, 0.0), grid_damp=0.999,
                   f_clamp=fclamp, damage_every=args.damage_every, device=dev)
    # placeholder elasticity/sigma_c -- overwritten per region below
    sim.add_object(x, vol, rho=500, E=args.E_flesh, nu=0.4,
                   fibers=None, alpha=0.0, percentage=1.0,
                   eta=args.eta, zeta=1.0, residual=0.005, allow_damage=True)

    ct = torch.tensor(c, device=dev, dtype=sim.dtype)
    rel = sim.x - ct
    r = rel.norm(dim=1)
    phi = r - R
    peel = phi >= -t_p
    interface = (phi >= -(t_p + t_i)) & ~peel
    flesh = ~peel & ~interface
    masks = dict(flesh=flesh, interface=interface, peel=peel)

    # ---- Step 2: per-particle materials ------------------------------------
    def lame(E, nu=0.4):
        return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

    if args.equal_E:                          # Block 1: strip modulus contrast
        args.E_int = args.E_peel = args.E_flesh
    for m, E in ((flesh, args.E_flesh), (interface, args.E_int), (peel, args.E_peel)):
        mu, lam = lame(E)
        sim.mu[m] = mu
        sim.lam[m] = lam
    sigc_int = args.sigc_frac * args.E_flesh  # designated failure zone
    sigc_by = sigc_int / args.rho if args.rho > 0 else 1e9  # bystanders: tough or unbreakable
    sim.sigma_c[:] = sigc_by
    sim.sigma_c[interface] = sigc_int

    # ---- Step 3: structural tensors (AFTER add_object -- it rebuilds A0) ---
    r_hat = rel / r.clamp(min=1e-9).unsqueeze(1)
    rr = r_hat.unsqueeze(2) * r_hat.unsqueeze(1)  # (N,3,3) r x r
    I3 = torch.eye(3, device=dev, dtype=sim.dtype)
    sim.set_structural_tensor(flesh, (I3 - rr)[flesh])             # flesh: orange recipe
    # interface (peel|interface) damage-DRIVER tensor A0, per --aniso:
    iface = peel | interface
    if args.aniso == "correct":
        A_if = rr                                  # n(x)n -> phi=(sigma_nn^+/sigma_c)^2
    elif args.aniso == "iso":
        A_if = I3.expand(sim.n, 3, 3)              # isotropic driver (no selection)
    else:                                          # "wrong": tangential isolation
        A_if = I3 - rr                             # protects normal, drives on shear
    sim.set_structural_tensor(iface, A_if[iface])
    # STRESS model, decoupled from the driver: registering the interface normals
    # switches substep to the directional split there.  --directional off leaves
    # the isotropic g(d) bonded baseline (matrix still holds the bond at d=1).
    if args.directional == "on":
        sim.set_interface_normals(iface, r_hat[iface])

    # ---- Step 4: clamp bottom, pull top peel cap at 45 degrees -------------
    bot = sim.x[:, 1] < c[1] - 0.55 * R
    cos_up = rel[:, 1] / r.clamp(min=1e-9)
    cap = peel & (cos_up > math.cos(math.radians(args.cap_deg)))
    sim.allow[bot] = False
    sim.allow[cap] = False  # grip regions never damage
    th = math.radians(args.pull_deg)  # angle off the +y pole normal
    pull_dir = torch.tensor([math.sin(th), math.cos(th), 0.0],
                            device=dev, dtype=sim.dtype)

    def grip(s, t):
        vmag = args.speed * min(t / args.ramp, 1.0)
        s.v[bot] = 0.0
        s.v[cap] = vmag * pull_dir
    sim.particle_bc.append(grip)

    print(f"[peel] R={R} spacing={spacing:.5f} dx={dx:.5f} l0={l0:.5f}")
    print(f"[peel] t_p={t_p:.4f} ({t_p/spacing:.1f} spacings)  "
          f"t_i={t_i:.4f} ({t_i/spacing:.1f} spacings, {t_i/l0:.1f} l0)")
    print(f"[peel] N={sim.n:,} flesh={int(flesh.sum()):,} interface={int(interface.sum()):,} "
          f"peel={int(peel.sum()):,} cap={int(cap.sum()):,} clamped={int(bot.sum()):,}")
    print(f"[peel] sigma_c(int)={sigc_int:.0f} sigma_c(bystanders)={sigc_by:.0f} "
          f"rho={sigc_int / sigc_by:.2g}"
          f"{' [existence mode: bystanders unbreakable]' if args.rho <= 0 else ''}")
    print(f"[peel] pull_deg={args.pull_deg:.0f} off normal  "
          f"pull_dir=({float(pull_dir[0]):.3f},{float(pull_dir[1]):.3f},0)  "
          f"speed={args.speed}")

    # per-run output folder whose name encodes the settings (so a sweep does not
    # overwrite itself).  e.g. out/peel/rho0.1_pull45_sigc0.05_ng64_ppcd2_R0.28
    tag = (f"rho{args.rho:g}_pull{args.pull_deg:g}_sigc{args.sigc_frac:g}"
           f"_ng{args.ngrid}_ppcd{args.ppcd:g}_R{args.R:g}")
    run_dir = os.path.join(args.out, tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[peel] outputs -> {run_dir}")
    n_sub = max(1, int((1.0 / args.fps) / args.dt))
    t0 = time.time()
    phi_peak = torch.zeros(3, device=dev, dtype=sim.dtype)    # [int, flesh, peel], on device
    phi_shear_peak = torch.zeros((), device=dev, dtype=sim.dtype)  # interface mode-II drive
    for f in range(args.frames):
        sim.run_frame(n_sub, f / args.fps)
        # running phi maxima accumulate on-device -- no host sync per frame
        ph = getattr(sim, "phi", None)
        if ph is not None:
            phf = torch.stack([ph[interface].max(), ph[flesh].max(), ph[peel].max()])
            phi_peak = torch.maximum(phi_peak, phf)
        psh = getattr(sim, "phi_shear", None)
        if psh is not None:
            phi_shear_peak = torch.maximum(phi_shear_peak, psh[interface].max())

        if (f % args.log_every == 0) or (f == args.frames - 1):
            di, dfl, dpe = sim.d[interface], sim.d[flesh], sim.d[peel]
            phf3 = phf if ph is not None else torch.full((3,), float("nan"), device=dev)
            # gather every scalar into ONE tensor -> a single device->host copy
            row = torch.stack([
                di.mean(), di.max(), (di > 0.5).float().mean(),
                dfl.max(), dpe.max(),
                phf3[0], phf3[1], phf3[2],
                sim.x[cap, 1].mean(),
                torch.isnan(sim.x).any().to(sim.dtype),
            ]).cpu().tolist()
            dim, dimx, brk, dflx, dpex, pi, pfl, ppe, capy, nan_ = row
            phs = "" if ph is None else f" phi_max[int,fl,pe]={pi:.2f},{pfl:.2f},{ppe:.2f}"
            print(f"  f{f:3d} d_int[mean,max]={dim:.3f},{dimx:.3f}"
                  f" broken_int={brk:.3f}"
                  f" d_flesh_max={dflx:.3f} d_peel_max={dpex:.3f}"
                  f"{phs} cap_y={capy:.3f} nan={bool(nan_)} ({time.time()-t0:.0f}s)", flush=True)
            if nan_:                                          # diverged: stop now
                print(f"  ABORT: non-finite state at frame {f} -- "
                      f"lower --dt / --speed / --E-peel, or check the LA self-test.")
                break
        if args.plot_every and (f % args.plot_every == 0 or f == args.frames - 1):
            save_slice(sim, masks, c, spacing, os.path.join(run_dir, f"slice_{f:03d}.png"))

    # ---- verdict ------------------------------------------------------------
    # NOTE on selectivity: the phase-field crack is a diffuse band of width
    # ~l0 that ignores region labels -- bystanders one spacing from a d=1 band
    # equilibrate at d ~ exp(-spacing/l0) by the Laplacian term alone. That
    # halo is the method working, not leakage. Failure means a bystander
    # actually BREAKS (d > 0.5), so that's what we test.
    broken_int_frac = float((sim.d[interface] > 0.5).float().mean())
    exist = broken_int_frac > 0.05
    n_out = int((sim.d[flesh] > 0.5).sum()) + int((sim.d[peel] > 0.5).sum())
    d_bystander_max = max(float(sim.d[flesh].max()), float(sim.d[peel].max()))
    halo_pred = math.exp(-spacing / l0)

    # routing: interface drive vs the toughest bystander drive.  With contrast and
    # geometry stripped (Block 1: rho=1, --equal-E) this isolates the structural
    # tensor's selectivity.  Use the RATIO; iso need not be ~1 (pole geometry
    # concentrates stress regardless), but correct should route >> iso.
    pk = phi_peak.cpu().tolist()                              # [int, flesh, peel]
    phi_bystander_max = max(pk[1], pk[2])
    routing = pk[0] / (phi_bystander_max + 1e-9)

    # grip-leak check (meaningful, unlike grip_d on the clamped cap which is
    # frozen by allow=False): max damage in the ALLOWED peel ring just below the
    # pulled cap.  High here = the pull is tearing at the grip, not delaminating.
    ring = peel & (~cap) & (cos_up > math.cos(math.radians(args.cap_deg + 15)))
    grip_d_max = float(sim.d[ring].max()) if int(ring.sum()) > 0 else 0.0

    # directional-stress observable: at damaged interface particles the normal
    # traction should collapse (normal_release -> 1) while the in-plane stress is
    # retained (inplane_keep -> 1).  Isotropic g(d) ALSO releases the normal under
    # tension, but inplane_keep -> 0 there -- so inplane_keep is the discriminant
    # that the directional split (not just damage) is doing the work.
    normal_release = inplane_keep = float("nan")
    tau_last = getattr(sim, "tau_last", None)
    if tau_last is not None and int(interface.sum()) > 0:
        sig = tau_last / sim.J_last.view(-1, 1, 1)            # degraded Cauchy stress
        nrm = r_hat
        Sn = torch.einsum('pij,pj->pi', sig, nrm)
        tn = (Sn * nrm).sum(1).abs()                          # |normal traction|
        Pn = nrm.unsqueeze(2) * nrm.unsqueeze(1)
        Qn = I3 - Pn
        sip = (Qn @ sig @ Qn).reshape(sim.n, -1).norm(dim=1)  # in-plane stress mag
        intact = interface & (sim.d < 0.1)
        broken = interface & (sim.d > 0.5)
        if int(intact.sum()) > 0 and int(broken.sum()) > 0:
            normal_release = 1.0 - float(tn[broken].mean() / (tn[intact].mean() + 1e-9))
            inplane_keep = float(sip[broken].mean() / (sip[intact].mean() + 1e-9))

    print(f"\n[peel] aniso={args.aniso} directional={args.directional} "
          f"equal_E={args.equal_E} seed={args.seed}")
    print(f"[peel] EXISTENCE (>5% interface broken): {'PASS' if exist else 'FAIL'}  "
          f"(broken_int_frac={broken_int_frac:.3f})")
    print(f"[peel] SELECTIVITY (no bystander breaks): {'PASS' if n_out == 0 else 'FAIL'}  "
          f"(broken_outside={n_out}, d_bystander_max={d_bystander_max:.2f}, halo~{halo_pred:.2f})")
    print(f"[peel] routing phi_int/phi_bystander={routing:.2f}  "
          f"(phi_int={pk[0]:.2f} phi_bystander={phi_bystander_max:.2f} phi_shear={float(phi_shear_peak):.2f})")
    print(f"[peel] directional: normal_release={normal_release:.2f} "
          f"inplane_keep={inplane_keep:.2f}  grip_d_max={grip_d_max:.3f}")
    if not exist:
        print("  -> lower --sigc-frac, raise --speed, or run more --frames")

    if args.summary_json:
        import json
        rec = dict(
            aniso=args.aniso, directional=args.directional, equal_E=bool(args.equal_E),
            seed=args.seed, rho=args.rho, pull_deg=args.pull_deg, sigc_frac=args.sigc_frac,
            speed=args.speed, ngrid=args.ngrid, ppcd=args.ppcd, frames=args.frames,
            phi_int_max=pk[0], phi_bystander_max=phi_bystander_max,
            phi_flesh_max=pk[1], phi_peel_max=pk[2], phi_shear_max=float(phi_shear_peak),
            routing=routing, broken_int_frac=broken_int_frac,
            d_int_max=float(sim.d[interface].max()), d_bystander_max=d_bystander_max,
            grip_d_max=grip_d_max, normal_release=normal_release, inplane_keep=inplane_keep,
            n_broken_outside=n_out, exist=bool(exist), select=bool(n_out == 0),
            nan=bool(torch.isnan(sim.x).any()), secs=round(time.time() - t0, 1),
        )
        with open(args.summary_json, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[peel] appended summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
