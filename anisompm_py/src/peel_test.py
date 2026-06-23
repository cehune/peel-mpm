"""Hand-tuned peel test on an ANALYTIC ball -- no PLY / assets needed.

Architecture (refactored): the EXPENSIVE simulation and the CHEAP observables
are split.  build_sim() + run_sim() run the physics and write a raw snapshot to
out/sims/<tag>.pt; derive.py turns that snapshot into out/results/<tag>.json.
The sim is cached on (config, hash(anisompm.py)) -- it re-runs ONLY when the
physics solver changes or the config is new; observables are always re-derived
cheaply (so changing a metric never re-runs a sim, and adding new configs never
touches existing ones).  See derive.py and run_gate.sh.

Steps:
1. three concentric regions by SDF threshold phi(X)=|X-c|-R (peel/interface/flesh),
   thicknesses auto-widened to >= 2.5 spacings.
2. per-particle materials; sigma_c(interface)=sigc_frac*E_flesh, bystanders /rho.
3. structural tensors (--aniso) + directional stress (--directional).
4. clamp bottom, pull a top cap; optionally pre-delaminate a pole patch (--notch-deg).

Run one cell:
    python3 src/peel_test.py --device cuda:0 --aniso correct --rho 0.1 --pull-deg 0
"""
import os, sys, math, time, json, hashlib, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from anisompm import AnisoMPM
from derive import derive, print_result, load_snapshot

GRID_LIM = 1.0


def _sha(path):
    try:
        return hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]
    except Exception:
        return "nofile"


SOLVER_SHA = _sha(os.path.join(os.path.dirname(__file__), "anisompm.py"))   # the PHYSICS
SETUP_SHA = _sha(os.path.abspath(__file__))                                 # the harness (info only)


def pick_device(spec):
    if spec and spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def config_dict(args):
    """Canonical, behavior-reflecting config (equal_E collapses the moduli)."""
    Ef = float(args.E_flesh)
    Ei = Ef if args.equal_E else float(args.E_int)
    Ep = Ef if args.equal_E else float(args.E_peel)
    return dict(aniso=args.aniso, directional=args.directional, rho=float(args.rho),
                pull_deg=float(args.pull_deg), equal_E=bool(args.equal_E), ngrid=int(args.ngrid),
                ppcd=float(args.ppcd), seed=int(args.seed), sigc_frac=float(args.sigc_frac),
                speed=float(args.speed), notch_deg=float(args.notch_deg), grip=str(args.grip),
                R=float(args.R), dt=float(args.dt), ramp=float(args.ramp), eta=float(args.eta),
                E_flesh=Ef, E_int=Ei, E_peel=Ep, f_clamp=str(args.f_clamp),
                frames=int(args.frames), fps=int(args.fps), damage_every=int(args.damage_every))


def config_tag(args):
    """Readable core + 6-char hash of the FULL config (collision-proof)."""
    cfg = config_dict(args)
    cid = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    core = (f"a{args.aniso}_dir{args.directional}_rho{args.rho:g}_pull{args.pull_deg:g}"
            f"_eq{int(bool(args.equal_E))}_ng{args.ngrid}_ppcd{args.ppcd:g}"
            f"_seed{args.seed}_notch{args.notch_deg:g}_grip{args.grip}")
    return f"{core}__{cid}"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ngrid", type=int, default=64)
    ap.add_argument("--ppcd", type=float, default=2.0)
    ap.add_argument("--R", type=float, default=0.28)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--dt", type=float, default=3e-4)
    ap.add_argument("--f-clamp", default="0.35,2.8", dest="f_clamp")
    ap.add_argument("--E-flesh", type=float, default=2e4, dest="E_flesh")
    ap.add_argument("--E-int", type=float, default=4e4, dest="E_int")
    ap.add_argument("--E-peel", type=float, default=1e5, dest="E_peel")
    ap.add_argument("--sigc-frac", type=float, default=0.05, dest="sigc_frac")
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--damage-every", type=int, default=1, dest="damage_every")
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--speed", type=float, default=0.12)
    ap.add_argument("--ramp", type=float, default=0.30)
    ap.add_argument("--cap-deg", type=float, default=30.0, dest="cap_deg")
    ap.add_argument("--grip", choices=["cap", "spread"], default="cap",
                    help="cap = rigid polar cap (stress concentrates at the pole edge); "
                         "spread = pull the WHOLE peel layer along pull_dir (distributed "
                         "mode-I load on the bond, no cap-edge singularity).")
    ap.add_argument("--pull-deg", type=float, default=45.0, dest="pull_deg")
    ap.add_argument("--notch-deg", type=float, default=0.0, dest="notch_deg",
                    help="pre-delaminate the interface within this many degrees of the pulled "
                         "pole (remove those particles) -- a traction-free pre-crack so the "
                         "delamination propagates from the interface instead of the cap "
                         "over-stressing the bulk.  0 (default) = no notch, identical to before.")
    ap.add_argument("--aniso", choices=["correct", "iso", "wrong"], default="correct")
    ap.add_argument("--directional", choices=["on", "off"], default="on")
    ap.add_argument("--equal-E", action="store_true", dest="equal_E")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-every", type=int, default=0, dest="plot_every")
    ap.add_argument("--log-every", type=int, default=4, dest="log_every")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "out", "peel"))
    ap.add_argument("--sims-dir", dest="sims_dir",
                    default=os.path.join(os.path.dirname(__file__), "..", "out", "sims"))
    ap.add_argument("--results-dir", dest="results_dir",
                    default=os.path.join(os.path.dirname(__file__), "..", "out", "results"))
    ap.add_argument("--force", action="store_true",
                    help="ignore the cache and re-run the sim even if a matching snapshot exists.")
    ap.add_argument("--skip-if-done", action="store_true", dest="skip_if_done",
                    help="(accepted for back-compat; caching is on by default now).")
    ap.add_argument("--summary-json", default="", dest="summary_json",
                    help="(optional) ALSO append the result as one json line here.")
    return ap.parse_args()


def ball_particles(center, R, spacing, jitter=0.3, seed=0):
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
    x = sim.x.detach().cpu().numpy(); d = sim.d.detach().cpu().numpy()
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


def build_sim(args, dev):
    """config -> (sim, geom).  Geometry, regions, materials, structural tensors,
    grips, and the optional pole notch.  Deterministic for a given config+seed."""
    c = np.array([0.5, 0.5, 0.5]) * GRID_LIM
    R = args.R; dx = GRID_LIM / args.ngrid; spacing = dx / args.ppcd; l0 = 0.5 * dx
    t_p = max(0.08 * R, 2.5 * spacing); t_i = max(0.07 * R, 2.5 * spacing)
    assert t_p + t_i < 0.7 * R, "shells eat the flesh core; raise ngrid/ppcd or R"

    pts = ball_particles(c, R, spacing, seed=args.seed)
    # --- pole notch: remove a wedge of INTERFACE particles under the pull pole --
    if args.notch_deg > 0:
        rel0 = pts - c[None]; r0 = np.linalg.norm(rel0, axis=1); phi0 = r0 - R
        peel0 = phi0 >= -t_p
        interface0 = (phi0 >= -(t_p + t_i)) & ~peel0
        cos_up0 = rel0[:, 1] / np.maximum(r0, 1e-9)
        notch0 = interface0 & (cos_up0 > math.cos(math.radians(args.notch_deg)))
        pts = pts[~notch0]
        print(f"[peel] notch: removed {int(notch0.sum()):,} interface particles "
              f"within {args.notch_deg:g} deg of the pole (pre-delamination)")

    x = torch.tensor(pts, device=dev)
    vol = torch.full((len(pts),), spacing ** 3, device=dev)
    fclamp = (None if str(args.f_clamp).lower() == "none"
              else tuple(float(v) for v in str(args.f_clamp).split(",")))
    sim = AnisoMPM(n_grid=args.ngrid, grid_lim=GRID_LIM, dt=args.dt, gravity=(0.0, 0.0, 0.0),
                   grid_damp=0.999, f_clamp=fclamp, damage_every=args.damage_every, device=dev)
    sim.add_object(x, vol, rho=500, E=args.E_flesh, nu=0.4, fibers=None, alpha=0.0,
                   percentage=1.0, eta=args.eta, zeta=1.0, residual=0.005, allow_damage=True)

    ct = torch.tensor(c, device=dev, dtype=sim.dtype)
    rel = sim.x - ct; r = rel.norm(dim=1); phi = r - R
    peel = phi >= -t_p
    interface = (phi >= -(t_p + t_i)) & ~peel
    flesh = ~peel & ~interface
    masks = dict(flesh=flesh, interface=interface, peel=peel)

    def lame(E, nu=0.4):
        return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    if args.equal_E:
        args.E_int = args.E_peel = args.E_flesh
    for m, E in ((flesh, args.E_flesh), (interface, args.E_int), (peel, args.E_peel)):
        mu, lam = lame(E); sim.mu[m] = mu; sim.lam[m] = lam
    sigc_int = args.sigc_frac * args.E_flesh
    sigc_by = sigc_int / args.rho if args.rho > 0 else 1e9
    sim.sigma_c[:] = sigc_by; sim.sigma_c[interface] = sigc_int

    r_hat = rel / r.clamp(min=1e-9).unsqueeze(1)
    rr = r_hat.unsqueeze(2) * r_hat.unsqueeze(1)
    I3 = torch.eye(3, device=dev, dtype=sim.dtype)
    sim.set_structural_tensor(flesh, (I3 - rr)[flesh])
    iface = peel | interface
    if args.aniso == "correct":
        A_if = rr
    elif args.aniso == "iso":
        A_if = I3.expand(sim.n, 3, 3)
    else:
        A_if = I3 - rr
    sim.set_structural_tensor(iface, A_if[iface])
    if args.directional == "on":
        sim.set_interface_normals(iface, r_hat[iface])

    bot = sim.x[:, 1] < c[1] - 0.55 * R
    cos_up = rel[:, 1] / r.clamp(min=1e-9)
    cap = peel & (cos_up > math.cos(math.radians(args.cap_deg)))
    # grip: 'cap' = rigid polar patch (stress concentrates at its edge);
    #       'spread' = the WHOLE peel layer pulled along pull_dir, so the load is
    #       introduced over the entire top shell -> distributed mode-I on the bond,
    #       no cap-edge singularity (the single pull_dir still spans theta_n).
    grip_region = peel if args.grip == "spread" else cap
    sim.allow[bot] = False; sim.allow[grip_region] = False
    th = math.radians(args.pull_deg)
    pull_dir = torch.tensor([math.sin(th), math.cos(th), 0.0], device=dev, dtype=sim.dtype)
    speed = args.speed; ramp = args.ramp

    def grip(s, t):
        vmag = speed * min(t / ramp, 1.0)
        s.v[bot] = 0.0; s.v[grip_region] = vmag * pull_dir
    sim.particle_bc.append(grip)

    ring = peel & (~cap) & (cos_up > math.cos(math.radians(args.cap_deg + 15)))
    halo_pred = math.exp(-spacing / l0); break_thresh = max(0.5, halo_pred + 0.25)

    print(f"[peel] R={R} spacing={spacing:.5f} dx={dx:.5f} l0={l0:.5f}")
    print(f"[peel] N={sim.n:,} flesh={int(flesh.sum()):,} interface={int(interface.sum()):,} "
          f"peel={int(peel.sum()):,} clamped={int(bot.sum()):,}  "
          f"grip={args.grip}({int(grip_region.sum()):,} particles)")
    print(f"[peel] sigma_c(int)={sigc_int:.0f} sigma_c(bystanders)={sigc_by:.0f} "
          f"rho={sigc_int / sigc_by:.2g}  pull_deg={args.pull_deg:.0f}  speed={args.speed}  "
          f"notch_deg={args.notch_deg:g}")

    geom = dict(interface=interface, flesh=flesh, peel=peel, cap=cap, ring=ring, bot=bot,
                r_hat=r_hat, x0=sim.x.clone(), pull_dir=pull_dir, masks=masks,
                break_thresh=break_thresh, halo_pred=halo_pred, spacing=spacing, l0=l0,
                c=c, R=R, t_p=t_p, t_i=t_i, sigc_int=sigc_int, sigc_by=sigc_by)
    return sim, geom


def run_sim(args, sim, geom, run_dir):
    """Run the frame loop; capture the delamination-ONSET state + peak reductions
    + final state into a snapshot dict (CPU).  Computes NO observables."""
    dev = sim.device
    interface = geom["interface"]; flesh = geom["flesh"]; peel = geom["peel"]; cap = geom["cap"]
    n_sub = max(1, int((1.0 / args.fps) / args.dt))
    t0 = time.time()
    phi_peak = torch.zeros(3, device=dev, dtype=sim.dtype)
    phi_shear_peak = torch.zeros((), device=dev, dtype=sim.dtype)
    phi_pp = torch.zeros(sim.n, device=dev, dtype=sim.dtype)
    psh_pp = torch.zeros(sim.n, device=dev, dtype=sim.dtype)
    peak_routing = torch.zeros((), device=dev, dtype=sim.dtype)
    onset_reached = False; onset_frame = -1
    d_snap = phipp_snap = pshpp_snap = tau_snap = J_snap = x_snap = None
    aborted = False
    for f in range(args.frames):
        sim.run_frame(n_sub, f / args.fps)
        ph = getattr(sim, "phi", None)
        if ph is not None:
            phf = torch.stack([ph[interface].max(), ph[flesh].max(), ph[peel].max()])
            phi_peak = torch.maximum(phi_peak, phf)
            phi_pp = torch.maximum(phi_pp, ph)
            peak_routing = torch.maximum(peak_routing, phf[0] / (torch.maximum(phf[1], phf[2]) + 1e-9))
        psh = getattr(sim, "phi_shear", None)
        if psh is not None:
            phi_shear_peak = torch.maximum(phi_shear_peak, psh[interface].max())
            psh_pp = torch.maximum(psh_pp, psh)
        if not onset_reached and float((sim.d[interface] > 0.5).float().mean()) > 0.05:
            onset_reached = True; onset_frame = f
            d_snap = sim.d.clone(); phipp_snap = phi_pp.clone(); pshpp_snap = psh_pp.clone()
            tl = getattr(sim, "tau_last", None); Jl = getattr(sim, "J_last", None)
            tau_snap = None if tl is None else tl.clone()
            J_snap = None if Jl is None else Jl.clone()
            x_snap = sim.x.clone()
            if args.plot_every:
                save_slice(sim, geom["masks"], geom["c"], geom["spacing"],
                           os.path.join(run_dir, f"slice_ONSET_f{f:03d}.png"))
        if (f % args.log_every == 0) or (f == args.frames - 1):
            di, dfl, dpe = sim.d[interface], sim.d[flesh], sim.d[peel]
            phf3 = phf if ph is not None else torch.full((3,), float("nan"), device=dev)
            row = torch.stack([di.mean(), di.max(), (di > 0.5).float().mean(), dfl.max(), dpe.max(),
                               phf3[0], phf3[1], phf3[2], sim.x[cap, 1].mean(),
                               torch.isnan(sim.x).any().to(sim.dtype)]).cpu().tolist()
            dim, dimx, brk, dflx, dpex, pi, pfl, ppe, capy, nanf = row
            phs = "" if ph is None else f" phi_max[int,fl,pe]={pi:.2f},{pfl:.2f},{ppe:.2f}"
            print(f"  f{f:3d} d_int[mean,max]={dim:.3f},{dimx:.3f} broken_int={brk:.3f}"
                  f" d_flesh_max={dflx:.3f} d_peel_max={dpex:.3f}{phs} cap_y={capy:.3f}"
                  f" nan={bool(nanf)} ({time.time()-t0:.0f}s)", flush=True)
            if nanf:
                aborted = True
                print(f"  ABORT: non-finite at frame {f} -- lower --dt / --speed / --E-peel.")
                break
        if args.plot_every and (f % args.plot_every == 0 or f == args.frames - 1):
            save_slice(sim, geom["masks"], geom["c"], geom["spacing"],
                       os.path.join(run_dir, f"slice_{f:03d}.png"))

    if onset_reached:
        d_e, phipp_e, pshpp_e, x_e, tau_e, J_e = d_snap, phipp_snap, pshpp_snap, x_snap, tau_snap, J_snap
    else:
        d_e, phipp_e, pshpp_e, x_e = sim.d, phi_pp, psh_pp, sim.x
        tau_e = getattr(sim, "tau_last", None); J_e = getattr(sim, "J_last", None)
    nan_state = bool(aborted or torch.isnan(sim.x).any())

    region = torch.zeros(sim.n, dtype=torch.int8)
    region[interface.cpu()] = 1; region[peel.cpu()] = 2     # flesh stays 0
    cpu = lambda t: None if t is None else t.detach().to("cpu")
    return dict(
        config=config_dict(args), solver_sha=SOLVER_SHA, setup_sha=SETUP_SHA,
        onset_frame=int(onset_frame), nan=nan_state, secs=round(time.time() - t0, 1), N=int(sim.n),
        region=region, cap=cpu(cap), ring=cpu(geom["ring"]),
        r_hat=cpu(geom["r_hat"]), x0=cpu(geom["x0"]), pull_dir=cpu(geom["pull_dir"]),
        break_thresh=float(geom["break_thresh"]), halo_pred=float(geom["halo_pred"]),
        d_e=cpu(d_e), phipp_e=cpu(phipp_e), pshpp_e=cpu(pshpp_e), x_e=cpu(x_e),
        tau_e=cpu(tau_e), J_e=cpu(J_e), d_f=cpu(sim.d),
        phi_peak=cpu(phi_peak), phi_shear_peak=float(phi_shear_peak), peak_routing=float(peak_routing),
    )


def main():
    args = parse_args()
    tag = config_tag(args)
    os.makedirs(args.sims_dir, exist_ok=True); os.makedirs(args.results_dir, exist_ok=True)
    snap_path = os.path.join(args.sims_dir, tag + ".pt")

    snap = None
    if os.path.exists(snap_path) and not args.force:
        cand = load_snapshot(snap_path)
        if str(cand.get("solver_sha")) == SOLVER_SHA and not bool(cand.get("nan")):
            print(f"[peel] cached sim (solver match): {tag}")
            snap = cand
        else:
            print(f"[peel] re-run (cache stale: solver "
                  f"{str(cand.get('solver_sha'))[:8]} vs {SOLVER_SHA[:8]}, nan={cand.get('nan')})")
    if snap is None:
        dev = pick_device(args.device); print(f"[peel] device={dev}  tag={tag}")
        sim, geom = build_sim(args, dev)
        run_dir = os.path.join(args.out, tag); os.makedirs(run_dir, exist_ok=True)
        snap = run_sim(args, sim, geom, run_dir)
        torch.save(snap, snap_path)
        print(f"[peel] snapshot -> {snap_path}")

    result = derive(snap)                                   # cheap; always fresh
    print_result(result)
    with open(os.path.join(args.results_dir, tag + ".json"), "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"[peel] result -> {os.path.join(args.results_dir, tag + '.json')}")
    if args.summary_json:                                   # optional back-compat log
        with open(args.summary_json, "a") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
