#!/usr/bin/env python3
"""derive.py -- compute peel-test observables from a cached sim snapshot.

The expensive simulation (peel_test.py) writes a snapshot to out/sims/<tag>.pt
holding the raw delamination-ONSET state + peak reductions + final state (the
"contract").  This module is the SINGLE SOURCE OF TRUTH for every verdict
observable -- EXISTENCE, SELECTIVITY@onset, routing(peak), the cos^2(theta)
angular law, the directional release/keep, the normal gap, the grip leak.

derive(snap) is a PURE function of the snapshot: no simulation, no GPU.  So
changing how an observable is computed re-derives in milliseconds from cached
snapshots instead of re-running the (minutes-long) sim:

    python3 src/derive.py out/sims              # re-derive ALL  -> out/results/*.json
    python3 src/derive.py out/sims/<tag>.pt     # just one

The sim re-runs only when the physics (anisompm.py) changes; observables are
always re-derived.  That is what keeps "never re-run unless functionality
changed" true while never serving a stale metric.
"""
import os, sys, json, glob
import torch


def derive(snap):
    """snap: dict of CPU tensors + scalars + 'config' (from peel_test.run_sim, or
    torch.load of a snapshot .pt).  Returns a json-serializable result dict.

    NOTE: the bodies below are the peel verdict code moved verbatim -- they MUST
    stay identical to what peel_test used to compute inline, so the refactor is
    behavior-preserving.  The only change is the source: snap['...'] not sim."""
    cfg = snap["config"]
    region = snap["region"]                       # int8 (N,): 0 flesh, 1 interface, 2 peel
    interface = region == 1
    flesh = region == 0
    peel = region == 2
    cap = snap["cap"]; ring = snap["ring"]
    r_hat = snap["r_hat"]; x0 = snap["x0"]; pull_dir = snap["pull_dir"]
    break_thresh = float(snap["break_thresh"]); halo_pred = float(snap["halo_pred"])
    onset_frame = int(snap["onset_frame"]); onset_reached = onset_frame >= 0
    d_e = snap["d_e"]; phipp_e = snap["phipp_e"]; pshpp_e = snap["pshpp_e"]
    x_e = snap["x_e"]; tau_e = snap.get("tau_e"); J_e = snap.get("J_e")
    d_f = snap["d_f"]
    phi_peak = snap["phi_peak"]
    phi_shear_peak = float(snap["phi_shear_peak"]); peak_routing = float(snap["peak_routing"])
    N = int(region.shape[0])
    I3 = torch.eye(3, dtype=d_e.dtype)

    # ---- EXISTENCE (achieved monotone max) + SELECTIVITY (read at onset) ------
    broken_int_frac = float((d_f[interface] > 0.5).float().mean())
    exist = bool(onset_reached) or broken_int_frac > 0.05
    n_out = int((d_e[flesh] > break_thresh).sum()) + int((d_e[peel] > break_thresh).sum())
    d_bystander_max = max(float(d_e[flesh].max()), float(d_e[peel].max()))         # at onset
    d_bystander_final = max(float(d_f[flesh].max()), float(d_f[peel].max()))       # over-pull tail

    # ---- routing (peak over the run) ------------------------------------------
    pk = phi_peak.tolist()                                  # [int, flesh, peel] running max
    phi_bystander_max = max(pk[1], pk[2])
    routing = float(peak_routing)

    # ---- Block 2: cos^2(theta) angular law (phi_int/phi_shear ~ cot^2) --------
    cth = (r_hat @ pull_dir).clamp(-1.0, 1.0)               # cos(theta_n) per particle
    loaded = interface & (cth > 0) & (phipp_e > 1e-6) & (pshpp_e > 1e-9)
    cos2_slope = float("nan"); cos2_r2 = float("nan"); cos2_n = int(loaded.sum())
    if cos2_n > 8:
        cos2 = cth[loaded] ** 2
        sin2 = (1.0 - cos2).clamp(min=1e-6)
        ratio_pred = cos2 / sin2                            # cot^2(theta_n)
        ratio_obs = phipp_e[loaded] / (pshpp_e[loaded] + 1e-12)
        m2 = (ratio_obs > 1e-6) & (ratio_pred > 1e-3) & (ratio_pred < 1e3)
        if int(m2.sum()) > 8:
            X = torch.log(ratio_pred[m2]); Y = torch.log(ratio_obs[m2])
            Xb = X - X.mean(); Yb = Y - Y.mean()
            cos2_slope = float((Xb @ Yb) / (Xb @ Xb + 1e-12))
            ss_res = float(((Yb - cos2_slope * Xb) ** 2).sum())
            cos2_r2 = 1.0 - ss_res / (float((Yb ** 2).sum()) + 1e-12)

    # ---- grip-leak (allowed peel ring just below the cap) ---------------------
    grip_d_max = float(d_e[ring].max()) if int(ring.sum()) > 0 else 0.0

    # ---- directional observable: normal_release / inplane_keep (at onset) -----
    normal_release = inplane_keep = float("nan")
    if tau_e is not None and J_e is not None and int(interface.sum()) > 0:
        sig = tau_e / J_e.view(-1, 1, 1)                    # degraded Cauchy stress (at onset)
        nrm = r_hat
        Sn = torch.einsum('pij,pj->pi', sig, nrm)
        tn = (Sn * nrm).sum(1).abs()                        # |normal traction|
        Pn = nrm.unsqueeze(2) * nrm.unsqueeze(1)
        Qn = I3 - Pn
        sip = (Qn @ sig @ Qn).reshape(N, -1).norm(dim=1)    # in-plane stress mag
        intact = interface & (d_e < 0.1)
        broken = interface & (d_e > 0.5)
        if int(intact.sum()) > 0 and int(broken.sum()) > 0:
            normal_release = 1.0 - float(tn[broken].mean() / (tn[intact].mean() + 1e-9))
            inplane_keep = float(sip[broken].mean() / (sip[intact].mean() + 1e-9))

    # ---- Block-D kinematic outcome: normal gap (at onset) ---------------------
    normal_gap = float("nan")
    d_peel_max = float(d_e[peel].max()) if int(peel.sum()) > 0 else float("nan")
    if int(interface.sum()) > 0 and int(flesh.sum()) > 0:
        fi = torch.nonzero(interface, as_tuple=False).squeeze(1)
        ff = torch.nonzero(flesh, as_tuple=False).squeeze(1)
        if ff.numel() > 20000:
            ff = ff[torch.randperm(ff.numel())[:20000]]

        def _nn(Xq, Xref, chunk=4096):
            out = torch.empty(Xq.shape[0], dtype=Xq.dtype)
            for i in range(0, Xq.shape[0], chunk):
                out[i:i + chunk] = torch.cdist(Xq[i:i + chunk], Xref).min(dim=1).values
            return out
        gap = (_nn(x_e[fi], x_e[ff]) - _nn(x0[fi], x0[ff])).clamp(min=0.0)
        br_i = d_e[fi] > 0.5
        normal_gap = float(gap[br_i].mean()) if int(br_i.sum()) > 0 else 0.0

    return dict(
        aniso=cfg["aniso"], directional=cfg["directional"], equal_E=bool(cfg["equal_E"]),
        seed=cfg["seed"], rho=cfg["rho"], pull_deg=cfg["pull_deg"], sigc_frac=cfg["sigc_frac"],
        speed=cfg["speed"], ngrid=cfg["ngrid"], ppcd=cfg["ppcd"], frames=cfg["frames"],
        notch_deg=cfg.get("notch_deg", 0.0), grip=cfg.get("grip", "cap"),
        phi_int_max=pk[0], phi_bystander_max=phi_bystander_max,
        phi_flesh_max=pk[1], phi_peel_max=pk[2], phi_shear_max=phi_shear_peak,
        routing=routing, broken_int_frac=broken_int_frac, onset_frame=onset_frame,
        cos2_slope=cos2_slope, cos2_r2=cos2_r2, cos2_n=cos2_n,
        d_int_max=float(d_f[interface].max()), d_bystander_max=d_bystander_max,
        d_bystander_final=d_bystander_final, d_peel_max=d_peel_max, break_thresh=break_thresh,
        grip_d_max=grip_d_max, normal_release=normal_release, inplane_keep=inplane_keep,
        normal_gap=normal_gap, halo_pred=halo_pred,
        n_broken_outside=n_out, exist=bool(exist), select=bool(n_out == 0),
        nan=bool(snap["nan"]), secs=float(snap["secs"]),
        solver_sha=str(snap.get("solver_sha", "")), setup_sha=str(snap.get("setup_sha", "")),
    )


def print_result(r):
    print(f"\n[peel] aniso={r['aniso']} directional={r['directional']} "
          f"equal_E={r['equal_E']} seed={r['seed']}")
    print(f"[peel] EXISTENCE (>5% interface broken): {'PASS' if r['exist'] else 'FAIL'}  "
          f"(broken_int_frac={r['broken_int_frac']:.3f}, delam onset frame={r['onset_frame']})")
    print(f"[peel] SELECTIVITY (no bystander breaks @ onset): {'PASS' if r['select'] else 'FAIL'}  "
          f"(broken_outside={r['n_broken_outside']}, d_bystander @onset={r['d_bystander_max']:.2f} "
          f"@final={r['d_bystander_final']:.2f}, thresh={r['break_thresh']:.2f}, halo~{r['halo_pred']:.2f})")
    print(f"[peel] routing phi_int/phi_bystander={r['routing']:.2f}  "
          f"(phi_int={r['phi_int_max']:.2f} phi_bystander={r['phi_bystander_max']:.2f} "
          f"phi_shear={r['phi_shear_max']:.2f})")
    print(f"[peel] Block2 angular law: phi_int/phi_shear ~ cot^2(theta) "
          f"slope={r['cos2_slope']:.2f} R2={r['cos2_r2']:.2f} n={r['cos2_n']} (want slope~+1 for correct)")
    print(f"[peel] Block-D outcome: normal_gap={r['normal_gap']:.4f} d_peel_max={r['d_peel_max']:.2f}  "
          f"| inplane_keep={r['inplane_keep']:.2f} (wiring smoke)  grip_d_max={r['grip_d_max']:.3f}")


def load_snapshot(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                                       # older torch: no weights_only kwarg
        return torch.load(path, map_location="cpu")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="re-derive peel observables from cached sim snapshots")
    ap.add_argument("snaps", nargs="+", help="snapshot .pt file(s) or a directory of them")
    ap.add_argument("--results-dir",
                    default=os.path.join(os.path.dirname(__file__), "..", "out", "results"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    paths = []
    for s in a.snaps:
        paths += sorted(glob.glob(os.path.join(s, "*.pt"))) if os.path.isdir(s) else [s]
    os.makedirs(a.results_dir, exist_ok=True)
    for p in paths:
        r = derive(load_snapshot(p))
        tag = os.path.splitext(os.path.basename(p))[0]
        with open(os.path.join(a.results_dir, tag + ".json"), "w") as f:
            json.dump(r, f, indent=1)
        if not a.quiet:
            print(f"  derived {tag}: exist={r['exist']} select={r['select']} "
                  f"routing={r['routing']:.1f} cos2={r['cos2_slope']:.2f}")
    print(f"[derive] {len(paths)} snapshot(s) -> {a.results_dir}")


if __name__ == "__main__":
    main()
