#!/usr/bin/env python3
"""Reduce the per-config result files to the gate verdict.

    python3 src/gate_verdict.py [out/results] [out/fd_check.jsonl]

Reads one JSON per config from the results DIR (written by derive.py) plus the
fd-check log, and computes per-block booleans (PASS_0, PASS_D, PASS_1..4) and
GATE, printing the evidence for each so a FAIL is diagnosable.  Decision rules
match run_gate.sh; thresholds are conservative and easy to retune at the top.

It also prints a DRIFT banner: if the result files were derived from more than
one solver/harness revision (mixed solver_sha / setup_sha) the verdict spans
inconsistent physics -- a partial-sync tell -- and you should re-run clean.
"""
import os, sys, json, glob

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "out/results"
FD_LOG = sys.argv[2] if len(sys.argv) > 2 else "out/fd_check.jsonl"

# ---- load peel results (one json per config) ------------------------------
peel = []
if os.path.isdir(RESULTS):
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        try:
            with open(p) as f:
                r = json.load(f)
        except Exception:
            continue
        if isinstance(r, dict) and "aniso" in r:
            peel.append(r)
else:                                                       # back-compat: a jsonl file
    peel = [json.loads(l) for l in open(RESULTS) if l.strip() and '"aniso"' in l]

# ---- load fd-check (math gate) result -------------------------------------
fd = []
if os.path.exists(FD_LOG):
    fd = [json.loads(l) for l in open(FD_LOG) if l.strip()]
fd = [r for r in fd if r.get("test") == "fd_check"] or fd


def match(r, kw):
    for k, v in kw.items():
        rv = r.get(k)
        if rv is None:
            return False
        if isinstance(v, bool):
            if bool(rv) != v:
                return False
        elif isinstance(v, (int, float)):
            if abs(float(rv) - float(v)) > 1e-9:
                return False
        elif rv != v:
            return False
    return True


def find(**kw):
    return [r for r in peel if match(r, kw)]


def one(**kw):
    m = find(**kw)
    return m[-1] if m else None


def ok(r):
    return r is not None and not r.get("nan", False)


lines = []
def say(s): lines.append(s)

# ---- DRIFT banner: are all results from the same solver/harness/protocol? --
solver_shas = sorted({r.get("solver_sha", "?") for r in peel})
setup_shas = sorted({r.get("setup_sha", "?") for r in peel})
notch_vals = sorted({round(float(r.get("notch_deg", 0.0)), 6) for r in peel})
# the loading protocol the matchers below pin to (the dominant notch in the dir)
NOTCH = (max(notch_vals, key=lambda v: sum(1 for r in peel
             if round(float(r.get("notch_deg", 0.0)), 6) == v)) if peel else 0.0)
drift = len(solver_shas) > 1 or len(setup_shas) > 1 or len(notch_vals) > 1
if drift:
    say("DRIFT  !!  results span multiple revisions -- verdict is NOT trustworthy:")
    if len(solver_shas) > 1:
        say(f"          solver_sha = {solver_shas}  (mixed PHYSICS -- re-run clean: RESUME=0)")
    if len(setup_shas) > 1:
        say(f"          setup_sha  = {setup_shas}  (mixed harness)")
    if len(notch_vals) > 1:
        say(f"          notch_deg  = {notch_vals}  (mixed loading protocol -- clear out/ and re-run)")
else:
    say(f"src OK  solver_sha={solver_shas[0] if solver_shas else '?'} "
        f"setup_sha={setup_shas[0] if setup_shas else '?'}  notch_deg={NOTCH:g}  "
        f"({len(peel)} configs)")

# ---- PASS_0: math gate + maximally-helped sanity --------------------------
fd_pass = bool(fd) and fd[-1].get("pass", False)
s = one(aniso="correct", directional="on", rho=0.1, pull_deg=0)
sanity = ok(s) and s["broken_int_frac"] > 0.05 and s["grip_d_max"] < 0.1
PASS_0 = bool(fd_pass and sanity)
say(f"PASS_0 {'PASS' if PASS_0 else 'FAIL'}  fd_pass={fd_pass}  "
    f"sanity={'ok' if sanity else 'BAD'}"
    + ("" if s is None else f" (broken_int={s['broken_int_frac']:.2f}, grip_d={s['grip_d_max']:.2f})"))

# ---- PASS_D: directional STRESS -- kinematic outcome, not the circular ratio --
don = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0, notch_deg=NOTCH)
dof = one(aniso="correct", directional="off", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0, notch_deg=NOTCH)
def g(r, k, d=0.0): return (r.get(k) if r and r.get(k) is not None else d)
PASS_D = bool(ok(don) and ok(dof)
              and g(don, "inplane_keep") > 0.5
              and g(don, "inplane_keep") > 2.0 * max(g(dof, "inplane_keep"), 1e-9)   # split fires
              and g(don, "normal_gap") > 0.0)                                        # sheet separates
say(f"PASS_D {'PASS' if PASS_D else 'FAIL'}  "
    + ("MISSING" if not (don and dof) else
       f"on(keep={g(don,'inplane_keep'):.2f},gap={g(don,'normal_gap'):.4f},d_peel={g(don,'d_peel_max'):.2f}) "
       f"off(keep={g(dof,'inplane_keep'):.2f},gap={g(dof,'normal_gap'):.4f},d_peel={g(dof,'d_peel_max'):.2f})"))

# ---- PASS_1: driver mechanism (correct selects; iso doesn't; wrong no-exist)
co = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0, notch_deg=NOTCH)
iso = one(aniso="iso", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0, notch_deg=NOTCH)
wr = one(aniso="wrong", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0, notch_deg=NOTCH)
sel_correct = ok(co) and co["routing"] > 2 and co["phi_int_max"] > 1 \
    and co["d_bystander_max"] < 0.5 and co["grip_d_max"] < 0.1
iso_no_sel = ok(iso) and (co and co["routing"] > 3 * iso["routing"] or (iso and iso["d_bystander_max"] > 0.5))
wrong_no_exist = ok(wr) and (wr["phi_int_max"] < 1 or wr["broken_int_frac"] < 0.05)
PASS_1 = bool(sel_correct and iso_no_sel and wrong_no_exist)
say(f"PASS_1 {'PASS' if PASS_1 else 'FAIL'}  "
    + ("MISSING" if not (co and iso and wr) else
       f"correct(route={co['routing']:.1f},sel={sel_correct}) "
       f"iso(route={iso['routing']:.1f}) wrong(phi_int={wr['phi_int_max']:.2f},brk={wr['broken_int_frac']:.2f})"))

# ---- PASS_2: within-run angular law  phi_int/phi_shear ~ cot^2(theta_n) -----
co2 = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0, notch_deg=NOTCH)
wr2 = one(aniso="wrong", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0, notch_deg=NOTCH)
def _f(r, k):
    v = r.get(k) if r else None
    return v if isinstance(v, (int, float)) and v == v else None   # None if missing/NaN
slope_c = _f(co2, "cos2_slope") if ok(co2) else None
slope_w = _f(wr2, "cos2_slope") if ok(wr2) else None
PASS_2 = bool(slope_c is not None and 0.6 < slope_c < 1.4
              and (slope_w is None or slope_w < 0.5))   # correct ~1, wrong clearly lower
say(f"PASS_2 {'PASS' if PASS_2 else 'FAIL'}  "
    + ("MISSING cos2_slope" if slope_c is None else
       f"cot^2 slope correct={slope_c:.2f} (want ~1, R2={_f(co2,'cos2_r2')}) "
       f"wrong={slope_w} (want <0.5)"))

# ---- PASS_3: toughness threshold rho* > 1 ---------------------------------
rhos = sorted(find(aniso="correct", directional="on", pull_deg=0, equal_E=False, notch_deg=NOTCH),
              key=lambda r: r["rho"])
rho_star = None
for r in rhos:
    if ok(r) and r["exist"] and r["select"]:
        rho_star = r["rho"]
PASS_3 = bool(rho_star is not None and rho_star > 1)
say(f"PASS_3 {'PASS' if PASS_3 else 'FAIL'}  rho*={rho_star} (need > 1)  "
    f"tested={[r['rho'] for r in rhos]}")

# ---- PASS_4: numerical invariance -----------------------------------------
base = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0, notch_deg=NOTCH)
fineg = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=96, ppcd=2.0, notch_deg=NOTCH)
finep = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.5, notch_deg=NOTCH)
def verdict(r): return (bool(r["exist"]), bool(r["select"])) if ok(r) else None
vs = [verdict(base), verdict(fineg), verdict(finep)]
PASS_4 = bool(all(v is not None for v in vs) and vs[0] == vs[1] == vs[2])
say(f"PASS_4 {'PASS' if PASS_4 else 'FAIL'}  (exist,select) base={vs[0]} fineGrid={vs[1]} fineParts={vs[2]}")

GATE = (not drift) and PASS_0 and PASS_D and PASS_1 and PASS_2 and PASS_3 and PASS_4
print("\n".join(lines))
print("-" * 64)
print(f"GATE = {'PASS' if GATE else 'FAIL'}   "
      f"(PASS_0={PASS_0} PASS_D={PASS_D} PASS_1={PASS_1} "
      f"PASS_2={PASS_2} PASS_3={PASS_3} PASS_4={PASS_4}"
      + ("" if not drift else " DRIFT") + ")")
sys.exit(0 if GATE else 1)
