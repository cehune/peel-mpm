#!/usr/bin/env python3
"""Reduce out/validate.jsonl to the gate verdict.

    python3 src/gate_verdict.py [out/validate.jsonl]

Computes per-block booleans (PASS_0, PASS_D, PASS_1..4) and GATE, printing the
evidence for each so a FAIL is diagnosable, not just a red light.  Decision rules
match run_gate.sh; thresholds are conservative and easy to retune at the top.
"""
import json, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "out/validate.jsonl"
recs = [json.loads(l) for l in open(PATH) if l.strip()]
fd = [r for r in recs if r.get("test") == "fd_check"]
peel = [r for r in recs if "aniso" in r]


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

# ---- PASS_0: math gate + maximally-helped sanity --------------------------
fd_pass = bool(fd) and fd[-1].get("pass", False)
s = one(aniso="correct", directional="on", rho=0.1, pull_deg=0)
sanity = ok(s) and s["broken_int_frac"] > 0.05 and s["grip_d_max"] < 0.1
PASS_0 = bool(fd_pass and sanity)
say(f"PASS_0 {'PASS' if PASS_0 else 'FAIL'}  fd_pass={fd_pass}  "
    f"sanity={'ok' if sanity else 'BAD'}"
    + ("" if s is None else f" (broken_int={s['broken_int_frac']:.2f}, grip_d={s['grip_d_max']:.2f})"))

# ---- PASS_D: directional STRESS -- kinematic outcome, not the circular ratio --
# The constitutive release is now proved deterministically by Gate 3
# (release_check). Block D's job is the SIM-LEVEL outcome: the split fires in the
# full solver (inplane_keep on>>off -- a wiring smoke) AND the sheet actually
# separates (a normal gap opens). inplane_keep alone is read off the split, so it
# is NOT used as the sole discriminant.
don = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0)
dof = one(aniso="correct", directional="off", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0)
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
co = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0)
iso = one(aniso="iso", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0)
wr = one(aniso="wrong", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0)
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
# The initiation criterion's LHS. NOT phi_int~cos^2 (the driver squares the
# normal traction -> phi_int~cos^4); the magnitude-free RATIO is the clean law.
# One pull_deg=0 run spans all theta over the shell. correct -> log-log slope ~ +1;
# wrong (shear driver) -> tracks sin^2 -> slope < 0; iso -> scatters (low R^2).
co2 = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0)
wr2 = one(aniso="wrong", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0)
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
rhos = sorted(find(aniso="correct", directional="on", pull_deg=0, equal_E=False),
              key=lambda r: r["rho"])
rho_star = None
for r in rhos:
    if ok(r) and r["exist"] and r["select"]:
        rho_star = r["rho"]
PASS_3 = bool(rho_star is not None and rho_star > 1)
say(f"PASS_3 {'PASS' if PASS_3 else 'FAIL'}  rho*={rho_star} (need > 1)  "
    f"tested={[r['rho'] for r in rhos]}")

# ---- PASS_4: numerical invariance -----------------------------------------
base = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0)
fineg = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=96, ppcd=2.0)
finep = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.5)
def verdict(r): return (bool(r["exist"]), bool(r["select"])) if ok(r) else None
vs = [verdict(base), verdict(fineg), verdict(finep)]
PASS_4 = bool(all(v is not None for v in vs) and vs[0] == vs[1] == vs[2])
say(f"PASS_4 {'PASS' if PASS_4 else 'FAIL'}  (exist,select) base={vs[0]} fineGrid={vs[1]} fineParts={vs[2]}")

GATE = PASS_0 and PASS_D and PASS_1 and PASS_2 and PASS_3 and PASS_4
print("\n".join(lines))
print("-" * 64)
print(f"GATE = {'PASS' if GATE else 'FAIL'}   "
      f"(PASS_0={PASS_0} PASS_D={PASS_D} PASS_1={PASS_1} "
      f"PASS_2={PASS_2} PASS_3={PASS_3} PASS_4={PASS_4})")
sys.exit(0 if GATE else 1)
