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

# ---- PASS_D: directional STRESS isolation (the new, load-bearing block) ----
don = one(aniso="correct", directional="on", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0)
dof = one(aniso="correct", directional="off", rho=1, pull_deg=0, equal_E=True, ngrid=64, ppcd=2.0)
PASS_D = bool(ok(don) and ok(dof)
              and don["normal_release"] > 0.5
              and don["inplane_keep"] > 0.5
              and don["inplane_keep"] > 2.0 * max(dof["inplane_keep"], 1e-9))
say(f"PASS_D {'PASS' if PASS_D else 'FAIL'}  "
    + ("MISSING" if not (don and dof) else
       f"on(release={don['normal_release']:.2f},keep={don['inplane_keep']:.2f}) "
       f"off(keep={dof['inplane_keep']:.2f}) -> keep must be on>>off"))

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

# ---- PASS_2: cos^2 direction signature ------------------------------------
ang = sorted(find(aniso="correct", directional="on", rho=1, equal_E=True),
             key=lambda r: r["pull_deg"])
ang = [a for a in ang if a["pull_deg"] in (0, 30, 60, 80) and a.get("ngrid") == 64]
PASS_2 = False
if len(ang) >= 3 and all(not a.get("nan") for a in ang):
    phis = [a["phi_int_max"] for a in ang]
    mono = all(phis[i] >= phis[i + 1] - 0.15 * (phis[0] + 1e-9) for i in range(len(phis) - 1))
    shear_ok = all((a["broken_int_frac"] < 0.05) or (a["phi_shear_max"] > 1)
                   for a in ang if a["pull_deg"] >= 60)
    PASS_2 = bool(mono and shear_ok)
    say(f"PASS_2 {'PASS' if PASS_2 else 'FAIL'}  phi_int(theta)={['%.2f'%p for p in phis]} "
        f"monotone={mono} high-angle-shear-explained={shear_ok}")
else:
    say("PASS_2 FAIL  MISSING angle sweep")

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
