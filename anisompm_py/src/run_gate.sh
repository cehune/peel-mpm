#!/usr/bin/env bash
# =============================================================================
# AnisoMPM delamination validation gate
# =============================================================================
# Run order matters: each stage can veto the next, so a confounded green is
# impossible.  Stages:
#   PREFLIGHT  LA self-test            -- is the analytic eig/polar sound here?
#   GATE 0a    1_fd_check (A,B,C)      -- math: dPsi/dF, d=0 reconstruction, AND
#                                         d=1 RELEASE (Gate 3 tells a peel from
#                                         its opposite; the others cannot).
#   GATE 0b    positive control        -- maximally-helped cell MUST delaminate.
#   GATE 0b'   aggressive smoke        -- harsh cell must stay finite (the gentle
#                                         control does not certify the harsh cells).
#   [NaN stop] if a smoke went non-finite, the ball is unstable at this
#              resolution -- STOP, the whole sweep below would be NaN.
#   BLOCK D    directional STRESS       -- split fires (smoke) AND sheet separates
#                                         (kinematic normal-gap; not the circular ratio).
#   BLOCK 1    driver mechanism         -- correct/iso/wrong differ ONLY in A0;
#                                         verdict keys on the routing RATIO.
#   BLOCK 2    angular law              -- within-run phi_int/phi_shear ~ cot^2(theta);
#                                         slope ~ +1 correct, < 0 wrong, scatter iso.
#   BLOCK 3    toughness threshold      -- SELECT holds for some rho* > 1.
#   BLOCK 4    numerics                 -- vary grid and particles SEPARATELY;
#                                         verdict invariant (halo-aware selectivity).
#   VERDICT    gate_verdict.py          -- reduces validate.jsonl to PASS_0..4,D + GATE.
#
# Equation consistency (vs Wolper et al. 2020): driver Phi=(A s+ : s+ A)/sigma_c^2
# (Eq.8), evolution Eq.11 with HISTORY-MAX drive, g(d)=(1-d)^2(1-r)+r (Eq.10),
# Laplacian via a CUBIC B-spline pass (Eq. under 4.3.1 mandates >= cubic),
# l0 = dx/2, zeta = 1.  See the consistency notes in anisompm.py.
#
# Usage:
#   DEV=cuda:0 bash src/run_gate.sh           # your NVIDIA box (recommended)
#   DEV=cuda:0 FRAMES=48 bash src/run_gate.sh # more frames if damage is slow
#   bash src/run_gate.sh                       # auto: cuda -> mps -> cpu
# Notes:
#   * f_clamp is pinned to none: the float32 analytic polar (~1e-2, see preflight)
#     makes the clamp REBUILD NaN on aggressive cells -- pre-existing; none is
#     stable and still develops damage.
#   * Gate 3 validates against RES (the residual peel_test ships at the interface),
#     not a different one -- validate what you run.
set -euo pipefail
cd "$(dirname "$0")/.."                          # -> anisompm_py
PY=python3
DEV=${DEV:-auto}                                 # cuda:0 / mps / cpu / auto
FRAMES=${FRAMES:-48}
NG=${NG:-64}
RES=${RES:-0.005}                                # = peel_test interface residual
J=out/validate.jsonl
S=src/peel_test.py
C="--device $DEV --ngrid $NG --frames $FRAMES --f-clamp none --plot-every 0 --summary-json $J"
mkdir -p out
: > "$J"                                          # one clean gate run

echo "############ PREFLIGHT -- analytic linear-algebra self-test ############"
# Informational: float32 polar is ~1e-2 for near-degenerate stretch (-> 'CHECK').
# Run on cpu -- it is a device-independent check of the analytic eig/polar math;
# the gate runs --f-clamp none so this does not gate, but a hard FAIL here means
# the eig path is broken and every Phi downstream is suspect.
$PY src/anisompm.py cpu 2>/dev/null | sed -n '/LA self-test/,$p' || true

echo "############ GATE 0a -- math (Gate1 dPsi/dF, Gate2 recon, Gate3 RELEASE) #"
$PY src/1_fd_check.py --tol 1e-6 --res "$RES" --summary-json "$J"

echo "############ GATE 0b -- positive control (must obviously delaminate) #####"
$PY $S --aniso correct --directional on --rho 0.1 --pull-deg 0 $C

echo "############ GATE 0b' -- aggressive stability smoke (harsh cell finite?) #"
$PY $S --aniso correct --directional on --rho 2 --pull-deg 80 $C

echo "############ NaN STOP -- abort the sweep if a smoke diverged #############"
$PY - "$J" <<'PYEOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
smk = [r for r in recs if r.get("aniso")][-2:]   # the two smoke cells
if any(r.get("nan") for r in smk) or not smk:
    print("STOP: a stability smoke produced NaN (ball unstable at this resolution).")
    print("      lower --dt/--speed or raise --ngrid; the sweep below would be all-NaN.")
    sys.exit(1)
print("stability smokes finite -- proceeding.")
PYEOF

echo "############ BLOCK D -- directional STRESS isolation #####################"
$PY $S --aniso correct --directional on  --rho 1 --pull-deg 0 --equal-E $C
$PY $S --aniso correct --directional off --rho 1 --pull-deg 0 --equal-E $C

echo "############ BLOCK 1 -- driver mechanism (rho1 + equal-E -> only A0) #####"
for A in correct iso wrong; do
  $PY $S --aniso "$A" --directional on --rho 1 --pull-deg 0 --equal-E $C
done

echo "############ BLOCK 2 -- angular law phi_int/phi_shear ~ cot^2(theta) #####"
# pull_deg=0 already spans every theta over the shell (the within-run fit); the
# extra angles are corroboration / mode-II logging, not separate fit points.
for a in 0 30 60 80; do
  $PY $S --aniso correct --directional on --rho 1 --pull-deg "$a" --equal-E $C
done

echo "############ BLOCK 3 -- toughness threshold (SELECT holds for rho* > 1) ##"
for r in 0.5 1 1.5 2; do
  $PY $S --aniso correct --directional on --rho "$r" --pull-deg 0 $C
done

echo "############ BLOCK 4 -- numerics (grid and particles varied SEPARATELY) ##"
$PY $S --aniso correct --directional on --rho 1 --pull-deg 0 --equal-E --ngrid 96 $C   # finer grid
$PY $S --aniso correct --directional on --rho 1 --pull-deg 0 --equal-E --ppcd 2.5 $C   # finer particles

echo "############ seed robustness at the load-bearing cells ###################"
for s in 1 2; do
  $PY $S --aniso iso     --directional on --rho 1   --pull-deg 0 --equal-E --seed "$s" $C
  $PY $S --aniso correct --directional on --rho 1.5 --pull-deg 0          --seed "$s" $C
done

echo "############ VERDICT #####################################################"
$PY src/gate_verdict.py "$J"
