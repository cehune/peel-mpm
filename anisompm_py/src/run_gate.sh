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
#   RESUME=0 DEV=cuda:0 bash src/run_gate.sh  # start clean (default RESUME=1 resumes)
#   bash src/run_gate.sh                       # auto: cuda -> mps -> cpu
# Preemption / restart: the gate is idempotent -- finished cells skip (--skip-if-done),
#   so after an interruption just re-run the SAME command and it continues.  For
#   AUTOMATIC restart submit with `sbatch --requeue` (bare srun does NOT requeue).
#   Each cell appends one atomic JSON line at the end, so a killed cell leaves no
#   half-record; it simply re-runs.  RESUME=0 wipes the log for a clean run.
# Notes:
#   * The eig/polar uses a batched-Jacobi eigensolver (no cusolver/LAPACK/MAGMA):
#     torch.linalg.eigh's CUDA batched path threw CUSOLVER_STATUS_INVALID_VALUE,
#     and the older analytic path collapsed on near-degenerate C (|V|~1e5 -> stress
#     ~1e23, the 'ball NaN').  f_clamp is at its default (0.35,2.8); LA self-test
#     PASSES.  If a cell still shows nan=true, lower --speed or --dt for that cell.
#   * Gate 3 validates against RES (the residual peel_test ships at the interface),
#     not a different one -- validate what you run.
set -euo pipefail
cd "$(dirname "$0")/.."                          # -> anisompm_py
PY=python3
DEV=${DEV:-auto}                                 # cuda:0 / mps / cpu / auto
FRAMES=${FRAMES:-48}
NG=${NG:-64}
RES=${RES:-0.005}                                # = peel_test interface residual
RESUME=${RESUME:-1}                              # 1 = skip finished cells (preempt-safe)
SPEED=${SPEED:-0.12}                             # pull speed; LOWER = gentler/more stable
DT=${DT:-3e-4}                                   # timestep;   LOWER = more stable
RAMP=${RAMP:-0.30}                               # pull ramp time; LONGER = gentler start
PLOT=${PLOT:-0}                                  # >0: save a cross-section PNG every N frames
J=out/validate.jsonl                             #     (+ one at delamination onset) per cell
S=src/peel_test.py                               #     into out/peel/<tag>/slice_*.png
C="--device $DEV --ngrid $NG --frames $FRAMES --speed $SPEED --dt $DT --ramp $RAMP --plot-every $PLOT --skip-if-done --summary-json $J"
mkdir -p out
if [ "$RESUME" = "0" ]; then
  : > "$J"; echo "[gate] RESUME=0 -> fresh run, cleared $J"
else
  echo "[gate] resume mode: keeping $J; finished cells skip. Set RESUME=0 to start clean"
  echo "       (do that after any change to the solver, so stale cells re-run)."
fi

echo "############ PREFLIGHT -- analytic linear-algebra self-test ############"
# Confirms the eig/polar path is sound (RESULT: PASS with the Jacobi eigensolver;
# the old analytic path printed 'CHECK' at ~1e-2 and blew up on the ball).
# Run on cpu -- device-independent check; a hard FAIL here means the eig path is
# broken and every Phi downstream is suspect.
$PY src/anisompm.py cpu 2>/dev/null | sed -n '/LA self-test/,$p' || true

echo "############ VERSION CHECK -- are the src files in sync on this machine? #"
# Catches a partial sync (e.g. a new run_gate.sh next to a stale peel_test.py)
# at the TOP, instead of dying on the first cell with 'unrecognized arguments'.
$PY src/peel_test.py --help 2>&1 | grep -q -- '--skip-if-done' || {
  echo "ERROR: src/peel_test.py on this machine is STALE (missing --skip-if-done)."
  echo "       You have a partial sync.  Re-copy the WHOLE src/ here so every file"
  echo "       is the same revision (anisompm.py, peel_test.py, 1_fd_check.py,"
  echo "       gate_verdict.py, run_gate.sh), then re-run.  Nothing below would be"
  echo "       trustworthy with mismatched files."
  exit 1
}
grep -q "delam onset frame" src/peel_test.py || {
  echo "ERROR: src/peel_test.py is missing the onset-grading code (older than run_gate.sh)."
  echo "       Partial sync -- re-copy the WHOLE src/ here, then re-run.  Without it the"
  echo "       verdict grades the over-pulled final frame, not the delamination event."
  exit 1
}
echo "ok: peel_test.py has the gate flags AND onset grading -- src in sync."

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
def find(rho, pull):                              # locate a smoke by CONFIG, not
    m = [r for r in recs if r.get("aniso") == "correct" and r.get("directional") == "on"
         and abs(r.get("rho", 9e9) - rho) < 1e-9 and abs(r.get("pull_deg", 9e9) - pull) < 1e-9]
    return m[-1] if m else None                   # position (robust to resume order)
smk = [find(0.1, 0), find(2, 80)]                 # 0b (gentle) and 0b' (aggressive)
if any(r is None for r in smk):
    print("STOP: a stability-smoke record is missing -- cannot certify the sweep."); sys.exit(1)
if any(r.get("nan") for r in smk):
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
