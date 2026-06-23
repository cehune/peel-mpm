#!/usr/bin/env bash
# =============================================================================
# AnisoMPM delamination validation gate  --  sim/observable SPLIT, 3 phases
# =============================================================================
# The EXPENSIVE simulation and the CHEAP observables are decoupled:
#
#   PHASE 1  run/cache sims   each cell runs peel_test.py, which writes a raw
#                             snapshot out/sims/<tag>.pt AND derives its result
#                             out/results/<tag>.json.  A sim is RE-RUN only if
#                             its snapshot is missing OR was made by a different
#                             solver (hash of anisompm.py) OR went NaN -- so a
#                             config never re-runs unless the PHYSICS changed.
#   PHASE 2  derive all       derive.py re-derives EVERY snapshot -> results.
#                             Cheap + pure, so changing a metric never re-runs a
#                             sim; you can also run just this after editing
#                             derive.py:  python3 src/derive.py out/sims
#   PHASE 3  verdict          gate_verdict.py reduces out/results/*.json to
#                             PASS_0..4,D + GATE, and refuses a green if the
#                             results span >1 solver/harness/notch revision.
#
# Each rung can veto the next, so a confounded green is impossible:
#   PREFLIGHT  LA self-test            -- is the eig/polar path sound here?
#   GATE 0a    1_fd_check (A,B,C)      -- math: dPsi/dF, d=0 reconstruction, AND
#                                         d=1 RELEASE (tells a peel from its opposite).
#   GATE 0b    positive control        -- maximally-helped cell MUST delaminate.
#   GATE 0b'   aggressive smoke        -- harsh cell must stay finite.
#   [NaN stop] if a smoke went non-finite, STOP (the sweep would be all-NaN).
#   BLOCK D    directional STRESS      -- split fires AND sheet separates (gap).
#   BLOCK 1    driver mechanism        -- correct/iso/wrong differ ONLY in A0.
#   BLOCK 2    angular law             -- within-run phi_int/phi_shear ~ cot^2(theta).
#   BLOCK 3    toughness threshold     -- SELECT holds for some rho* > 1.
#   BLOCK 4    numerics                -- vary grid and particles SEPARATELY.
#
# Equation consistency (vs Wolper et al. 2020): driver Phi=(A s+ : s+ A)/sigma_c^2
# (Eq.8); evolution Eq.11 with HISTORY-MAX drive; g(d)=(1-d)^2(1-r)+r (Eq.10);
# Laplacian via a CUBIC B-spline pass (>= cubic, 4.3.1); l0 = dx/2; zeta = 1.
#
# Usage:
#   DEV=cuda:0 bash src/run_gate.sh              # your NVIDIA box (recommended)
#   RESUME=0 DEV=cuda:0 bash src/run_gate.sh     # clean slate (wipe out/sims+results)
#   NOTCH=8 RESUME=0 DEV=cuda:0 bash src/run_gate.sh   # pole-seeded loading protocol
#   bash src/run_gate.sh                          # auto: cuda -> mps -> cpu
# Caching: with RESUME=1 (default) finished cells are served from cache; after a
#   solver change they re-run automatically (snapshot solver-hash mismatch).  So
#   after an interruption just re-run the SAME command -- it continues.  For
#   AUTOMATIC SLURM restart submit with `sbatch --requeue`.  RESUME=0 wipes the
#   sim+result dirs for a fully fresh run.
# Notch: NOTCH>0 removes a wedge of interface particles under the pulled pole (a
#   traction-free pre-crack), so delamination propagates from the interface
#   instead of the cap over-stressing the bulk.  It is a distinct loading
#   protocol -> its own snapshots/results (the tag carries notch_deg); the
#   verdict pins whatever notch the dir is dominated by and flags mixes.
set -euo pipefail
cd "$(dirname "$0")/.."                          # -> anisompm_py
PY=python3
DEV=${DEV:-auto}                                 # cuda:0 / mps / cpu / auto
FRAMES=${FRAMES:-48}
NG=${NG:-64}
RES=${RES:-0.005}                                # = peel_test interface residual
RESUME=${RESUME:-1}                              # 1 = serve cache; 0 = wipe + fresh
SPEED=${SPEED:-0.12}                             # pull speed; LOWER = gentler/more stable
DT=${DT:-3e-4}                                   # timestep;   LOWER = more stable
RAMP=${RAMP:-0.30}                               # pull ramp time; LONGER = gentler start
PLOT=${PLOT:-0}                                  # >0: save a cross-section PNG every N frames
NOTCH=${NOTCH:-0}                                # >0: pole pre-crack (degrees); loading protocol
SIMS=out/sims                                    # cached snapshots  (<tag>.pt)
RES_DIR=out/results                              # per-config observables (<tag>.json)
FD_LOG=out/fd_check.jsonl                        # math-gate record (read by verdict)
S=src/peel_test.py
C="--device $DEV --ngrid $NG --frames $FRAMES --speed $SPEED --dt $DT --ramp $RAMP \
   --plot-every $PLOT --notch-deg $NOTCH --sims-dir $SIMS --results-dir $RES_DIR"
mkdir -p out "$SIMS" "$RES_DIR"
if [ "$RESUME" = "0" ]; then
  rm -f "$SIMS"/*.pt "$RES_DIR"/*.json "$FD_LOG" 2>/dev/null || true
  echo "[gate] RESUME=0 -> fresh run, wiped $SIMS and $RES_DIR"
else
  echo "[gate] resume mode: serving cache in $SIMS; solver changes re-run automatically."
  echo "       (Set RESUME=0 to wipe and start clean.)  notch_deg=$NOTCH"
fi

echo "############ PREFLIGHT -- analytic linear-algebra self-test ############"
# Confirms the eig/polar path is sound (PASS with the Jacobi eigensolver). Run on
# cpu -- device-independent; a hard FAIL means every Phi downstream is suspect.
$PY src/anisompm.py cpu 2>/dev/null | sed -n '/LA self-test/,$p' || true

echo "############ VERSION CHECK -- are the src files in sync on this machine? #"
# Cheap upfront partial-sync catch (the verdict's DRIFT banner is the backstop).
$PY src/peel_test.py --help 2>&1 | grep -q -- '--notch-deg' || {
  echo "ERROR: src/peel_test.py is STALE (missing --notch-deg).  Partial sync --"
  echo "       re-copy the WHOLE src/ (anisompm.py, peel_test.py, derive.py,"
  echo "       1_fd_check.py, gate_verdict.py, run_gate.sh) so every file is the"
  echo "       same revision, then re-run."
  exit 1
}
test -f src/derive.py && grep -q "delam onset frame" src/derive.py || {
  echo "ERROR: src/derive.py is missing or stale (no onset grading).  Partial sync"
  echo "       -- re-copy the WHOLE src/ here, then re-run."
  exit 1
}
echo "ok: peel_test.py has --notch-deg AND derive.py has onset grading -- src in sync."

echo "############ GATE 0a -- math (Gate1 dPsi/dF, Gate2 recon, Gate3 RELEASE) #"
$PY src/1_fd_check.py --tol 1e-6 --res "$RES" --summary-json "$FD_LOG"

echo "############ PHASE 1 -- run/cache sims (solver-hash skip) ################"
echo "## GATE 0b -- positive control (must obviously delaminate) ##############"
$PY $S --aniso correct --directional on --rho 0.1 --pull-deg 0 $C

echo "## GATE 0b' -- aggressive stability smoke (harsh cell finite?) ##########"
$PY $S --aniso correct --directional on --rho 2 --pull-deg 80 $C

echo "## NaN STOP -- abort the sweep if a smoke diverged #####################"
$PY - "$RES_DIR" <<'PYEOF'
import json, sys, glob, os
recs = []
for p in glob.glob(os.path.join(sys.argv[1], "*.json")):
    try: recs.append(json.load(open(p)))
    except Exception: pass
def find(rho, pull):                              # locate a smoke by CONFIG, not position
    m = [r for r in recs if r.get("aniso") == "correct" and r.get("directional") == "on"
         and abs(r.get("rho", 9e9) - rho) < 1e-9 and abs(r.get("pull_deg", 9e9) - pull) < 1e-9]
    return m[-1] if m else None
smk = [find(0.1, 0), find(2, 80)]                 # 0b (gentle) and 0b' (aggressive)
if any(r is None for r in smk):
    print("STOP: a stability-smoke result is missing -- cannot certify the sweep."); sys.exit(1)
if any(r.get("nan") for r in smk):
    print("STOP: a stability smoke produced NaN (ball unstable at this resolution).")
    print("      lower --dt/--speed or raise --ngrid; the sweep below would be all-NaN.")
    sys.exit(1)
print("stability smokes finite -- proceeding.")
PYEOF

echo "## BLOCK D -- directional STRESS isolation #############################"
$PY $S --aniso correct --directional on  --rho 1 --pull-deg 0 --equal-E $C
$PY $S --aniso correct --directional off --rho 1 --pull-deg 0 --equal-E $C

echo "## BLOCK 1 -- driver mechanism (rho1 + equal-E -> only A0) #############"
for A in correct iso wrong; do
  $PY $S --aniso "$A" --directional on --rho 1 --pull-deg 0 --equal-E $C
done

echo "## BLOCK 2 -- angular law phi_int/phi_shear ~ cot^2(theta) #############"
# pull_deg=0 already spans every theta over the shell (the within-run fit); the
# extra angles are corroboration / mode-II logging, not separate fit points.
for a in 0 30 60 80; do
  $PY $S --aniso correct --directional on --rho 1 --pull-deg "$a" --equal-E $C
done

echo "## BLOCK 3 -- toughness threshold (SELECT holds for rho* > 1) ##########"
for r in 0.5 1 1.5 2; do
  $PY $S --aniso correct --directional on --rho "$r" --pull-deg 0 $C
done

echo "## BLOCK 4 -- numerics (grid and particles varied SEPARATELY) #########"
$PY $S --aniso correct --directional on --rho 1 --pull-deg 0 --equal-E --ngrid 96 $C   # finer grid
$PY $S --aniso correct --directional on --rho 1 --pull-deg 0 --equal-E --ppcd 2.5 $C   # finer particles

echo "## seed robustness at the load-bearing cells ##########################"
for s in 1 2; do
  $PY $S --aniso iso     --directional on --rho 1   --pull-deg 0 --equal-E --seed "$s" $C
  $PY $S --aniso correct --directional on --rho 1.5 --pull-deg 0          --seed "$s" $C
done

echo "############ PHASE 2 -- re-derive ALL snapshots -> results (cheap) ######"
# Pure + fast: guarantees every result reflects the CURRENT derive.py even for
# cached sims and any orphan snapshots in $SIMS.
$PY src/derive.py "$SIMS" --results-dir "$RES_DIR" --quiet

echo "############ PHASE 3 -- VERDICT #########################################"
$PY src/gate_verdict.py "$RES_DIR" "$FD_LOG"
