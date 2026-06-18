#!/usr/bin/env bash
# =============================================================================
# AnisoMPM delamination validation gate  (revised)
# =============================================================================
# Fixes vs the original plan:
#   * commands actually run -- every flag below exists in peel_test.py now
#     (--aniso, --directional, --equal-E, --seed) and 1_fd_check.py is the real
#     file name.
#   * BLOCK D is NEW: it isolates the directional STRESS (the thing we added),
#     which the original plan never tested -- all its observables were downstream
#     of the damage DRIVER (phi), which is identical whether the stress split is
#     on or off.  Block D holds the driver fixed and flips --directional, and the
#     verdict keys on inplane_keep (only the directional split preserves it).
#   * routing is compared as a RATIO (correct vs iso), not "iso ~ 1": pole
#     geometry concentrates stress regardless of A0.
#   * grip leak is measured in the ALLOWED peel ring below the cap, not on the
#     clamped cap (whose damage is frozen to 0 and can never fail the check).
#   * seed robustness at the load-bearing cells.
#
# Each run appends one JSON record to out/validate.jsonl; gate_verdict.py reduces
# it to PASS_0..4 + PASS_D + GATE.
#
# Usage:   DEV=mps FRAMES=48 bash src/run_gate.sh      # mac GPU, full
#          DEV=cuda:0 bash src/run_gate.sh             # cuda box
#          bash src/run_gate.sh                        # cpu, default FRAMES
# Notes:
#   * On a GPU each run is seconds; on CPU it is minutes -- use a GPU for the
#     full gate, or FRAMES=12 NG-small for a quick smoke.
#   * peel_test.py hardcodes f_clamp=(0.35,2.8).  In float32 the analytic polar
#     decomposition is only ~1e-2 accurate (see the LA self-test) and the clamp
#     rebuild can NaN on aggressive configs.  If you see nan=true, that is the
#     pre-existing f_clamp/float32 issue, not the gate -- rerun that cell with a
#     gentler --speed or move the polar step to float64.
set -e
cd "$(dirname "$0")/.."                       # -> anisompm_py
PY=python3
DEV=${DEV:-cpu}
FRAMES=${FRAMES:-32}
J=out/validate.jsonl
S=src/peel_test.py
C="--device $DEV --frames $FRAMES --plot-every 0 --f-clamp none --summary-json $J"
mkdir -p out
: > "$J"                                       # one clean gate run

echo "### GATE 0a -- math gate (directional_pk1 == dPsi/dF + Option B reconstruction)"
$PY src/1_fd_check.py --tol 1e-6 --summary-json "$J"

echo "### GATE 0b -- maximally-helped positive control (must obviously delaminate)"
$PY $S --aniso correct --directional on --rho 0.1 --pull-deg 0 $C

echo "### BLOCK D -- directional STRESS isolation (driver fixed, flip the stress model)"
$PY $S --aniso correct --directional on  --rho 1 --pull-deg 0 --equal-E $C
$PY $S --aniso correct --directional off --rho 1 --pull-deg 0 --equal-E $C

echo "### BLOCK 1 -- driver mechanism (rho1 + equal-E: correct/iso/wrong differ ONLY in A0)"
for A in correct iso wrong; do
  $PY $S --aniso $A --directional on --rho 1 --pull-deg 0 --equal-E $C
done

echo "### BLOCK 2 -- direction signature (phi_int ~ cos^2(theta); phi_shear logged)"
for a in 0 30 60 80; do
  $PY $S --aniso correct --directional on --rho 1 --pull-deg $a --equal-E $C
done

echo "### BLOCK 3 -- toughness threshold (SELECT must hold for some rho* > 1)"
for r in 0.5 1 1.5 2; do
  $PY $S --aniso correct --directional on --rho $r --pull-deg 0 $C
done

echo "### BLOCK 4 -- numerics (vary grid and particles SEPARATELY; verdict must be invariant)"
$PY $S --aniso correct --directional on --rho 1 --pull-deg 0 --equal-E --ngrid 96 $C   # finer grid only
$PY $S --aniso correct --directional on --rho 1 --pull-deg 0 --equal-E --ppcd 2.5 $C   # finer particles only

echo "### seed robustness at the load-bearing cells (Block 1 iso, Block 3 threshold)"
for s in 1 2; do
  $PY $S --aniso iso     --directional on --rho 1   --pull-deg 0 --equal-E --seed $s $C
  $PY $S --aniso correct --directional on --rho 1.5 --pull-deg 0          --seed $s $C
done

echo "### VERDICT"
$PY src/gate_verdict.py "$J"
