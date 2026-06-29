# Runbook — slab → peel → directional extension → render harness

This covers the clean arc we built (validate the AnisoMPM port, then head toward
neural crack rendering). Run everything from `anisompm_py/`. Outputs land in
`out/<area>/`. CPU is fine for all of it.

> Other scripts in `src/` (`orange_*.py`, `render.py`, `squeeze_*.py`, `sweep.py`,
> `tune_*.py`, `test_*.py`, `slab_test.py`, the ball gate `peel_test.py` /
> `run_gate.sh` / `gate_verdict.py` / `derive.py` / `make_figs.py`) are earlier or
> separate work and are **not** part of this arc. The four files below are.

---

## What we've done

1. **Solver (`anisompm.py`)** — MLS-MPM + anisotropic phase-field damage, with the
   directional-stress extension (Option B). Self-test: `python3 src/anisompm.py cpu`.
2. **`slab.py` — directional toughness** (the paper's core claim, quantified): pull a
   fibered bar, rotate the grain. Strong along the grain (≈unbreakable), weak across.
3. **`peel.py` — Fig. 4 reproduction** (animations): transverse-iso peels a clean
   *strip*; isotropic tears a *small piece*.
4. **`cohesive.py` — your directional extension, posed cleanly**: two-material bond,
   Option B on vs off. Measured `inplane_keep` ≈ 5.9 (on) vs 0.6 (off).
5. **`expose.py` — render harness, STEP 1**: pull the freshly-exposed crack face out
   of a peel, per frame, with stable labels + outward normals. (Done.)

Next: STEP 2 baseline render of that face, STEP 3 a direction-sensitive metric, then
the A/B (pre-baked interior vs fracture-conditioned appearance).

---

## 1. slab.py — directional toughness

Pull a bar, rotate the fiber; measure strength/damage vs fiber angle.
`--axis x|y` (pull axis), `--one-ended` (clamp one end, pull the other). Mode tag is
`x2` (x, both ends) or `y1` (y, one end).

```bash
# the headline result: strength vs fiber angle (runs the sweep, then plots+renders)
python3 src/slab.py sweep                      # x, both ends   -> out/slab/strength_x2.png, render_x2.png
python3 src/slab.py sweep --axis y --one-ended # vertical, one end -> strength_y1.png, render_y1.png

# one stress-strain curve / re-draw without re-running:
python3 src/slab.py one  --deg 45              # single angle
python3 src/slab.py plot   --axis x            # redraw strength curve from cached curve_*.npz
python3 src/slab.py render --axis x            # redraw particle picture from cached curves
```
Outputs (`out/slab/`): `strength_<mode>.png` (strength vs angle + stress–strain),
`render_<mode>.png` (the bars, damage-colored), `curve_<mode>_*.npz` (raw curves).

## 2. peel.py — delamination animations (Fig. 4)

```bash
python3 src/peel.py --mode peel                # transverse-iso -> clean STRIP (Fig.4 middle)
python3 src/peel.py --mode peel --iso          # isotropic     -> small PIECE  (Fig.4 right)
python3 src/peel.py --mode split               # two layers pulled apart (pre-broken bond)
```
Outputs (`out/slab/`): `peel.gif` + `peel_strip.png` (and `peel_iso.*`, `split.*`).
Useful knobs: `--lift` (peel speed), `--frames`, `--notch-len`, `--grip-len`.

## 3. cohesive.py — directional extension, on vs off

Stiff top + soft bulk + weak bond; Option B directional release toggled, damage
driver held fixed. Prints the `inplane_keep` discriminant.

```bash
python3 src/cohesive.py --directional on       # bond opens, in-plane KEPT
python3 src/cohesive.py --directional off      # isotropic g(d), bond fully soft
```
Outputs (`out/cohesive/`): `on.gif`/`off.gif` + `*_strip.png`; the run prints
`normal_release` and `inplane_keep` (use the ON-vs-OFF *ratio*; the absolute
`normal_release` is confounded by the moving crack front).

## 4. expose.py — render harness STEP 1 (exposed crack face)

Runs the flap peel and labels, per frame, the particles that **lost a bonded
neighbor** = the new crack surface (monotone-latched so labels don't flicker), with
an outward normal each.

```bash
python3 src/expose.py
```
Outputs (`out/render/`): `exposed.npz` (per frame: positions `x`, `exposed` mask,
`normals`, `d`, `spacing`) — the input for STEP 2 — and `expose_strip.png` (sanity
viz: red = exposed crack face).

---

## Where we are / next

- Done: solver validated, toughness quantified, Fig. 4 reproduced, directional
  extension shown, and STEP 1 (exposed-face extraction) producing clean labels.
- Next: **STEP 2** render the exposed face (CPU point-splat stand-in here; `gsplat`
  on GPU later — it's CUDA-only). **STEP 3** a metric sensitive to directional
  structure on the face. Then the **A/B**: pre-baked-interior vs
  fracture-state-conditioned appearance, scored against an authored synthetic target.
