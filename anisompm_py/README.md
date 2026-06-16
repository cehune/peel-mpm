# AnisoMPM in Python, driving the trained orange 3D Gaussian Splatting

A from-scratch **PyTorch / GPU** reimplementation of **AnisoMPM** (Wolper et al.,
*"AnisoMPM: Animating Anisotropic Damage Mechanics"*, SIGGRAPH 2020), ported
from the C++ `ziran2020` codebase, and wired up to physically fracture the
**trained orange 3DGS** model from `FruitNinja3DInterior`.

Everything runs in pure PyTorch — no CUDA extensions are compiled (there is no
`nvcc` on this machine), including a self-contained Gaussian-splatting renderer.

## What it does

1. Loads the trained `orange.ply` (3.93 M Gaussians) and fills its interior with
   ~210 k simulation particles.
2. Runs an **MLS-MPM** solve with the **anisotropic phase-field damage** model
   that is AnisoMPM's contribution.
3. Assigns **radial fibers** (as in `ziran2020/.../examples/orange.h`) so the
   fruit tends to split into orange-segment wedges.
4. Carries every Gaussian on its bound particle's local affine motion
   (`x_p + F_p·off0`, covariance `F_p·Σ0·F_pᵀ`) and renders with the original
   `orange.json` camera.
5. Sweeps the key AnisoMPM parameters and writes side-by-side comparison videos.

## The model (ported from C++ `AnisoFractureSimulation.h`)

Per particle we track a damage variable `d ∈ [0,1]`. Each step:

```
F = U S Vᵀ ,  R = U Vᵀ                         (polar rotation)
σ = Cauchy stress (fixed-corotated),  σ⁺ = tensile part (positive eigenvalues)
A = I + α (a ⊗ a),   a = R·a₀                  (structural tensor, a₀ = ref fiber)
φ = tr( (A σ⁺)² ) / σ_c²                        (anisotropic driving function)
ḋ = (1/η) · ⟨ (1-d)·ζ·⟨φ-1⟩  −  (d − l₀²·∇²d) ⟩  (Macaulay brackets ⟨·⟩ = relu)
g(d) = (1-d)²(1-k_r) + k_r                       (stiffness degradation)
```

`g` multiplies the deviatoric (and tensile-volumetric) stress, so cracks soften
the material in tension/shear while it still resists compression. `∇²d` is a grid
Laplacian (transfer `d` to the grid, gather with the B-spline's 2nd derivatives).

Parameter meaning (the "controls"):

| symbol | code | meaning |
|---|---|---|
| `E`, `ν` | `E`, `nu` | Young's modulus / Poisson — stiffness |
| `σ_c` | `percentage` → `σ_c = percentage·E` | critical stress (brittleness) |
| `η` | `eta` | damage viscosity/rate (brittle ↔ gummy) |
| `α`, `a₀` | `alpha`, fibers | anisotropy strength + fiber field (α=−1 transverse iso) |
| `l₀` | `l0_scale·dx` | phase-field length scale (crack width) |
| `k_r` | `residual` | residual stiffness of fully-broken material |

## Files

| file | role |
|---|---|
| `src/anisompm.py` | the AnisoMPM solver (MLS-MPM + phase-field damage) + colliders |
| `src/render.py` | self-contained PyTorch EWA Gaussian-splatting renderer (chunked, memory-bounded) |
| `src/orange_setup.py` | load orange.ply, fill interior particles, radial fibers, bind Gaussians |
| `src/run_orange.py` | scenarios (`squeeze`, `crush`, `split`), Gaussian deformation, frame rendering |
| `src/sweep.py` | parameter-control study → per-condition + side-by-side comparison videos |
| `src/test_block.py` | solver validation (drop stability + notched-bar fracture localization) |

## Running

```bash
# validation (block tests)
python3 src/test_block.py --drop

# parameter sweep across both GPUs, then compose comparisons
python3 src/sweep.py --device cuda:0 --conditions aniso,iso,brittle,tough &
python3 src/sweep.py --device cuda:1 --conditions eta_fast,eta_slow,soft,stiff &
wait
python3 src/sweep.py --compose          # builds compare_*.mp4 from cached frames
```

## Validation summary

* **Rest**: a block at rest stays exactly put (F=I), confirming a
  momentum-consistent transfer.
* **Free-fall**: velocity tracks `−g·t` exactly (rigid translation).
* **Drop**: a soft block falls, contacts a ground plane and settles (bounded
  velocity, no NaN).
* **Notched bar pull**: damage localizes at the notch (`d_notch ≫ d_far`) and the
  bar tears in two — the canonical AnisoMPM mode-I behavior.

## Notes / engineering choices

* **Linear algebra on Blackwell (sm_120)**: the batched cuSOLVER `eigh` is broken
  on this GPU, so the symmetric eigendecomposition (for σ⁺) is derived from the
  (fast, working) SVD.
* **Stability**: a deformation-gradient singular-value clamp `f_clamp` bounds
  extreme local stretch (unphysical for a solid), preventing both numerical
  blow-up and renderer footprint explosions; damage may be updated every *k*
  substeps for speed.
* **The trained orange's stored Gaussian scales are degenerate** (a dense point
  cloud), so each point is rendered as a small isotropic world-space splat that
  deforms with its particle's `F`.
* **Squeeze scenario** uses contact-free clamped caps (prescribed velocity),
  which is far more stable than resolving sphere/plane contact against the sparse
  curved surface.
