# Gate rationale — what the AnisoMPM delamination gate proves, and why

This document is the single source of truth for *why* the validation gate is built the
way it is: the one claim it serves, what each block proves (and cannot), what every
figure means, and which threshold is principled versus merely tuned. If a number in
`gate_verdict.py` looks arbitrary, it is explained here — or it is flagged as arbitrary.

**TL;DR of epistemic status.**

- **Solidly proven** whenever it greens: the *math* (`0a`) and the *angular law* (`Block 2`).
  These are tied to ground truth (finite differences) or to theory (the $\cot^2\theta$
  initiation law), and the angular law survives the wrong-model contrast.
- **Conditional** on a *fair loading protocol*: everything from `Block 1` selectivity
  upward. The loading protocol (cap vs spread vs clamp) demonstrably dominates the
  selectivity outcome and is resolution-dependent. A green here means "the model *can*
  peel directionally under *this* load," not "under all loads."
- **Not in scope:** matching the paper's animations quantitatively (there is nothing to
  match), or visual realism.

---

## 1. The single claim

The paper (Wolper et al., *AnisoMPM*, SIGGRAPH 2020) biases the phase-field damage
*driving force* with a structural tensor $A$ so that cracks prefer a direction. For a
bond, $A$ emphasizes the normal, so normal tension breaks it while in-plane shear does
not. Everything in the gate exists to prove exactly one thing:

> Damage is **routed to the bond along its normal by the structural tensor**, the
> directional stress split does the **mechanical** work, and **flipping the tensor flips
> the outcome** — i.e. the anisotropy is *causal*, not incidental.

The constitutive pieces being exercised (all in `src/anisompm.py`, equations from the paper):

- driving force $\displaystyle \Phi(\sigma^+) = \frac{1}{\sigma_c^2}\,\big(A\,\sigma^+ : \sigma^+ A\big)$  [Eq. 8]
- structural tensor $A = I + \alpha_1\,a_1\!\otimes a_1 + \alpha_2\,a_2\!\otimes a_2$  [Eq. 9]
- evolution $d_{n+1} = \min\!\big(1,\; d + \tfrac{\Delta t}{\eta}\,\langle (1-d)\,\tilde D - (d - l_0^2\,\Delta d)\rangle\big)$  [Eq. 11], with **history-max** drive $\tilde D$
- degradation $g(d) = (1-d)^2(1-r) + r$  [Eq. 10]
- $l_0 = \Delta x/2$, $\zeta = 1$; Laplacian $\Delta d$ via a **cubic** B-spline pass (paper §4.3.1 mandates $\ge$ cubic)
- directional stress (Option B, shipped): split by the *deformed* normal $m = F n_0 / |F n_0|$,
  with $P = m\!\otimes m$, $Q = I - P$, and $\tau = \tau_\parallel + g_\mathrm I\,\tau_\perp + g_{\mathrm{II}}\,\tau_\mathrm{shear}$ under a tension gate.

---

## 2. Particle taxonomy — target, handle, body (read this first)

Every particle gets one **region** label from the signed-distance partition
$\phi(X) = |X - c| - R$, with shell thicknesses $t_p = \max(0.08R,\,2.5\,s)$ and
$t_i = \max(0.07R,\,2.5\,s)$ ($s$ = particle spacing):

| region | role | meaning |
|---|---|---|
| **interface** | **target** | the bond — the shell we *want* to delaminate |
| **peel** | **handle** | the outer top layer — the thing you grip and pull |
| **flesh** | **body** | the core — the true innocent that must stay intact |

On top of the region label, two boundary conditions **freeze** particles (`allow=False`,
so their damage $d$ is pinned at 0 and they move rigidly): the **clamp** (bottom,
$y < c_y - 0.55R$) and the **grip**. The grip is `cap` (a $30^\circ$ polar patch of peel)
or `spread` (the *entire* peel layer) — see §8.

**"Bystander" must mean the body, but the metric currently lumps handle+body.** This is
the most important subtlety and was a real source of confusion. Illustrative counts for a
small ng24/ppcd1.5 ball ($N=4{,}337$: flesh 540, interface 1289, peel 2508):

| grip | frozen | **live bystanders** (can take damage) |
|---|---|---|
| cap | 738 | **2,402** = 540 flesh + **1,862 free peel** |
| spread | 2,600 | **540** = flesh only (all peel is the handle) |

A "bystander break" in the gate is *any* non-interface particle with $d > $ `break_thresh`.
Under `cap` that pool includes 1,862 free-peel particles; under `spread` it is the 540
body particles only. So selectivity numbers are not directly comparable across grips —
the *denominator* changes. The clean question selectivity should ask is **"does the body
stay intact while the bond breaks,"** watching flesh, not the handle.

---

## 3. The claim ladder — each rung kills a way the rung above could be a false positive

Run order matters; a confounded green is impossible because each stage can veto the next.
Mapping is to `run_gate.sh` blocks and the `PASS_*` booleans in `gate_verdict.py`.

**`0a` math is the paper's math** — `src/1_fd_check.py`, deterministic, no simulation:
1. $P = \partial\Psi/\partial F$ vs central finite differences (~$10^{-10}$): the stress is
   genuinely the gradient of the claimed energy.
2. reconstruction at $d=0$: the directional split is a true decomposition (adds/loses nothing).
3. **release at $d=1$**: normal + shear tractions vanish, in-plane is retained. *This is the
   check that tells a peel from its mirror image* — it verifies *which* component releases.
   Reconstruction (2) is blind to a $P\leftrightarrow Q$ swap; this is not.
   → *Proves the constitutive code is correct against ground truth. Proves nothing about emergent peeling.* (`PASS_0`, with the sanity control below.)

**`0b`/`0b′` runs and stays finite** — a maximally-helped cell must obviously delaminate; a
harsh cell must not go NaN. Numerical hygiene, not physics.

**`D` the split does mechanical work** — with contrast stripped (`--equal-E`, $\rho=1$), a
normal **gap** opens at the interface and `inplane_keep` fires only with directional stress
on. → *Proves directional stress (not just scalar damage) moves material.* (`PASS_D`.)

**`1` the structural tensor is the causal lever** — correct vs iso vs wrong, identical in
*everything but $A_0$* ($\rho=1$, `--equal-E` strip modulus and toughness contrast). If only
$A_0$ changed and only correct selects, the anisotropy *is* the cause. → *The central
experiment.* (`PASS_1`.)

**`2` the angular dependence matches theory** — within one run the magnitude-free ratio obeys
$$\frac{\phi_{int}}{\phi_{shear}} \sim \cot^2\theta,$$
because the driver squares an already-quadratic normal traction (so $\phi_{int}\sim\cos^4\theta$,
*not* $\cos^2$). correct → log-log slope $\approx +1$; wrong (shear-driven) → negative; iso →
scatter. → *The strongest single check:* a functional form that is hard to fake, and the wrong
model yields the *opposite sign* under identical loading. (`PASS_2`.)

**`3` robust to adverse toughness** — selectivity holds for some $\rho^\* > 1$, i.e. the bond is
*tougher* than bystanders and still delaminates. Rules out "it only works because the bond is
weak." (`PASS_3`.)

**`4` not a discretization artifact** — vary grid (`ngrid`) and particle density (`ppcd`)
*separately*; the (exist, select) verdict is invariant. (`PASS_4`.)

The fully-green statement is therefore: *"my code computes the paper's constitutive model
correctly, and that model — in my implementation — produces directional selective
delamination caused by the structural tensor, with the theoretically-predicted angular
law, robust to numerics and to an adverse toughness ratio."*

---

## 4. Observables and the onset-grading rule (`src/derive.py`)

- **exist** — onset reached, or final `broken_int_frac` $> 0.05$. Uses the monotone maximum
  (damage is irreversible, so existence is a high-water mark).
- **select** — `n_broken_outside == 0`: no flesh/peel particle exceeds `break_thresh`,
  **read at onset** (see below).
- **routing** — peak over the run of $\phi_{int}/\phi_{bystander}$. A *drive* ratio (cause),
  not a damage outcome.
- **cos2_slope** — log-log slope of $\phi_{int}/\phi_{shear}$ vs $\cot^2\theta$ over loaded
  interface particles.
- **normal_release / inplane_keep** — from the degraded Cauchy stress at onset: normal
  traction collapses, in-plane is retained. `inplane_keep` is read off the split itself, so
  it is only a wiring smoke, not a sole discriminant.
- **normal_gap** — increase in nearest-neighbour distance from interface to flesh: the sheet
  physically lifts.

**Why grade at onset, not the final frame.** $d$ is irreversible and monotone. If you keep
pulling after the bond has parted, the bystanders eventually take damage too — that is
*over-pull*, an artifact of pulling past the event, not a selectivity failure of the model.
So the over-pull-sensitive metrics (select, cos2, release, gap, grip) are snapshotted at the
**first frame where $>5\%$ of the interface is broken** (the onset); EXISTENCE uses the
monotone max. Empirically the same run can read `select=PASS` at onset and `FAIL` at the
final frame — the figures (§5) show exactly this.

---

## 5. What each figure means — and its limit (`src/make_figs.py`)

- **2 · angular law** — the load-bearing figure. Magnitude-free, and correct vs wrong give
  *opposite slopes from the same loading* (ng24: $+0.81$ vs $-0.33$). *Limit:* a fit over
  loaded interface particles — needs a genuine spread of $\theta$, which the `spread` grip
  weakens (it muted wrong to $+0.19$, still $<0.5$ but less emphatic).
- **3 · routing** — separation of *drives* $\phi_{int}$ vs $\phi_{bystander}$ over time, and
  the bottom panel exposes over-pull (bystander damage crossing the break line late).
  *Limit:* $\phi$ is the driver, not the outcome, and "bystander" is the lumped quantity.
- **1 · slices** — the actual damage field on deformed particles: *where* damage goes.
  *Limit:* one config, one mid-plane slice, qualitative.
- **4 · histogram** — non-interface damage at onset vs final; the visual case for onset
  grading. *Limit:* same handle/body lumping.

---

## 6. Thresholds — principled vs tuned

| threshold | value | status |
|---|---|---|
| $l_0 = \Delta x/2$, $\zeta = 1$ | — | **principled** (from paper) |
| cubic Laplacian kernel | — | **principled** (paper §4.3.1 mandates $\ge$ cubic) |
| `break_thresh` $= \max(0.5,\; e^{-s/l_0} + 0.25)$ | ~0.51 | **principled** — $e^{-s/l_0}$ is the diffuse-band "halo" a bystander one spacing from a $d{=}1$ band equilibrates to from the Laplacian *alone*; the threshold sits above it so the halo (method, not leakage) is not miscounted as a break |
| cot² slope target | $+1$ | **principled** (initiation law, §3) |
| $\rho^\* > 1$ | 1 | **principled** — definitionally "bond is tougher than bystanders" |
| $d > 0.5$ = "broken" | 0.5 | convention — midpoint of $[0,1]$; $g(0.5)=0.25$ stiffness ("mostly gone") |
| onset $= 5\%$ interface broken | 0.05 | **arbitrary** — "enough of the bond has parted that delamination has begun" |
| `routing > 2`, iso margin $3\times$ | 2, 3 | **arbitrary** margins |
| slope band $0.6$–$1.4$ (correct), $<0.5$ (wrong) | — | tolerance around the principled $+1$ |
| $\sigma_c^{int} = 0.05\,E$, $\eta = 0.02$, `f_clamp`$=(0.35, 2.8)$ | — | **tuned** (delamination rate / numerical stability) |

**Why the arbitrary ones are tolerable — the contrasts defense.** The gate's conclusions
come from *differences between cells that share every threshold*, not from absolute lines.
correct gives $+0.81$, wrong gives $-0.33$, iso scatters — under the *same* thresholds. If a
conclusion is a *difference* between cells with identical thresholds, the threshold value
cancels out of the conclusion. The arbitrary numbers decide *cosmetic* pass/fail; the science
lives in correct-vs-wrong-vs-iso.

**Where this defense is weakest.** `PASS_1`/`PASS_3`'s absolute lines (`routing > 2`,
`d_bystander < 0.5`, `grip_d < 0.1`) are *not* contrasts — they are tuned lines a tuned loading
can cross. Treat those as the soft links and lean on the contrasts (Blocks 1-correct-vs-wrong,
2) for the real verdict.

---

## 7. What the gate does NOT prove

- **Not a benchmark against the paper.** Wolper's experiments are per-shot-tuned animations
  with no quantitative selectivity numbers; there is nothing to match. This is an
  *internal-consistency + causal-isolation* gate.
- **A pass is conditional on a *fair* load** (§8), not a claim that the model peels under all
  loads. "Fair" is a judgment call.
- **It says nothing about visual realism.**

---

## 8. The loading-protocol caveat — the live frontier

The selectivity blocks are at the mercy of how load is introduced. Evidence (ng24/ppcd1.5,
$\rho=0.1$ correct, identical except grip):

| | interface drive $\phi$ | flesh (body) $\phi$ | peel (handle) $\phi$ | broken bystanders @ final |
|---|---|---|---|---|
| **cap** | 18,011 | **0.02** | **20.0** | 11 — *all peel*, at the cap edge ($\cos\theta\!\approx\!0.80$), $d\to0.945$ |
| **spread** | 1,992 | 0.38 | 0.64 | **0** |

Reading: the rigid `cap` concentrates stress at its *edge* and tears the **handle** (free peel
just outside the cap); the **body** was never stressed at this resolution ($\phi_{flesh}=0.02$).
`spread` (whole peel = handle) removes that two ways — **physically** (no edge: peel drive
$20\to0.64$) and by **reclassification** (the free peel leaves the live-bystander pool, $2402\to540$).
Both are legitimate, but they must be reported separately, not conflated into one "0.94 → 0.10."

**Resolution dependence / the open question.** The earlier ng64 cluster run showed
$\phi_{flesh}\sim10^4$ at onset — a *body* over-stress this ng24 run does **not** reproduce. So
whether the cap cooks the *body* (not just the handle) is unproven at the scale runnable on CPU.
The decisive ng64 test is: **does $\phi_{flesh}$ fall sub-critical ($<1$) with `spread`?** If yes,
the cap was the whole story. If $\phi_{flesh}$ is still $\sim10^4$, spreading the *peel* grip did
not reach the *body*, and the **clamp** (the next sharpest BC edge — `spread`'s only residual
damage sits there, at $\cos\theta\approx-0.95$) is the real culprit and needs the same treatment.

---

## File map

| file | role |
|---|---|
| `src/anisompm.py` | the solver (constitutive model + MLS-MPM). The "physics"; its hash is the sim-cache key. |
| `src/1_fd_check.py` | `0a` math gate (FD, reconstruction, release). |
| `src/peel_test.py` | `build_sim` + `run_sim` → cached snapshot `out/sims/<tag>.pt`. Loading knobs: `--grip {cap,spread}`, `--notch-deg`. |
| `src/derive.py` | snapshot → observables → `out/results/<tag>.json`. Pure; re-derivable without re-simulating. |
| `src/gate_verdict.py` | results dir → `PASS_0..4`, `PASS_D`, `GATE`; drift banner on mixed solver/harness/notch/grip. |
| `src/run_gate.sh` | 3-phase: run/cache sims → derive all → verdict. `GRIP=`, `NOTCH=`, `RESUME=`, `DEV=`. |
| `src/make_figs.py` | the four figures from real run data. |
