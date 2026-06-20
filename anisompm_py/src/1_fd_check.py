#!/usr/bin/env python3
"""
1_fd_check.py -- the validation gate for the directional damage extension.

It checks BOTH constitutive forms that anisompm.py ships:

  Option A (variational):  directional_pk1 == d energy / d F, by central finite
                           differences.  If the coded stress is not the gradient
                           of the coded energy, every Phi in peel_test is
                           meaningless and a "passing" peel gate would be noise.

  Option B (operator split): directional_kirchhoff reconstructs the plain
                           corotated Kirchhoff stress at d=0.  Option B is not
                           the gradient of one potential (it is the field-
                           standard split, like Miehe / Amor), so the dPsi/dF
                           test does not apply to it -- the reconstruction
                           identity is its analogue.

Both gate the pipeline: exit 0 only if BOTH pass.

    python3 src/1_fd_check.py --tol 1e-6 --summary-json out/validate.jsonl

This imports the REAL functions from anisompm, so it tests the code the solver
actually runs (energy/directional_pk1 share gI/gII with the solver -- that
shared degradation is what makes the FD identity meaningful).
"""
import argparse, json, os, sys, time
import torch

torch.set_default_dtype(torch.float64)   # finite differences need double precision
sys.path.insert(0, os.path.dirname(__file__))

# The functions under test -- imported, not redefined, so the gate guards the
# real solver code.  energy() and directional_pk1() use anisompm's module-level
# gI/gII, the same pair the directional stress is built from.
from anisompm import (energy, directional_pk1, reconstruction_check,  # noqa: E402
                      release_check, gI, gII)


def fd_gradient(F, n0, d, params, eps):
    """Central-difference d energy / d F, component by component."""
    P_num = torch.zeros_like(F)
    for i in range(3):
        for j in range(3):
            Fp = F.clone(); Fm = F.clone()
            Fp[:, i, j] += eps
            Fm[:, i, j] -= eps
            P_num[:, i, j] = (energy(Fp, n0, d, **params)
                              - energy(Fm, n0, d, **params)) / (2 * eps)
    return P_num


def option_a_fd(args):
    """Gate 1: directional_pk1 == d energy/dF."""
    torch.manual_seed(args.seed)
    params = dict(mu=10.0, lam=20.0, gamma_I=5.0, gamma_II=3.0)
    # random, well-conditioned, near-identity F (avoids SVD degeneracy at F=I)
    F = torch.eye(3).repeat(args.n, 1, 1) + 0.15 * torch.randn(args.n, 3, 3)
    F = F[torch.linalg.det(F) > 0.2]
    n0 = torch.randn(F.shape[0], 3)
    n0 = n0 / n0.norm(dim=-1, keepdim=True)

    worst_rel = 0.0
    print("Gate 1  (Option A):  directional_pk1 == d energy/dF")
    for d_val in (0.0, 0.5, 0.9):              # undamaged, mid, near-broken
        d = torch.full((F.shape[0],), d_val)
        P_an = directional_pk1(F, n0, d, **params)
        P_fd = fd_gradient(F, n0, d, params, args.eps)
        err = (P_an - P_fd).abs().max().item()
        scale = P_an.abs().max().item() + 1e-12
        rel = err / scale
        worst_rel = max(worst_rel, rel)
        print(f"   d={d_val:>4}:  max_abs_err={err:.3e}   max_rel_err={rel:.3e}")
    ok = worst_rel < args.tol
    print(f"   -> {'PASS' if ok else 'FAIL'}  (worst rel err {worst_rel:.3e}, tol {args.tol:.0e})")
    return ok, worst_rel


def option_b_reconstruction(args):
    """Gate 2: directional_kirchhoff rebuilds corotated tau at d=0."""
    torch.manual_seed(args.seed + 1)
    print("\nGate 2  (Option B):  directional_kirchhoff reconstructs corotated tau at d=0")
    F = torch.eye(3).repeat(args.n, 1, 1) + 0.15 * torch.randn(args.n, 3, 3)
    F = F[torch.linalg.det(F) > 0.2]
    U, S, Vh = torch.linalg.svd(F)
    R = U @ Vh
    J = torch.linalg.det(F)
    n0 = torch.randn(F.shape[0], 3)
    n0 = n0 / n0.norm(dim=-1, keepdim=True)
    err = reconstruction_check(F, R, J, n0, mu=10.0, lam=20.0, verbose=False)
    tol = 1e-9                                  # float64; this is an exact identity
    ok = err < tol
    print(f"   max|split(d=0) - corotated| = {err:.3e}")
    print(f"   -> {'PASS' if ok else 'FAIL'}  (tol {tol:.0e})")
    print("   (NB: d=0 reconstruction is direction-BLIND -- a P<->Q swapped split")
    print("    that keeps the normal and sheds the in-plane passes this identically.")
    print("    Gate 3 is what tells a peel from its opposite.)")
    return ok, err


def option_b_release(args):
    """Gate 3: directional_kirchhoff RELEASES the normal + interfacial-shear
    traction at d=1 while PRESERVING the in-plane stress -- the property that
    defines a peel, and the one Gate 2 cannot see.  Run against the shipped
    split, it PASSes the correct model and FAILs a P<->Q swap (proved in
    validate_goal.py)."""
    torch.manual_seed(args.seed + 2)
    print("\nGate 3  (Option B):  releases normal+shear, keeps in-plane at d=1")
    F = torch.eye(3).repeat(args.n, 1, 1) + 0.15 * torch.randn(args.n, 3, 3)
    F = F[torch.linalg.det(F) > 0.2]
    U, S, Vh = torch.linalg.svd(F)
    R = U @ Vh
    J = torch.linalg.det(F)
    n0 = torch.randn(F.shape[0], 3)
    n0 = n0 / n0.norm(dim=-1, keepdim=True)
    ok, nk, ik, sk = release_check(F, R, J, n0, mu=10.0, lam=20.0, res=args.res, verbose=False)
    print(f"   normal_kept={nk:.3f}  shear_kept={sk:.3f}  (want ~{args.res:g})  "
          f"inplane_kept={ik:.3f}  (want ~1)")
    print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok, nk, ik, sk


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tol', type=float, default=1e-6, help='max relative error to pass Gate 1')
    ap.add_argument('--n', type=int, default=64, help='random samples')
    ap.add_argument('--eps', type=float, default=1e-6, help='FD step')
    ap.add_argument('--res', type=float, default=1e-3,
                    help='solver residual k_r to validate Gate 3 against '
                         '(match your add_object residual)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--summary-json', default=None)
    args = ap.parse_args()

    okA, worstA = option_a_fd(args)
    okB, errB = option_b_reconstruction(args)
    okC, nk, ik, sk = option_b_release(args)
    ok = okA and okB and okC
    print(f"\nFD CHECK {'PASS' if ok else 'FAIL'}  "
          f"(A worst rel {worstA:.2e}; B recon {errB:.2e}; "
          f"C release n/s={nk:.3f}/{sk:.3f} keep={ik:.3f})")

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or '.', exist_ok=True)
        with open(args.summary_json, 'a') as f:
            f.write(json.dumps({
                "test": "fd_check", "t": time.time(),
                "optionA_max_rel_err": worstA, "optionA_tol": args.tol, "optionA_pass": bool(okA),
                "optionB_recon_err": errB, "optionB_pass": bool(okB),
                "optionC_normal_kept": nk, "optionC_shear_kept": sk,
                "optionC_inplane_kept": ik, "optionC_pass": bool(okC),
                "pass": bool(ok),
            }) + "\n")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
