"""
AnisoMPM in PyTorch
===================

A faithful, self-contained reimplementation of the AnisoMPM solver
(Wolper et al., SIGGRAPH 2020 -- "AnisoMPM: Animating Anisotropic Damage
Mechanics"), ported from the C++ `ziran2020` codebase.

It couples a standard MLS-MPM (Hu et al. 2018) momentum solve with the
anisotropic phase-field damage model that is the paper's contribution:

  structural tensor :  A = I + alpha * (a (x) a)         a = R a0   (current fiber)
  driving function  :  phi = tr( (A sigma+)^2 ) / sigma_c^2
  damage evolution  :  d_dot = (1/eta) < (1-d) zeta <phi-1>  -  (d - l0^2 lap(d)) >
  degradation       :  g(d) = (1-d)^2 (1-k_r) + k_r

where <.> is the Macaulay bracket (relu), sigma+ is the tensile (positive
eigenvalue) part of the Cauchy stress, R is the rotation from the polar
decomposition of the deformation gradient F, and lap(d) is the grid Laplacian
of the damage field.  The degradation g multiplies the deviatoric (and tensile
volumetric) stress so that cracks soften the material in tension/shear but the
material still resists compression.

Everything runs batched on the GPU in float32/float64.

Reference (C++): ziran2020/Projects/anisofracture/AnisoFractureSimulation.h
                 ziran2020/Projects/anisofracture/examples/orange.h
"""

from __future__ import annotations
import math
import torch

def sym_eig(A: torch.Tensor):
    """Analytic eigendecomposition of a batch of symmetric 3x3 matrices.

    Closed form (Cardano for the eigenvalues, null-space products + a cross
    product for the eigenvectors).  No LAPACK / cuSOLVER / SVD, so it runs
    natively on every backend including Apple MPS (where torch.linalg.svd is
    unimplemented and silently falls back to CPU -- a host<->device copy every
    substep).  It is also correct on indefinite/degenerate stress, unlike the
    old SVD sign-recovery trick which mis-signed balanced shear (eigs +/-s with
    equal singular values).  Returns (evals ascending, evecs as columns)."""
    a00 = A[:, 0, 0]; a11 = A[:, 1, 1]; a22 = A[:, 2, 2]
    a01 = A[:, 0, 1]; a02 = A[:, 0, 2]; a12 = A[:, 1, 2]
    q = (a00 + a11 + a22) / 3.0
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = torch.sqrt(torch.clamp(p2 / 6.0, min=1e-30))
    iso = (p2 < 1e-18)                                       # A ~ q I (degenerate)
    bp = 1.0 / p
    b00 = (a00 - q) * bp; b11 = (a11 - q) * bp; b22 = (a22 - q) * bp
    b01 = a01 * bp; b02 = a02 * bp; b12 = a12 * bp
    detB = (b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02))
    r = torch.clamp(0.5 * detB, -1.0, 1.0)
    phi = torch.acos(r) / 3.0
    TWO_PI_3 = 2.0943951023931953                            # 2*pi/3
    e1 = q + 2.0 * p * torch.cos(phi)                        # largest
    e3 = q + 2.0 * p * torch.cos(phi + TWO_PI_3)             # smallest
    e2 = 3.0 * q - e1 - e3
    I3 = torch.eye(3, device=A.device, dtype=A.dtype).expand_as(A)

    def _vec(li, lj):                                        # eigvec of the third eval
        M = (A - li.view(-1, 1, 1) * I3) @ (A - lj.view(-1, 1, 1) * I3)
        nrm = M.norm(dim=1)                                  # per-column norms (N,3)
        k = nrm.argmax(dim=1)                                # most reliable column
        v = torch.gather(M, 2, k.view(-1, 1, 1).expand(-1, 3, 1))[:, :, 0]
        return v / (v.norm(dim=1, keepdim=True) + 1e-30)

    v1 = _vec(e2, e3)                                        # eigvec for e1
    v3 = _vec(e1, e2)                                        # eigvec for e3
    v2 = torch.cross(v3, v1, dim=1)
    v2 = v2 / (v2.norm(dim=1, keepdim=True) + 1e-30)
    v3 = torch.cross(v1, v2, dim=1)                          # re-orthonormalise
    if iso.any():                                            # fall back to I basis
        im = iso.view(-1, 1)
        ex = torch.tensor([1.0, 0, 0], device=A.device, dtype=A.dtype)
        ey = torch.tensor([0, 1.0, 0], device=A.device, dtype=A.dtype)
        ez = torch.tensor([0, 0, 1.0], device=A.device, dtype=A.dtype)
        v1 = torch.where(im, ex, v1); v2 = torch.where(im, ey, v2); v3 = torch.where(im, ez, v3)
    evals = torch.stack([e3, e2, e1], dim=1)                 # ascending
    Q = torch.stack([v3, v2, v1], dim=2)                     # columns match evals
    return evals, Q


# quadratic B-spline 2nd-derivative stencil (wrt the local fx coordinate)
_D2 = (1.0, -2.0, 1.0)


class AnisoMPM:
    def __init__(
        self,
        n_grid: int = 96,
        grid_lim: float = 1.0,
        dt: float = 1e-4,
        gravity=(0.0, -9.8, 0.0),
        grid_damp: float = 1.0,
        f_clamp=None,                 # optional (smin, smax) on F singular values
        damage_every: int = 1,        # update phase-field damage every k substeps
        device: str = "cuda:0",
        dtype=torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_grid = int(n_grid)
        self.grid_lim = float(grid_lim)
        self.dx = grid_lim / n_grid
        self.inv_dx = 1.0 / self.dx
        self.dt = float(dt)
        self.gravity = torch.tensor(gravity, device=self.device, dtype=dtype)
        self.grid_damp = float(grid_damp)
        self.f_clamp = f_clamp
        self.damage_every = max(1, int(damage_every))
        self._step = 0

        # 27-neighbour offsets for the 3x3x3 quadratic kernel
        off = torch.arange(3, device=self.device)
        gx, gy, gz = torch.meshgrid(off, off, off, indexing="ij")
        self.offsets = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], 1)  # (27,3) int

        self.colliders = []          # list of callables(state, time)->None acting on grid_v
        self.particle_bc = []        # list of callables(self, time)->None acting on particle v before p2g
        self.n = 0

    # ------------------------------------------------------------------ setup
    def add_object(
        self,
        x: torch.Tensor,            # (N,3) initial positions, world == grid coords
        vol: torch.Tensor,          # (N,) per-particle volume
        rho: float = 500.0,
        E: float = 1.0e4,
        nu: float = 0.4,
        fibers: torch.Tensor | None = None,   # (N,3) reference fiber a0 (unit) or None=isotropic
        alpha: float = -1.0,        # anisotropy weight (alpha=0 isotropic, -1 transverse iso)
        percentage: float = 0.15,   # damage threshold -> sigma_c (fraction of failure stress)
        l0_scale: float = 0.5,      # l0 = l0_scale * dx
        eta: float = 0.01,          # damage viscosity / rate
        zeta: float = 1.0,
        residual: float = 0.001,    # k_r residual stiffness
        allow_damage: bool = True,
    ):
        dev, dt = self.device, self.dtype
        x = x.to(dev, dt)
        vol = vol.to(dev, dt)
        N = x.shape[0]

        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        new = {}
        new["x"] = x
        new["v"] = torch.zeros_like(x)
        new["F"] = torch.eye(3, device=dev, dtype=dt).expand(N, 3, 3).contiguous()
        new["C"] = torch.zeros(N, 3, 3, device=dev, dtype=dt)
        new["vol0"] = vol.clone()
        new["mass"] = vol * rho
        new["mu"] = torch.full((N,), mu, device=dev, dtype=dt)
        new["lam"] = torch.full((N,), lam, device=dev, dtype=dt)
        new["d"] = torch.zeros(N, device=dev, dtype=dt)
        new["lap"] = torch.zeros(N, device=dev, dtype=dt)

        if fibers is None or alpha == 0.0:
            new["a0"] = torch.zeros(N, 3, device=dev, dtype=dt)
            new["alpha"] = torch.zeros(N, device=dev, dtype=dt)
        else:
            f = fibers.to(dev, dt)
            f = f / (f.norm(dim=1, keepdim=True) + 1e-12)
            new["a0"] = f
            new["alpha"] = torch.full((N,), float(alpha), device=dev, dtype=dt)

        # Reference structural tensor A0 (the single source of truth for the
        # damage filter).  The driver uses the *current* tensor A = R A0 R^T,
        # which co-rotates A0 by the polar rotation R.  Fibers are just one way
        # to build A0:  A0 = I + alpha (a0 (x) a0).  set_structural_tensor() can
        # overwrite A0 per particle with any symmetric 3x3 -- e.g. the rank-1
        # projector A0 = n (x) n, which makes phi = (sigma_nn^+/sigma_c)^2 so the
        # ONLY damage driver is the normal traction (delamination bands), with no
        # single-fiber "protect one direction" limitation.
        a0_ = new["a0"]; al_ = new["alpha"]
        aa0 = a0_.unsqueeze(2) * a0_.unsqueeze(1)                       # (N,3,3)
        eyeN = torch.eye(3, device=dev, dtype=dt).expand(N, 3, 3)
        new["A0"] = eyeN + al_.view(-1, 1, 1) * aa0                     # I + alpha a0 a0^T

        # critical stress.  The AnisoMPM C++ derives sigma_c from a "percentage"
        # of the energy/stress at failure of the elastic model.  We use the
        # common surrogate sigma_c = percentage * E, which reproduces the same
        # qualitative control (small percentage -> brittle, large -> tough).
        sigma_c = max(percentage, 1e-8) * E
        new["sigma_c"] = torch.full((N,), float(sigma_c), device=dev, dtype=dt)
        new["eta"] = torch.full((N,), float(eta), device=dev, dtype=dt)
        new["zeta"] = float(zeta)
        new["l0"] = l0_scale * self.dx
        new["residual"] = float(residual)
        new["allow_damage"] = bool(allow_damage)

        self._append(new)
        return self

    def _append(self, new):
        if self.n == 0:
            self.x = new["x"]; self.v = new["v"]; self.F = new["F"]; self.C = new["C"]
            self.vol0 = new["vol0"]; self.mass = new["mass"]
            self.mu = new["mu"]; self.lam = new["lam"]
            self.d = new["d"]; self.lap = new["lap"]
            self.a0 = new["a0"]; self.alpha = new["alpha"]
            self.A0 = new["A0"]
            self.sigma_c = new["sigma_c"]; self.eta = new["eta"]
            # per-object scalars promoted to per-particle where they may differ
            N = new["x"].shape[0]
            self.zeta = torch.full((N,), new["zeta"], device=self.device, dtype=self.dtype)
            self.l0 = torch.full((N,), new["l0"], device=self.device, dtype=self.dtype)
            self.residual = torch.full((N,), new["residual"], device=self.device, dtype=self.dtype)
            self.allow = torch.full((N,), new["allow_damage"], device=self.device, dtype=torch.bool)
        else:
            cat = lambda a, b: torch.cat([a, b], 0)
            self.x = cat(self.x, new["x"]); self.v = cat(self.v, new["v"])
            self.F = cat(self.F, new["F"]); self.C = cat(self.C, new["C"])
            self.vol0 = cat(self.vol0, new["vol0"]); self.mass = cat(self.mass, new["mass"])
            self.mu = cat(self.mu, new["mu"]); self.lam = cat(self.lam, new["lam"])
            self.d = cat(self.d, new["d"]); self.lap = cat(self.lap, new["lap"])
            self.a0 = cat(self.a0, new["a0"]); self.alpha = cat(self.alpha, new["alpha"])
            self.A0 = cat(self.A0, new["A0"])
            self.sigma_c = cat(self.sigma_c, new["sigma_c"]); self.eta = cat(self.eta, new["eta"])
            N = new["x"].shape[0]
            self.zeta = cat(self.zeta, torch.full((N,), new["zeta"], device=self.device, dtype=self.dtype))
            self.l0 = cat(self.l0, torch.full((N,), new["l0"], device=self.device, dtype=self.dtype))
            self.residual = cat(self.residual, torch.full((N,), new["residual"], device=self.device, dtype=self.dtype))
            self.allow = cat(self.allow, torch.full((N,), new["allow_damage"], device=self.device, dtype=torch.bool))
        self.n = self.x.shape[0]

    def set_radial_fibers(self, center, zero_dim: int = 1, alpha: float = -1.0):
        """Radial fibers about an axis (as in orange.h radialFibers(center, zeroDim)).

        Fibers point radially from `center`, with the component along `zero_dim`
        set to zero so they lie in the plane perpendicular to that axis.  With
        alpha<0 this makes the material weak across radial planes -> it splits
        into wedge-shaped segments (orange-peel structure)."""
        c = torch.as_tensor(center, device=self.device, dtype=self.dtype)
        r = self.x - c
        r[:, zero_dim] = 0.0
        nrm = r.norm(dim=1, keepdim=True)
        r = torch.where(nrm > 1e-9, r / (nrm + 1e-12), torch.zeros_like(r))
        self.a0 = r
        self.alpha = torch.full((self.n,), float(alpha), device=self.device, dtype=self.dtype)
        # keep A0 in sync: A0 = I + alpha a0 a0^T  (so A = R A0 R^T reproduces the
        # old A = I + alpha (R a0)(R a0)^T exactly).
        aa = self.a0.unsqueeze(2) * self.a0.unsqueeze(1)
        eyeN = torch.eye(3, device=self.device, dtype=self.dtype).expand(self.n, 3, 3)
        self.A0 = eyeN + self.alpha.view(-1, 1, 1) * aa
        return self

    def set_structural_tensor(self, mask, A0):
        """Overwrite the reference structural tensor A0 for the masked particles.

        A0 is the per-particle damage filter in the *reference* frame; the driver
        uses the co-rotated A = R A0 R^T each step.  Pass any symmetric 3x3 per
        particle.  Common recipes:
            A0 = I - r r^T   (radial fiber, alpha=-1): protect the radial axis.
            A0 = r r^T       (rank-1 projector): isolate the normal traction, so
                              phi = (sigma_nn^+ / sigma_c)^2 -- the delamination
                              filter, with the whole tangent plane protected.

        mask : (N,) bool                which particles to overwrite
        A0   : (N,3,3) or (M,3,3)       new reference tensors; only mask rows used
        """
        mask = torch.as_tensor(mask, device=self.device).bool()
        A = torch.as_tensor(A0, device=self.device, dtype=self.dtype)
        if A.dim() == 2:                                   # one tensor for all masked
            A = A.unsqueeze(0).expand(int(mask.sum()), 3, 3)
        if A.shape[0] == int(mask.sum()):                  # only the masked rows given
            full = self.A0.clone()
            full[mask] = A
            A = full
        self.A0 = torch.where(mask.view(-1, 1, 1), A, self.A0)
        return self

    def set_normal_isolation(self, mask, normals):
        """Convenience wrapper: isolate the normal traction via A0 = n (x) n.

        Equivalent to set_structural_tensor(mask, n n^T) with n normalised.
        `normals` may be (N,3) or (M,3) for the M masked rows."""
        mask = torch.as_tensor(mask, device=self.device).bool()
        nrm = torch.as_tensor(normals, device=self.device, dtype=self.dtype)
        if nrm.shape[0] == int(mask.sum()):
            full = torch.zeros(self.n, 3, device=self.device, dtype=self.dtype)
            full[mask] = nrm
            nrm = full
        nrm = nrm / (nrm.norm(dim=1, keepdim=True) + 1e-12)
        nn = nrm.unsqueeze(2) * nrm.unsqueeze(1)
        return self.set_structural_tensor(mask, nn[mask])

    # ---------------------------------------------------------------- kernels
    def _weights(self, x):
        """Quadratic B-spline weights, derivatives placeholders, base node.
        Returns base (N,3 int), w (N,3,3) per-axis weights, w2 (N,3,3) 2nd deriv
        (wrt fx), and fx (N,3)."""
        Xp = x * self.inv_dx
        base = torch.floor(Xp - 0.5).to(torch.long)          # (N,3)
        fx = Xp - base.to(self.dtype)                         # (N,3) in [0.5,1.5]
        w = torch.stack([
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ], dim=1)                                             # (N,3nodes,3dims)
        w2 = torch.tensor(_D2, device=self.device, dtype=self.dtype)  # (3,)
        return base, fx, w, w2

    def _node_index(self, nodes):
        """Flatten (M,3) integer node coords to linear index, clamped to grid."""
        ng = self.n_grid
        nc = nodes.clamp(0, ng - 1)
        return (nc[:, 0] * ng + nc[:, 1]) * ng + nc[:, 2]

    # ------------------------------------------------------------------- step
    def substep(self, time: float = 0.0):
        dev, dt = self.device, self.dtype
        ng = self.n_grid
        ngrid3 = ng * ng * ng
        inv_dx = self.inv_dx
        dtt = self.dt

        # particle velocity boundary conditions (e.g. grippers dragging skin)
        for bc in self.particle_bc:
            bc(self, time)

        # ---- constitutive: polar decomposition, stress, damage driving ------
        # Analytic polar via the symmetric eig of C = F^T F (no SVD -> MPS-safe).
        # F = R U with U = V diag(S) V^T.  We must get R *exactly orthogonal* for
        # ANY stretch (the SVD's U@Vh was orthogonal for free).  Key fact: the
        # columns of (F V) are F v_i, which are mutually orthogonal with norms S_i
        # (since (F v_i).(F v_j) = v_i^T C v_j = S_i^2 delta_ij).  So normalising
        # each column of F V by its OWN norm yields an orthonormal frame, and
        # R = normalize_cols(F V) @ V^T is orthogonal regardless of S magnitude.
        # (Dividing by a *clamped* S instead leaves R non-orthogonal when the
        #  element stretches past the clamp -> stress blow-up.  Do not do that.)
        F = self.F
        C = F.transpose(1, 2) @ F                             # right Cauchy-Green (SPD)
        lamC, V = sym_eig(C)                                  # ascending eigenvalues
        S = torch.sqrt(torch.clamp(lamC, min=1e-12))          # singular values = |F v_i|
        Vt = V.transpose(1, 2)
        W = F @ V                                             # columns F v_i (orthogonal)
        What = W / (W.norm(dim=1, keepdim=True) + 1e-12)      # orthonormalise columns
        R = What @ Vt                                         # exactly orthogonal polar rotation
        if self.f_clamp is not None:                          # strain-limit (stability)
            S = S.clamp(self.f_clamp[0], self.f_clamp[1])
            F = R @ (V @ torch.diag_embed(S) @ Vt)            # clamp stretch only (R untouched)
            self.F = F
        J = (S[:, 0] * S[:, 1] * S[:, 2]).clamp(min=1e-6)
        mu = self.mu.view(-1, 1, 1)
        lam = self.lam.view(-1, 1, 1)
        I3 = torch.eye(3, device=dev, dtype=dt).expand_as(F)

        # Kirchhoff stress, fixed corotated:  tau = 2mu (F-R) F^T + lam (J-1) J I
        tau_dev = 2.0 * mu * (F - R) @ F.transpose(1, 2)
        tau_vol = (lam[:, 0, 0] * (J - 1.0) * J).view(-1, 1, 1) * I3

        # --- damage update (explicit), uses *undegraded* Cauchy stress -------
        do_damage = self.allow.any() and (self._step % self.damage_every == 0)
        if do_damage:
            dt_dmg = dtt * self.damage_every          # lumped rate over skipped steps
            tau_full = tau_dev + tau_vol
            sigma = tau_full / J.view(-1, 1, 1)               # Cauchy
            sigma = 0.5 * (sigma + sigma.transpose(1, 2))     # symmetrize
            evals, evecs = sym_eig(sigma)                     # ascending
            evals_p = evals.clamp(min=0.0)                    # tensile part
            sigma_p = evecs @ torch.diag_embed(evals_p) @ evecs.transpose(1, 2)

            # current structural tensor: co-rotate the reference A0 by the polar
            # rotation R.  A0 = I + alpha a0 a0^T reproduces the fiber recipe, while
            # set_structural_tensor() can have replaced A0 with any tensor (e.g. the
            # rank-1 projector r r^T that isolates the normal traction).
            A = R @ self.A0 @ R.transpose(1, 2)               # (N,3,3)

            M = A @ sigma_p
            phi = (M @ M).diagonal(dim1=1, dim2=2).sum(1) / (self.sigma_c ** 2)
            self.phi = phi                                    # expose for diagnostics
            dTilde = self.zeta * torch.relu(phi - 1.0)
            resist = self.d - (self.l0 ** 2) * self.lap
            ddot = torch.relu((1.0 - self.d) * dTilde - resist)
            dnew = self.d + (dt_dmg / self.eta) * ddot
            dnew = torch.clamp(dnew, min=0.0, max=1.0)
            self.d = torch.where(self.allow, torch.maximum(dnew, self.d), self.d)

        # degradation g(d): soften deviatoric & tensile-volumetric stress only
        g = (1.0 - self.d) ** 2 * (1.0 - self.residual) + self.residual   # (N,)
        gv = g.view(-1, 1, 1)
        vol_deg = torch.where((J >= 1.0).view(-1, 1, 1), gv, torch.ones_like(gv))
        tau = gv * tau_dev + vol_deg * tau_vol

        # ---- P2G -------------------------------------------------------------
        base, fx, w, w2 = self._weights(self.x)
        grid_mv = torch.zeros(ngrid3, 3, device=dev, dtype=dt)
        grid_m = torch.zeros(ngrid3, device=dev, dtype=dt)
        grid_d = torch.zeros(ngrid3, device=dev, dtype=dt)
        grid_dw = torch.zeros(ngrid3, device=dev, dtype=dt)

        stress_coeff = (-dtt * self.vol0 * 4.0 * inv_dx * inv_dx).view(-1, 1, 1)
        affine = stress_coeff * tau + self.mass.view(-1, 1, 1) * self.C   # (N,3,3)
        mv = self.mass.view(-1, 1) * self.v                                # (N,3)

        for k in range(27):
            ox, oy, oz = self.offsets[k]
            nodes = base + self.offsets[k]
            idx = self._node_index(nodes)
            wk = w[:, ox, 0] * w[:, oy, 1] * w[:, oz, 2]                    # (N,)
            dpos = (self.offsets[k].to(dt) - fx) * self.dx                  # (N,3)
            contrib = wk.unsqueeze(1) * (mv + (affine @ dpos.unsqueeze(-1)).squeeze(-1))
            grid_mv.index_add_(0, idx, contrib)
            grid_m.index_add_(0, idx, wk * self.mass)
            grid_d.index_add_(0, idx, wk * self.d)
            grid_dw.index_add_(0, idx, wk)

        # ---- grid update -----------------------------------------------------
        mask = grid_m > 1e-12
        inv_m = torch.zeros_like(grid_m)
        inv_m[mask] = 1.0 / grid_m[mask]
        grid_v = grid_mv * inv_m.unsqueeze(1)
        grid_v = grid_v + dtt * self.gravity.unsqueeze(0)          # gravity
        if self.grid_damp != 1.0:
            grid_v = grid_v * self.grid_damp
        grid_v[~mask] = 0.0

        # normalize damage field on grid (for Laplacian)
        gd = torch.zeros_like(grid_d)
        dwm = grid_dw > 1e-12
        gd[dwm] = grid_d[dwm] / grid_dw[dwm]

        # colliders act on grid velocity (positions of active nodes)
        if self.colliders:
            node_lin = torch.nonzero(mask, as_tuple=False).squeeze(1)
            ix = node_lin // (ng * ng)
            iy = (node_lin // ng) % ng
            iz = node_lin % ng
            pos = torch.stack([ix, iy, iz], 1).to(dt) * self.dx
            vsel = grid_v[node_lin]
            for col in self.colliders:
                vsel = col(pos, vsel, time)
            grid_v[node_lin] = vsel

        # ---- G2P + Laplacian -------------------------------------------------
        new_v = torch.zeros_like(self.v)
        new_C = torch.zeros_like(self.C)
        lap = torch.zeros(self.n, device=dev, dtype=dt)
        for k in range(27):
            ox, oy, oz = self.offsets[k]
            nodes = base + self.offsets[k]
            idx = self._node_index(nodes)
            wk = w[:, ox, 0] * w[:, oy, 1] * w[:, oz, 2]
            gvk = grid_v[idx]                                     # (N,3)
            dpos = (self.offsets[k].to(dt) - fx) * self.dx
            new_v = new_v + wk.unsqueeze(1) * gvk
            new_C = new_C + wk.unsqueeze(1).unsqueeze(2) * (gvk.unsqueeze(2) * dpos.unsqueeze(1))
            # Laplacian weight = inv_dx^2 ( d2x wy wz + wx d2y wz + wx wy d2z )
            lapw = inv_dx * inv_dx * (
                w2[ox] * w[:, oy, 1] * w[:, oz, 2]
                + w[:, ox, 0] * w2[oy] * w[:, oz, 2]
                + w[:, ox, 0] * w[:, oy, 1] * w2[oz]
            )
            lap = lap + lapw * gd[idx]

        new_C = new_C * (4.0 * inv_dx * inv_dx)
        self.v = new_v
        self.C = new_C
        self.lap = lap
        self.x = self.x + dtt * self.v
        self.F = (I3 + dtt * self.C) @ self.F

        # keep particles inside the domain (soft clamp)
        pad = 1.5 * self.dx
        self.x = self.x.clamp(pad, self.grid_lim - pad)
        self._step += 1

    def run_frame(self, n_sub: int, time0: float = 0.0):
        for s in range(n_sub):
            self.substep(time0 + s * self.dt)


# --------------------------------------------------------------------- helpers
def _resolve(vsel, vcol, n, mode, friction):
    """Resolve grid velocities against a collider whose outward normal (toward
    the material side) is `n` and whose velocity is `vcol`.  Removes the inward
    (penetrating) normal component; sticky pins to the collider velocity."""
    if mode == "sticky":
        return vcol.expand_as(vsel)
    vrel = vsel - vcol
    vn = (vrel * n).sum(1, keepdim=True)          # along outward normal
    inward = vn.clamp(max=0.0)                    # negative => penetrating
    vtan = vrel - vn * n
    if friction > 0.0:
        vtmag = vtan.norm(dim=1, keepdim=True)
        scale = (1.0 + friction * inward / (vtmag + 1e-9)).clamp(min=0.0)
        vtan = vtan * scale
    return vcol + vtan + (vn - inward) * n        # keep outward normal + (damped) tangential


def sphere_collider(center_fn, radius, mode="sticky", friction=0.0):
    """Grid collider for a (possibly moving) sphere; outward normal is radial."""
    def col(pos, v, time):
        c = torch.as_tensor(center_fn(time), device=pos.device, dtype=pos.dtype)
        rel = pos - c
        dist = rel.norm(dim=1)
        inside = dist < radius
        if not inside.any():
            return v
        dt_fd = 1e-4
        c2 = torch.as_tensor(center_fn(time + dt_fd), device=pos.device, dtype=pos.dtype)
        vcol = (c2 - c) / dt_fd
        n = rel[inside] / (dist[inside].unsqueeze(1) + 1e-9)
        vv = v.clone()
        vv[inside] = _resolve(v[inside], vcol, n, mode, friction)
        return vv
    return col


def halfspace_collider(origin_fn, normal, mode="slip", friction=0.0):
    """Half-space collider.  Material occupies the side the outward `normal`
    points toward; the solid obstacle is on the opposite side.  A grid node is
    'inside' (penetrated) when it lies on the obstacle side, i.e. (pos-o).n<0."""
    nrm = torch.tensor(normal, dtype=torch.float64)
    nrm = (nrm / nrm.norm()).tolist()

    def col(pos, v, time):
        n = torch.as_tensor(nrm, device=pos.device, dtype=pos.dtype)
        o = torch.as_tensor(origin_fn(time), device=pos.device, dtype=pos.dtype)
        sd = ((pos - o) * n).sum(1)
        inside = sd < 0.0
        if not inside.any():
            return v
        dt_fd = 1e-4
        o2 = torch.as_tensor(origin_fn(time + dt_fd), device=pos.device, dtype=pos.dtype)
        vcol = (o2 - o) / dt_fd
        vv = v.clone()
        vv[inside] = _resolve(v[inside], vcol, n.unsqueeze(0), mode, friction)
        return vv
    return col


# --------------------------------------------------------------------- LA self-test
def _la_selftest(device="cpu", n=20000, seed=0):
    """Validate the analytic linear algebra against torch's LAPACK reference.

    Run me on your machine to confirm the MPS / CPU path is correct:
        python3 src/anisompm.py            # cpu
        python3 src/anisompm.py mps        # Apple GPU
    Reference (eigh/svd) is always computed on CPU in float64; the analytic
    sym_eig runs on the requested device/dtype."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    # symmetric eig: reconstruction + comparison to eigh
    B = torch.randn(n, 3, 3, dtype=torch.float64)
    Asym = 0.5 * (B + B.transpose(1, 2))
    # include nasty cases: pure shear (degenerate singular values) + isotropic
    Asym[0] = torch.tensor([[0., 2, 0], [2, 0, 0], [0, 0, 0]], dtype=torch.float64)
    Asym[1] = 3.0 * torch.eye(3, dtype=torch.float64)
    ev, Q = sym_eig(Asym.to(dev, torch.float32))
    ev, Q = ev.cpu().double(), Q.cpu().double()           # .cpu() first: MPS has no float64
    recon = (Q @ torch.diag_embed(ev) @ Q.transpose(1, 2) - Asym).abs().amax()
    ortho = (Q.transpose(1, 2) @ Q - torch.eye(3).double()).abs().amax()
    ev_ref = torch.linalg.eigvalsh(Asym)
    ev_err = (ev.sort(1).values - ev_ref).abs().amax()
    # polar decomposition: same column-normalised formula as substep.  Tested on
    # a hard mix incl. near-singular random F -- R must be orthogonal regardless
    # of stretch (this is the property the clamped-1/S version lost).
    torch.manual_seed(seed + 1)
    Q1 = torch.linalg.qr(torch.randn(n, 3, 3, dtype=torch.float64)).Q
    sv = 0.2 + 7.8 * torch.rand(n, 3, dtype=torch.float64)    # wide stretch range [0.2, 8]
    F = Q1 @ torch.diag_embed(sv) @ torch.linalg.qr(torch.randn(n, 3, 3, dtype=torch.float64)).Q.transpose(1, 2)
    F = F[torch.linalg.det(F) > 0]
    C = (F.transpose(1, 2) @ F).to(dev, torch.float32)
    lam, Vv = sym_eig(C); lam, Vv = lam.cpu().double(), Vv.cpu().double()
    Vt = Vv.transpose(1, 2)
    W = F @ Vv
    Rp = (W / (W.norm(dim=1, keepdim=True) + 1e-12)) @ Vt      # column-normalised polar R
    r_ortho = (Rp.transpose(1, 2) @ Rp - torch.eye(3).double()).abs().amax()
    r_svd = (Rp - (lambda u, s, vh: u @ vh)(*torch.linalg.svd(F))).abs().amax()
    print(f"[LA self-test on {device}, float32 analytic vs float64 LAPACK]")
    print(f"  sym_eig  reconstruction  max|Δ| = {float(recon):.2e}")
    print(f"  sym_eig  orthonormality  max|Δ| = {float(ortho):.2e}")
    print(f"  sym_eig  eigenvalue vs eigh     = {float(ev_err):.2e}")
    print(f"  polar R  orthogonality   max|Δ| = {float(r_ortho):.2e}")
    print(f"  polar R  vs SVD U@Vh     max|Δ| = {float(r_svd):.2e}")
    ok = max(float(recon), float(ortho), float(ev_err), float(r_ortho), float(r_svd)) < 1e-3
    print("  RESULT:", "PASS" if ok else "CHECK (float32 tolerance ~1e-3)")
    return ok


if __name__ == "__main__":
    import sys
    _la_selftest(device=sys.argv[1] if len(sys.argv) > 1 else "cpu")
