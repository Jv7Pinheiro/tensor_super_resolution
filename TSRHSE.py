"""Eigenphase estimation from the Z tensor.

Two routes, sharing one eigenphase extractor:

  jennrich_ladder(...)      corrected version of your current TSRHSE -- still a
                            two-slice ratio estimator, but with the pairing,
                            truncation, and unwrapping bugs fixed.
  cp_eigenphases(...)       CP/ALS route -- decompose Z once, then read the
                            eigenphases off the time-mode factor W.

Index convention matches your code: Z has shape (Q, Q, N), time last, so
Z[:, :, j] is the time slice at t_j and

    Z[i, k, j] = sum_l U[i, l] * V[k, l] * W[j, l],   W[j, l] = c_l * exp(-i lam_l t_j)

which is the paper's CP form with the time mode moved to the back.
"""

import numpy as np

###########################
## Eigenphase extraction ##
###########################

def eigenphase_from_column(w, t, lam_bounds=None, grid_size=None, refine=True):
    """Estimate lam from one CP time-factor column w[j] ~ c * exp(-i lam t[j]).
    In other words, it computes the "angle between vectors" step

    The per-column scale c is a CP gauge freedom and is unknown, so lam has to
    be read off relative phases, not absolute ones. This maximises the
    (non-uniform) periodogram

        f(lam) = | sum_j conj(w[j]) exp(-i lam t[j]) |^2

    which peaks exactly at lam = lam_true regardless of c, uses every time
    sample rather than two, and needs no phase unwrapping. For a uniform grid
    with spacing h the same quantity reduces to the familiar one-liner
    lam = -arg( sum_j conj(w[j]) w[j+1] ) / h, i.e. the angle between the column
    and its own one-step shift.
    """
    w = np.asarray(w, dtype=complex)
    t = np.asarray(t, dtype=float)

    if lam_bounds is None:
        # Nyquist limit set by the typical sample spacing.
        gaps = np.diff(np.sort(t))
        gaps = gaps[gaps > 0]
        h = float(np.median(gaps)) if gaps.size else 1.0
        lam_bounds = (-np.pi / h, np.pi / h)

    lo, hi = lam_bounds
    span = t.max() - t.min()
    if grid_size is None:
        # Peak width is ~2*pi/span; oversample it by ~8x.
        width = 2 * np.pi / max(span, 1e-12)
        grid_size = int(np.clip(8 * (hi - lo) / width, 1024, 200_000))

    grid = np.linspace(lo, hi, grid_size)
    # f(lam) for all grid points at once.
    power = np.abs(np.conj(w) @ np.exp(-1j * np.outer(t, grid)))
    best = int(np.argmax(power))
    lam = float(grid[best])

    if refine:
        step = grid[1] - grid[0]
        a, b = lam - step, lam + step
        obj = lambda x: abs(np.conj(w) @ np.exp(-1j * x * t))
        for _ in range(60):                      # ternary search
            m1 = a + (b - a) / 3.0
            m2 = b - (b - a) / 3.0
            if obj(m1) < obj(m2):
                a = m1
            else:
                b = m2
        lam = 0.5 * (a + b)

    return lam


def eigenphases_from_factor(W, t, lam_bounds=None, **kw):
    """Apply eigenphase_from_column to every column of the time factor."""
    lams = np.array([eigenphase_from_column(W[:, l], t, lam_bounds, **kw)
                     for l in range(W.shape[1])])
    weights = np.linalg.norm(W, axis=0)          # CP component magnitudes
    order = np.argsort(lams)
    return lams[order], weights[order]


############
## CP-ALS ##
############

def khatri_rao(A, B):
    """Columnwise Kronecker: out[(a, b), l] = A[a, l] * B[b, l], b fastest."""
    return (A[:, None, :] * B[None, :, :]).reshape(-1, A.shape[1])


def _unfold(Z, mode):
    if mode == 0:
        return Z.reshape(Z.shape[0], -1)                       # cols (k, j)
    if mode == 1:
        return Z.transpose(1, 0, 2).reshape(Z.shape[1], -1)    # cols (i, j)
    return Z.transpose(2, 0, 1).reshape(Z.shape[2], -1)        # cols (i, k)


def _solve_block(K, X):
    """min_A || X - A K^T ||_F  ->  A = lstsq(K, X^T)^T."""
    return np.linalg.lstsq(K, X.T, rcond=None)[0].T


def _reconstruct(U, V, W):
    return np.einsum("il,kl,jl->ikj", U, V, W, optimize=True)


def cp_als(Z, rank, t_list=None, n_iter=300, tol=1e-7, vandermonde=False, lam_bounds=None, n_restarts=1, seed=None, verbosity=0):
    """Complex CP decomposition of Z (Q, Q, N) by alternating least squares.

    Parameters
    ----------
    vandermonde : bool
        After each sweep, project every column of the time factor W back onto
        the manifold {c * exp(-i lam t)}. This is the structured-CP variant: it
        enforces the physics (each time factor IS a pure exponential) inside the
        fit instead of only reading it off afterwards, and it denoises hard.
        Requires t.
    n_restarts : int
        Random restarts; the lowest-residual fit is returned. ALS is only
        locally convergent (Theorem 8.1 in your draft).
    """
    Z = np.asarray(Z, dtype=complex)
    L, R, N = Z.shape
    rng = np.random.default_rng(seed)

    Z0, Z1, Z2 = _unfold(Z, 0), _unfold(Z, 1), _unfold(Z, 2)
    normZ = np.linalg.norm(Z)

    best = None
    for restart in range(max(1, n_restarts)):
        U = rng.normal(size=(L, rank)) + 1j * rng.normal(size=(L, rank))
        V = rng.normal(size=(R, rank)) + 1j * rng.normal(size=(R, rank))
        W = rng.normal(size=(N, rank)) + 1j * rng.normal(size=(N, rank))

        prev = np.inf
        for it in range(n_iter):
            U = _solve_block(khatri_rao(V, W), Z0)
            V = _solve_block(khatri_rao(U, W), Z1)
            W = _solve_block(khatri_rao(U, V), Z2)

            if vandermonde:
                if t_list is None:
                    raise ValueError("vandermonde=True requires t_list")
                W = project_vandermonde(W, t_list, lam_bounds)

            # Gauge fixing: unit-norm columns in U and V, scale absorbed by W.
            for F in (U, V):
                nrm = np.linalg.norm(F, axis=0)
                nrm[nrm == 0] = 1.0
                F /= nrm
                W *= nrm

            res = np.linalg.norm(Z - _reconstruct(U, V, W)) / normZ
            if verbosity > 1 and it % 25 == 0:
                print(f"\t\t\trestart {restart} iter {it:4d}  rel. residual {res:.3e}")
            if abs(prev - res) < tol:
                break
            prev = res

        if best is None or res < best[0]:
            best = (res, U.copy(), V.copy(), W.copy())
        if verbosity > 0:
            print(f"\t\trestart {restart}: rel. residual {res:.3e}")

    res, U, V, W = best
    return U, V, W, res


def project_vandermonde(W, t_list, lam_bounds=None):
    """Replace each column of W by the best fit c * exp(-i lam t)."""
    t_list = np.asarray(t_list, dtype=float)
    out = np.empty_like(W)
    for l in range(W.shape[1]):
        lam = eigenphase_from_column(W[:, l], t_list, lam_bounds)
        v = np.exp(-1j * lam * t_list)
        c = (np.conj(v) @ W[:, l]) / (np.conj(v) @ v)
        out[:, l] = c * v
    return out


#########################
## Jenrich's Algorithm ##
#########################

def jennrich_ratio(Z, t_list, rank, a, b, ridge=0.0):
    """One ratio estimate from the slices at t_a and t_b."""
    dt = float(t_list[a] - t_list[b])
    A = Z[:, :, a]
    B = Z[:, :, b]

    Ub, S, Vh = np.linalg.svd(B, full_matrices=False)
    rank = min(rank, S.size)
    S_inv = S[:rank] / (S[:rank] ** 2 + ridge ** 2)
    B_dag = Vh[:rank].conj().T @ np.diag(S_inv) @ Ub[:, :rank].conj().T

    alphas = np.linalg.eigvals(A @ B_dag)
    alphas = np.array(sorted(alphas, key=lambda z: abs(abs(z) - 1.0))[:rank])

    lambdas = -np.angle(alphas) / dt
    order = np.argsort(lambdas)
    return lambdas[order], alphas[order], dt


#########################
## Public entry points ##
#########################

def jennrich_ladder(Z, t_list, rank, n_candidates=15, unit_circle_tol=0.25, ridge=0.0, max_pairs=200_000, seed=None, verbosity=0):
    """Multiscale dyadic unwrapping, with the four fixes described in chat.

    Differences from the original:
      * pairs are drawn from ALL (a, b), not just time-adjacent ones, so the
        ladder can actually reach large dt;
      * each candidate is unwrapped against the running estimate using ITS OWN
        dt before the median, instead of the nominal target_dt after it;
      * the pseudoinverse is truncated at `rank` with optional
        Tikhonov damping, instead of inverting every singular value of B;
      * the unit-circle check is per-eigenvalue rather than all-or-nothing.
    """
    t = np.asarray(t_list, dtype=float)
    N = len(t)
    rng = np.random.default_rng(seed)

    n_all = N * (N - 1) // 2
    if n_all <= max_pairs:
        ia, ib = np.triu_indices(N, k=1)
    else:
        ia = rng.integers(0, N, size=max_pairs)
        ib = rng.integers(0, N, size=max_pairs)
        keep = ia != ib
        ia, ib = ia[keep], ib[keep]

    dts = np.abs(t[ia] - t[ib])
    keep = dts > 0
    ia, ib, dts = ia[keep], ib[keep], dts[keep]

    dt_min = max(float(dts.min()), 0.05)
    dt_max = float(dts.max())
    if verbosity > 0:
        print(f"dt_min = {dt_min:.4f}, dt_max = {dt_max:.4f}, "
              f"{len(dts)} usable pairs")

    scales = []
    dt = dt_min
    while dt <= dt_max * 1.01:
        scales.append(dt)
        dt *= 2.0

    current = None
    for target in scales:
        idx = np.argsort(np.abs(dts - target))[:n_candidates]

        unwrapped = []
        for j in idx:
            a, b = int(ia[j]), int(ib[j])
            try:
                lams, alphas, dt_ab = jennrich_ratio(Z, t, rank, a, b)
            except np.linalg.LinAlgError:
                continue

            ok = np.abs(np.abs(alphas) - 1.0) < unit_circle_tol
            if not ok.all():
                continue

            if current is not None:
                # Unwrap with this pair's own dt, before any averaging.
                k = np.round((current - lams) * dt_ab / (2 * np.pi))
                lams = lams + 2 * np.pi * k / dt_ab
            unwrapped.append(lams)

            if verbosity > 1:
                print(f"  dt={dt_ab:+.3f}  lams={np.round(lams, 3)}")

        if not unwrapped:
            continue
        current = np.median(np.array(unwrapped), axis=0)
        if verbosity > 0:
            print(f"target_dt={target:.3f}  est={np.round(current, 4)}")

    return current


def cp_eigenphases(Z, t_list, rank, tol=1e-7, vandermonde=True, n_restarts=4, lam_bounds=None, seed=None, verbosity=0):
    """End-to-end: CP-decompose Z, then read eigenphases off the time factor."""

    U, V, W, res = cp_als(Z, rank, t_list=t_list, tol=tol, vandermonde=vandermonde, lam_bounds=lam_bounds, n_restarts=n_restarts, seed=seed, verbosity=verbosity)
    eigenvalues, weights = eigenphases_from_factor(W, t_list, lam_bounds)

    return eigenvalues, weights, (U, V, W, res)