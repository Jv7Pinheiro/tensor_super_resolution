"""Parallel generation of the Z tensor for QPE experiments.

The module provides two backends for computing the same Z tensor:

    _compute_chunk
        Builds generalized Hadamard test circuits with Qiskit and estimates
        the real and imaginary parts of the matrix element from finite-shot
        measurements.

    _compute_chunk_numerically
        Evaluates
            Z[l, r, n] = <psi_l| exp(-i M t_n) |phi_r>
        directly using NumPy. This backend computes the tensor exactly and
        does not use shot sampling.

The task space is flattened over (l, r, t), allowing work to be distributed
across the probe-pair and time dimensions. Workers receive integer index
ranges rather than pickled NumPy slices, and t_list is shipped once per
worker through the initializer. Results are assembled into a single
(L, R, N) tensor after the workers finish their assigned chunks.

For the circuit backend, the exact ancilla marginal is obtained directly
from the statevector and the shot noise is drawn with NumPy when
fast_sampling=True. This avoids constructing a counts dictionary for every
circuit. Set fast_sampling=False to use Qiskit's Statevector.sample_counts()
path instead.

For the numeric backend, M is assumed to be the Hamiltonian and the
propagator is exp(-i M t). Probe vectors are stored as COLUMNS of U_list
and V_list, matching U_list[:, l] and V_list[:, r] in the workers.

The is_unitary flag determines whether M is treated as unitary or
Hermitian. The circuit backend forwards this flag to
algorithms.GeneralizedHadamardTest. The numeric backend uses it to select
the appropriate spectral decomposition and validates M accordingly.

compare_backends() provides a direct check that the numerical backend
agrees with the exact amplitudes produced by the circuit backend.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import qiskit as qk

import algorithms


#################
## Worker Side ##
#################

_worker_data = {}

def _initialize_worker(M, U_list, V_list, t_list, is_unitary, shots, fast_sampling, seed, method):
    """Runs once per worker process. Heavy objects are pickled once, here."""
    _worker_data["M"] = M
    _worker_data["U_list"] = U_list
    _worker_data["V_list"] = V_list
    _worker_data["t_list"] = t_list
    _worker_data["is_unitary"] = is_unitary
    _worker_data["shots"] = shots
    _worker_data["fast_sampling"] = fast_sampling
    _worker_data["seed"] = seed
    _worker_data["rng"] = np.random.default_rng() # Fallback stream, used only when seed is None (non-reproducible mode).

    if method == "numeric":
        _precompute_numeric(M, U_list, V_list, t_list, is_unitary)


########################
## Scheduling Helpers ##
########################

def _split_bounds(n, k):
    """
    Split range(n) into k contiguous, near-equal (start, stop) pairs.
    Where k is the number of workers.
    """
    k = max(1, min(int(k), n))
    base, extra = divmod(n, k)
    bounds = []
    start = 0
    for i in range(k):
        stop = start + base + (1 if i < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def _build_tasks(L, R, N, workers, tasks_per_worker, n_splits=None):
    """Flat task list covering every (l, r, t-slice).

    If L * R is already large compared to the core count, each (l, r) pair is a
    single task and t is not split at all -- that gives good balance with the
    least IPC. If L * R is small (e.g. 4x4 on 64 cores), t gets split enough to
    keep every core busy.

    Pass n_splits to pin the t-axis split count instead, which makes the task
    boundaries independent of the worker count (see the seed note in
    generate_multiple_Z_tensors).
    """
    if n_splits is None:
        pairs = L * R
        # Below we are basically calculating ceil(desired number of tasks / number of (L, R) pairs)
        target = max(1, int(workers) * int(tasks_per_worker))
        n_splits = max(1, -(-target // pairs))      # ceil division
    n_splits = min(max(1, int(n_splits)), N)

    bounds = _split_bounds(N, n_splits)

    tasks = []
    for l in range(L):
        for r in range(R):
            for start, stop in bounds:
                if stop > start:
                    tasks.append((l, r, start, stop))
    return tasks


def _limit_blas_threads():
    """Stop each worker's BLAS from spawning its own thread pool.

    With 64 processes each defaulting to 64 BLAS threads you get ~4096 threads
    fighting over 64 cores. Set in the parent; spawned children inherit it.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


##############
## Checkers ##
##############

def _check_hermitian(M, tol=1e-8):
    asym = np.max(np.abs(M - M.conj().T))
    if asym > tol * max(1.0, float(np.max(np.abs(M)))):
        raise ValueError(
            f"M is not Hermitian (max |M - M^dagger| = {asym:.3e}).\nPass is_unitary=True if M is meant to be unitary instead.")


def _check_unitary(M, tol=1e-8):
    d = M.shape[0]
    resid = np.max(np.abs(M @ M.conj().T - np.eye(d)))
    if resid > tol * max(1.0, float(np.max(np.abs(M)))):
        raise ValueError(
            f"M is not unitary (max |M M^dagger - I| = {resid:.3e}).\nPass is_unitary=False if M is meant to be Hermitian instead.")


#####################
## Numeric Methods ##
#####################

def _precompute_numeric(M, U_list, V_list, t_list, is_unitary):
    """Diagonalise M once per worker and cache the pieces every entry reuses.

    Single flag, matching your is_matrix_unitary() convention: M is assumed to
    be either unitary or Hermitian, never both, never neither, so is_unitary
    alone decides the branch (the old separate `hermitian` flag is gone --
    it's just `not is_unitary` now, and derived internally so the two can't
    drift out of sync again).

    With M = W diag(lam) W^-1,

        <psi_l| exp(-i M t_n) |phi_r> = sum_k (U^dagger W)[l, k] * exp(-i lam_k t_n) * (W^-1 V)[k, r]

    Both Hermitian and unitary matrices are normal, so in both cases W can be
    taken unitary and the expensive/ill-conditioned W^-1 replaced by the cheap,
    stable W^dagger:
      - Hermitian (is_unitary=False): eigh gives W unitary directly, lam real.
      - Unitary (is_unitary=True): a Schur decomposition of a normal matrix is
        exactly diagonal, so its unitary factor plays the same role eigh's W
        plays for the Hermitian case; lam sits on the unit circle instead of
        the real line. Falls back to eig + solve (still correct, just less
        stable, and only exactly right for non-degenerate eigenvalues) if
        scipy is not importable.
    """
    if is_unitary:
        _check_unitary(M)
        try:
            from scipy.linalg import schur
            T, W = schur(M, output="complex")      # exact diagonal: M normal
            evals = np.diag(T).astype(complex)
            B = W.conj().T @ V_list                # W guaranteed unitary
        except ImportError:
            evals, W = np.linalg.eig(M)
            B = np.linalg.solve(W, V_list)         # W not guaranteed unitary
    else:
        _check_hermitian(M)
        evals, W = np.linalg.eigh(M)
        evals = evals.astype(complex)
        B = W.conj().T @ V_list                    # W^-1 == W^dagger

    A = U_list.conj().T @ W                        # (L, d)

    # (d, N) phase table, shared by every (l, r). 
    # Small d is the Hilbert dimension, so for example 16 x N complex for a 16x16 Hamiltonian.
    phases = np.exp(-1j * np.outer(evals, t_list))

    _worker_data["num_A"] = np.ascontiguousarray(A)
    _worker_data["num_B"] = np.ascontiguousarray(B)
    _worker_data["num_phases"] = np.ascontiguousarray(phases)


def _amplitudes_numerically(l, r, start, stop):
    """Exact <psi_l| exp(-i M t_n) |phi_r> for n in [start, stop)."""
    A = _worker_data["num_A"]               # (L, d)
    B = _worker_data["num_B"]               # (d, R)
    phases = _worker_data["num_phases"]     # (d, N)

    # Fold the two probe-dependent factors together first: one length-d vector.
    coeff = A[l, :] * B[:, r]
    return coeff @ phases[:, start:stop]    # (stop - start,)


def _compute_chunk_numerically(task):
    """Evaluate Z[l, r, start:stop] exactly, matching Z_{i,k}(t) = <u_i| exp(-i H t) |v_k> directly."""
    l, r, start, stop = task
 
    Z_chunk = _amplitudes_numerically(l, r, start, stop)
 
    return l, r, start, Z_chunk


#####################
## Circuit Methods ##
#####################

def _p0_from_statevector(data):
    """Exact P(ancilla qubit 0 == '0') for a qiskit Statevector's raw array.

    Qiskit indexes statevector amplitudes little-endian, so qubit 0 is the least
    significant bit of the array index: every even index has qubit 0 in |0>.
    Summing |amp|^2 over those indices is exactly the marginal that
    sample_counts(shots, [0]) samples from.
    """
    amps = data[0::2]
    p0 = float(np.vdot(amps, amps).real)
    # Guard against tiny negative / >1 values from floating point.
    return min(1.0, max(0.0, p0))


def _run_circuit_probability(M, U, V, t, img, is_unitary):
    qc = algorithms.GeneralizedHadamardTest(M, U, V, t, img=img, is_unitary=is_unitary)
    state = qk.quantum_info.Statevector.from_instruction(qc)
    return state, _p0_from_statevector(np.asarray(state.data))


def _task_rng(l, r, start):
    """Stream derived from the task coordinates, so results are reproducible
    regardless of how tasks get scheduled."""
    seed = _worker_data["seed"]
    if seed is None:
        return _worker_data["rng"]
    return np.random.default_rng([seed, l, r, start])


def _compute_chunk(task):
    """Evaluate Z[l, r, start:stop] for one (l, r) pair and one slice of t.
 
    Guarded by _worker_data["want"] ("single", "poly", or "both"): the circuit
    simulation (building the Hadamard test and reading off p_real/p_img) is
    unavoidable either way, since single and poly are two different samplings
    of the SAME probability -- but the sampling step itself only runs for the
    tensor(s) actually requested, so with want="poly" the code never draws the
    1-shot Bernoulli (fast path) or calls sample_counts(1, ...) (slow path).
    """
    l, r, start, stop = task
 
    M = _worker_data["M"]
    U = _worker_data["U_list"][:, l]
    V = _worker_data["V_list"][:, r]
    t_list = _worker_data["t_list"]
    is_unitary = _worker_data["is_unitary"]
    shots = _worker_data["shots"]
    fast_sampling = _worker_data["fast_sampling"]
 
    rng = _task_rng(l, r, start)
    n = stop - start
 
    if fast_sampling:
        p_real = np.empty(n, dtype=float)
        p_img = np.empty(n, dtype=float)
 
        for i, t in enumerate(t_list[start:stop]):
            _, p_real[i] = _run_circuit_probability(M, U, V, t, False, is_unitary)
            _, p_img[i] = _run_circuit_probability(M, U, V, t, True, is_unitary)
 
        real = 2.0 * (rng.binomial(shots, p_real) / shots) - 1.0
        img = 2.0 * (rng.binomial(shots, p_img) / shots) - 1.0
        Z_chunk = real + 1j * img
 
    else:
        # Original code path, kept verbatim in spirit for A/B validation.
        real = np.empty(n, dtype=float)
        img = np.empty(n, dtype=float)
 
        for i, t in enumerate(t_list[start:stop]):
            state_re, _ = _run_circuit_probability(M, U, V, t, False, is_unitary)
            real[i] = 2 * (state_re.sample_counts(shots, [0]).get("0", 0) / shots) - 1
 
            state_im, _ = _run_circuit_probability(M, U, V, t, True, is_unitary)
            img[i] = 2 * (state_im.sample_counts(shots, [0]).get("0", 0) / shots) - 1
 
        Z_chunk = (real + 1j * img)
 
    return l, r, start, Z_chunk


########################
## Public Entry Point ##
########################
 
def generate_Z_tensor(M, N, U_list, V_list, L, R, t_list, is_unitary=True, shots=1500, workers=4, tasks_per_worker=4, n_splits=None, fast_sampling=True, limit_blas_threads=True, seed=None, progress=False, method="circuit"):
    """Build and return a single Z tensor of shape (L, R, N).

    The tensor contains

        Z[l, r, n] = <psi_l| exp(-i M t_n) |phi_r>,

    with the circuit backend estimating the matrix elements using finite-shot
    generalized Hadamard tests and the numeric backend evaluating them exactly.

    Parameters
    ----------
    M : array-like
        Matrix used by the generalized Hadamard test or, for method="numeric",
        the Hamiltonian defining the propagator exp(-i M t).

    N : int
        Number of time points. Must match len(t_list).

    U_list : array-like
        Probe vectors for the left side, stored as columns with shape (d, L).

    V_list : array-like
        Probe vectors for the right side, stored as columns with shape (d, R).

    L, R : int
        Number of left and right probe vectors.

    t_list : array-like
        Time values at which the Z tensor is evaluated.

    is_unitary : bool, default=True
        Determines how M is treated. If True, M is assumed to be unitary;
        if False, M is assumed to be Hermitian. The circuit backend forwards
        this flag to GeneralizedHadamardTest, while the numeric backend uses it
        to select the corresponding spectral decomposition and validation.

    shots : int, default=1500
        Number of measurement shots used by the circuit backend. Ignored by
        the numeric backend, which computes the tensor exactly.

    workers : int, default=4
        Maximum number of worker processes.

    tasks_per_worker : int, default=4
        Target number of tasks per worker. Higher values provide finer-grained
        load balancing at the cost of additional inter-process communication.

    n_splits : int or None, default=None
        Number of chunks into which the t axis is divided. If None, the split
        count is chosen adaptively from the number of workers, tasks_per_worker,
        and the number of probe pairs.

        When seed is specified, providing an explicit n_splits fixes the task
        boundaries and therefore allows reproducibility across different worker
        counts.

    fast_sampling : bool, default=True
        Circuit backend only. If True, draw shot noise directly from the exact
        ancilla probabilities using NumPy. If False, use
        Statevector.sample_counts().

    limit_blas_threads : bool, default=True
        If True, limit each worker's BLAS thread count to one thread to avoid
        excessive thread oversubscription.

    seed : int or None, default=None
        Random seed for reproducible circuit-backend shot noise. Task-local
        random streams are keyed by (l, r, start), so results do not depend on
        the order in which tasks complete. The adaptive task boundaries can
        still depend on the worker count unless n_splits is specified.

    progress : bool, default=False
        If True, print task completion progress.

    method : {"circuit", "numeric"}, default="circuit"
        Backend used to generate the tensor. "circuit" builds generalized
        Hadamard tests and applies finite-shot sampling. "numeric" evaluates
        the matrix elements directly and exactly using the spectral
        decomposition of M.

    Returns
    -------
    Z : ndarray
        Complex array of shape (L, R, N) containing the generated Z tensor.
    """

    if method not in ("circuit", "numeric"):
        raise ValueError(f"unknown method {method!r}")

    if len(t_list) != N:
        raise ValueError(f"N ({N}) must match len(t_list) ({len(t_list)})")

    if U_list.ndim != 2 or U_list.shape[1] != L or V_list.shape[1] != R:
        raise ValueError(
            f"expected probes in columns: U_list is {U_list.shape} (want (d, {L})), "
            f"V_list is {V_list.shape} (want (d, {R}))")
 
    Z = np.zeros((L, R, N), dtype=complex) 
 
    if method == "numeric":
        # Check once, here, before spawning workers: an exception raised inside
        # the pool initializer surfaces as an opaque BrokenProcessPool instead
        # of the informative ValueError from _check_unitary/_check_hermitian.
        if is_unitary:
            _check_unitary(M)
        else:
            _check_hermitian(M)
 
    tasks = _build_tasks(L, R, N, workers, tasks_per_worker, n_splits)
    workers = max(1, min(int(workers), len(tasks)))
 
    if limit_blas_threads:
        _limit_blas_threads()
 
    # spawn is important because the worker processes import modules
    # without re-executing main.py.
    context = mp.get_context("spawn")
 
    initargs = (M, U_list, V_list, t_list, is_unitary, shots, fast_sampling, seed, method)
 
    worker_fn = _compute_chunk if method == "circuit" else _compute_chunk_numerically
 
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_initialize_worker, initargs=initargs) as executor:
        # Everything is submitted up front, so a worker that finishes early
        # immediately picks up the next (l, r, t-slice) rather than waiting for
        # its peers to finish the current (l, r).
        futures = [executor.submit(worker_fn, task) for task in tasks]
 
        done = 0
        for future in as_completed(futures):
            l, r, start, chunk = future.result()
            chunk_len = len(chunk)
            stop = start + chunk_len

            Z[l, r, start:stop] = chunk
 
            done += 1
            if progress and (done % max(1, len(tasks) // 100) == 0 or done == len(tasks)):
                pct = 100.0 * done / len(tasks)
                print(f"\rZ tensor: {pct:5.1f}%  ({done}/{len(tasks)} tasks)", end="", flush=True)
 
        if progress:
            print()
 
    return Z

def generate_G_tensor(Z, t_list, thetas, workers=4, tasks_per_worker=4, max_block_mb=128, limit_blas_threads=True, progress=False):
    """Build G of shape (L, R, J) from a Z tensor of shape (L, R, N).
 
        G[l, r, j] = (1/N) * sum_n Z[l, r, n] * exp(i * thetas[j] * t_list[n])
 
    This is Eq. 18 of the QFAMES paper, evaluated at every theta in `thetas`
    for every (l, r): the same contraction algorithms.QFAMES does internally
    (Z[l, r, :].dot(np.exp(1j * np.outer(t_list, x))) / N), done for all fibers
    at once.
 
    Parameters
    ----------
    Z : array-like, shape (L, R, N)
        Z tensor, e.g. from generate_Z_tensor.
 
    t_list : array-like, shape (N,)
        The same time samples that were used to build Z.
 
    thetas : array-like, shape (J,)
        Grid of theta values at which to evaluate G. A grid step of dx (from
        QFAMES_setup, dx = q / T_max) gives about 1 / (q) points across one
        1/T_max-wide bump, which is plenty to draw it smoothly. Only the part
        of the spectrum you want to look at needs to be covered.
 
    workers : int, default=4
        Number of worker processes. With workers=1 everything runs in this
        process (no pool is spawned) and BLAS is free to use all its threads.
 
    tasks_per_worker : int, default=4
        Target number of theta blocks per worker, for load balancing.
 
    max_block_mb : float, default=128
        Upper bound on the size of the exponential matrix each worker builds at
        once (N x block complex entries). More blocks are used automatically if
        the target number of blocks would exceed this.
 
    limit_blas_threads : bool, default=True
        Same meaning as in generate_Z_tensor; only applies when workers > 1.
 
    progress : bool, default=False
        If True, print block completion progress.
 
    Returns
    -------
    G : ndarray
        Complex array of shape (L, R, J). Use np.linalg.norm(G, axis=(0, 1))
        for ||G(theta)||_F, and np.linalg.svd(G[:, :, j]) for the singular
        values of G at theta = thetas[j].
    """
    Z = np.asarray(Z, dtype=complex)
    t_list = np.ascontiguousarray(np.asarray(t_list, dtype=float))
    thetas = np.ascontiguousarray(np.asarray(thetas, dtype=float))
 
    if Z.ndim != 3:
        raise ValueError(f"Z must have shape (L, R, N), got {Z.shape}")
    L, R, N = Z.shape
    if len(t_list) != N:
        raise ValueError(f"Z has N = {N} time points but len(t_list) = {len(t_list)}")
    if thetas.ndim != 1 or len(thetas) == 0:
        raise ValueError("thetas must be a non-empty 1-D array")
    J = len(thetas)
 
    # (l, r) fibers become the rows of one matrix; row index is l * R + r.
    Z2d = np.ascontiguousarray(Z.reshape(L * R, N))
 
    # Block the theta axis: enough blocks to keep every worker busy, and enough
    # that no block's exponential matrix (N x block complex) exceeds the budget.
    max_cols = max(1, int(max_block_mb * 2**20 // (16 * N)))
    n_blocks = max(int(workers) * int(tasks_per_worker), -(-J // max_cols))
    tasks = [(start, stop, thetas[start:stop]) for start, stop in _split_bounds(J, n_blocks) if stop > start]
 
    G2d = np.empty((L * R, J), dtype=complex)
 
    if int(workers) <= 1:
        _initialize_G_worker(Z2d, t_list)
        try:
            for done, task in enumerate(tasks, start=1):
                start, block = _compute_G_block(task)
                G2d[:, start:start + block.shape[1]] = block
                if progress:
                    print(f"\rG tensor: {100.0 * done / len(tasks):5.1f}%  ({done}/{len(tasks)} blocks)", end="", flush=True)
        finally:
            _G_data.clear()
        if progress:
            print()
        return G2d.reshape(L, R, J)
 
    workers = max(1, min(int(workers), len(tasks)))
 
    if limit_blas_threads:
        _limit_blas_threads()
 
    context = mp.get_context("spawn")
 
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_initialize_G_worker, initargs=(Z2d, t_list)) as executor:
        futures = [executor.submit(_compute_G_block, task) for task in tasks]
 
        done = 0
        for future in as_completed(futures):
            start, block = future.result()
            G2d[:, start:start + block.shape[1]] = block
 
            done += 1
            if progress and (done % max(1, len(tasks) // 100) == 0 or done == len(tasks)):
                print(f"\rG tensor: {100.0 * done / len(tasks):5.1f}%  ({done}/{len(tasks)} blocks)", end="", flush=True)
 
        if progress:
            print()
 
    return G2d.reshape(L, R, J)


################
## Validation ##
################

def compare_backends(M, U_list, V_list, t_list, n_samples=12, is_unitary=True, tol=1e-8, seed=0):
    """Check the numeric backend reproduces the circuits' exact amplitudes.

    Runs single-process, no sampling: for random (l, r, n) it reads the exact
    ancilla marginals off both Hadamard test circuits, converts them to
    Re z and Im z, and compares against the numpy contraction. Run this once
    before trusting method="numeric" -- and run it once with is_unitary=True
    specifically, since that path assumes the circuit computes exp(-i M t)
    for unitary M just as it does for Hermitian M. If GeneralizedHadamardTest
    actually does something else in that regime (e.g. controlled powers of M
    rather than continuous-time evolution), this will show up here as a
    systematic mismatch rather than noise.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t_list, dtype=float)
    U = np.asarray(U_list)
    V = np.asarray(V_list)

    _worker_data["t_list"] = t
    _precompute_numeric(M, U, V, t, is_unitary)

    worst = 0.0
    print(f"{'l':>3} {'r':>3} {'n':>5} {'Re circ':>13} {'Re numpy':>13} "
          f"{'Im circ':>13} {'Im numpy':>13}")

    for _ in range(n_samples):
        l = int(rng.integers(U.shape[1]))
        r = int(rng.integers(V.shape[1]))
        n = int(rng.integers(len(t)))

        vals = []
        for img in (False, True):
            _, p0 = _run_circuit_probability(M, U[:, l], V[:, r], t[n], img, is_unitary)
            vals.append(2.0 * p0 - 1.0)

        z = _amplitudes_numerically(l, r, n, n + 1)[0]
        worst = max(
            worst,
            abs(vals[0] - z.real), # |Re(Z_circuit) - Re(Z_numeric)|
            abs(vals[1] - z.imag)  # |Im(Z_circuit) - Im(Z_numeric)|
            )

        print(f"{l:>3} {r:>3} {n:>5} {vals[0]:>13.9f} {z.real:>13.9f}\n{vals[1]:>13.9f} {z.imag:>13.9f}")

    verdict = "PASS" if worst < tol else "FAIL -- conventions differ"
    print(f"\nworst absolute discrepancy: {worst:.3e}  ({verdict})")
    return worst


########################
## Main (for testing) ##
########################

if __name__ == "__main__":
    import hamiltonians
    import aux_functions

    Ham = hamiltonians.get_hamiltonian("belldiagonal4x4")
    M = (np.pi / (4 * np.linalg.norm(Ham))) * Ham
    is_unitary = aux_functions.is_matrix_unitary(M)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    L = Ham.shape[0]
    R = Ham.shape[1]
    U_list = eigenvectors[:, 0:L]
    V_list = eigenvectors[:, 0:R]
    t_list = aux_functions.generate_t_list(1600, 1200, 1)
    result = compare_backends(M, U_list, V_list, t_list, 100, is_unitary=is_unitary)
    print(result)