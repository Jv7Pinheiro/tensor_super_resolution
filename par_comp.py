"""Parallel generation of the Z tensor for QPE experiments.
 
Drop-in replacement for the original par_comp.py. The public signature of
generate_multiple_Z_tensors() is unchanged except for new keyword-only options,
all of which default to the old behaviour where behaviour could differ.
 
Three changes relative to the original:
 
1. The task space is flattened over (l, r, t) instead of being parallelised only
   over t inside a serial L x R double loop. There is now exactly one
   synchronisation barrier at the very end of the whole run, instead of L * R
   barriers.
 
2. Workers receive integer index ranges instead of pickled numpy slices. t_array
   is shipped once per worker in the initializer.
 
3. The measurement step computes the exact marginal probability of the ancilla
   from the statevector and draws the shot noise with numpy, instead of calling
   Statevector.sample_counts(). This is distributionally identical (see
   _p0_from_statevector) and avoids building a counts dict per circuit.
   Set fast_sampling=False to fall back to the original sample_counts() path.
"""

"""Parallel generation of the Z tensor for QPE experiments.

Two backends, same task decomposition and same output statistics:

  _compute_chunk              builds the generalized Hadamard test circuits with
                              qiskit and reads the ancilla marginal off the
                              statevector.
  _compute_chunk_numerically  evaluates Z[l, r, n] = <psi_l| exp(-i M t_n) |phi_r>
                              directly with numpy.

Select with method="circuit" (default) or method="numeric" in
generate_multiple_Z_tensors(). compare_backends() checks that the two agree
before you rely on the numeric one.

Convention note for the numeric backend: it assumes M is the Hamiltonian and the
propagator is exp(-i M t), with probe vectors stored as COLUMNS of U_list and
V_list (matching U_list[:, l] in the worker). The is_unitary flag is passed to
algorithms.GeneralizedHadamardTest and is not used by the numeric path.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import qiskit as qk

import algorithms


_worker_data = {}


#################
## Worker side ##
#################

def _initialize_worker(M, U_list, V_list, t_array, is_unitary, shots,
                       fast_sampling, seed, method, noiseless):
    """Runs once per worker process. Heavy objects are pickled once, here."""
    _worker_data["M"] = M
    _worker_data["U_list"] = U_list
    _worker_data["V_list"] = V_list
    _worker_data["t_array"] = t_array
    _worker_data["is_unitary"] = is_unitary
    _worker_data["shots"] = shots
    _worker_data["fast_sampling"] = fast_sampling
    _worker_data["seed"] = seed
    _worker_data["noiseless"] = noiseless
    # Fallback stream, used only when seed is None (non-reproducible mode).
    _worker_data["rng"] = np.random.default_rng()

    if method == "numeric":
        _precompute_numeric(M, U_list, V_list, t_array, is_unitary)


def _check_hermitian(M, tol=1e-8):
    asym = np.max(np.abs(M - M.conj().T))
    if asym > tol * max(1.0, float(np.max(np.abs(M)))):
        raise ValueError(
            f"M is not Hermitian (max |M - M^dagger| = {asym:.3e}). "
            "Pass is_unitary=True if M is meant to be unitary instead.")


def _check_unitary(M, tol=1e-8):
    d = M.shape[0]
    resid = np.max(np.abs(M @ M.conj().T - np.eye(d)))
    if resid > tol * max(1.0, float(np.max(np.abs(M)))):
        raise ValueError(
            f"M is not unitary (max |M M^dagger - I| = {resid:.3e}). "
            "Pass is_unitary=False if M is meant to be Hermitian instead.")


def _precompute_numeric(M, U_list, V_list, t_array, is_unitary):
    """Diagonalise M once per worker and cache the pieces every entry reuses.

    Single flag, matching your is_matrix_unitary() convention: M is assumed to
    be either unitary or Hermitian, never both, never neither, so is_unitary
    alone decides the branch (the old separate `hermitian` flag is gone --
    it's just `not is_unitary` now, and derived internally so the two can't
    drift out of sync again).

    With M = W diag(lam) W^-1,

        <psi_l| exp(-i M t_n) |phi_r>
            = sum_k (U^dagger W)[l, k] * exp(-i lam_k t_n) * (W^-1 V)[k, r]

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
    M = np.asarray(M, dtype=complex)
    U = np.asarray(U_list, dtype=complex)          # (d, L), probes in columns
    V = np.asarray(V_list, dtype=complex)          # (d, R)

    if is_unitary:
        _check_unitary(M)
        try:
            from scipy.linalg import schur
            T, W = schur(M, output="complex")      # exact diagonal: M normal
            evals = np.diag(T).astype(complex)
            B = W.conj().T @ V                      # W guaranteed unitary
        except ImportError:
            evals, W = np.linalg.eig(M)
            B = np.linalg.solve(W, V)               # W not guaranteed unitary
    else:
        _check_hermitian(M)
        evals, W = np.linalg.eigh(M)
        evals = evals.astype(complex)
        B = W.conj().T @ V                          # W^-1 == W^dagger

    A = U.conj().T @ W                              # (L, d)

    # (d, N) phase table, shared by every (l, r). Small: d is the Hilbert
    # dimension, so 16 x N complex for a 16x16 Hamiltonian.
    phases = np.exp(-1j * np.outer(evals, t_array))

    _worker_data["num_A"] = np.ascontiguousarray(A)
    _worker_data["num_B"] = np.ascontiguousarray(B)
    _worker_data["num_phases"] = np.ascontiguousarray(phases)


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


def _sample_from_amplitudes(z, rng, shots):
    """Turn exact amplitudes into the same two tensors the circuits produce.

    A generalized Hadamard test whose ancilla marginal is P(0) = (1 + Re z)/2
    returns 2*P(0) - 1 as its estimate of Re z; the imaginary circuit does the
    same for Im z. One shot is a Bernoulli draw, `shots` shots a Binomial mean.
    """
    p_real = np.clip(0.5 * (1.0 + z.real), 0.0, 1.0)
    p_img = np.clip(0.5 * (1.0 + z.imag), 0.0, 1.0)

    n = z.shape[0]
    real_single = 2.0 * (rng.random(n) < p_real) - 1.0
    img_single = 2.0 * (rng.random(n) < p_img) - 1.0

    real_poly = 2.0 * (rng.binomial(shots, p_real) / shots) - 1.0
    img_poly = 2.0 * (rng.binomial(shots, p_img) / shots) - 1.0

    return (real_single + 1j * img_single), (real_poly + 1j * img_poly)


def _compute_chunk(task):
    """Evaluate Z[l, r, start:stop] for one (l, r) pair and one slice of t."""
    l, r, start, stop = task

    M = _worker_data["M"]
    U = _worker_data["U_list"][:, l]
    V = _worker_data["V_list"][:, r]
    t_array = _worker_data["t_array"]
    is_unitary = _worker_data["is_unitary"]
    shots = _worker_data["shots"]
    fast_sampling = _worker_data["fast_sampling"]

    rng = _task_rng(l, r, start)
    n = stop - start

    if fast_sampling:
        p_real = np.empty(n, dtype=float)
        p_img = np.empty(n, dtype=float)

        for i, t in enumerate(t_array[start:stop]):
            _, p_real[i] = _run_circuit_probability(M, U, V, t, False, is_unitary)
            _, p_img[i] = _run_circuit_probability(M, U, V, t, True, is_unitary)

        # One vectorised draw per chunk instead of four per t.
        real_single = 2.0 * (rng.random(n) < p_real) - 1.0
        img_single = 2.0 * (rng.random(n) < p_img) - 1.0

        real_poly = 2.0 * (rng.binomial(shots, p_real) / shots) - 1.0
        img_poly = 2.0 * (rng.binomial(shots, p_img) / shots) - 1.0

    else:
        # Original code path, kept verbatim in spirit for A/B validation.
        real_single = np.empty(n, dtype=float)
        img_single = np.empty(n, dtype=float)
        real_poly = np.empty(n, dtype=float)
        img_poly = np.empty(n, dtype=float)

        for i, t in enumerate(t_array[start:stop]):
            state_re, _ = _run_circuit_probability(M, U, V, t, False, is_unitary)
            real_single[i] = 2 * state_re.sample_counts(1, [0]).get("0", 0) - 1
            real_poly[i] = 2 * (state_re.sample_counts(shots, [0]).get("0", 0)
                                / shots) - 1

            state_im, _ = _run_circuit_probability(M, U, V, t, True, is_unitary)
            img_single[i] = 2 * state_im.sample_counts(1, [0]).get("0", 0) - 1
            img_poly[i] = 2 * (state_im.sample_counts(shots, [0]).get("0", 0)
                               / shots) - 1

    Z_single_chunk = real_single + 1j * img_single
    Z_poly_chunk = real_poly + 1j * img_poly

    return l, r, start, Z_single_chunk, Z_poly_chunk


def _amplitudes_numerically(l, r, start, stop):
    """Exact <psi_l| exp(-i M t_n) |phi_r> for n in [start, stop)."""
    A = _worker_data["num_A"]               # (L, d)
    B = _worker_data["num_B"]               # (d, R)
    phases = _worker_data["num_phases"]     # (d, N)

    # Fold the two probe-dependent factors together first: one length-d vector.
    coeff = A[l, :] * B[:, r]
    return coeff @ phases[:, start:stop]    # (stop - start,)


def _compute_chunk_numerically(task):
    """Drop-in replacement for _compute_chunk using numpy instead of qiskit.

    Same task granularity, same return signature, same shot statistics, so the
    scheduling layer does not change at all. The work per chunk is a single
    (d,) x (d, chunk) product against the cached phase table, which is why this
    backend is orders of magnitude faster than building circuits.
    """
    l, r, start, stop = task

    z = _amplitudes_numerically(l, r, start, stop)

    if _worker_data["noiseless"]:
        # Both tensors carry the exact amplitudes -- useful as ground truth.
        return l, r, start, z.copy(), z.copy()

    rng = _task_rng(l, r, start)
    Z_single_chunk, Z_poly_chunk = _sample_from_amplitudes(
        z, rng, _worker_data["shots"])

    return l, r, start, Z_single_chunk, Z_poly_chunk


########################
## Scheduling helpers ##
########################

def _split_bounds(n, k):
    """Split range(n) into k contiguous, near-equal (start, stop) pairs."""
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


########################
## Public entry point ##
########################

def generate_multiple_Z_tensors(M, N, U_list, V_list, L, R, t_list,
                                is_unitary=True, shots=750, workers=4,
                                tasks_per_worker=4,
                                n_splits=None,
                                fast_sampling=True,
                                limit_blas_threads=True,
                                seed=None,
                                progress=False,
                                method="circuit",
                                noiseless=False):
    """Build the single-shot and poly-shot Z tensors of shape (L, R, N).

    Parameters beyond the original signature
    ----------------------------------------
    tasks_per_worker : int
        Target number of tasks per worker. Higher means finer-grained load
        balancing but more IPC. 4 is a reasonable default; raise it if you see
        stragglers at the end of a run, lower it if IPC dominates.
    n_splits : int or None
        Pin how many chunks the t axis is cut into, overriding the adaptive
        rule above. Needed only for the reproducibility caveat under `seed`.
    fast_sampling : bool
        Circuit backend only. Draw shot noise with numpy from the exact ancilla
        probability instead of calling Statevector.sample_counts().
    limit_blas_threads : bool
        Pin each worker's BLAS to one thread.
    seed : int or None
        If given, results are reproducible across runs and independent of the
        order in which tasks happen to complete. They are NOT independent of
        the worker count on their own: each task's stream is keyed by its
        (l, r, start), and the adaptive rule above makes `start` depend on
        `workers`. Pass an explicit n_splits to fix the task boundaries and get
        the same tensors from any number of workers.
    progress : bool
        Print completion percentage.
    method : {"circuit", "numeric"}
        "circuit" builds generalized Hadamard tests in qiskit. "numeric"
        evaluates the matrix elements directly with numpy.
    is_unitary : bool
        Now the single flag for both backends, matching your is_matrix_unitary
        convention: M is assumed unitary if True, Hermitian if False -- never
        both, never neither. The circuit backend forwards it to
        GeneralizedHadamardTest as before; the numeric backend uses it to pick
        eigh (Hermitian) vs. a Schur decomposition (unitary) and validates M
        against the corresponding assumption before running (see
        _check_hermitian / _check_unitary).
    noiseless : bool
        Numeric backend only. Return the exact amplitudes in both tensors
        instead of sampled ones -- ground truth for accuracy studies.
    """
    if method not in ("circuit", "numeric"):
        raise ValueError(f"unknown method {method!r}")

    N = int(N)
    t_array = np.ascontiguousarray(np.asarray(t_list))

    if len(t_array) != N:
        raise ValueError(f"N ({N}) must match len(t_list) ({len(t_array)})")

    U_arr = np.asarray(U_list)
    V_arr = np.asarray(V_list)
    if U_arr.ndim != 2 or U_arr.shape[1] != L or V_arr.shape[1] != R:
        raise ValueError(
            f"expected probes in columns: U_list is {U_arr.shape} (want (d, {L})), "
            f"V_list is {V_arr.shape} (want (d, {R}))")

    Z_tensor_single = np.zeros((L, R, N), dtype=complex)
    Z_tensor_poly = np.zeros((L, R, N), dtype=complex)

    if method == "numeric":
        # Check once, here, before spawning workers: an exception raised inside
        # the pool initializer surfaces as an opaque BrokenProcessPool instead
        # of the informative ValueError from _check_unitary/_check_hermitian.
        M_arr = np.asarray(M, dtype=complex)
        _check_unitary(M_arr) if is_unitary else _check_hermitian(M_arr)

    tasks = _build_tasks(L, R, N, workers, tasks_per_worker, n_splits)
    workers = max(1, min(int(workers), len(tasks)))

    if limit_blas_threads:
        _limit_blas_threads()

    # spawn is important because the worker processes import modules
    # without re-executing main.py.
    context = mp.get_context("spawn")

    initargs = (M, U_list, V_list, t_array, is_unitary, shots,
                fast_sampling, seed, method, noiseless)

    worker_fn = _compute_chunk if method == "circuit" else _compute_chunk_numerically

    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_initialize_worker,
                             initargs=initargs) as executor:

        # Everything is submitted up front, so a worker that finishes early
        # immediately picks up the next (l, r, t-slice) rather than waiting for
        # its peers to finish the current (l, r).
        futures = [executor.submit(worker_fn, task) for task in tasks]

        done = 0
        for future in as_completed(futures):
            l, r, start, single_chunk, poly_chunk = future.result()
            stop = start + len(single_chunk)

            Z_tensor_single[l, r, start:stop] = single_chunk
            Z_tensor_poly[l, r, start:stop] = poly_chunk

            done += 1
            if progress and (done % max(1, len(tasks) // 100) == 0
                             or done == len(tasks)):
                pct = 100.0 * done / len(tasks)
                print(f"\rZ tensor: {pct:5.1f}%  ({done}/{len(tasks)} tasks)",
                      end="", flush=True)

        if progress:
            print()

    return Z_tensor_single, Z_tensor_poly


################
## Validation ##
################

def compare_backends(M, U_list, V_list, t_list, n_samples=12, is_unitary=True,
                     tol=1e-8, seed=0):
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

    _worker_data["t_array"] = t
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
            _, p0 = _run_circuit_probability(M, U[:, l], V[:, r], t[n], img,
                                             is_unitary)
            vals.append(2.0 * p0 - 1.0)

        z = _amplitudes_numerically(l, r, n, n + 1)[0]
        worst = max(worst, abs(vals[0] - z.real), abs(vals[1] - z.imag))

        print(f"{l:>3} {r:>3} {n:>5} {vals[0]:>13.9f} {z.real:>13.9f} "
              f"{vals[1]:>13.9f} {z.imag:>13.9f}")

    verdict = "PASS" if worst < tol else "FAIL -- conventions differ"
    print(f"\nworst absolute discrepancy: {worst:.3e}  ({verdict})")
    return worst


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