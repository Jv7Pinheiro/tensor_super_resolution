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
 
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
 
import numpy as np
import qiskit as qk
 
import algorithms
 
 
_worker_data = {}
 
 
# --------------------------------------------------------------------------- #
# Worker side
# --------------------------------------------------------------------------- #
 
def _initialize_worker(M, U_list, V_list, t_array, is_unitary, shots,
                       fast_sampling, seed):
    """Runs once per worker process. Heavy objects are pickled once, here."""
    _worker_data["M"] = M
    _worker_data["U_list"] = U_list
    _worker_data["V_list"] = V_list
    _worker_data["t_array"] = t_array
    _worker_data["is_unitary"] = is_unitary
    _worker_data["shots"] = shots
    _worker_data["fast_sampling"] = fast_sampling
    _worker_data["seed"] = seed
    # Fallback stream, used only when seed is None (non-reproducible mode).
    _worker_data["rng"] = np.random.default_rng()
 
 
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
    qc = algorithms.GeneralizedHadamardTest(M, U, V, t, img=img,
                                            is_unitary=is_unitary)
    state = qk.quantum_info.Statevector.from_instruction(qc)
    return state, _p0_from_statevector(np.asarray(state.data))
 
 
def _compute_chunk(task):
    """Evaluate Z[l, r, start:stop] for one (l, r) pair and one slice of t."""
    l, r, start, stop = task
 
    M = _worker_data["M"]
    U = _worker_data["U_list"][l]
    V = _worker_data["V_list"][r]
    t_array = _worker_data["t_array"]
    is_unitary = _worker_data["is_unitary"]
    shots = _worker_data["shots"]
    fast_sampling = _worker_data["fast_sampling"]
    seed = _worker_data["seed"]
 
    # Deriving the stream from the task coordinates rather than from the worker
    # makes results reproducible regardless of how tasks get scheduled.
    if seed is None:
        rng = _worker_data["rng"]
    else:
        rng = np.random.default_rng([seed, l, r, start])
 
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
 
 
# --------------------------------------------------------------------------- #
# Scheduling helpers
# --------------------------------------------------------------------------- #
 
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
 
 
def _build_tasks(L, R, N, workers, tasks_per_worker):
    """Flat task list covering every (l, r, t-slice).
 
    If L * R is already large compared to the core count, each (l, r) pair is a
    single task and t is not split at all -- that gives good balance with the
    least IPC. If L * R is small (e.g. 4x4 on 64 cores), t gets split enough to
    keep every core busy.
    """
    pairs = L * R
    target = max(1, int(workers) * int(tasks_per_worker))
    n_splits = max(1, -(-target // pairs))          # ceil division
    n_splits = min(n_splits, N)
 
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
 
 
# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
 
def generate_multiple_Z_tensors(M, N, U_list, V_list, L, R, t_list,
                                is_unitary=True, shots=750, workers=4,
                                tasks_per_worker=4,
                                fast_sampling=True,
                                limit_blas_threads=True,
                                seed=None,
                                progress=False):
    """Build the single-shot and poly-shot Z tensors of shape (L, R, N).
 
    Parameters beyond the original signature
    ----------------------------------------
    tasks_per_worker : int
        Target number of tasks per worker. Higher means finer-grained load
        balancing but more IPC. 4 is a reasonable default; raise it if you see
        stragglers at the end of a run, lower it if IPC dominates.
    fast_sampling : bool
        Draw shot noise with numpy from the exact ancilla probability instead of
        calling Statevector.sample_counts(). Distributionally identical.
    limit_blas_threads : bool
        Pin each worker's BLAS to one thread. Leave on unless you know your
        circuits benefit from threaded BLAS.
    seed : int or None
        If given, results are reproducible and independent of worker count and
        task scheduling order.
    progress : bool
        Print completion percentage to stderr.
    """
    N = int(N)
    t_array = np.ascontiguousarray(np.asarray(t_list))
 
    if len(t_array) != N:
        raise ValueError(f"N ({N}) must match len(t_list) ({len(t_array)})")
 
    Z_tensor_single = np.zeros((L, R, N), dtype=complex)
    Z_tensor_poly = np.zeros((L, R, N), dtype=complex)
 
    tasks = _build_tasks(L, R, N, workers, tasks_per_worker)
    workers = max(1, min(int(workers), len(tasks)))
 
    if limit_blas_threads:
        _limit_blas_threads()
 
    # spawn is important because the worker processes import modules
    # without re-executing main.py.
    context = mp.get_context("spawn")
 
    initargs = (M, U_list, V_list, t_array, is_unitary, shots,
                fast_sampling, seed)
 
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_initialize_worker,
                             initargs=initargs) as executor:
 
        # Everything is submitted up front, so a worker that finishes early
        # immediately picks up the next (l, r, t-slice) rather than waiting for
        # its peers to finish the current (l, r).
        futures = [executor.submit(_compute_chunk, task) for task in tasks]
 
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