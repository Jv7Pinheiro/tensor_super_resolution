import algorithms

import numpy as np
import qiskit as qk
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

_worker_data = {}

def _initialize_worker(M, U_list, V_list, is_unitary, shots):
    _worker_data["M"] = M
    _worker_data["U_list"] = U_list
    _worker_data["V_list"] = V_list
    _worker_data["is_unitary"] = is_unitary
    _worker_data["shots"] = shots


def _compute_t_chunk(task):
    l, r, start, t_chunk = task

    M = _worker_data["M"]
    U = _worker_data["U_list"][l]
    V = _worker_data["V_list"][r]
    is_unitary = _worker_data["is_unitary"]
    shots = _worker_data["shots"]

    chunk_length = len(t_chunk)

    Z_single_chunk = np.empty(chunk_length, dtype=complex)
    Z_poly_chunk = np.empty(chunk_length, dtype=complex)

    for local_index, t in enumerate(t_chunk):

        # Real part
        qc_re = algorithms.GeneralizedHadamardTest(M, U, V, t, img=False, is_unitary=is_unitary)

        state_re = qk.quantum_info.Statevector.from_instruction(qc_re)

        counts_real_single = state_re.sample_counts(1, [0])
        counts_real_poly = state_re.sample_counts(shots, [0])

        prob_0_real_single = counts_real_single.get("0", 0)
        prob_0_real_poly = counts_real_poly.get("0", 0) / shots

        real_single = 2 * prob_0_real_single - 1
        real_poly = 2 * prob_0_real_poly - 1

        # Imaginary part
        qc_im = algorithms.GeneralizedHadamardTest(M, U, V, t, img=True, is_unitary=is_unitary)

        state_im = qk.quantum_info.Statevector.from_instruction(qc_im)

        counts_img_single = state_im.sample_counts(1, [0])
        counts_img_poly = state_im.sample_counts(shots, [0])

        prob_0_img_single = counts_img_single.get("0", 0)
        prob_0_img_poly = counts_img_poly.get("0", 0) / shots

        img_single = 2 * prob_0_img_single - 1
        img_poly = 2 * prob_0_img_poly - 1

        Z_single_chunk[local_index] = real_single + 1j * img_single
        Z_poly_chunk[local_index] = real_poly + 1j * img_poly

    return start, Z_single_chunk, Z_poly_chunk

def generate_multiple_Z_tensors(M, N, U_list, V_list, L, R, t_list, is_unitary=True, shots=750, workers=4):
    N = int(N)
    t_array = np.asarray(t_list)

    if len(t_array) != N:
        raise ValueError(f"N ({N}) must match len(t_list) ({len(t_array)})")

    workers = min(int(workers), N)

    Z_tensor_single = np.zeros((L, R, N), dtype=complex)
    Z_tensor_poly = np.zeros((L, R, N), dtype=complex)


    # spawn is important because the worker processes import modules
    # without re-executing main.py.
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_initialize_worker, initargs=(M, U_list, V_list, is_unitary, shots)) as executor:

        for l in range(L):
            for r in range(R):
                # Exactly one approximately equal chunk per worker.
                chunks = np.array_split(t_array, workers)

                tasks = []
                start = 0

                for t_chunk in chunks:
                    tasks.append((l, r, start, t_chunk))
                    start += len(t_chunk)

                for start, single_chunk, poly_chunk in executor.map(_compute_t_chunk, tasks):
                    stop = start + len(single_chunk)

                    Z_tensor_single[l, r, start:stop] = single_chunk
                    Z_tensor_poly[l, r, start:stop] = poly_chunk

    return Z_tensor_single, Z_tensor_poly