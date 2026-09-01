import numpy as np


def _belldiagonal_4x4():
    return np.array(
        [
            [-2, 0, 0, -1],
            [0, 3, -1, 0],
            [0, -1, 3, 0],
            [-1, 0, 0, -2],
        ],
        dtype=float,
    )


def _belldiagonal_16x16():
    return np.kron(_belldiagonal_4x4(), _belldiagonal_4x4())


HAMILTONIANS = {
    "belldiagonal": _belldiagonal_4x4,
    "belldiagonal4x4": _belldiagonal_4x4,
    "belldiagonal16x16": _belldiagonal_16x16,
    "belldiagonal_16x16": _belldiagonal_16x16,
}


def get_hamiltonian(name):
    if name is None:
        raise ValueError("Hamiltonian name cannot be None.")

    key = str(name).strip()
    if not key:
        raise ValueError("Hamiltonian name cannot be empty.")

    normalized = key.lower().replace(" ", "")
    canon = {
        "belldiagonal": "belldiagonal",
        "belldiagonal4x4": "belldiagonal4x4",
        "belldiagonal16x16": "belldiagonal16x16",
        "belldiagonal_16x16": "belldiagonal16x16",
    }.get(normalized, normalized)

    if canon not in HAMILTONIANS:
        available = "\n\t".join(sorted(HAMILTONIANS.keys()))
        raise ValueError(f"Unknown Hamiltonian '{name}'. Available options:\n\t{available}")

    return HAMILTONIANS[canon]()
