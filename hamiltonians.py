import re
import numpy as np

####################
## Pauli Matrices ##
####################

_I = np.eye(2, dtype=complex)

_X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

_Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

_Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

#########################
## Auxiliary Functions ##
#########################

def _operator_on_site(A, site, n_qubits):
    """
    Construct the n-qubit operator with A acting on `site` and
    identity acting on every other qubit.

    Qubit ordering follows the computational basis convention used
    by the existing hard-coded Hamiltonians.
    """
    operators = [_I] * n_qubits
    operators[site] = A

    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)

    return result


def _two_site_operator(A, site1, B, site2, n_qubits):
    """
    Construct the n-qubit operator with A on site1, B on site2,
    and identity elsewhere.
    """
    operators = [_I] * n_qubits
    operators[site1] = A
    operators[site2] = B

    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)

    return result


#####################
## My Hamiltonians ##
#####################

def _belldiagonal_4x4():
    return np.array([
        [-2, 0, 0, -1],
        [0, 3, -1, 0],
        [0, -1, 3, 0],
        [-1, 0, 0, -2],
    ],dtype=float)


def _belldiagonal_16x16():
    return np.kron(_belldiagonal_4x4(), _belldiagonal_4x4())

def _diagonal_8x8():
    return np.array([
        [8, 0, 0, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 6, 0, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ],dtype=float)


def _diagonal_16x16():
    return np.array([
        [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],dtype=float)

############################
## Important Hamiltonians ##
############################

def _TFIM(n_qubits):
    """
    Open-boundary transverse-field Ising model:
    """
    if n_qubits < 1: raise ValueError("Number of qubits must be at least 1.")

    dimension = 2 ** n_qubits
    H = np.zeros((dimension, dimension), dtype=complex)

    # -Z_i Z_{i+1}
    for i in range(n_qubits - 1):
        H -= _two_site_operator(_Z, i, _Z, i + 1, n_qubits)

    # -X_i
    for i in range(n_qubits):
        H -= _operator_on_site(_X, i, n_qubits)

    return H.real


def _XXZ(n_qubits, delta=0.5):
    """
    Open-boundary XXZ model:
    """
    if n_qubits < 1: raise ValueError("Number of qubits must be at least 1.")

    dimension = 2 ** n_qubits
    H = np.zeros((dimension, dimension), dtype=complex)

    for i in range(n_qubits - 1):
        H += _two_site_operator( _X, i, _X, i + 1, n_qubits)
        H += _two_site_operator(_Y, i, _Y, i + 1, n_qubits)
        H += delta * _two_site_operator(_Z, i, _Z, i + 1, n_qubits)

    return H.real


HAMILTONIANS = {
    "belldiagonal4x4": _belldiagonal_4x4,
    "belldiagonal16x16": _belldiagonal_16x16,
    "diagonal8x8": _diagonal_8x8,
    "diagonal16x16": _diagonal_16x16,
}


def get_hamiltonian(name):
    if name is None:
        raise ValueError("Hamiltonian name cannot be None.")

    key = str(name).strip()
    if not key:
        raise ValueError("Hamiltonian name cannot be empty.")

    normalized = key.lower().replace(" ", "").replace("_", "")

    ##########################
    ## Dynamic Hamiltonians ##
    ##########################

    match = re.fullmatch(r"(tfim|xxz)(\d+)x(\d+)", normalized)
    if match:
        model = match.group(1)
        rows = int(match.group(2))
        cols = int(match.group(3))
        N = rows

        if rows != cols: raise ValueError(f"Hamiltonian dimension must be square, got {rows}x{cols}.")
        if N < 2 or (N & (N - 1)) != 0: raise ValueError(f"{model.upper()} dimension must be a power of 2, got {N}.")

        n_qubits = N.bit_length() - 1

        if model == "tfim": 
            return _TFIM(n_qubits)
        elif model == "xxz": 
            return _XXZ(n_qubits)

    #######################
    ## Fixed Hamiltnians ##
    #######################
    if normalized not in HAMILTONIANS:
        available = "\n\t".join(sorted(HAMILTONIANS.keys()))

        raise ValueError(f"Unknown Hamiltonian '{name}'. Available options:\n\t{available}")

    return HAMILTONIANS[normalized]()