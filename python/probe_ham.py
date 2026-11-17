import sys
import numpy as np
np.set_printoptions(linewidth=500)
import hamiltonians

# ── Hamiltonian registry ──────────────────────────────────────────────────────
#   Each entry: (label, model_string, degeneracy_score)
#   Score 0 = fully degenerate (XXZ h=0 δ=1), 100 = no degeneracy (TFIM g=0.5)

HAMILTONIANS = [
    ("XXZ h=0 δ=1",  "XXZ",   0,  dict(h=0, delta=1)),
    # ("XXZ h=0 δ=0",  "XXZ",  25,  dict(h=0, delta=0)),
    ("XXZ h=0 δ=2",  "XXZ",  33,  dict(h=0, delta=2)),
    # ("XXZ h=1 δ=-1", "XXZ",  50,  dict(h=1, delta=-1)),
    # ("TFIM g=0.1",   "TFIM", 66,  dict(g=0.1)),
    # ("TFIM g=1.0",   "TFIM", 80,  dict(g=1.0)),
    ("TFIM g=0.5",   "TFIM", 100, dict(g=0.5)),
]

#        4q   5q   6q  7q    8q    9q    10q
SIZES = [16,  32,  64, 128,  256,  512,  1024]


# ── Name builders ─────────────────────────────────────────────────────────────

def ham_name(entry, size):
    _, model, _, params = entry
    if model == "XXZ":
        return f"XXZ_(h={params['h']})-(d={params['delta']})_{size}x{size}"
    elif model == "TFIM":
        return f"TFIM_(g={params['g']})_{size}x{size}"
    raise ValueError(f"Unknown model: {model}")


# ── Spectrum analysis ─────────────────────────────────────────────────────────

def get_H_info(M):
    eigenvalues = np.sort(np.real(np.linalg.eig(M)[0]))

    tol = 1e-10
    groups, counts = [], []
    i = 0
    while i < len(eigenvalues):
        j = i + 1
        while j < len(eigenvalues) and abs(eigenvalues[j] - eigenvalues[i]) < tol:
            j += 1
        groups.append(np.mean(eigenvalues[i:j]))
        counts.append(j - i)
        i = j

    counts = np.array(counts)
    unique_counts, freq = np.unique(counts, return_counts=True)
    dist_str = "  ".join(f"{m}×{f}" for m, f in zip(unique_counts, freq))

    return len(counts), counts.max(), counts.mean(), dist_str


# ── Table printer ─────────────────────────────────────────────────────────────

def print_table(rows, col_headers, row_headers):
    """
    rows        : list of lists of cell strings
    col_headers : list of column header strings
    row_headers : list of row header strings (left margin)
    """
    # column widths: max of header and all cell contents
    rw = max(len(h) for h in row_headers)
    cw = [max(len(col_headers[j]), max(len(rows[i][j]) for i in range(len(rows))))
          for j in range(len(col_headers))]

    sep = "─" * (rw + 2) + "┼" + "┼".join("─" * (w + 2) for w in cw)

    # header
    header = " " * (rw + 2) + "│" + "│".join(f" {col_headers[j]:^{cw[j]}} " for j in range(len(col_headers)))
    print(header)
    print(sep)

    for i, row_h in enumerate(row_headers):
        line = f" {row_h:<{rw}} │" + "│".join(f" {rows[i][j]:^{cw[j]}} " for j in range(len(col_headers)))
        print(line)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for entry in HAMILTONIANS:
        label, model, score, _ = entry
        print(f"\n{'═'*90}")
        print(f"  {label}  (degeneracy score: {score}/100)")
        print(f"{'═'*90}")

        col_headers = ["size", "n_unique", "max_mult", "avg_mult", "distribution  (mult × count)"]
        rows = []
        row_headers = []

        for size in SIZES:
            name = ham_name(entry, size)
            try:
                Ham = hamiltonians.get_hamiltonian(name)
            except ValueError as exc:
                print(f"  [skip] {name}: {exc}", file=sys.stderr)
                continue

            M = (np.pi / (4 * np.linalg.norm(Ham))) * Ham
            n_unique, max_mult, avg_mult, dist = get_H_info(M)

            row_headers.append(f"{size}×{size}")
            rows.append([
                name.split("_")[-1],      # just "16x16" etc — name already in header
                str(n_unique),
                str(max_mult),
                f"{avg_mult:.2f}",
                dist,
            ])
            # also keep "size" column for clarity when printed standalone
            rows[-1][0] = f"{size}×{size}"

        print_table(rows, col_headers, row_headers)

    print()


if __name__ == "__main__":
    main()