import matplotlib.pyplot as plt
import numpy as np


def grid_dimensions(n_panels):
    """Return a near-square (rows, columns) layout for ``n_panels`` plots."""
    if n_panels <= 0:
        return 0, 0

    n_columns = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_columns))
    return n_rows, n_columns


def plot_G(pred_evals, svs, tau, path, true_multiplicities=None):
    """Plot predicted eigenvalues and singular values.

    ``true_multiplicities`` is supplied by the caller because it belongs to
    the problem setup, not to the plotting module.
    """
    pred_evals = np.real(pred_evals)
    svs = np.real(svs)
    K = svs.shape[1]
    n_rows, n_columns = grid_dimensions(svs.shape[1])
    fig, subplot_axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.5 * n_columns, 3.5 * n_rows),
        squeeze=False)
    axes = subplot_axes.ravel()

    for ax, singular_values, index in zip(axes, svs.T, range(K)):
        x = np.arange(1, len(singular_values) + 1)
        ax.axhline(tau, color="red", linestyle="--", linewidth=2, label=r"$\tau$")
        ax.bar(x, singular_values)
        ax.set_xticks(x)
        # ax.legend(fontsize=10)
        title = rf"SV of $G$ at $\lambda \approx {pred_evals[index]:.4f}$"
        if true_multiplicities is not None:
            closest_eigenvalue = min(
                true_multiplicities,
                key=lambda eigenvalue: abs(eigenvalue - pred_evals[index]),
            )
            ax.set_title(
                title
                + f"\ntrue multiplicity: {true_multiplicities[closest_eigenvalue]}"
            )
        else:
            ax.set_title(title)

    for ax in axes[svs.shape[1]:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)


def plot_sv_vs_theta(Z, t_list, T, tau, true_multiplicities, path, n_points=200):
    """
    Experiment 2: for each true eigenvalue λ_k, sweep theta over the window
    [λ_k - 5/T, λ_k + 5/T], compute W(theta) and its singular values at each
    point, and plot all SVs as curves vs. (theta - λ_k).  A vertical line marks
    theta = λ_k and a horizontal dashed line marks tau.

    This shows (a) whether the gap between signal and noise SVs is sharp at the
    true eigenvalue, (b) how quickly it collapses as theta drifts away, and
    (c) whether tau is in the right place relative to both groups.

    Parameters
    ----------
    Z                   : (L, R, N) tensor
    t_list              : (N,) array of time samples
    T                   : T_max used for this run
    tau                 : singular-value threshold
    true_multiplicities : dict  {eigenvalue: multiplicity}
    path                : where to save the figure (pdf/png)
    n_points            : number of theta values in the sweep window (default 200)
    """
    L = Z.shape[0]
    R = Z.shape[1]
    N = Z.shape[2]

    true_evals = sorted(true_multiplicities.keys())
    n_evals = len(true_evals)

    # Near-square grid of subplots, one per true eigenvalue
    n_cols = int(np.ceil(np.sqrt(n_evals)))
    n_rows = int(np.ceil(n_evals / n_cols))

    fig, subplot_axes = plt.subplots(n_rows, n_cols,
                                     figsize=(5 * n_cols, 3.5 * n_rows),
                                     squeeze=False)
    axes = subplot_axes.ravel()

    half_window = 5.0 / T  # ± 5/T around each true eigenvalue

    for ax, lam in zip(axes, true_evals):
        m = true_multiplicities[lam]
        deltas = np.linspace(-half_window, half_window, n_points)
        thetas = lam + deltas

        # sv_grid[i, j] = j-th singular value at thetas[i]
        min_dim = min(L, R)
        sv_grid = np.zeros((n_points, min_dim))

        W = np.zeros((L, R), dtype='complex')
        for i, theta in enumerate(thetas):
            for l in range(L):
                for r in range(R):
                    W[l, r] = Z[l, r, :].dot(np.exp(1j * t_list * theta)) / N
            sv_grid[i] = np.linalg.svd(W, compute_uv=False)

        # Plot each SV as a curve; colour signal SVs (index < m) differently
        for j in range(min_dim):
            is_signal = j < m
            ax.plot(deltas, sv_grid[:, j],
                    color='steelblue' if is_signal else 'lightsteelblue',
                    linewidth=2.0 if is_signal else 0.8,
                    label=rf"$\sigma_{j+1}$" if j < 4 else None)  # legend cap at 4

        ax.axvline(0, color='black', linestyle=':', linewidth=1.2, label=r'$\lambda_k$')
        ax.axhline(tau, color='red', linestyle='--', linewidth=1.5, label=r'$\tau$')
        ax.set_title(rf"$\lambda \approx {lam:.4f}$,  true mult: {m}")
        ax.legend(fontsize=7, ncol=2)

    for ax in axes[n_evals:]:
        ax.set_visible(False)

    # Shared axis labels — bottom row only for x, leftmost column only for y
    for ax in axes[n_evals:]:
        ax.set_visible(False)

    # X-axis label: only on the bottom-row axes
    n_bottom_row_start = (n_rows - 1) * n_cols
    for ax in axes[n_bottom_row_start : n_bottom_row_start + n_cols]:
        if ax.get_visible():
            ax.set_xlabel(r'$\theta - \lambda_k$', fontsize=11)

    # Y-axis label: only on the leftmost column axes
    for i in range(n_rows):
        ax = axes[i * n_cols]
        if ax.get_visible():
            ax.set_ylabel(r'Singular value $\sigma_j$', fontsize=11)

    fig.suptitle(r"Sing Vals of $W(\theta)$ vs $\delta = \theta - \lambda_k$", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=150)
