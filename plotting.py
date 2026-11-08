import matplotlib.pyplot as plt
import numpy as np


def grid_dimensions(n_panels):
    """Return a near-square (rows, columns) layout for ``n_panels`` plots."""
    if n_panels <= 0:
        return 0, 0

    n_columns = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_columns))
    return n_rows, n_columns


def plot_G(pred_evals, svs, tau, true_multiplicities=None):
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
    fig.savefig(f"data/figures/attempt.pdf", bbox_inches="tight")