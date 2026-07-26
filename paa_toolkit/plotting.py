"""
plotting.py

Publication-quality plotting utilities for

Pattern Analysis and Applications (Springer)

Author:
PAA Publication Toolkit
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------
# Journal Style
# ---------------------------------------------------------------------

class SpringerStyle:
    """
    Plot configuration used across the manuscript.
    """

    # Figure sizes (inches)
    SINGLE_COLUMN = (3.35, 2.70)
    DOUBLE_COLUMN = (6.90, 3.60)
    DPI = 600
    CMAP = "viridis"
    FONT_FAMILY = "DejaVu Sans"
    TITLE_SIZE = 9
    VALUE_SIZE = 7
    GRID_COLOR = "white"
    GRID_WIDTH = 0.8
    VALUE_FORMAT = "{:.2f}"
    ACCURACY_LIMITS = (0.90, 1.00)
    KAPPA_LIMITS = (0.80, 1.00)
    LABEL_SIZE = 10
    TICK_SIZE = 10
    ANNOTATION_SIZE = 9
    COLORBAR_SIZE = 10
    AXIS_LABEL_SIZE = 11


STYLE = SpringerStyle()


# ---------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------

CLASSIFIER_NAMES = {
    "RandomForest": "RF",
    "SVM-lineal": "SVM-Lin",
    "SVM-polinomial": "SVM-Poly",
    "SVM-RBF": "SVM-RBF",
    "MLP": "MLP",
    "kNN": "kNN"
}

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def apply_style():

    plt.rcParams.update({
        "font.family": STYLE.FONT_FAMILY,
        "font.size": STYLE.TICK_SIZE,
        "axes.titlesize": STYLE.TITLE_SIZE,
        "axes.labelsize": STYLE.LABEL_SIZE,
        "xtick.labelsize": STYLE.TICK_SIZE,
        "ytick.labelsize": STYLE.TICK_SIZE,
        "figure.dpi": STYLE.DPI
    })


def save_figure(fig, folder, filename):

    folder = Path(folder)

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        folder / f"{filename}.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        folder / f"{filename}.svg",
        bbox_inches="tight"
    )

    fig.savefig(
        folder / f"{filename}.png",
        dpi=STYLE.DPI,
        bbox_inches="tight"
    )


def classifier_labels(columns):
    labels = []
    for c in columns:
        labels.append(
            CLASSIFIER_NAMES.get(c, c)
        )

    return labels


def build_matrix(df, metric):
    required = [
        "feature_set",
        "classifier",
        metric
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    duplicated = df.duplicated(
        subset=[
            "feature_set",
            "classifier"
        ]
    )

    if duplicated.any():
        raise ValueError(
            "Duplicated FeatureSet/Classifier pairs."
        )

    matrix = (
        df.pivot(
            index="feature_set",
            columns="classifier",
            values=metric
        )
        .sort_index()

    )

    return matrix

    # ---------------------------------------------------------------------
# Heatmap Engine
# ---------------------------------------------------------------------

def draw_heatmap(
    matrix,
    title="",
    colorbar_label="",
    limits=None,
    figsize=None,
    ax=None,
    show_colorbar=True
):
    """
    Draw a publication-quality heatmap.
    Parameters
    ----------
    matrix : pandas.DataFrame
    title : str
    colorbar_label : str
    limits : tuple(min,max)
    figsize : tuple
    """

    apply_style()

    if figsize is None:
        figsize = STYLE.DOUBLE_COLUMN

    created_figure = False

    if ax is None:
        if figsize is None:
            figsize = STYLE.DOUBLE_COLUMN
        fig, ax = plt.subplots(figsize=figsize)

        created_figure = True

    else:
        fig = ax.figure

    cmap = plt.get_cmap(STYLE.CMAP)

    norm = Normalize(
        vmin=limits[0],
        vmax=limits[1]
    )

    mesh = ax.pcolormesh(
        matrix.values,
        cmap=cmap,
        norm=norm,
        edgecolors=STYLE.GRID_COLOR,
        linewidth=STYLE.GRID_WIDTH,
        shading="flat"
    )

    ax.set_xticks(
        np.arange(matrix.shape[1]) + 0.5
    )

    ax.set_yticks(
        np.arange(matrix.shape[0]) + 0.5
    )

    ax.set_xticklabels(
        classifier_labels(matrix.columns),
        rotation=35,
        ha="right"
    )

    ax.set_yticklabels(
        matrix.index
    )

    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(matrix.shape[0], 0)

    ax.set_xlabel("Classifier")
    ax.set_ylabel("Feature Set")
    if title:
        ax.set_title(
            title,
            fontsize=STYLE.TITLE_SIZE
        )

    # --------------------------------------------------
    # Cell values
    # --------------------------------------------------

    best_value = np.nanmax(matrix.values)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if np.isnan(value):
                continue
            color = "white"

            if norm(value) < 0.45:
                color = "black"

            text = STYLE.VALUE_FORMAT.format(value)

            weight = (
                "bold"
                if np.isclose(
                    value,
                    matrix.values.max()
                )
                else "normal"
            )

            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=STYLE.ANNOTATION_SIZE,
                fontweight=weight,
                color=color
            )

    # --------------------------------------------------
    # Highlight best cell
    # --------------------------------------------------

    row, col = np.argwhere(
        matrix.values == best_value
    )[0]


    if show_colorbar:
        cbar = fig.colorbar(
            mesh,
            ax=ax,
            fraction=0.04,
            pad=0.02
        )

        cbar.set_label(
            colorbar_label,
            fontsize=STYLE.COLORBAR_SIZE
        )

        cbar.ax.tick_params(
            labelsize=STYLE.COLORBAR_SIZE
        )
        
        cbar.set_label(
            colorbar_label,
            fontsize=STYLE.COLORBAR_SIZE
        )

        cbar.ax.tick_params(
            labelsize=STYLE.COLORBAR_SIZE
        )

    if created_figure:
        fig.tight_layout()

    return fig, mesh

    # ---------------------------------------------------------------------
# Accuracy Heatmap
# ---------------------------------------------------------------------

def plot_accuracy_heatmap(
    df,
    output_folder=None,
    filename="figure_accuracy_heatmap",
    show=False
):
    """
    Generate Accuracy heatmap.
    """

    matrix = build_matrix(
        df,
        "acc_mean"
    )

    fig, _ = draw_heatmap(
        matrix=matrix,
        title="",
        colorbar_label="Accuracy",
        limits=STYLE.ACCURACY_LIMITS
    )

    if output_folder is not None:

        save_figure(
            fig,
            output_folder,
            filename
        )

    if show:

        plt.show()

    return fig


# ---------------------------------------------------------------------
# Kappa Heatmap
# ---------------------------------------------------------------------

def plot_kappa_heatmap(
    df,
    output_folder=None,
    filename="figure_kappa_heatmap",
    show=False
):
    """
    Generate Cohen's Kappa heatmap.
    """

    matrix = build_matrix(
        df,
        "kappa_mean"
    )

    fig, _ = draw_heatmap(
        matrix=matrix,
        title="",
        colorbar_label="Kappa",
        limits=STYLE.KAPPA_LIMITS
    )

    if output_folder is not None:

        save_figure(
            fig,
            output_folder,
            filename
        )

    if show:

        plt.show()

    return fig

    # ---------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------

def generate_standard_figures(
    datasets,
    figures_folder
):
    """
    Generate every standard figure supported by the toolkit.
    """

    generated = []

    if "binary" in datasets:

        plot_accuracy_heatmap(
            datasets["binary"],
            figures_folder,
            filename="binary_accuracy_heatmap"
        )

        plot_kappa_heatmap(
            datasets["binary"],
            figures_folder,
            filename="binary_kappa_heatmap"
        )

        generated.extend([
            "binary_accuracy_heatmap",
            "binary_kappa_heatmap"
        ])

    if "multiclass" in datasets:

        plot_accuracy_heatmap(
            datasets["multiclass"],
            figures_folder,
            filename="multiclass_accuracy_heatmap"
        )

        plot_kappa_heatmap(
            datasets["multiclass"],
            figures_folder,
            filename="multiclass_kappa_heatmap"
        )

        generated.extend([
            "multiclass_accuracy_heatmap",
            "multiclass_kappa_heatmap"
        ])

    return generated