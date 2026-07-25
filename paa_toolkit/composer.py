"""
composer.py

Compose multi-panel figures for the manuscript.
"""

import matplotlib.pyplot as plt

from .plotting import (
    build_matrix,
    draw_heatmap,
    STYLE,
    save_figure,
)

class PaperComposer:

    def __init__(self, project):

        self.project = project

    def compose_performance_heatmaps(self):

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(7.2, 6.2),
            constrained_layout=True
        )

        # Binary Accuracy

        draw_heatmap(
            build_matrix(
                self.project.datasets["binary"],
                "acc_mean"
            ),
            ax=axes[0,0],
            limits=STYLE.ACCURACY_LIMITS,
            show_colorbar=False
        )

        axes[0,0].set_title("(a)")

        # Binary Kappa

        draw_heatmap(
            build_matrix(
                self.project.datasets["binary"],
                "kappa_mean"
            ),
            ax=axes[0,1],
            limits=STYLE.KAPPA_LIMITS,
            show_colorbar=False
        )

        axes[0,1].set_title("(b)")