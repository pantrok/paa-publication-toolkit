"""
leakage.py

Utilities for generating tables and figures
for the leakage analysis.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LeakageAnalysis:

    def __init__(
        self,
        project,
        results_df,
        stats_df
    ):

        self.MODE_NAMES = {
            "binary": "Binary",
            "multiclass": "Multiclass"
        }
        self.CLASSIFIER_NAMES = {
            "RandomForest": "RF",
            "SVM-lineal": "SVM-L",
            "SVM-polinomial": "SVM-P",
            "SVM-RBF": "SVM-RBF",
            "MLP": "MLP",
            "kNN": "kNN"
        }
        self.CONDITION_NAMES = {
            "subject_dependent": "Subject-dependent",
            "subject-dependent": "Subject-dependent",
            "LOBO": "LOBO",
            "lobo": "LOBO",
            "LOCO": "LOCO",
            "loco": "LOCO",
            "G-1s": "G-1s",
            "S-1s": "S-1s"
        }
        self.project = project
        self.results = results_df.copy()
        self.stats = stats_df.copy()
        self.output_tables = Path(
            project.tables
        )
        self.output_figures = Path(
            project.figures
        )

    def summary(self):

        print("=" * 70)
        print("Leakage Results")
        print("=" * 70)
        print(self.results.shape)
        print()
        print(self.results.columns.tolist())
        print()
        print("=" * 70)
        print("Leakage Statistics")
        print("=" * 70)
        print(self.stats.shape)
        print()
        print(self.stats.columns.tolist())
            # --------------------------------------------------------

    def generate_summary_table(self):
        rows = []
        grouped = self.results.groupby(
            [
                "mode",
                "feature_set",
                "condition"
            ]
        )

        for (mode, feature_set, condition), group in grouped:
            idx = group["acc_mean"].idxmax()
            best = group.loc[idx]
            rows.append({
                "Mode": mode,
                "Features": feature_set,
                "Condition": condition,
                "Classifier": best["classifier"],
                "Accuracy":
                    f'{best["acc_mean"]:.3f} $\\pm$ {best["acc_std"]:.3f}',
                "Kappa":
                    f'{best["kappa_mean"]:.3f} $\\pm$ {best["kappa_std"]:.3f}'
            })
        summary = pd.DataFrame(rows)

        summary["Mode"] = (
            summary["Mode"]
            .map(self.MODE_NAMES)
            .fillna(summary["Mode"])
        )

        summary["Classifier"] = (
            summary["Classifier"]
            .map(self.CLASSIFIER_NAMES)
            .fillna(summary["Classifier"])
        )

        summary["Condition"] = (
            summary["Condition"]
            .map(self.CONDITION_NAMES)
            .fillna(summary["Condition"])
        )

        mode_order = {
            "Binary": 0,
            "Multiclass": 1
        }

        feature_order = {
            "B": 0,
            "C": 1,
            "D": 2,
            "E": 3
        }

        summary["mode_order"] = (
            summary["Mode"]
            .map(mode_order)
        )

        summary["feature_order"] = (
            summary["Features"]
            .map(feature_order)
        )

        summary = summary.sort_values(
            [
                "mode_order",
                "feature_order",
                "Condition"
            ]
        )

        summary = summary.drop(
            columns=[
                "mode_order",
                "feature_order"
            ]
        )

        best_acc = summary["Accuracy"].iloc[
            summary["Accuracy"].str.extract(
                r"([0-9.]+)"
            ).astype(float)[0].idxmax()
        ]

        summary.loc[
            summary["Accuracy"] == best_acc,
            "Accuracy"
        ] = (
            "\\textbf{"
            + best_acc
            + "}"
        )

        latex = summary.to_latex(
            index=False,
            escape=False,
            column_format="llllcc",
            caption=(
                "Best classification performance obtained "
                "for each feature set under the evaluated "
                "data partitioning protocols."
            ),
            label="tab:leakage-performance"
        )

        outfile = self.output_tables / "leakage_summary.tex"
        outfile.write_text(
            latex,
            encoding="utf8"
        )

        print("✓ leakage_summary.tex generated")

        return summary

        # --------------------------------------------------------

    def _plot_metric(
        self,
        metric,
        ylabel,
        filename
    ):

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9,4),
            sharey=True
        )

        modes = ["binary", "multiclass"]

        colors = [
            "#4C72B0",
            "#55A868",
            "#C44E52"
        ]

        for ax, mode in zip(axes, modes):
            data = self.results[
                self.results["mode"] == mode
            ]
            pivot = (
                data
                .groupby(
                    [
                        "feature_set",
                        "condition"
                    ]
                )[metric]
                .max()
                .unstack()
            )

            x = np.arange(
                len(pivot.index)
            )

            width = 0.8 / len(pivot.columns)

            for i, condition in enumerate(
                pivot.columns
            ):

                ax.bar(
                    x + i * width,
                    pivot[condition],
                    width,
                    label=condition,
                    color=colors[
                        i % len(colors)
                    ]
                )

            ax.set_xticks(
                x + width
            )

            ax.set_xticklabels(
                pivot.index
            )

            ax.set_title(
                self.MODE_NAMES.get(
                    mode,
                    mode
                )
            )

            ax.set_xlabel(
                "Feature set"
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

        axes[0].set_ylabel(
            ylabel
        )

        handles, labels = (
            axes[0]
            .get_legend_handles_labels()
        )

        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels)
        )

        fig.tight_layout()

        outfile = (
            self.output_figures /
            filename
        )

        fig.savefig(
            outfile.with_suffix(".pdf"),
            bbox_inches="tight"
        )

        fig.savefig(
            outfile.with_suffix(".png"),
            dpi=600,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"✓ {filename}")

        # --------------------------------------------------------

    def plot_leakage_accuracy(self):
        self._plot_metric(
            metric="acc_mean",
            ylabel="Accuracy",
            filename="leakage_accuracy"
        )

    # --------------------------------------------------------

    def plot_leakage_kappa(self):
        self._plot_metric(
            metric="kappa_mean",
            ylabel="Cohen's Kappa",
            filename="leakage_kappa"
        )

        # --------------------------------------------------------

    def plot_effect_size(self):

        df = self.stats.copy()

        df["Comparison"] = (
            df["feature_set"]
            + "-"
            + df["classifier"]
        )

        df["Mode"] = (
            df["mode"]
            .map(self.MODE_NAMES)
            .fillna(df["mode"])
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10,6),
            sharex=True
        )

        modes = ["Binary", "Multiclass"]

        for ax, mode in zip(axes, modes):

            data = (
                df[df["Mode"] == mode]
                .sort_values(
                    "cohens_d",
                    ascending=True
                )
            )

            colors = [
                "tab:red" if s else "tab:gray"
                for s in data["significant"]
            ]

            ax.barh(

                data["Comparison"],

                data["cohens_d"],

                color=colors

            )

            ax.axvline(
                0,
                color="black",
                linewidth=1
            )

            ax.set_title(mode)

            ax.set_xlabel("Cohen's d")

            ax.grid(
                axis="x",
                alpha=0.3
            )

        fig.suptitle(
            "Effect size between leakage evaluation protocols"
        )

        fig.tight_layout()

        outfile = (
            self.output_figures /
            "leakage_effect_size"
        )

        fig.savefig(
            outfile.with_suffix(".pdf"),
            bbox_inches="tight"
        )

        fig.savefig(
            outfile.with_suffix(".png"),
            dpi=600,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("✓ leakage_effect_size")

    # --------------------------------------------------------

    # --------------------------------------------------------

    def plot_effect_size_summary(self):

        df = self.stats.copy()

        # Magnitud absoluta del efecto
        df["abs_d"] = df["cohens_d"].abs()

        summary = (
            df.groupby(
                ["mode", "feature_set"]
            )
            .agg(
                mean_d=("abs_d", "mean"),
                std_d=("abs_d", "std"),
                significant=("significant", "sum"),
                total=("significant", "count")
            )
            .reset_index()
        )

        summary["Mode"] = (
            summary["mode"]
            .map(self.MODE_NAMES)
            .fillna(summary["mode"])
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8,4),
            sharex=True
        )

        for ax, mode in zip(
            axes,
            ["Binary", "Multiclass"]
        ):

            data = (
                summary[
                    summary["Mode"] == mode
                ]
                .sort_values("mean_d")
            )

            labels = []

            for _, row in data.iterrows():

                labels.append(
                    f'{row["feature_set"]} ({int(row["significant"])}/{int(row["total"])})'
                )

            ax.barh(

                labels,

                data["mean_d"],

                xerr=data["std_d"],

                capsize=3

            )

            ax.set_title(mode)

            ax.set_xlabel("|Cohen's d|")

            ax.grid(
                axis="x",
                alpha=0.3
            )

            # Umbrales clásicos
            ax.axvline(
                0.2,
                color="gray",
                linestyle="--",
                linewidth=1
            )

            ax.axvline(
                0.5,
                color="gray",
                linestyle=":",
                linewidth=1
            )

            ax.axvline(
                0.8,
                color="gray",
                linestyle="-.",
                linewidth=1
            )

        fig.suptitle(
            "Magnitude of the leakage effect by feature set"
        )

        fig.tight_layout()

        outfile = (
            self.output_figures /
            "leakage_effect_summary"
        )

        fig.savefig(
            outfile.with_suffix(".pdf"),
            bbox_inches="tight"
        )

        fig.savefig(
            outfile.with_suffix(".png"),
            dpi=600,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("✓ leakage_effect_summary")

    def generate_statistics_table(self):

        df = self.stats.copy()

        # -----------------------------
        # Display names
        # -----------------------------

        df["Mode"] = (
            df["mode"]
            .map(self.MODE_NAMES)
            .fillna(df["mode"])
        )

        df["Features"] = df["feature_set"]

        df["Classifier"] = (
            df["classifier"]
            .map(self.CLASSIFIER_NAMES)
            .fillna(df["classifier"])
        )

        # Comparison label

        df["Comparison"] = (
            df["cond_A"]
            + " vs "
            + df["cond_B"]
        )

        df["Adjusted $p$"] = (
            df["p_bonf"]
            .map(lambda x: f"{x:.4f}")
        )

        df["Cohen's $d$"] = (
            df["cohens_d"]
            .map(lambda x: f"{x:.3f}")
        )

        df["Significant"] = (
            df["significant"]
            .map(
                {
                    True: "Yes",
                    False: "No"
                }
            )
        )

        summary = df[
            [
                "Mode",
                "Features",
                "Classifier",
                "Comparison",
                "Adjusted $p$",
                "Cohen's $d$",
                "Significant"
            ]
        ].copy()

        mode_order = {
            "Binary": 0,
            "Multiclass": 1
        }

        feature_order = {
            "B": 0,
            "C": 1,
            "D": 2,
            "E": 3
        }

        classifier_order = {
            "RF": 0,
            "kNN": 1,
            "MLP": 2,
            "SVM-L": 3,
            "SVM-P": 4,
            "SVM-RBF": 5
        }

        summary["m"] = summary["Mode"].map(mode_order)

        summary["f"] = summary["Features"].map(feature_order)

        summary["c"] = summary["Classifier"].map(classifier_order)

        summary = summary.sort_values(
            [
                "m",
                "f",
                "c"
            ]
        )

        summary = summary.drop(
            columns=[
                "m",
                "f",
                "c"
            ]
        )

        latex = summary.to_latex(

            index=False,

            escape=False,

            column_format="lllclcc",

            caption=(
                "Statistical comparison between the "
                "segment-wise and group-wise validation "
                "protocols using the Wilcoxon signed-rank "
                "test with Bonferroni correction."
            ),

            label="tab:leakage-statistics"

        )

        outfile = (
            self.output_tables /
            "leakage_statistics.tex"
        )

        outfile.write_text(
            latex,
            encoding="utf8"
        )

        print(
            "✓ leakage_statistics.tex generated"
        )

        return summary

    # --------------------------------------------------------

    def generate_significant_statistics_table(self):

        summary = self.generate_statistics_table()

        significant = summary[
            summary["Significant"] == "Yes"
        ].copy()

        significant = significant.drop(
            columns=["Significant"]
        )

        significant = significant[
            [
                "Mode",
                "Features",
                "Classifier",
                "Adjusted $p$",
                "Cohen's $d$"
            ]
        ]

        significant["Cohen's $d$"] = (
            significant["Cohen's $d$"]
            .apply(
                lambda x: (
                    "\\textbf{"
                    + x
                    + "}"
                    if abs(float(x)) >= 0.5
                    else x
                )
            )
        )

        latex = significant.to_latex(

            index=False,

            escape=False,

            column_format="lllcc",

            caption=(
                "Statistically significant performance "
                "differences between the segment-wise "
                "and group-wise validation protocols "
                "after Bonferroni correction."
            ),

            label="tab:leakage-significant"

        )

        outfile = (
            self.output_tables /
            "leakage_significant.tex"
        )

        outfile.write_text(
            latex,
            encoding="utf8"
        )

        print(
            "✓ leakage_significant.tex generated"
        )

        return significant