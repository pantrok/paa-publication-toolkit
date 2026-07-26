"""
tables.py

Generate LaTeX tables compatible with
Springer sn-jnl.cls.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class TableGenerator:

    def __init__(self, project):

        self.project = project

        self.output = Path(project.tables)

    # --------------------------------------------------------

    @staticmethod
    def format_metric(mean, std):

        if pd.isna(mean):
            return "--"

        if pd.isna(std):
            return f"{mean:.3f}"

        return f"{mean:.3f} $\\pm$ {std:.3f}"

    # --------------------------------------------------------

    @staticmethod
    def bold(text):

        return f"\\textbf{{{text}}}"

    # --------------------------------------------------------

    @staticmethod
    def save(filename, latex):

        Path(filename).write_text(
            latex,
            encoding="utf8"
        )

            # --------------------------------------------------------

    def dataframe_to_latex(
        self,
        df,
        caption,
        label
    ):

        header = r"""\begin{table}
\centering
\caption{%s}
\label{%s}
\begin{tabular}{llcc}
\toprule
Feature set &
Classifier &
Accuracy &
Kappa\\
\midrule
""" % (caption, label)

        body = ""

        best_accuracy = df["acc_mean"].max()

        for _, row in df.iterrows():

            acc = self.format_metric(
                row["acc_mean"],
                row["acc_std"]
            )

            if np.isclose(
                row["acc_mean"],
                best_accuracy
            ):
                acc = self.bold(acc)

            kappa = self.format_metric(
                row["kappa_mean"],
                row["kappa_std"]
            )

            body += (
                f'{row["feature_set"]} & '
                f'{row["classifier"]} & '
                f'{acc} & '
                f'{kappa} \\\\\n'
            )

        footer = r"""\bottomrule
\end{tabular}
\end{table}
"""

        return header + body + footer

            # --------------------------------------------------------

    def generate_binary(self):

        if "binary" not in self.project.datasets:

            print("Binary dataset not found.")

            return

        df = self.project.datasets["binary"].copy()

        latex = self.dataframe_to_latex(
            df=df,
            caption="Performance obtained for the binary classification task.",
            label="tab:binary_results"
        )

        self.save(
            self.output / "binary_results.tex",
            latex
        )

        print("✓ binary_results.tex generated")

    # --------------------------------------------------------

    def generate_multiclass(self):

        if "multiclass" not in self.project.datasets:

            print("Multiclass dataset not found.")

            return

        df = self.project.datasets["multiclass"].copy()

        latex = self.dataframe_to_latex(
            df=df,
            caption="Performance obtained for the multiclass classification task.",
            label="tab:multiclass_results"
        )

        self.save(
            self.output / "multiclass_results.tex",
            latex
        )

        print("✓ multiclass_results.tex generated")

            # --------------------------------------------------------

    def generate_all(self):

        self.output.mkdir(
            parents=True,
            exist_ok=True
        )

        self.generate_binary()

        self.generate_multiclass()

        self.generate_best_tables()

        print()

        print("=" * 60)

        print("Tables successfully generated.")

        print("=" * 60)


            # --------------------------------------------------------

    def best_per_feature_set(self, df):

        rows = []

        for feature_set, group in df.groupby("feature_set"):

            idx = group["acc_mean"].idxmax()

            rows.append(df.loc[idx])

        best = pd.DataFrame(rows)

        preferred_order = ["B", "C", "D", "E"]

        if set(preferred_order).issubset(best["feature_set"].unique()):

            best["feature_set"] = pd.Categorical(
                best["feature_set"],
                categories=preferred_order,
                ordered=True
            )

            best = best.sort_values("feature_set")

        return best.reset_index(drop=True)

            # --------------------------------------------------------

    def generate_best_tables(self):

        configs = [

            (
                "binary",
                "binary_best_results.tex",
                "Best performance obtained for each feature set in the binary classification task.",
                "tab:binary_best"
            ),

            (
                "multiclass",
                "multiclass_best_results.tex",
                "Best performance obtained for each feature set in the multiclass classification task.",
                "tab:multiclass_best"
            )

        ]

        for dataset, filename, caption, label in configs:

            if dataset not in self.project.datasets:

                continue

            df = self.best_per_feature_set(
                self.project.datasets[dataset]
            )

            latex = self.dataframe_to_latex(
                df,
                caption,
                label
            )

            self.save(
                self.output / filename,
                latex
            )

            print(f"✓ {filename} generated")

            # --------------------------------------------------------

    