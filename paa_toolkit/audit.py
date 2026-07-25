"""
Dataset auditing utilities.
"""

import pandas as pd


def audit_dataframe(name: str, df: pd.DataFrame):

    print("=" * 80)
    print(name.upper())
    print("=" * 80)

    print(f"\nShape: {df.shape}")

    print("\nColumns:\n")

    for c in df.columns:
        print(" -", c)

    print("\nMissing values:")

    print(df.isna().sum())

    print("\nData types:")

    print(df.dtypes)

    print("\nFirst rows:")

    display(df.head())