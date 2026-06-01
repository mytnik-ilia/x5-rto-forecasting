import numpy as np
import pandas as pd

ID_COL = "new_id"


def create_lag_features(df, target_col="rto"):
    df = df.copy().sort_values([ID_COL, "month"])
    grp = df.groupby(ID_COL)[target_col]

    for lag in [1, 2, 3, 4, 5, 6, 9, 12]:
        df[f"lag_{lag}"] = grp.shift(lag)

    return df


def create_rolling_features(df, target_col="rto"):
    df = df.copy().sort_values([ID_COL, "month"])
    grp = df.groupby(ID_COL)[target_col]

    for w in [2, 3, 6, 9, 12]:
        df[f"roll_mean_{w}"] = grp.transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean()
        )

        df[f"roll_std_{w}"] = grp.transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).std()
        )

        df[f"roll_min_{w}"] = grp.transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).min()
        )

        df[f"roll_max_{w}"] = grp.transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).max()
        )

        df[f"roll_med_{w}"] = grp.transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=1).median()
        )

    return df


def create_ratio_features(df):
    df = df.copy()

    df["ratio_1_2"] = df["lag_1"] / (df["lag_2"] + 1e-6)
    df["ratio_1_3"] = df["lag_1"] / (df["lag_3"] + 1e-6)
    df["ratio_1_6"] = df["lag_1"] / (df["lag_6"] + 1e-6)

    df["diff_1_2"] = df["lag_1"] - df["lag_2"]
    df["diff_1_3"] = df["lag_1"] - df["lag_3"]

    return df


def create_date_features(df):
    df = df.copy()

    df["month_num"] = df["month"].dt.month
    df["year"] = df["month"].dt.year

    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    return df


def create_interaction_features(df):
    df = df.copy()

    if {"avg_promo_items", "foot_traffic"}.issubset(df.columns):
        df["promo_x_traffic"] = (
            df["avg_promo_items"] * df["foot_traffic"]
        )

    if {"avg_items_in_check", "working_hours"}.issubset(df.columns):
        df["items_x_hours"] = (
            df["avg_items_in_check"] * df["working_hours"]
        )

    if {"grocery_500m", "pyaterochka_500m"}.issubset(df.columns):
        df["competition_idx"] = (
            df["grocery_500m"] + df["pyaterochka_500m"]
        )

    return df


def build_features(df):
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_ratio_features(df)
    df = create_date_features(df)
    df = create_interaction_features(df)

    return df
