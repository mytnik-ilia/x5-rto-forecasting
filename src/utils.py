import numpy as np


def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = (
        (y_true > 0)
        & ~np.isnan(y_true)
        & ~np.isnan(y_pred)
    )

    return (
        100.0
        * np.mean(
            np.abs(
                (y_pred[mask] - y_true[mask])
                / y_true[mask]
            )
        )
    )


def score_to_points(mape_score):
    return round(
        100 * ((100 - min(mape_score, 100)) / 100) ** 2,
        2
    )


def reduce_memory(df):
    for col in df.columns:

        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

        if df[col].dtype == "int64":
            df[col] = df[col].astype("int32")

    return df
