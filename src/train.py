import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import LabelEncoder

from features import build_features

SEED = 42

CAT_COLS = [
    "open_date_cat",
    "floor_area_cat",
    "city",
    "region",
    "alcohol_license"
]


def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    return df


def encode_categories(df):
    df = df.copy()
    encoders = {}

    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(
            df[col].astype(str).fillna("NA")
        )
        encoders[col] = le

    return df, encoders


def train_model(df):
    sorted_months = sorted(df["month"].unique())
    val_month = sorted_months[-1]

    train_df = df[df["month"] < val_month].copy()
    valid_df = df[df["month"] == val_month].copy()

    features = [
        c for c in df.columns
        if c not in ["rto", "month"]
    ]

    X_train = train_df[features]
    y_train = train_df["rto"]

    X_valid = valid_df[features]
    y_valid = valid_df["rto"]

    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.01,
        depth=8,
        loss_function="MAPE",
        eval_metric="MAPE",
        random_seed=SEED,
        verbose=500,
        early_stopping_rounds=300
    )

    model.fit(
        train_pool,
        eval_set=valid_pool
    )

    model.save_model("models/catboost_model.cbm")

    print("Model saved to models/catboost_model.cbm")

    return model


if __name__ == "__main__":
    df = load_data()
    df = build_features(df)
    df, encoders = encode_categories(df)

    train_model(df)
