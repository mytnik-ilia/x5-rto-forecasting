import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder

from features import build_features

CAT_COLS = [
    "open_date_cat",
    "floor_area_cat",
    "city",
    "region",
    "alcohol_license"
]


def load_test(path="data/test.csv"):
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    return df


def prepare_features(df):
    df = build_features(df)

    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(
            df[col].astype(str).fillna("NA")
        )

    return df


def run_inference():
    test_df = load_test()
    test_df = prepare_features(test_df)

    feature_cols = [
        c for c in test_df.columns
        if c not in ["month"]
    ]

    X_test = test_df[feature_cols]

    model = CatBoostRegressor()
    model.load_model("models/catboost_model.cbm")

    preds = model.predict(X_test)

    submission = pd.DataFrame({
        "new_id": test_df["new_id"],
        "rto": preds
    })

    submission.to_csv(
        "submissions/submission.csv",
        index=False
    )

    print("Submission saved to submissions/submission.csv")


if __name__ == "__main__":
    run_inference()
