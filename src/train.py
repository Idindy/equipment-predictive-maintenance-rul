from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from simulate_data import simulate_equipment
from features import add_rolling_features, feature_columns


def machine_split(df, test_fraction=0.25, seed=42):
    rng = np.random.default_rng(seed)
    units = df["unit_id"].unique()
    rng.shuffle(units)
    n_test = max(1, int(len(units) * test_fraction))
    test_units = set(units[:n_test])
    return df[~df.unit_id.isin(test_units)], df[df.unit_id.isin(test_units)]


def main():
    Path("outputs").mkdir(exist_ok=True)
    data_path = Path("data/simulated_equipment.csv")
    df = pd.read_csv(data_path) if data_path.exists() else simulate_equipment()
    df = add_rolling_features(df, window=10)
    train_df, test_df = machine_split(df)
    cols = feature_columns(df)
    X_train, y_train = train_df[cols], train_df["rul"]
    X_test, y_test = test_df[cols], test_df["rul"]

    model = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "r2": float(r2_score(y_test, pred)),
        "train_units": int(train_df.unit_id.nunique()),
        "test_units": int(test_df.unit_id.nunique()),
    }
    print(json.dumps(metrics, indent=2))
    Path("outputs/metrics.json").write_text(json.dumps(metrics, indent=2))
    importance = pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(
        "outputs/feature_importance.csv", index=False
    )


if __name__ == "__main__":
    main()
