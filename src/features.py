import pandas as pd

SENSORS = ["temperature", "vibration", "pressure", "motor_current", "efficiency"]


def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.sort_values(["unit_id", "cycle"]).copy()
    grouped = df.groupby("unit_id", group_keys=False)
    for col in SENSORS:
        df[f"{col}_roll_mean"] = grouped[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = grouped[col].transform(
            lambda x: x.rolling(window, min_periods=2).std()
        ).fillna(0.0)
        df[f"{col}_trend"] = grouped[col].transform(
            lambda x: x.diff(window).fillna(0.0) / max(window, 1)
        )
    return df


def feature_columns(df: pd.DataFrame):
    excluded = {"unit_id", "rul"}
    return [c for c in df.columns if c not in excluded]
