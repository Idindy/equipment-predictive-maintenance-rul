from pathlib import Path
import pandas as pd

CMAPSS_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmapss_train(path: str) -> pd.DataFrame:
    """Load a NASA C-MAPSS train_FD00x.txt file and add RUL labels."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    df["rul"] = max_cycles - df["cycle"]
    return df


if __name__ == "__main__":
    example = Path("data/train_FD001.txt")
    if example.exists():
        print(load_cmapss_train(str(example)).head())
    else:
        print("Place a NASA C-MAPSS train_FD001.txt file in data/ to use this loader.")
