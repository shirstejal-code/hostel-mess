"""Shared helpers: feature building, time-based split, cost settings."""
import numpy as np
import pandas as pd

MEALS = ["breakfast", "lunch", "dinner"]
CAPACITY = 1000          # hostel residents
COST_WASTE = 40          # Rs lost per plate cooked but not eaten (Co)
COST_SHORT = 60          # Rs penalty per student who finds food finished (Cu)

FEATURES = [
    "dow", "month", "is_weekend", "is_holiday", "is_exam", "is_event",
    "rain_mm", "temp_c", "menu_score",
    "meal_breakfast", "meal_lunch", "meal_dinner",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    X = pd.DataFrame(index=df.index)
    X["dow"] = d.dt.dayofweek
    X["month"] = d.dt.month
    X["is_weekend"] = (X["dow"] >= 5).astype(int)
    for c in ["is_holiday", "is_exam", "is_event", "rain_mm", "temp_c", "menu_score"]:
        X[c] = df[c].values
    for m in MEALS:
        X["meal_" + m] = (df["meal"] == m).astype(int).values
    return X[FEATURES]


def split_by_date(df: pd.DataFrame, train=0.6, calib=0.2):
    """Chronological split: train / calibration / test (no shuffling, no leakage)."""
    dates = np.sort(df["date"].unique())
    n = len(dates)
    a, b = dates[int(n * train)], dates[int(n * (train + calib))]
    return df[df["date"] < a], df[(df["date"] >= a) & (df["date"] < b)], df[df["date"] >= b]
