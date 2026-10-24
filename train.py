"""Train the attendance forecaster and store calibration residuals."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from common import make_features, split_by_date
from decision import prediction_interval

MODEL_PATH = "models/model.joblib"


def main():
    df = pd.read_csv("data/mess_data.csv")
    train, calib, test = split_by_date(df)

    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=15, random_state=0)
    model.fit(make_features(train), train["attendance"])

    # Calibration residuals (actual - predicted) give us the uncertainty.
    calib_pred = model.predict(make_features(calib))
    residuals = (calib["attendance"].values - calib_pred)

    test_pred = model.predict(make_features(test))
    mae = mean_absolute_error(test["attendance"], test_pred)
    mape = np.mean(np.abs(test["attendance"] - test_pred) / test["attendance"].clip(lower=1)) * 100
    lo, hi = prediction_interval(test_pred, residuals, 0.8)
    coverage = np.mean((test["attendance"].values >= lo) & (test["attendance"].values <= hi)) * 100

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "residuals": residuals}, MODEL_PATH)
    print(f"Test MAE: {mae:.1f} students | MAPE: {mape:.2f}% | 80% interval coverage: {coverage:.1f}%")


if __name__ == "__main__":
    main()
