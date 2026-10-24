"""Compare forecasting models, each followed by the same decision layer."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from common import make_features, split_by_date, COST_WASTE, COST_SHORT
from decision import optimal_quantity, plate_cost, prediction_interval


def main():
    df = pd.read_csv("data/mess_data.csv")
    train, calib, test = split_by_date(df)
    Xtr, Xca, Xte = (make_features(d) for d in (train, calib, test))
    models = {
        "Ridge regression": Ridge(alpha=1.0),
        "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "Gradient boosting (ours)": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                                  max_leaf_nodes=15, random_state=0),
    }
    rows = []
    for name, m in models.items():
        m.fit(Xtr, train["attendance"])
        res = calib["attendance"].values - m.predict(Xca)
        pred = m.predict(Xte)
        actual = test["attendance"].values
        qty = optimal_quantity(pred, res, COST_SHORT, COST_WASTE)
        cost = plate_cost(qty, actual, COST_SHORT, COST_WASTE)[0].sum()
        lo, hi = prediction_interval(pred, res, 0.8)
        rows.append({"Model": name,
                     "MAE (students)": round(mean_absolute_error(actual, pred), 1),
                     "80% coverage (%)": round(np.mean((actual >= lo) & (actual <= hi)) * 100, 1),
                     "Decision cost (Rs)": int(cost)})
    out = pd.DataFrame(rows)
    out.to_csv("results/model_comparison.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
