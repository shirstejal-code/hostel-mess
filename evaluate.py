"""Simulate the decision policy on the held-out test period vs. baselines."""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import make_features, split_by_date, COST_WASTE, COST_SHORT, CAPACITY
from decision import optimal_quantity, plate_cost


def main():
    df = pd.read_csv("data/mess_data.csv").sort_values(["date", "meal"]).reset_index(drop=True)
    train, _, test = split_by_date(df)
    art = joblib.load("models/model.joblib")
    model, residuals = art["model"], art["residuals"]

    df["lag1"] = df.groupby("meal")["attendance"].shift(1)       # same meal yesterday
    test = df.loc[test.index].dropna(subset=["lag1"]).copy()
    hist_mean = train.groupby("meal")["attendance"].mean()

    pred = model.predict(make_features(test))
    actual = test["attendance"].values
    cu, co = COST_SHORT, COST_WASTE

    policies = {
        "Historical avg + 10% buffer": np.ceil(test["meal"].map(hist_mean).values * 1.10),
        "Same as yesterday": test["lag1"].values,
        "Point forecast only": np.ceil(pred),
        "Forecast + decision layer (ours)": optimal_quantity(pred, residuals, cu, co),
    }

    rows = []
    for name, cooked in policies.items():
        cost, waste, short = plate_cost(cooked, actual, cu, co)
        rows.append({"Policy": name, "Wasted plates": int(waste.sum()),
                     "Shortage (students)": int(short.sum()),
                     "Total cost (Rs)": int(cost.sum())})
    res = pd.DataFrame(rows)
    base = res.loc[res["Policy"] == "Same as yesterday", "Total cost (Rs)"].iloc[0]
    res["Saving vs yesterday (%)"] = ((base - res["Total cost (Rs)"]) / base * 100).round(1)

    os.makedirs("results", exist_ok=True)
    res.to_csv("results/summary.csv", index=False)
    print(res.to_string(index=False))

    # Plot 1: cost comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(res["Policy"], res["Total cost (Rs)"], color=["#999", "#999", "#6a9fd8", "#2e7d32"])
    ax.set_xlabel("Total cost on test period (Rs)")
    ax.set_title("Cost of each cooking policy (lower is better)")
    plt.tight_layout(); plt.savefig("results/cost_comparison.png", dpi=150); plt.close()

    # Plot 2: forecast vs actual (last 30 days of dinner)
    d = test[test["meal"] == "dinner"].tail(30)
    p = model.predict(make_features(d))
    q = optimal_quantity(p, residuals, cu, co)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d["date"], d["attendance"], "o-", label="Actual", color="black")
    ax.plot(d["date"], p, "--", label="Forecast", color="#1565c0")
    ax.plot(d["date"], q, "-", label="Plates to cook", color="#2e7d32")
    ax.set_xticks(range(0, len(d), 5)); ax.set_xticklabels(d["date"].iloc[::5], rotation=30)
    ax.set_ylabel("Students / plates"); ax.set_title("Dinner: forecast vs actual vs decision")
    ax.legend(); plt.tight_layout(); plt.savefig("results/forecast_vs_actual.png", dpi=150); plt.close()
    print("Saved results/summary.csv, cost_comparison.png, forecast_vs_actual.png")


if __name__ == "__main__":
    main()
