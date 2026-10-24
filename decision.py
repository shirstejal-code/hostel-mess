"""Decision layer: turn a forecast + its uncertainty into a cooking quantity.

Newsvendor logic: cook the quantile of demand at the critical ratio
    CR = Cu / (Cu + Co)
where Cu = cost of a shortage, Co = cost of a wasted plate.
The demand distribution is pred + calibration residuals (split-conformal style).
"""
import numpy as np


def optimal_quantity(pred, residuals, cu, co):
    cr = cu / (cu + co)
    return np.maximum(0, np.ceil(np.asarray(pred) + np.quantile(residuals, cr))).astype(int)


def prediction_interval(pred, residuals, level=0.8):
    lo = np.quantile(residuals, (1 - level) / 2)
    hi = np.quantile(residuals, 1 - (1 - level) / 2)
    return np.maximum(0, np.asarray(pred) + lo), np.asarray(pred) + hi


def plate_cost(cooked, actual, cu, co):
    cooked, actual = np.asarray(cooked), np.asarray(actual)
    waste = np.maximum(cooked - actual, 0)
    short = np.maximum(actual - cooked, 0)
    return co * waste + cu * short, waste, short


def expected_cost_curve(pred, residuals, cu, co, span=150):
    """Expected cost for a range of cooking quantities (for plotting)."""
    qs = np.arange(max(0, int(pred) - span), int(pred) + span)
    demand = pred + residuals
    costs = [plate_cost(np.full_like(demand, q), demand, cu, co)[0].mean() for q in qs]
    return qs, np.array(costs)
