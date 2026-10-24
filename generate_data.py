"""Generate a realistic synthetic mess-attendance dataset.

Replace data/mess_data.csv with your own logs (same columns) to use real data:
date, meal, rain_mm, temp_c, is_holiday, is_exam, is_event, menu_score, attendance
"""
import os
import numpy as np
import pandas as pd
from common import CAPACITY, MEALS


def main(seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-07-01", "2026-09-30")
    base = {"breakfast": 0.62, "lunch": 0.70, "dinner": 0.86}
    rows = []
    for d in dates:
        m, dow = d.month, d.dayofweek
        temp = 28 + 4 * np.sin((m - 3) / 12 * 2 * np.pi) + rng.normal(0, 1.5)
        p_rain = 0.45 if m in (6, 7, 8, 9, 10, 11) else 0.08
        rain = rng.exponential(12) if rng.random() < p_rain else 0.0
        in_break = (m == 12 and d.day >= 20) or (m == 1 and d.day <= 8) or (m in (5, 6) and d.day >= 1 and (m == 5 and d.day >= 10 or m == 6 and d.day <= 20))
        holiday = int(in_break or rng.random() < 0.03)
        exam = int((m == 10 and d.day <= 14) or (m == 3 and d.day <= 14) or (m == 12 and 1 <= d.day <= 14))
        event = int(rng.random() < 0.04)
        for meal in MEALS:
            menu = int(rng.integers(1, 6))
            f = base[meal]
            if dow >= 5:
                f += {"breakfast": -0.15, "lunch": 0.05, "dinner": -0.06}[meal]
            if holiday:
                f -= 0.35
            if exam:
                f += {"breakfast": -0.08, "lunch": 0.0, "dinner": 0.05}[meal]
            if event and meal != "breakfast":
                f += 0.08
            f += 0.0006 * min(rain, 60)
            f += 0.04 * (menu - 3)
            f += -0.004 * max(temp - 32, 0)
            f *= rng.normal(1.0, 0.04)
            att = int(np.clip(round(f * CAPACITY), 0, CAPACITY))
            rows.append(dict(date=d.strftime("%Y-%m-%d"), meal=meal, rain_mm=round(rain, 1),
                             temp_c=round(temp, 1), is_holiday=holiday, is_exam=exam,
                             is_event=event, menu_score=menu, attendance=att))
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("data/mess_data.csv", index=False)
    print(f"Saved data/mess_data.csv with {len(df)} rows")


if __name__ == "__main__":
    main()
