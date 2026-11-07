"""Generates data/sample/sample_metrics.xlsx: 30 days of normal business
activity, ending with a deliberate critical revenue anomaly (matching the
Appendix example in the PRD: revenue drops ~32% while traffic stays flat).

Run with: python data/sample/generate_sample_data.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_DAYS = 30
dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=N_DAYS, freq="D")

traffic = np.random.normal(10000, 400, N_DAYS).round()
conversion_rate = np.random.normal(3.2, 0.15, N_DAYS).round(2)
orders = (traffic * conversion_rate / 100).round()
revenue = (orders * np.random.normal(100, 5, N_DAYS)).round(2)
cost = (revenue * np.random.normal(0.4, 0.02, N_DAYS)).round(2)
refunds = (revenue * np.random.normal(0.03, 0.01, N_DAYS)).clip(min=0).round(2)

# Inject the critical anomaly on the last day: revenue falls ~32%,
# traffic stays roughly flat, conversion drops - matches the PRD appendix.
traffic[-1] = traffic[-2] * 1.01
conversion_rate[-1] = conversion_rate[-2] * 0.68
orders[-1] = round(traffic[-1] * conversion_rate[-1] / 100)
revenue[-1] = round(revenue[-2] * 0.68, 2)
cost[-1] = round(cost[-2] * 0.98, 2)
refunds[-1] = round(refunds[-2] * 1.05, 2)

df = pd.DataFrame(
    {
        "Date": dates,
        "Revenue": revenue,
        "Orders": orders,
        "Conversion Rate": conversion_rate,
        "Traffic": traffic,
        "Cost": cost,
        "Refunds": refunds,
    }
)

out_path = os.path.join(os.path.dirname(__file__), "sample_metrics.xlsx")
df.to_excel(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
