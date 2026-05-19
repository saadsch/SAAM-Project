import pandas as pd
import numpy as np


returns = pd.read_csv(
    "data/processed/returns.csv",
    index_col=0,
    parse_dates=True
)

returns.index = pd.to_datetime(returns.index)


monthly_returns = (
    (1 + returns)
    .resample("M")
    .prod()
    - 1
)

monthly_returns = monthly_returns.dropna(how="all")

# Sauvegarder
monthly_returns.to_csv("data/processed/monthly_returns.csv")

print("monthly_returns.csv créé ✅")
print(monthly_returns.head())