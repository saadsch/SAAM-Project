import pandas as pd
import numpy as np

# Charger les returns journaliers
returns = pd.read_csv(
    "data/processed/returns.csv",
    index_col=0,
    parse_dates=True
)

# Vérifier que l'index est bien une date
returns.index = pd.to_datetime(returns.index)

# Transformer les returns journaliers en returns mensuels composés
monthly_returns = (
    (1 + returns)
    .resample("M")
    .prod()
    - 1
)

# Supprimer les mois entièrement vides
monthly_returns = monthly_returns.dropna(how="all")

# Sauvegarder
monthly_returns.to_csv("data/processed/monthly_returns.csv")

print("monthly_returns.csv créé ✅")
print(monthly_returns.head())