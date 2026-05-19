import pandas as pd
import numpy as np

# Charger le fichier
df = pd.read_csv('data/processed/Prices.csv', index_col=0, parse_dates=True)

# Identifier les cellules < 0.5 AVANT remplacement
mask = df < 0.5

# Trouver les ISIN concernés (colonnes avec au moins une valeur < 0.5)
affected_isins = df.columns[mask.any()]

# Remplacer les valeurs < 0.5 par NaN
df = df.where(df >= 0.5, np.nan)

# Sauvegarder
df.to_csv('data/processed/prices_clean.csv')

# Print des ISIN concernés
print("ISIN avec au moins une valeur < 0.5 (remplacée par NaN) :")
print(list(affected_isins))

print(f"\nNombre d'entreprises concernées : {len(affected_isins)}")
print("prices_clean.csv créé avec succès ✅")