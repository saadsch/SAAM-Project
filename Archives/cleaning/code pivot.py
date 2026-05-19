import pandas as pd

# Charger le fichier returns
df = pd.read_csv('data/processed/clean_revenues_pacific.csv')

# Colonnes d'identité
id_cols = ['NAME', 'ISIN']

# Colonnes de dates
date_cols = df.columns.drop(id_cols)

# Transformer en format long
df_long = df.melt(
    id_vars=id_cols,
    value_vars=date_cols,
    var_name='Date',
    value_name='Return'
)

# Convertir Date en format datetime
df_long['Date'] = pd.to_datetime(df_long['Date'])

# Pivot : lignes = Date, colonnes = ISIN
df_pivot = df_long.pivot(
    index='Date',
    columns='ISIN',
    values='Return'
)

# Trier par date
df_pivot = df_pivot.sort_index()

# Sauvegarder
df_pivot.to_csv('data/processed/revenues_pivot.csv')

print("Pivot créé : Revenues.csv ✅")