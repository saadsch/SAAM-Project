import pandas as pd
from pathlib import Path

# Paths
input_file = Path("data/Risk_Free_Rate_2025.xlsx")
output_file = Path("data/processed/Risk_Free_Rate_2025.csv")

# Read Excel file
df = pd.read_excel(input_file)

# Save as CSV
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Saved CSV to: {output_file}")
print(df.head())