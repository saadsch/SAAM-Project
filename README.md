# SAAM — Pacific Carbon-Aware Allocation

HEC Lausanne, *Sustainability-Aware Asset Management* (Prof. E. Jondeau).
Region: Pacific. Implementation window: 2014-01 → 2025-12 (144 monthly returns).

The single authoritative deliverable is `src/SAAM_Project_FINAL.ipynb`. It
contains all four analytical parts (MV, VW, 50% carbon reduction, net-zero),
the validation checks, the limitations discussion and the LLM disclosure.

## How to run

From the repository root:

```
# 1. Create a local virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install the required packages
python -m pip install -r requirements.txt

# 5. Register the project kernel for Jupyter / VS Code
python -m ipykernel install --user --name saam-project \
    --display-name "Python (SAAM Project)"

# 6. Open SAAM_Project_FINAL.ipynb and select the
#    "Python (SAAM Project)" kernel before running.

# 7. Execute the notebook end-to-end from the command line
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    src/SAAM_Project_FINAL.ipynb
```

Top-to-bottom execution regenerates every CSV in `outputs/tables/` and every
PNG in `outputs/figures/` from the cleaned data in `data/processed/`. No cell
relies on a hard-coded result.

## Repository layout

```
data/
    raw/         # original Datastream/Refinitiv files (xlsx kept as Archives)
    processed/   # cleaned prices, returns, emissions, revenues, market caps,
                 # risk-free series
src/
    SAAM_Project_FINAL.ipynb   # AUTHORITATIVE final notebook (run this)
    data_cleaning.ipynb        # PDF Section 2.1 cleaning pipeline
    saam_core.py               # optimizer, drift, summary statistics
    05_part3.py                # Part 3 driver (50% carbon reduction)
    06_part4.py                # Part 4 driver (net-zero trajectory)
    MVP-construction.ipynb     # legacy standalone Part 1 (returns-eligible
                               #   universe, kept as a backup reference)
    vwp.ipynb                  # legacy standalone Part 2 (returns-eligible
                               #   universe, kept as a backup reference)
outputs/
    tables/       # final CSVs displayed inside the notebook
    figures/      # final PNGs displayed inside the notebook
    cleanup_log.md
    final_deliverables_index.md
    validation_checklist_part3_part4.md
requirements.txt
README.md
```

## Universe note

`SAAM_Project_FINAL.ipynb` restricts every part — including Parts I and II — to
the **carbon-eligible** universe (firms with valid Scope-1 emissions, revenues
and market cap at the rebalance date). The legacy notebooks
`MVP-construction.ipynb` and `vwp.ipynb` run on the broader **returns-eligible**
universe; their summary statistics therefore differ from the final notebook and
are kept only as a backup reference.
