# SAAM Final Notebook Package

This folder is the minimal reproducible notebook package for the SAAM project.

## Contents

- `SAAM_NOTEBOOK.ipynb`: final notebook; run all cells from top to bottom.
- `data/`: only the input files required by the notebook.
- `requirements.txt`: Python packages needed to run the notebook.

The notebook creates `outputs/tables/` and `outputs/figures/` automatically when it runs.

## Run

From this folder:

```powershell
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 SAAM_NOTEBOOK.ipynb
```

The notebook uses relative paths only. It does not import any files from `src/`, `Archives/`, or other project folders.
