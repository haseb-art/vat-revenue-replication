# Revenue Mobilization and VAT Adoption: Replication Code

Replication code for the manuscript "Did VAT Adoption Change Revenue Mobilization and Tax Structure? Evidence from 195 Countries, 1980–2024."

## Data

The analysis uses the IMF World Revenue Longitudinal Database (WoRLD), 2026 release. The file is public and is available from the IMF data portal. Download the Excel workbook and place it in `data/` under the name `world-imf2026.xlsx`. The file is not redistributed here.

## Requirements

Python 3.10 or later, with the packages in `requirements.txt`:

    pip install -r requirements.txt

## Running

From the repository root:

    bash run_all.sh

This runs, in order: data preparation, descriptive statistics, two-way fixed-effects regressions, the Callaway–Sant'Anna event study with bootstrap standard errors, and figure generation. Outputs are written to `out/`. The bootstrap takes the longest; set `BOOT_REPS` (default 300) to a smaller value for a quick check.

## Files

`code/config.py` sets paths. `code/00_load_data.py` reads the workbook and caches it. `code/common.py` builds the panel, the absorbing VAT indicator, and the estimators. `code/01_descriptives.py` produces the descriptive tables in Section 4. `code/02_twfe.py` runs the two-way fixed-effects models and robustness checks. `code/03_callaway_santanna.py` runs the group-time event study. `code/04_figures.py` produces Figures 1–5. Figure 6 (Bangladesh vs. peers) is produced by the same script.

## Coding decisions

The VAT indicator is absorbing: it switches on in the first year a VAT is recorded and stays on. This corrects a coding gap in 2024 and gaps within some country series. The comparison group for the event study is countries that never record a VAT. Countries with a VAT in 1980 are excluded from the adoption comparison. Standard errors in the two-way models are clustered by country with a small-sample adjustment. Standard errors in the event study come from a country-level bootstrap.

## Reproducibility notes

Results are deterministic for the two-way models. The bootstrap uses a fixed random seed (42), so the reported standard errors are reproducible on the same software versions. Different numpy versions may change the last decimal place.

## Citation

[Authors], [Year]. [Title]. [Journal]. Code: [repository URL].
