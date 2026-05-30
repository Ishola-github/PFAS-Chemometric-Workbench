# Chemometric Workbench (RUO)

Research-use-only (RUO) workbench for multivariate analysis of analytical
chemistry data: PCA exploration, spectral preprocessing, and multivariate
outlier detection (Hotelling's T² and Q-residual / SPE), with a roadmap toward
PLS regression, explainable AI (SHAP / applicability domain), and LC-MS/MS QA
review.

> **Governance / scope:** This is a research and training utility. It is **not**
> EPA-certified, **not** ISO-certified software, **not** validated for regulatory
> submission, and is **intentionally separate** from any governed reproducibility
> or frozen-evidence release. All flags and model outputs are *suggested
> interpretations* that require qualified analyst review.

## Features (V1)

- **Data import** — upload a CSV (samples × variables) or use the built-in
  synthetic NIR-like spectral demo (with injected outliers).
- **Preprocessing** — mean-centering, autoscaling, SNV, MSC, Savitzky-Golay
  (smoothing and 1st derivative).
- **PCA explorer** — scree (explained / cumulative variance), interactive score
  plot, and loading plot.
- **Outlier detection** — Hotelling's T² vs Q-residual influence plot with
  statistically derived control limits (F-distribution for T², Jackson-Mudholkar
  for Q), a flagged-sample table, and CSV export of diagnostics.

## Roadmap

| Version | Adds |
|---------|------|
| V1 (this) | PCA, preprocessing, T²/Q outlier detection |
| V2 | PLS regression, cross-validation, RMSE, prediction dashboard |
| V3 | SHAP, feature importance, applicability domain, uncertainty flags |
| V4 | LC-MS/MS QA review: retention-time monitoring, peak-area drift, batch QC |
| V5 | PFAS chemometric module (screening + QA + sustainability metrics) |

## Setup

```powershell
cd C:\Users\techj\Downloads\chemometric-workbench
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # or run python directly: .\.venv\Scripts\python.exe -m ...
pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501

To regenerate the example dataset:

```powershell
python sample_data.py   # writes examples/example_spectra.csv
```

## Expected CSV format

- One row per sample.
- Numeric **feature** columns (e.g. wavelengths or peak areas) used to build the model.
- Optional metadata columns (e.g. `sample_id`, `group`, `target`) — these can be
  excluded from the feature set and used for colouring in the score plot.

## Repository layout

```text
chemometric-workbench/
├── app.py                 # Streamlit UI (V1)
├── chemometrics_core.py   # preprocessing + PCA + T²/Q diagnostics
├── sample_data.py         # synthetic spectral dataset generator
├── requirements.txt
├── examples/              # generated example CSV
├── README.md
├── LICENSE
└── CITATION.cff
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff). This is an RUO utility; cite it as research
software, not as a validated analytical method.
