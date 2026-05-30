# Changelog

All notable changes to this project are documented in this file.

This project follows a lightweight Keep a Changelog / SemVer style.
Research-use-only (RUO): this software is not validated regulatory software.

## [Unreleased]

### Planned
- SHAP and feature-importance views
- Applicability domain and uncertainty flags
- LC/MS/MS QA review module (retention-time and peak-area drift)
- Batch anomaly detection dashboard

## [v0.2.0] - 2026-05-30

### Added
- PLS Regression tab in `app.py` with:
  - Target column selection
  - Train/test split controls
  - K-fold CV controls
  - RMSE, MAE, and R² metrics (train/test/CV)
  - Predicted-vs-actual plot
  - Residual plot
  - Component sweep plot (CV RMSE vs components)
  - Prediction CSV export
  - Model PKL export

### Added
- `run_pls_regression(...)` workflow in `chemometrics_core.py`.
- Metric helpers: RMSE, MAE, R².
- CV out-of-fold predictions and component sweep utility outputs.

### Changed
- README updated to reflect V2 status and current feature set.

## [v0.1.0] - 2026-05-30

### Added
- Initial Chemometric Workbench MVP:
  - CSV import + synthetic demo dataset
  - Preprocessing (mean-centering, autoscaling, SNV, MSC, Savitzky-Golay)
  - PCA explorer (scree, score, loading plots)
  - T²/Q multivariate outlier diagnostics and CSV export
- Governance and citation metadata:
  - RUO/non-regulatory scope language
  - `LICENSE`, `CITATION.cff`, `.gitignore`, `.streamlit/config.toml`

