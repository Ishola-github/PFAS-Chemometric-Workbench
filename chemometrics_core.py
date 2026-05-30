"""Core chemometrics routines for the Chemometric Workbench (RUO).

Research-use-only. Implements common spectral/analytical preprocessing,
PCA with multivariate outlier diagnostics (Hotelling's T-squared and
Q-residuals / SPE), with statistically derived control limits.

Nothing here is a validated/regulatory method; results require analyst review.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import f as f_dist
from scipy.stats import norm
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

PREPROCESS_OPTIONS = [
    "none",
    "mean-center",
    "autoscale",
    "SNV",
    "MSC",
    "Savitzky-Golay (smooth)",
    "Savitzky-Golay (1st derivative)",
]


def mean_center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def autoscale(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: per-row center and scale (scatter correction)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def msc(X: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Multiplicative Scatter Correction against a reference (mean) spectrum."""
    ref = X.mean(axis=0) if reference is None else reference
    corrected = np.empty_like(X, dtype=float)
    for i in range(X.shape[0]):
        slope, intercept = np.polyfit(ref, X[i, :], 1)
        if slope == 0:
            slope = 1.0
        corrected[i, :] = (X[i, :] - intercept) / slope
    return corrected


def savgol(X: np.ndarray, window: int = 11, polyorder: int = 2, deriv: int = 0) -> np.ndarray:
    """Row-wise Savitzky-Golay smoothing / derivative."""
    n_features = X.shape[1]
    window = min(window, n_features if n_features % 2 == 1 else n_features - 1)
    if window < 3:
        return X.astype(float)
    if window % 2 == 0:
        window += 1
    polyorder = min(polyorder, window - 1)
    return savgol_filter(X, window_length=window, polyorder=polyorder, deriv=deriv, axis=1)


def apply_preprocess(X: np.ndarray, method: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if method == "none":
        return X
    if method == "mean-center":
        return mean_center(X)
    if method == "autoscale":
        return autoscale(X)
    if method == "SNV":
        return snv(X)
    if method == "MSC":
        return msc(X)
    if method == "Savitzky-Golay (smooth)":
        return savgol(X, deriv=0)
    if method == "Savitzky-Golay (1st derivative)":
        return savgol(X, deriv=1)
    raise ValueError(f"Unknown preprocessing method: {method}")


# --------------------------------------------------------------------------- #
# PCA with multivariate diagnostics
# --------------------------------------------------------------------------- #


@dataclass
class PCAResult:
    scores: np.ndarray            # (n_samples, n_components)
    loadings: np.ndarray          # (n_components, n_features)
    explained_variance_ratio: np.ndarray  # per retained component
    eigenvalues_all: np.ndarray   # all eigenvalues (for Q-residual limit)
    n_components: int
    t2: np.ndarray                # Hotelling's T^2 per sample
    q: np.ndarray                 # Q-residual (SPE) per sample
    t2_limit: float
    q_limit: float
    confidence: float
    feature_names: list[str] = field(default_factory=list)

    def outlier_mask(self) -> np.ndarray:
        """True where a sample exceeds either control limit."""
        return (self.t2 > self.t2_limit) | (self.q > self.q_limit)

    def flag_table(self) -> list[dict]:
        rows = []
        for i in range(self.scores.shape[0]):
            t2_flag = self.t2[i] > self.t2_limit
            q_flag = self.q[i] > self.q_limit
            if t2_flag and q_flag:
                verdict = "T2 + Q outlier"
            elif t2_flag:
                verdict = "T2 outlier (extreme within model)"
            elif q_flag:
                verdict = "Q outlier (poor model fit)"
            else:
                verdict = "in-model"
            rows.append(
                {
                    "index": i,
                    "T2": float(self.t2[i]),
                    "T2_limit": float(self.t2_limit),
                    "Q": float(self.q[i]),
                    "Q_limit": float(self.q_limit),
                    "verdict": verdict,
                }
            )
        return rows


def _t2_limit(n_samples: int, n_components: int, confidence: float) -> float:
    """Hotelling's T^2 control limit via the F-distribution."""
    k, n = n_components, n_samples
    if n - k <= 0:
        return float("inf")
    fval = f_dist.ppf(confidence, k, n - k)
    return (k * (n - 1) / (n - k)) * fval


def _q_limit(residual_eigenvalues: np.ndarray, confidence: float) -> float:
    """Q-residual (SPE) limit via the Jackson-Mudholkar approximation."""
    eig = np.asarray(residual_eigenvalues, dtype=float)
    eig = eig[eig > 1e-12]
    if eig.size == 0:
        return 0.0
    theta1 = eig.sum()
    theta2 = (eig**2).sum()
    theta3 = (eig**3).sum()
    if theta2 == 0:
        return float(theta1)
    h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2**2)
    if h0 == 0:
        h0 = 1e-6
    c_alpha = norm.ppf(confidence)
    term = (
        c_alpha * np.sqrt(2.0 * theta2 * h0**2) / theta1
        + 1.0
        + theta2 * h0 * (h0 - 1.0) / theta1**2
    )
    return float(theta1 * term ** (1.0 / h0))


def run_pca(
    X: np.ndarray,
    n_components: int = 2,
    preprocess: str = "autoscale",
    confidence: float = 0.95,
    feature_names: list[str] | None = None,
) -> PCAResult:
    """Fit PCA and compute Hotelling's T^2 and Q-residual diagnostics."""
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    Xp = apply_preprocess(X, preprocess)

    max_components = max(1, min(n_samples - 1, n_features))
    n_components = int(max(1, min(n_components, max_components)))

    # Full PCA to obtain the complete eigenvalue spectrum for the Q-limit.
    full = PCA(n_components=max_components)
    full.fit(Xp)
    eigenvalues_all = full.explained_variance_  # eigenvalues of the covariance

    model = PCA(n_components=n_components)
    scores = model.fit_transform(Xp)
    loadings = model.components_  # (n_components, n_features)

    # Hotelling's T^2 = sum_k (t_k^2 / lambda_k)
    lam = model.explained_variance_
    lam_safe = np.where(lam > 1e-12, lam, 1e-12)
    t2 = np.sum((scores**2) / lam_safe, axis=1)

    # Q-residual (SPE) = squared reconstruction error in preprocessed space
    X_centered = Xp - model.mean_
    reconstruction = scores @ loadings
    residuals = X_centered - reconstruction
    q = np.sum(residuals**2, axis=1)

    residual_eigs = eigenvalues_all[n_components:]
    t2_lim = _t2_limit(n_samples, n_components, confidence)
    q_lim = _q_limit(residual_eigs, confidence)

    if feature_names is None:
        feature_names = [f"feature_{j}" for j in range(n_features)]

    return PCAResult(
        scores=scores,
        loadings=loadings,
        explained_variance_ratio=model.explained_variance_ratio_,
        eigenvalues_all=eigenvalues_all,
        n_components=n_components,
        t2=t2,
        q=q,
        t2_limit=t2_lim,
        q_limit=q_lim,
        confidence=confidence,
        feature_names=list(feature_names),
    )


# --------------------------------------------------------------------------- #
# PLS regression (V2)
# --------------------------------------------------------------------------- #


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def run_pls_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    preprocess: str = "autoscale",
    n_components: int = 2,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Fit/train/test/CV PLS model and return metrics + prediction tables."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if X.shape[0] < 8:
        raise ValueError("Need at least 8 rows for train/test + CV.")
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match X columns.")

    Xp = apply_preprocess(X, preprocess)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        Xp,
        y,
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
    )

    max_allowed = max(1, min(X_train.shape[0] - 1, X_train.shape[1]))
    k = int(max(1, min(n_components, max_allowed)))

    model = PLSRegression(n_components=k, scale=False)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train).reshape(-1)
    y_pred_test = model.predict(X_test).reshape(-1)

    metrics_train = _metrics(y_train, y_pred_train)
    metrics_test = _metrics(y_test, y_pred_test)

    # Out-of-fold CV predictions across all rows (same k).
    cv_folds = int(max(3, min(cv_folds, len(y) - 1)))
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    y_oof = np.zeros_like(y, dtype=float)
    for tr, va in kf.split(Xp):
        max_k_fold = max(1, min(len(tr) - 1, Xp.shape[1]))
        k_fold = int(min(k, max_k_fold))
        m = PLSRegression(n_components=k_fold, scale=False)
        m.fit(Xp[tr], y[tr])
        y_oof[va] = m.predict(Xp[va]).reshape(-1)
    metrics_cv = _metrics(y, y_oof)

    # Component sweep for CV RMSE/R2.
    sweep_rows = []
    max_sweep = int(max(1, min(20, Xp.shape[1], len(y) - 2)))
    for c in range(1, max_sweep + 1):
        oof = np.zeros_like(y, dtype=float)
        for tr, va in kf.split(Xp):
            c_fold = int(min(c, max(1, min(len(tr) - 1, Xp.shape[1]))))
            m = PLSRegression(n_components=c_fold, scale=False)
            m.fit(Xp[tr], y[tr])
            oof[va] = m.predict(Xp[va]).reshape(-1)
        sweep_rows.append(
            {
                "components": c,
                "cv_rmse": rmse(y, oof),
                "cv_r2": r2(y, oof),
            }
        )

    pred_train = np.array(["train"] * len(y_train), dtype=object)
    pred_test = np.array(["test"] * len(y_test), dtype=object)
    pred_oof = np.array(["cv_oof"] * len(y), dtype=object)

    pred_df = np.concatenate(
        [
            np.column_stack([idx_train, pred_train, y_train, y_pred_train, y_train - y_pred_train]),
            np.column_stack([idx_test, pred_test, y_test, y_pred_test, y_test - y_pred_test]),
            np.column_stack([np.arange(len(y)), pred_oof, y, y_oof, y - y_oof]),
        ],
        axis=0,
    )

    pred_df = np.asarray(pred_df, dtype=object)
    pred_table = {
        "index": pred_df[:, 0].astype(int),
        "set": pred_df[:, 1].astype(str),
        "actual": pred_df[:, 2].astype(float),
        "predicted": pred_df[:, 3].astype(float),
        "residual": pred_df[:, 4].astype(float),
    }

    return {
        "model": model,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "metrics_cv": metrics_cv,
        "pred_df": pred_table,
        "component_sweep_df": sweep_rows,
        "meta": {
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "n_components": int(k),
            "preprocess": preprocess,
            "test_size": float(test_size),
            "cv_folds": int(cv_folds),
            "random_state": int(random_state),
            "feature_names": list(feature_names),
        },
    }
