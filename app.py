"""Chemometric Workbench (RUO) - Streamlit app.

V1: data import, preprocessing, PCA explorer (scree / score / loading plots),
and multivariate outlier detection (Hotelling's T-squared vs Q-residual).

Research-use-only. Not a validated or regulatory tool; analyst review required.
"""
from __future__ import annotations

import io
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from chemometrics_core import PREPROCESS_OPTIONS, run_pca, run_pls_regression
from sample_data import make_spectral_dataset

st.set_page_config(page_title="Chemometric Workbench (RUO)", layout="wide")

st.title("Chemometric Workbench (RUO)")
st.caption(
    "Research/training utility for multivariate analysis of analytical data: "
    "PCA exploration and outlier diagnostics. Not EPA/ISO certified, not for "
    "regulatory submission, and independent of any governed reproducibility release. "
    "All flags are suggested interpretations requiring analyst review."
)

HELPER_COLS = {"_injected_outlier"}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_dataframe() -> pd.DataFrame | None:
    st.sidebar.header("1. Data")
    source = st.sidebar.radio(
        "Data source",
        ["Synthetic spectral demo", "Upload CSV"],
        help="The demo generates NIR-like spectra with injected outliers.",
    )
    if source == "Synthetic spectral demo":
        n = st.sidebar.slider("Samples", 40, 300, 120, 10)
        seed = st.sidebar.number_input("Random seed", value=7, step=1)
        return make_spectral_dataset(n_samples=int(n), seed=int(seed))

    upload = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if upload is None:
        st.info("Upload a CSV, or switch to the synthetic spectral demo in the sidebar.")
        return None
    return pd.read_csv(upload)


df = load_dataframe()
if df is None:
    st.stop()

numeric_cols = [c for c in df.columns if c not in HELPER_COLS and pd.api.types.is_numeric_dtype(df[c])]
meta_candidates = [c for c in df.columns if c not in numeric_cols or c in {"target"}]

st.sidebar.header("2. Columns")
default_features = [c for c in numeric_cols if c != "target"]
feature_cols = st.sidebar.multiselect(
    "Feature columns (X)",
    options=numeric_cols,
    default=default_features,
    help="Numeric variables used to build the PCA model.",
)
color_options = ["(none)"] + [c for c in df.columns if c not in feature_cols]
_preferred = next((c for c in ("group", "_injected_outlier", "sample_id") if c in color_options), None)
color_by = st.sidebar.selectbox(
    "Colour points by",
    color_options,
    index=color_options.index(_preferred) if _preferred else 0,
)

st.sidebar.header("3. Model")
preprocess = st.sidebar.selectbox("Preprocessing", PREPROCESS_OPTIONS, index=PREPROCESS_OPTIONS.index("autoscale"))
max_pc = max(2, min(len(feature_cols), len(df) - 1))
n_components = st.sidebar.slider("PCA components", 2, int(min(10, max_pc)), 2)
confidence = st.sidebar.select_slider("Confidence limit", options=[0.90, 0.95, 0.99], value=0.95)

if len(feature_cols) < 2:
    st.warning("Select at least two feature columns to run PCA.")
    st.stop()
if len(df) < 3:
    st.warning("Need at least three samples.")
    st.stop()

X = df[feature_cols].to_numpy(dtype=float)
if not np.isfinite(X).all():
    st.warning("Feature matrix contains missing/non-finite values; filling with column means.")
    col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_mean, inds[1])

result = run_pca(X, n_components=n_components, preprocess=preprocess, confidence=confidence, feature_names=feature_cols)

tab_data, tab_pca, tab_out, tab_pls, tab_about = st.tabs(
    ["Data", "PCA explorer", "Outlier detection", "PLS regression", "About"]
)

# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", df.shape[0])
    c2.metric("Feature columns", len(feature_cols))
    c3.metric("Preprocessing", preprocess)
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(f"Showing first {min(50, len(df))} of {len(df)} rows.")

# --------------------------------------------------------------------------- #
with tab_pca:
    st.subheader("Explained variance (scree)")
    evr = result.explained_variance_ratio
    scree = pd.DataFrame(
        {
            "PC": [f"PC{i+1}" for i in range(len(evr))],
            "explained_variance": evr,
            "cumulative": np.cumsum(evr),
        }
    )
    fig_scree = go.Figure()
    fig_scree.add_bar(x=scree["PC"], y=scree["explained_variance"], name="Explained")
    fig_scree.add_scatter(x=scree["PC"], y=scree["cumulative"], name="Cumulative", mode="lines+markers")
    fig_scree.update_layout(yaxis_title="Variance ratio", height=360, legend_orientation="h")
    st.plotly_chart(fig_scree, use_container_width=True)

    st.subheader("Score plot")
    cc1, cc2 = st.columns(2)
    pc_x = cc1.selectbox("X axis", [f"PC{i+1}" for i in range(result.n_components)], index=0)
    pc_y = cc2.selectbox("Y axis", [f"PC{i+1}" for i in range(result.n_components)], index=min(1, result.n_components - 1))
    ix, iy = int(pc_x[2:]) - 1, int(pc_y[2:]) - 1
    score_df = pd.DataFrame({pc_x: result.scores[:, ix], pc_y: result.scores[:, iy]})
    if color_by != "(none)":
        score_df[color_by] = df[color_by].values
    score_df["outlier"] = np.where(result.outlier_mask(), "outlier", "in-model")
    fig_score = px.scatter(
        score_df,
        x=pc_x,
        y=pc_y,
        color=color_by if color_by != "(none)" else "outlier",
        symbol="outlier",
        hover_data=score_df.columns,
        height=480,
    )
    fig_score.add_hline(y=0, line_dash="dot", line_color="grey")
    fig_score.add_vline(x=0, line_dash="dot", line_color="grey")
    st.plotly_chart(fig_score, use_container_width=True)

    st.subheader("Loading plot")
    show_pcs = st.multiselect(
        "Components to show", [f"PC{i+1}" for i in range(result.n_components)], default=[pc_x, pc_y]
    )
    fig_load = go.Figure()
    feat_axis = list(range(len(feature_cols)))
    spectral_like = len(feature_cols) > 30
    for pc in show_pcs:
        k = int(pc[2:]) - 1
        fig_load.add_scatter(
            x=feat_axis,
            y=result.loadings[k, :],
            mode="lines" if spectral_like else "lines+markers",
            name=pc,
        )
    tickvals = feat_axis if not spectral_like else None
    fig_load.update_layout(
        height=400,
        xaxis_title="feature",
        yaxis_title="loading",
        xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=feature_cols) if tickvals else {},
    )
    st.plotly_chart(fig_load, use_container_width=True)

# --------------------------------------------------------------------------- #
with tab_out:
    st.subheader("Hotelling's T\u00b2 vs Q-residual (influence plot)")
    st.caption(
        "Top-right quadrant = strong outliers. High T\u00b2 = extreme but consistent with the model; "
        f"high Q = poor fit to the model. Limits at {int(confidence*100)}% confidence."
    )
    influence = pd.DataFrame(result.flag_table())
    label_col = "sample_id" if "sample_id" in df.columns else None
    influence["label"] = df[label_col].values if label_col else influence["index"].astype(str)
    fig_inf = px.scatter(
        influence,
        x="T2",
        y="Q",
        color="verdict",
        hover_name="label",
        height=480,
        labels={"T2": "Hotelling's T\u00b2", "Q": "Q-residual (SPE)"},
    )
    fig_inf.add_vline(x=result.t2_limit, line_dash="dash", line_color="red")
    fig_inf.add_hline(y=result.q_limit, line_dash="dash", line_color="red")
    st.plotly_chart(fig_inf, use_container_width=True)

    flagged = influence[influence["verdict"] != "in-model"].copy()
    cc1, cc2 = st.columns(2)
    cc1.metric("Flagged samples", len(flagged))
    cc2.metric("Total samples", len(influence))
    st.markdown("**Flagged samples (suggested review)**")
    show = flagged if len(flagged) else influence
    cols = ["label", "T2", "T2_limit", "Q", "Q_limit", "verdict"]
    st.dataframe(show[cols].round(4), use_container_width=True)

    csv_buf = io.StringIO()
    influence_out = influence.copy()
    influence_out.to_csv(csv_buf, index=False)
    st.download_button(
        "Download diagnostics (CSV)",
        csv_buf.getvalue(),
        file_name="pca_outlier_diagnostics.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------- #
with tab_pls:
    st.subheader("PLS regression")
    st.caption(
        "Train/test/CV PLS workflow for quantitative targets (RUO). "
        "Outputs are suggested decision support; analyst review required."
    )

    numeric_targets = [c for c in numeric_cols if c not in feature_cols] + [c for c in feature_cols if c == "target"]
    numeric_targets = list(dict.fromkeys(numeric_targets))
    if not numeric_targets:
        st.info("No numeric target column is available. Add a numeric target to run PLS.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        target_col = c1.selectbox("Target column (y)", options=numeric_targets, index=0)
        test_size = c2.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
        cv_folds = c3.slider("CV folds", 3, 10, 5, 1)
        random_state = c4.number_input("Random state", value=42, step=1)

        max_pls = max(1, min(20, len(feature_cols), len(df) - 2))
        n_comp_pls = st.slider("PLS components", 1, int(max_pls), min(3, int(max_pls)))

        y = df[target_col].to_numpy(dtype=float)
        valid_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        dropped = int((~valid_mask).sum())
        X_pls = X[valid_mask]
        y_pls = y[valid_mask]
        idx_pls = np.where(valid_mask)[0]
        label_series = (
            df.loc[valid_mask, "sample_id"].astype(str).values
            if "sample_id" in df.columns
            else idx_pls.astype(str)
        )

        if dropped > 0:
            st.warning(f"Dropped {dropped} rows with missing/non-finite X or y values for PLS.")
        if len(y_pls) < 8:
            st.warning("Need at least 8 valid rows to run PLS train/test + CV.")
        else:
            pls = run_pls_regression(
                X=X_pls,
                y=y_pls,
                feature_names=feature_cols,
                preprocess=preprocess,
                n_components=int(n_comp_pls),
                test_size=float(test_size),
                cv_folds=int(cv_folds),
                random_state=int(random_state),
            )

            mtrain = pls["metrics_train"]
            mtest = pls["metrics_test"]
            mcv = pls["metrics_cv"]

            mt1, mt2, mt3, mt4 = st.columns(4)
            mt1.metric("Test RMSE", f"{mtest['rmse']:.4g}")
            mt2.metric("Test R²", f"{mtest['r2']:.4f}")
            mt3.metric("CV RMSE", f"{mcv['rmse']:.4g}")
            mt4.metric("CV R²", f"{mcv['r2']:.4f}")

            st.caption(
                f"Train RMSE={mtrain['rmse']:.4g}, Train R²={mtrain['r2']:.4f}, "
                f"MAE(test)={mtest['mae']:.4g}"
            )

            pred = pd.DataFrame(pls["pred_df"])
            pred["sample_id"] = label_series[pred["index"].to_numpy()]

            chart_df = pred[pred["set"].isin(["train", "test"])].copy()
            fig_pa = px.scatter(
                chart_df,
                x="actual",
                y="predicted",
                color="set",
                hover_name="sample_id",
                height=430,
                labels={"actual": "Actual", "predicted": "Predicted"},
            )
            if not chart_df.empty:
                mn = float(min(chart_df["actual"].min(), chart_df["predicted"].min()))
                mx = float(max(chart_df["actual"].max(), chart_df["predicted"].max()))
                fig_pa.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx, line=dict(dash="dash", color="gray"))
            st.plotly_chart(fig_pa, use_container_width=True)

            fig_res = px.scatter(
                chart_df,
                x="predicted",
                y="residual",
                color="set",
                hover_name="sample_id",
                height=380,
                labels={"predicted": "Predicted", "residual": "Residual (actual - predicted)"},
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_res, use_container_width=True)

            sweep = pd.DataFrame(pls["component_sweep_df"])
            fig_sweep = px.line(
                sweep,
                x="components",
                y="cv_rmse",
                markers=True,
                height=320,
                labels={"components": "PLS components", "cv_rmse": "CV RMSE"},
            )
            fig_sweep.add_vline(x=int(n_comp_pls), line_dash="dot", line_color="gray")
            st.plotly_chart(fig_sweep, use_container_width=True)

            st.markdown("**Predictions table**")
            st.dataframe(
                pred.sort_values(["set", "index"])[["sample_id", "set", "actual", "predicted", "residual"]].round(5),
                use_container_width=True,
            )

            csv_buf = io.StringIO()
            pred.to_csv(csv_buf, index=False)
            st.download_button(
                "Download PLS predictions (CSV)",
                csv_buf.getvalue(),
                file_name="pls_predictions.csv",
                mime="text/csv",
            )

            model_payload = {
                "model": pls["model"],
                "meta": pls["meta"],
                "target_col": target_col,
            }
            st.download_button(
                "Download PLS model (PKL)",
                data=pickle.dumps(model_payload),
                file_name="pls_model.pkl",
                mime="application/octet-stream",
            )

            st.markdown("### Model explainability (PLS coefficient influence)")
            inf = pd.DataFrame(pls["influence_table"])
            top_n = st.slider("Top influential features", 5, 40, 20, 1, key="pls_topn")
            inf_top = inf.head(top_n).copy()
            fig_inf = px.bar(
                inf_top.iloc[::-1],
                x="std_abs_coef",
                y="feature",
                orientation="h",
                height=420,
                labels={"std_abs_coef": "Standardized |coefficient|", "feature": "Feature"},
            )
            st.plotly_chart(fig_inf, use_container_width=True)
            buf_inf = io.StringIO()
            inf.to_csv(buf_inf, index=False)
            st.download_button(
                "Download feature influence (CSV)",
                buf_inf.getvalue(),
                file_name="pls_feature_influence.csv",
                mime="text/csv",
            )

            st.markdown("### Applicability domain (AD)")
            ad = pd.DataFrame(pls["ad_table"])
            ad["sample_id"] = label_series[ad["index"].to_numpy()]
            ad["domain_flag"] = np.where(ad["in_domain"], "in_domain", "out_of_domain")
            ad1, ad2 = st.columns(2)
            ad1.metric("In-domain", int(ad["in_domain"].sum()))
            ad2.metric("Out-of-domain", int((~ad["in_domain"]).sum()))
            fig_ad = px.scatter(
                ad,
                x="mahalanobis",
                y="leverage",
                color="domain_flag",
                hover_name="sample_id",
                height=420,
            )
            fig_ad.add_vline(x=float(ad["mahal_limit"].iloc[0]), line_dash="dash", line_color="red")
            fig_ad.add_hline(y=float(ad["lev_limit"].iloc[0]), line_dash="dash", line_color="red")
            st.plotly_chart(fig_ad, use_container_width=True)

            st.markdown("### Uncertainty and review flags")
            u = pd.DataFrame(pls["uncertainty_table"])
            u["sample_id"] = label_series[u["index"].to_numpy()]
            vc = u["uncertainty_level"].value_counts()
            u1, u2, u3, u4 = st.columns(4)
            u1.metric("LOW", int(vc.get("LOW", 0)))
            u2.metric("MEDIUM", int(vc.get("MEDIUM", 0)))
            u3.metric("HIGH", int(vc.get("HIGH", 0)))
            u4.metric("Review required", int(u["review_required"].sum()))
            fig_u = px.histogram(
                u,
                x="uncertainty_level",
                color="uncertainty_level",
                category_orders={"uncertainty_level": ["LOW", "MEDIUM", "HIGH"]},
                height=300,
            )
            st.plotly_chart(fig_u, use_container_width=True)

            st.markdown("### Consolidated export")
            pred_export = pred.merge(
                ad[["index", "mahalanobis", "mahal_limit", "leverage", "lev_limit", "in_domain"]],
                on="index",
                how="left",
            ).merge(
                u[["index", "pi_half_width", "uncertainty_level", "review_required"]],
                on="index",
                how="left",
            )
            buf_all = io.StringIO()
            pred_export.to_csv(buf_all, index=False)
            st.download_button(
                "Download PLS predictions + AD + uncertainty (CSV)",
                buf_all.getvalue(),
                file_name="pls_predictions_with_ad_uncertainty.csv",
                mime="text/csv",
            )

# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("About this tool")
    st.markdown(
        """
**Chemometric Workbench (RUO)** is a research/training utility for multivariate
analysis of analytical chemistry data.

**Current capabilities**
- CSV import or synthetic spectral demo data
- Preprocessing: mean-centering, autoscaling, SNV, MSC, Savitzky-Golay (smooth / 1st derivative)
- PCA: scree, score, and loading plots
- Multivariate outlier detection: Hotelling's T\u00b2 and Q-residual (SPE) with
  statistically derived control limits (F-distribution and Jackson-Mudholkar)
- PLS regression: train/test/CV metrics (RMSE, MAE, R\u00b2), predicted vs actual,
  residual plots, component sweep, and model/prediction export
- Explainability + confidence: coefficient-based feature influence, applicability
  domain checks (Mahalanobis + leverage), uncertainty levels, and review flags

**Roadmap:** optional SHAP panel, batch anomaly detection, and an LC-MS/MS QA
review module.

**Governance:** Research-use-only. Not EPA/ISO certified, not validated for
regulatory submission, and intentionally separate from any governed
reproducibility/evidence release. All outputs require qualified analyst review.
        """
    )
