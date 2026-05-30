"""Synthetic analytical datasets for the Chemometric Workbench (RUO).

Generates NIR-like spectra built from a few latent chemical components plus
noise, with deliberately injected Hotelling-T2 and Q-residual outliers so the
diagnostics in the app have something meaningful to find. Fully synthetic;
not real measurement data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _gaussian_band(x: np.ndarray, center: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def make_spectral_dataset(
    n_samples: int = 120,
    n_features: int = 200,
    seed: int = 7,
) -> pd.DataFrame:
    """Return a DataFrame: sample_id, group, target, then wavelength columns.

    - 3 pure-component spectra (Gaussian band sets) mixed by random concentrations
    - 2 latent groups with different mean composition (separable in PCA score space)
    - additive noise + random multiplicative scatter (so SNV/MSC are meaningful)
    - injected outliers: a few Q-outliers (anomalous band) and T2-outliers (extreme score)
    """
    rng = np.random.default_rng(seed)
    wl = np.linspace(1000.0, 2500.0, n_features)  # nm

    pure = np.vstack(
        [
            _gaussian_band(wl, 1200, 60, 1.0) + _gaussian_band(wl, 1900, 90, 0.6),
            _gaussian_band(wl, 1450, 70, 1.0) + _gaussian_band(wl, 2100, 80, 0.5),
            _gaussian_band(wl, 1700, 50, 0.9) + _gaussian_band(wl, 2300, 100, 0.7),
        ]
    )

    groups = rng.integers(0, 2, size=n_samples)
    conc = np.zeros((n_samples, 3))
    for i in range(n_samples):
        if groups[i] == 0:
            conc[i] = rng.dirichlet([6, 2, 1])
        else:
            conc[i] = rng.dirichlet([1, 2, 6])

    spectra = conc @ pure

    # target: a property linearly related to component concentrations (for PLS later)
    target = 10.0 * conc[:, 0] + 4.0 * conc[:, 1] + 25.0 * conc[:, 2]
    target += rng.normal(0, 0.4, size=n_samples)

    # multiplicative scatter + baseline offset + noise
    scatter = rng.normal(1.0, 0.05, size=(n_samples, 1))
    offset = rng.normal(0.0, 0.02, size=(n_samples, 1))
    noise = rng.normal(0.0, 0.01, size=spectra.shape)
    spectra = spectra * scatter + offset + noise

    # inject outliers
    n_q = max(2, n_samples // 40)
    q_idx = rng.choice(n_samples, size=n_q, replace=False)
    for idx in q_idx:
        spike_center = rng.uniform(1300, 2200)
        spectra[idx] += _gaussian_band(wl, spike_center, 25, 0.8)  # anomalous band -> Q outlier

    remaining = np.setdiff1d(np.arange(n_samples), q_idx)
    t2_idx = rng.choice(remaining, size=max(2, n_samples // 50), replace=False)
    for idx in t2_idx:
        spectra[idx] *= 1.9  # extreme but in-subspace magnitude -> T2 outlier

    cols = [f"{w:.0f}nm" for w in wl]
    df = pd.DataFrame(spectra, columns=cols)
    df.insert(0, "target", np.round(target, 3))
    df.insert(0, "group", np.where(groups == 0, "A", "B"))
    df.insert(0, "sample_id", [f"S{i+1:03d}" for i in range(n_samples)])

    injected = np.zeros(n_samples, dtype=object)
    injected[:] = ""
    for idx in q_idx:
        injected[idx] = "Q"
    for idx in t2_idx:
        injected[idx] = "T2"
    df["_injected_outlier"] = injected
    return df


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path(__file__).resolve().parent / "examples"
    out.mkdir(exist_ok=True)
    make_spectral_dataset().to_csv(out / "example_spectra.csv", index=False)
    print("wrote examples/example_spectra.csv")
