"""Panel construction and estimators shared across scripts."""
import numpy as np
import pandas as pd
from config import WORLD_PKL

YEARS = np.arange(1980, 2025)
YEAR_IDX = {y: i for i, y in enumerate(YEARS)}


def load_panel():
    """Return the country-year panel with the absorbing VAT indicator and derived shares."""
    df = pd.read_pickle(WORLD_PKL).rename(columns={"ISO3": "iso", "year": "yr"})
    df["vat_raw"] = (df["SalGType"] == "VAT").astype(int)
    df = df.sort_values(["iso", "yr"])
    df["vat"] = df.groupby("iso")["vat_raw"].cummax()
    df["trade_sh"] = df.TaxTra / df.TaxRev * 100
    df["inc_sh"] = df.TaxInc / df.TaxRev * 100
    df["sal_sh"] = df.TaxSal / df.TaxRev * 100
    df["LIDC"] = df.Incomegroup.str.contains("LIDC").astype(int)
    df["EME"] = df.Incomegroup.str.contains("EME").astype(int)
    df["resource"] = (df.ResourceRich == "Resource-rich").astype(int)
    return df


def adoption_map(df):
    """Map each country to its adoption year, 'always' (VAT in 1980), or None (never)."""
    out = {}
    for iso, g in df.groupby("iso"):
        v = g.set_index("yr").vat_raw.reindex(YEARS).fillna(0)
        ones = v[v == 1].index
        if len(ones) == 0:
            out[iso] = None
            continue
        y0 = int(ones.min())
        if y0 == 1980:
            out[iso] = "always"
        elif (y0 - 1) in v.index and v[y0 - 1] == 0:
            out[iso] = y0
    return out


def _demean(X, c, t, maxit=500, tol=1e-10):
    """Two-way demeaning by alternating projections (unbalanced panels)."""
    X = X.astype(float).copy()
    for _ in range(maxit):
        old = X.copy()
        for f in (c, t):
            X = X - pd.DataFrame(X).groupby(f.values).transform("mean").values
        if np.max(np.abs(X - old)) < tol:
            break
    return X


def twfe(d, y, xs, cluster="iso"):
    """Two-way FE with country-clustered SEs and a small-sample adjustment."""
    d = d.dropna(subset=[y] + xs).copy()
    Z = _demean(np.hstack([d[[y]].values, d[xs].values]), d["iso"], d["yr"])
    yy, XX = Z[:, 0], Z[:, 1:]
    XtX = XX.T @ XX
    b = np.linalg.solve(XtX, XX.T @ yy)
    e = yy - XX @ b
    G = d[cluster].nunique()
    N = len(d)
    K = len(xs) + d.iso.nunique() + d.yr.nunique() - 2
    bread = np.linalg.inv(XtX)
    meat = np.zeros((len(xs), len(xs)))
    for _, idx in d.groupby(cluster).indices.items():
        s = XX[idx].T @ e[idx]
        meat += np.outer(s, s)
    adj = G / (G - 1) * (N - 1) / (N - K)
    V = adj * bread @ meat @ bread
    se = np.sqrt(np.diag(V))
    return pd.DataFrame({"coef": b, "se": se, "t": b / se}, index=xs), N, G


def cs_estimate(Y, treated_idx, treated_cohort, control_idx, evs=range(-5, 9)):
    """Group-time event-study estimator (Callaway & Sant'Anna, never-treated controls).

    Y is a (units x years) array with NaN for missing. Each cohort's change from year g-1
    is compared with the same change for never-adopters, then averaged over cohorts with
    weights equal to cohort size.
    """
    out = {}
    for e in evs:
        num, den = 0.0, 0.0
        for c in np.unique(treated_cohort):
            members = treated_idx[treated_cohort == c]
            t, b = c + e, c - 1
            if not (1980 <= t <= 2024 and b >= 1980):
                continue
            ti, bi = YEAR_IDX[t], YEAR_IDX[b]
            m = members[~np.isnan(Y[members, ti]) & ~np.isnan(Y[members, bi])]
            cc = control_idx[~np.isnan(Y[control_idx, ti]) & ~np.isnan(Y[control_idx, bi])]
            if len(m) == 0 or len(cc) < 5:
                continue
            dT = np.mean(Y[m, ti] - Y[m, bi])
            dC = np.mean(Y[cc, ti] - Y[cc, bi])
            num += len(m) * (dT - dC)
            den += len(m)
        out[e] = num / den if den > 0 else np.nan
    return out
