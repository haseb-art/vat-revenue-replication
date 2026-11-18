"""Joint country bootstrap for the group-time event study.

One set of bootstrap draws resamples countries jointly (adopters and never-adopters together), and every
estimate in Tables 3, 4A and 4B and in Figure 5 uses it. Estimates use the Callaway & Sant'Anna (2021)
group-time comparison with never-adopters ('never') or not-yet-treated countries ('nyt') as controls, and
subgroup estimates for low-income ('lidc') and other ('nonlidc') adopters.

Outputs:
  out/joint_bootstrap_full.pkl  all specifications, with bootstrap draws (for 06_relative_magnitudes.py)
  out/joint_bootstrap.pkl       {variable: (estimates, standard errors)} for never-adopters (for Figure 5)
"""
import pickle
import warnings
import numpy as np
import pandas as pd
from common import load_panel, YEARS
from config import BOOT_REPS, SEED, OUT_DIR

warnings.filterwarnings("ignore")
EVS = list(range(-5, 9))
YI = {y: i for i, y in enumerate(YEARS)}

df = load_panel()
adopt, lidc = {}, {}
for iso, g in df.groupby("iso"):
    v = g.set_index("yr").vat_raw.reindex(YEARS).fillna(0)
    ones = v[v == 1].index
    lidc[iso] = int(g.LIDC.iloc[0])
    if len(ones) == 0:
        adopt[iso] = np.inf                      # never-adopter
        continue
    y0 = int(ones.min())
    if y0 == 1980:
        adopt[iso] = None                        # VAT in 1980: excluded from adoption comparison
    elif (y0 - 1) in v.index and v[y0 - 1] == 0:
        adopt[iso] = y0
    else:
        adopt[iso] = None
units = [i for i in sorted(adopt) if adopt[i] is not None]
ay = np.array([adopt[i] for i in units], dtype=float)
lid = np.array([lidc[i] for i in units])
print(f"Units: {len(units)} (adopters {np.isfinite(ay).sum()}, never-adopters {np.isinf(ay).sum()}); reps {BOOT_REPS}", flush=True)


def att(Y, ay, lid, mode, tmask=None):
    """Group-time ATT by event time, cohort-size weighted."""
    out = {}
    coh = np.unique(ay[np.isfinite(ay)])
    for e in EVS:
        num = den = 0.0
        for c in coh:
            t, b = int(c + e), int(c - 1)
            if not (1980 <= t <= 2024 and b >= 1980):
                continue
            ti, bi = YI[t], YI[b]
            ok = ~np.isnan(Y[:, ti]) & ~np.isnan(Y[:, bi])
            mem = ok & (ay == c)
            if tmask is not None:
                mem &= tmask
            if mode == "never":
                ctl = ok & np.isinf(ay)
            else:  # not-yet-treated plus never-treated, excluding the cohort itself
                ctl = ok & ((np.isinf(ay)) | (ay > t)) & (ay != c)
            if mem.sum() == 0 or ctl.sum() < 5:
                continue
            dT = np.mean(Y[mem, ti] - Y[mem, bi])
            dC = np.mean(Y[ctl, ti] - Y[ctl, bi])
            num += mem.sum() * (dT - dC)
            den += mem.sum()
        out[e] = num / den if den > 0 else np.nan
    return out


def panel(var):
    P = df.pivot_table(index="iso", columns="yr", values=var, aggfunc="first").reindex(index=units, columns=YEARS)
    return P.values


rng = np.random.default_rng(SEED)
res = {}
for var in ["TaxRev", "TaxSal", "trade_sh", "TotRev"]:
    Y = panel(var)
    spec = {
        "never": dict(mode="never"),
        "nyt": dict(mode="nyt"),
        "lidc": dict(mode="never", group=1),
        "nonlidc": dict(mode="never", group=0),
    }
    est = {k: att(Y, ay, lid, v["mode"], None if "group" not in v else (lid == v["group"])) for k, v in spec.items()}
    draws = {k: {e: [] for e in EVS} for k in spec}
    for _ in range(BOOT_REPS):
        idx = rng.choice(len(units), len(units))
        Yb, ayb, lb = Y[idx], ay[idx], lid[idx]
        for k, v in spec.items():
            tm = None if "group" not in v else (lb == v["group"])
            r = att(Yb, ayb, lb, v["mode"], tm)
            for e in EVS:
                draws[k][e].append(r[e])
    se = {k: {e: float(np.nanstd(draws[k][e])) for e in EVS} for k in spec}
    res[var] = {
        "est": est,
        "se": se,
        # shape (BOOT_REPS, len(EVS)); column j corresponds to event time EVS[j]
        "draws": {k: np.array([draws[k][e] for e in EVS]).T for k in spec},
        "evs": EVS,
    }
    print(f"\n{var}", flush=True)
    for e in [0, 1, 2, 3, 4]:
        print(f"  e={e}: never {est['never'][e]:.3f} ({se['never'][e]:.3f}); nyt {est['nyt'][e]:.3f}", flush=True)

with open(OUT_DIR / "joint_bootstrap_full.pkl", "wb") as f:
    pickle.dump(res, f)
with open(OUT_DIR / "joint_bootstrap.pkl", "wb") as f:
    pickle.dump({v: (res[v]["est"]["never"], res[v]["se"]["never"]) for v in res}, f)
print("\nSaved joint bootstrap outputs to", OUT_DIR, flush=True)
