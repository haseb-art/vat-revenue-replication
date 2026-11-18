"""Relative-magnitudes sensitivity analysis (Rambachan & Roth, 2023) for Table 4B.

Post-adoption violations delta_e are free. Pre-period violations equal the observed pre-adoption
coefficients (tau_pre = 0). The restriction |second difference of delta| <= M * B at every center
t = -1..7 defines the identified set, with B the largest pre-period second difference (delta_{-1} = 0).
For each event time e (0..4) and for the average of e = 0..4, a linear program (HiGHS) gives the range of
tau_e = beta_e - delta_e. The approximate 95% interval widens that range by 1.96 standard errors, taken
from the joint bootstrap. M* is the largest M for which the interval excludes zero (bisection).
These are approximations: they do not use the covariance-based confidence sets of the original paper.
"""
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from config import OUT_DIR

res = pickle.load(open(OUT_DIR / "joint_bootstrap_full.pkl", "rb"))
EVS = list(range(-5, 9))
PRE = [-5, -4, -3, -2]
POST = list(range(0, 9))


def constraints(beta, M, B):
    known = {e: beta[e] for e in PRE}
    known[-1] = 0.0
    A, b = [], []
    for t in range(-1, 8):
        coef = np.zeros(len(POST))
        const = 0.0
        for k, w in [(t + 1, 1.0), (t, -2.0), (t - 1, 1.0)]:
            if k >= 0:
                coef[k] += w
            else:
                const += w * known[k]
        A.append(coef); b.append(M * B - const)
        A.append(-coef); b.append(M * B + const)
    return np.array(A), np.array(b)


def tau_set(beta, M, B, evs):
    """Identified set for the average of tau over event times evs."""
    A, b = constraints(beta, M, B)
    c = np.zeros(len(POST))
    for e in evs:
        c[e] += 1.0 / len(evs)
    lo = linprog(c=c, A_ub=A, b_ub=b, bounds=[(None, None)] * len(POST), method="highs")
    hi = linprog(c=-c, A_ub=A, b_ub=b, bounds=[(None, None)] * len(POST), method="highs")
    if lo.status != 0 or hi.status != 0:
        raise RuntimeError("LP infeasible")
    base = np.mean([beta[e] for e in evs])
    # tau_avg = base - delta_avg; delta_avg ranges over [lo.fun, -hi.fun]
    return base + hi.fun, base - lo.fun


def excludes_zero(beta, M, B, evs, se):
    tlo, thi = tau_set(beta, M, B, evs)
    return (tlo - 1.96 * se) > 0 or (thi + 1.96 * se) < 0


def breakdown(beta, B, evs, se):
    if not excludes_zero(beta, 0.0, B, evs, se):
        return 0.0
    lo, hi = 0.0, 20.0
    if excludes_zero(beta, hi, B, evs, se):
        return np.inf
    for _ in range(50):
        mid = (lo + hi) / 2
        if excludes_zero(beta, mid, B, evs, se):
            lo = mid
        else:
            hi = mid
    return lo


rows = []
for var, label in [("TaxRev", "Tax revenue"), ("TaxSal", "Sales & production tax"),
                   ("trade_sh", "Trade-tax share"), ("TotRev", "Total revenue")]:
    r = res[var]
    est, se = r["est"]["never"], r["se"]["never"]
    beta = {e: float(est[e]) for e in EVS}
    beta[-1] = 0.0
    B = max(abs(beta[t + 1] - 2 * beta[t] + beta[t - 1]) for t in [-4, -3, -2])
    # average over e = 0..4: standard error from the bootstrap draws of that average.
    # draws columns follow EVS, so event time e is column EVS.index(e).
    cols = [EVS.index(e) for e in range(0, 5)]
    avg_draws = r["draws"]["never"][:, cols].mean(axis=1)
    se_avg = float(np.nanstd(avg_draws, ddof=1))
    for label_e, evs, s in [(str(e), [e], float(se[e])) for e in range(0, 5)] + [("Average 0-4", list(range(0, 5)), se_avg)]:
        base = float(np.mean([beta[e] for e in evs]))
        tlo0, thi0 = tau_set(beta, 0.0, B, evs)
        mstar = breakdown(beta, B, evs, s)
        rows.append(dict(outcome=label, event_time=label_e, estimate=round(base, 2), se=round(s, 2),
                         benchmark_B=round(B, 2), tau_M0=round(tlo0, 2), breakdown_M=round(mstar, 2)))

tab = pd.DataFrame(rows)
tab.to_csv(OUT_DIR / "06_relative_magnitudes.csv", index=False)
with open(OUT_DIR / "06_relative_magnitudes.txt", "w") as f:
    f.write(tab.to_string(index=False) + "\n")
print(tab.to_string(index=False))
