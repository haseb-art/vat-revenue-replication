"""Heterogeneity-robust event study (Callaway & Sant'Anna) with country-level bootstrap SEs."""
import pickle
import sys
import numpy as np
from common import load_panel, adoption_map, cs_estimate, YEARS
from config import BOOT_REPS, SEED, CS_RESULTS

df = load_panel()
adopt = adoption_map(df)
isos = sorted(adopt)
never = [i for i in isos if adopt[i] is None]
treated = {i: adopt[i] for i in isos if isinstance(adopt[i], int)}
tr = np.array([isos.index(i) for i in treated])
tc = np.array([treated[i] for i in treated])
ct = np.array([isos.index(i) for i in never])
print(f"Adopters: {len(treated)}  Never-adopters: {len(never)}  Bootstrap reps: {BOOT_REPS}", flush=True)

rng = np.random.default_rng(SEED)
res = {}
for var in ["TaxRev", "TaxSal", "trade_sh", "TotRev"]:
    P = df.pivot_table(index="iso", columns="yr", values=var, aggfunc="first").reindex(index=isos, columns=YEARS)
    Y = P.values
    est = cs_estimate(Y, tr, tc, ct)
    boots = {e: [] for e in est}
    for _ in range(BOOT_REPS):
        st = rng.choice(np.arange(len(tr)), len(tr))
        sn = rng.choice(np.arange(len(ct)), len(ct))
        r = cs_estimate(Y, tr[st], tc[st], ct[sn])
        for e in est:
            boots[e].append(r[e])
    se = {e: np.nanstd(boots[e]) for e in est}
    res[var] = (est, se)
    print(f"\nCS event study: {var}", flush=True)
    for e in sorted(est):
        print(f"  e={e:3d}  ATT={est[e]:7.3f}  se={se[e]:.3f}", flush=True)

with open(CS_RESULTS, "wb") as f:
    pickle.dump(res, f)
print("\nSaved", CS_RESULTS)
