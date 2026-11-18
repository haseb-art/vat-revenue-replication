"""Descriptive statistics: revenue by income group, trade-tax share, composition, VAT counts, Bangladesh."""
import numpy as np
import pandas as pd
from common import load_panel

df = load_panel()
inc = {"AE": "Advanced", "EME": "Emerging", "LIDC": "Low income"}
df["g"] = df.Incomegroup.map(lambda s: "AE" if "Advanced" in s else "EME" if "Emerging" in s else "LIDC")

print("Mean TotRev and TaxRev by income group (unweighted)")
for y in [1990, 2000, 2010, 2020, 2024]:
    sub = df[df.yr == y]
    parts = []
    for k in ["AE", "EME", "LIDC"]:
        s = sub[sub.g == k]
        parts.append(f"{k}: TotRev {s.TotRev.mean():.2f} (n={s.TotRev.notna().sum()}), "
                     f"TaxRev {s.TaxRev.mean():.2f} (n={s.TaxRev.notna().sum()})")
    print(y, " | ".join(parts))

print("\nMedian trade-tax share of tax revenue")
for y in [1990, 2000, 2010, 2020, 2024]:
    s = df[(df.yr == y)].dropna(subset=["trade_sh"])
    print(y, round(s.trade_sh.median(), 1), "n =", len(s))

print("\nTax composition, balanced component sample")
for y in [2000, 2010, 2020, 2024]:
    for k in ["AE", "EME", "LIDC"]:
        s = df[(df.yr == y) & (df.g == k)].dropna(subset=["TaxRev", "TaxInc", "TaxSal", "TaxTra"])
        s = s[s.TaxRev > 0]
        if len(s):
            print(y, k, "n =", len(s), "income", round(s.inc_sh.mean(), 1),
                  "sales", round(s.sal_sh.mean(), 1), "trade", round(s.trade_sh.mean(), 1))

print("\nVAT count: raw vs absorbing")
for y in [1980, 1990, 2000, 2010, 2020, 2024]:
    s = df[df.yr == y]
    print(y, "raw", int(s.vat_raw.sum()), "absorbing", int(s.vat.sum()))

bgd = df[(df.iso == "BGD") & df.yr.isin([1990, 2000, 2010, 2020, 2024])]
print("\nBangladesh")
print(bgd[["yr", "TotRev", "TaxRev", "TaxInc", "TaxSal", "TaxTra", "Grants"]].round(2).to_string(index=False))
peers = df[(df.yr == 2020) & (df.Region == "Asia and pacific") & (df.g == "LIDC")].TotRev.dropna()
print("Asia-Pacific LIDC median TotRev 2020:", round(peers.median(), 2), "n =", len(peers))
