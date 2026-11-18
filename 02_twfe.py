"""Two-way fixed-effects estimates and robustness checks (Table 3 TWFE rows, Table 4)."""
import pandas as pd
from common import load_panel, twfe

df = load_panel()
full = df[df.yr.between(1980, 2024)]


def show(title, res):
    r, N, G = res
    print(f"\n{title}  N={N}  countries={G}")
    for k, row in r.iterrows():
        print(f"  {k:10s} {row.coef:8.3f}  ({row.se:.3f})  t={row.t:6.2f}")


show("(A) Tax revenue", twfe(full, "TaxRev", ["vat"]))
show("(B) Total revenue", twfe(full, "TotRev", ["vat"]))
show("(C) Trade tax", twfe(full, "TaxTra", ["vat"]))
show("(D) Income tax", twfe(full, "TaxInc", ["vat"]))
show("(E) Sales and production tax", twfe(full, "TaxSal", ["vat"]))
show("(F) Trade-tax share (pp)", twfe(full, "trade_sh", ["vat"]))
show("(G) Income-tax share (pp)", twfe(full, "inc_sh", ["vat"]))
show("(I) Heterogeneity by income group", twfe(
    full.assign(vatL=full.vat * full.LIDC, vatE=full.vat * full.EME), "TaxRev", ["vat", "vatL", "vatE"]))

print("\nRobustness, tax revenue")
show("Raw VAT coding", twfe(full.assign(vat=full.vat_raw), "TaxRev", ["vat"]))
show("1990-2024", twfe(full[full.yr >= 1990], "TaxRev", ["vat"]))
show("2000-2023", twfe(full[full.yr.between(2000, 2023)], "TaxRev", ["vat"]))
show("Excluding resource-rich", twfe(full[full.resource == 0], "TaxRev", ["vat"]))
sub = full[full.yr.between(1990, 2020)]
ok = sub.groupby("iso").apply(lambda g: g.TaxRev.notna().all() and g.TotRev.notna().all(), include_groups=False)
bal = ok[ok].index
show("Balanced 1990-2020", twfe(full[full.iso.isin(bal) & full.yr.between(1990, 2020)], "TaxRev", ["vat"]))
w = full.copy()
for v in ["TaxRev", "TotRev"]:
    lo, hi = w[v].quantile([0.01, 0.99])
    w[v] = w[v].clip(lo, hi)
show("Winsorized 1/99, tax revenue", twfe(w, "TaxRev", ["vat"]))
