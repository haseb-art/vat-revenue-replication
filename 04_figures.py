"""Figures 1–6 for the manuscript."""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_panel
from config import OUT_DIR, CS_RESULTS

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
df = load_panel()
df["g"] = df.Incomegroup.map(lambda s: "AE" if "Advanced" in s else "EME" if "Emerging" in s else "LIDC")
C = {"AE": "#1f4e79", "EME": "#c55a11", "LIDC": "#548235"}
G = {"AE": "Advanced economy", "EME": "Emerging market", "LIDC": "Low-income developing"}

# Figure 1
fig, ax = plt.subplots(figsize=(7, 4.2))
for k in ["AE", "EME", "LIDC"]:
    s = df[(df.g == k) & df.yr.between(1980, 2024)].groupby("yr").TotRev.mean()
    ax.plot(s.index, s.values, color=C[k], lw=2, label=G[k])
ax.set_title("Figure 1. Mean total revenue by income group, % of GDP")
ax.set_xlabel("Year"); ax.set_ylabel("% of GDP"); ax.legend(frameon=False); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT_DIR / "fig1_totrev_by_group.png"); plt.close(fig)

# Figure 2
fig, ax = plt.subplots(figsize=(7, 4.2))
d2 = df[df.yr.between(1990, 2024)].dropna(subset=["trade_sh"])
s = d2.groupby("yr").trade_sh.median(); n = d2.groupby("yr").size()
ax.plot(s.index, s.values, color="#7f1d1d", lw=2)
for y in [1990, 2000, 2010, 2020, 2024]:
    ax.annotate(f"n={n[y]}", (y, s[y]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
ax.set_title("Figure 2. Median trade-tax share of tax revenue, 1990–2024")
ax.set_xlabel("Year"); ax.set_ylabel("% of tax revenue"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT_DIR / "fig2_trade_share_median.png"); plt.close(fig)

# Figure 3
rows = []
for y in [2000, 2010, 2020, 2024]:
    for k in ["AE", "EME", "LIDC"]:
        s = df[(df.yr == y) & (df.g == k)].dropna(subset=["TaxRev", "TaxInc", "TaxSal", "TaxTra"])
        s = s[s.TaxRev > 0]
        if len(s):
            rows.append((y, k, s.inc_sh.mean(), s.sal_sh.mean(), s.trade_sh.mean()))
t = pd.DataFrame(rows, columns=["yr", "g", "inc", "sal", "trade"])
fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(t))
ax.bar(x, t.inc, color="#1f4e79", label="Income tax")
ax.bar(x, t.sal, bottom=t.inc, color="#9dc3e6", label="Sales & production (incl. VAT)")
ax.bar(x, t.trade, bottom=t.inc + t.sal, color="#c55a11", label="Trade taxes")
ax.set_xticks(x); ax.set_xticklabels([f"{r.yr}\n{r.g}" for r in t.itertuples()], fontsize=7)
ax.set_ylim(0, 100); ax.set_ylabel("% of tax revenue")
ax.set_title("Figure 3. Tax composition by income group (balanced component sample)")
ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
fig.text(0.01, 0.01, "Remainder to 100% is property and other taxes. Unweighted means.", fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT_DIR / "fig3_composition.png"); plt.close(fig)

# Figure 4
fig, ax = plt.subplots(figsize=(7, 4))
raw = df.groupby("yr").vat_raw.sum(); ab = df.groupby("yr").vat.sum()
ax.plot(raw.index, raw.values, color="#999", lw=2, label="Raw VAT field (as published)")
ax.plot(ab.index, ab.values, color="#1f4e79", lw=2, ls="--", label="Absorbing VAT (used in estimation)")
ax.set_title("Figure 4. Countries with a VAT, 1980–2024"); ax.set_ylabel("Countries")
ax.legend(frameon=False); ax.grid(alpha=.25)
fig.text(0.01, 0.01, "The 2024 drop reflects missing coding for countries that still had a VAT in 2023.", fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT_DIR / "fig4_vat_coding.png"); plt.close(fig)

# Figure 5
with open(CS_RESULTS, "rb") as f:
    res = pickle.load(f)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharex=True)
titles = {"TaxRev": "Tax revenue (% GDP)", "TaxSal": "Sales & production tax (% GDP)",
          "trade_sh": "Trade-tax share (pp)", "TotRev": "Total revenue (% GDP)"}
for ax, v in zip(axes, ["TaxRev", "TaxSal", "trade_sh", "TotRev"]):
    est, se = res[v]
    e = np.array(sorted(est)); b = np.array([est[k] for k in e]); s = np.array([se[k] for k in e])
    ax.axhline(0, color="#999", lw=0.8); ax.axvline(-0.5, color="#999", lw=0.8, ls=":")
    ax.errorbar(e, b, yerr=1.96 * s, fmt="o", ms=3.5, color="#1f4e79", ecolor="#9dc3e6", capsize=2)
    ax.set_title(titles[v], fontsize=9); ax.set_xlabel("Years relative to VAT adoption", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.2)
axes[0].set_ylabel("ATT vs. never-adopters (95% CI)")
fig.suptitle("Figure 5. Heterogeneity-robust event study of VAT adoption (year −1 normalized to zero)", fontsize=10)
fig.text(0.01, 0.01, "Bootstrap standard errors, clustered by country. Never-adopting countries are the comparison group.", fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 0.92)); fig.savefig(OUT_DIR / "fig5_event_study.png", dpi=150); plt.close(fig)

# Figure 6
bgd = df[df.iso == "BGD"].set_index("yr")
peers = df[(df.Region == "Asia and pacific") & (df.g == "LIDC") & df.yr.between(1990, 2024)]
pm = peers.groupby("yr").TotRev.median()
fig, ax = plt.subplots(figsize=(7, 4.2))
b = bgd.loc[1990:2024]
ax.plot(b.index, b.TotRev, color="#7f1d1d", lw=2, label="Bangladesh total revenue")
ax.plot(b.index, b.TaxRev, color="#7f1d1d", lw=1.5, ls=":", label="Bangladesh tax revenue")
ax.plot(pm.index, pm.values, color="#548235", lw=2, label="Asia-Pacific LIDC median (total)")
ax.set_title("Figure 6. Bangladesh and Asia-Pacific low-income peers, % of GDP")
ax.set_xlabel("Year"); ax.set_ylabel("% of GDP"); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT_DIR / "fig6_bangladesh_peers.png"); plt.close(fig)
print("Figures 1-6 written to", OUT_DIR)
