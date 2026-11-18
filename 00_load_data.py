"""Read the WoRLD workbook once and cache it as a pickle."""
import pandas as pd
from config import WORLD_XLSX, WORLD_PKL

if __name__ == "__main__":
    if not WORLD_XLSX.exists():
        raise SystemExit(f"Missing {WORLD_XLSX}. Download WoRLD 2026 from the IMF data portal.")
    df = pd.read_excel(WORLD_XLSX, sheet_name="Data")
    df.to_pickle(WORLD_PKL)
    print("Cached", df.shape, "to", WORLD_PKL)
