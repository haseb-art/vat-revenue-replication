"""Paths and settings shared by all scripts. Paths are relative to the repository root."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
WORLD_XLSX = DATA_DIR / "world-imf2026.xlsx"
WORLD_PKL = DATA_DIR / "world.pkl"
CS_RESULTS = OUT_DIR / "cs_results.pkl"
BOOT_REPS = int(os.environ.get("BOOT_REPS", "300"))
SEED = 42

OUT_DIR.mkdir(exist_ok=True)
