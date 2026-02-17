# FILE: api/app/features/tennis_training_combined.py
"""
Build a combined training CSV by merging simple rolling features with TA rolling features.

Inputs:
  - tennis_training_simple.csv (from tennis_training_simple.py)
  - tennis_training_ta.csv     (from tennis_training_dataset_ta.py)
Output:
  - tennis_training_combined.csv
"""

from __future__ import annotations
import argparse
import logging
import pathlib
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SIMPLE = PROJECT_ROOT / "tennis_training_simple.csv"
DEFAULT_TA = PROJECT_ROOT / "tennis_training_ta.csv"
DEFAULT_OUT = PROJECT_ROOT / "tennis_training_combined.csv"

TA_FEATURES = [
    "d_last5_hold",
    "d_last5_break",
    "d_last10_hold",
    "d_last10_break",
    "d_surf_last10_hold",
    "d_surf_last10_break",
    "d_last10_aces_pg",
    "d_surf_last10_aces_pg",
    "d_last10_df_pg",
    "d_surf_last10_df_pg",
    "d_last10_tb_match_rate",
    "d_last10_tb_win_pct",
    "d_surf_last10_tb_match_rate",
    "d_surf_last10_tb_win_pct",
]


def build_combined(simple_path: pathlib.Path, ta_path: pathlib.Path, out_path: pathlib.Path) -> None:
    if not simple_path.exists():
        raise FileNotFoundError(f"Missing simple CSV: {simple_path}")
    if not ta_path.exists():
        raise FileNotFoundError(f"Missing TA CSV: {ta_path}")

    simple = pd.read_csv(simple_path)
    ta = pd.read_csv(ta_path)

    if "match_id" not in simple.columns or "match_id" not in ta.columns:
        raise ValueError("Both CSVs must include match_id for merge.")

    # Keep only required TA columns
    ta_keep = ["match_id"] + [c for c in TA_FEATURES if c in ta.columns]
    ta = ta[ta_keep].copy()

    merged = simple.merge(ta, on="match_id", how="inner")

    logger.info("Simple rows: %d", len(simple))
    logger.info("TA rows: %d", len(ta))
    logger.info("Merged rows: %d", len(merged))

    merged.to_csv(out_path, index=False)
    logger.info("Wrote combined CSV -> %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simple", type=str, default=str(DEFAULT_SIMPLE))
    ap.add_argument("--ta", type=str, default=str(DEFAULT_TA))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    build_combined(pathlib.Path(args.simple), pathlib.Path(args.ta), pathlib.Path(args.out))


if __name__ == "__main__":
    main()
