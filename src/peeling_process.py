"""Recalculate peeling plateau forces and adhesion energy from packaged data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


RAW_COLUMNS = ["Time (s)", "Load (N)", "Displacement (mm)"]
CONFIG_COLUMNS = [
    "test_id",
    "sheet_name",
    "liquid",
    "drying_time_h",
    "specimen_width_mm",
    "thickness_um",
    "peel_speed_mm_per_min",
    "plateau_start_mm_zeroed",
    "plateau_end_mm_zeroed",
]
OPTIONAL_CONFIG_COLUMNS = ["planned_drying_time_h", "peel_angle_deg"]
ANALYSIS_COLUMNS = [
    "plateau_force_mean_n",
    "plateau_force_sd_n",
    "plateau_point_count",
    "gamma_n_per_m",
    "first_displacement_offset_mm",
    "data_point_count",
    "record_duration_s",
]
RESULT_COLUMNS = CONFIG_COLUMNS + ANALYSIS_COLUMNS

LIQUID_OUTPUT_ORDER = {
    "ethanol": 0,
    "ipa": 1,
    "glycerol": 2,
    "water": 3,
    "oil": 4,
}


def prepare_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Return numeric peeling data with displacement relative to its first valid point."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    missing = set(RAW_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Raw peeling sheet is missing columns: {sorted(missing)}")
    prepared = frame.loc[:, RAW_COLUMNS].copy()
    for column in RAW_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=RAW_COLUMNS).reset_index(drop=True)
    if prepared.empty:
        raise ValueError("Raw peeling sheet has no complete numeric rows")
    first_displacement = float(prepared.loc[0, "Displacement (mm)"])
    prepared["Displacement zeroed (mm)"] = (
        prepared["Displacement (mm)"] - first_displacement
    )
    return prepared


def analyze_curve(
    frame: pd.DataFrame,
    plateau_start_mm: float,
    plateau_end_mm: float,
    width_mm: float,
) -> dict[str, float | int]:
    """Calculate the inclusive plateau mean and 90-degree peeling energy."""
    values = (plateau_start_mm, plateau_end_mm, width_mm)
    if not all(isinstance(value, (int, float, np.number)) and math.isfinite(float(value)) for value in values):
        raise ValueError("Plateau bounds and specimen width must be finite numbers")
    if float(width_mm) <= 0:
        raise ValueError("Specimen width must be positive")
    if float(plateau_end_mm) < float(plateau_start_mm):
        raise ValueError("Plateau end must not precede plateau start")

    prepared = prepare_curve(frame)
    displacement = prepared["Displacement zeroed (mm)"]
    mask = displacement.between(
        float(plateau_start_mm), float(plateau_end_mm), inclusive="both"
    )
    plateau_load = prepared.loc[mask, "Load (N)"]
    if plateau_load.empty:
        raise ValueError("Configured plateau interval contains no data points")
    mean_force = float(plateau_load.mean())
    sd_force = float(plateau_load.std(ddof=1)) if len(plateau_load) > 1 else 0.0
    first_offset = float(prepared.loc[0, "Displacement (mm)"])
    duration = float(prepared["Time (s)"].iloc[-1] - prepared["Time (s)"].iloc[0])
    return {
        "plateau_force_mean_n": mean_force,
        "plateau_force_sd_n": sd_force,
        "plateau_point_count": int(len(plateau_load)),
        "gamma_n_per_m": mean_force / (float(width_mm) / 1000.0),
        "first_displacement_offset_mm": first_offset,
        "data_point_count": int(len(prepared)),
        "record_duration_s": duration,
    }


def load_configuration(path: Path) -> pd.DataFrame:
    config = pd.read_excel(path, sheet_name="peeling_config")
    missing = set(CONFIG_COLUMNS).difference(config.columns)
    if missing:
        raise ValueError(f"Peeling configuration is missing columns: {sorted(missing)}")
    columns = [key for key in config if key in CONFIG_COLUMNS + OPTIONAL_CONFIG_COLUMNS]
    config = config.loc[:, columns].copy()
    if any(config[key].isna().any() or config[key].duplicated().any()
           for key in ["test_id", "sheet_name"]):
        raise ValueError("Peeling configuration requires unique, nonmissing test IDs")
    return config.sort_values("test_id", kind="stable").reset_index(drop=True)


def sort_results_for_output(frame: pd.DataFrame) -> pd.DataFrame:
    """Group like liquids and order experimental conditions deterministically."""
    ordered = frame.copy()
    ordered["_liquid_name"] = ordered["liquid"].astype(str).str.strip().str.lower()
    ordered["_liquid_order"] = ordered["_liquid_name"].map(LIQUID_OUTPUT_ORDER).fillna(99)
    ordered = ordered.sort_values(
        [
            "_liquid_order",
            "_liquid_name",
            "thickness_um",
            "drying_time_h",
            "peel_speed_mm_per_min",
            "test_id",
        ],
        kind="stable",
    )
    return ordered.drop(columns=["_liquid_order", "_liquid_name"]).reset_index(drop=True)


def process_workbook(data_path: Path, config_path: Path) -> pd.DataFrame:
    """Process exactly the test sheets named in the configuration workbook."""
    config = load_configuration(config_path)
    excel = pd.ExcelFile(data_path)
    data_sheets = {name for name in excel.sheet_names if name.startswith("T")}
    configured_sheets = set(config["sheet_name"].astype(str))
    if data_sheets != configured_sheets:
        missing = sorted(configured_sheets - data_sheets)
        unexpected = sorted(data_sheets - configured_sheets)
        raise ValueError(
            f"Data/config sheet mismatch; missing={missing}, unexpected={unexpected}"
        )

    rows: list[dict[str, object]] = []
    for record in config.to_dict("records"):
        frame = pd.read_excel(excel, sheet_name=str(record["sheet_name"]))
        analysis = analyze_curve(
            frame,
            float(record["plateau_start_mm_zeroed"]),
            float(record["plateau_end_mm_zeroed"]),
            float(record["specimen_width_mm"]),
        )
        rows.append({**record, **analysis})
    results = pd.DataFrame(rows).reindex(columns=list(config.columns) + ANALYSIS_COLUMNS)
    return sort_results_for_output(results)


def build_argument_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "data" / "peeling" / "peeling_data.xlsx",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "peeling_config.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "outputs" / "peeling" / "peeling_results.xlsx",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    results = process_workbook(args.data, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_excel(args.output, sheet_name="peeling_results", index=False)
    print(f"Processed {len(results)} peeling tests into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
