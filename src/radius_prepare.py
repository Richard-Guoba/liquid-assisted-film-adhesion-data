from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import numpy as np
import cv2

from video_radius_analysis import (
    DISPLACEMENT_COL,
    FORCE_SAMPLE_ORDER,
    TIME_COL,
    VIDEO_EXTENSIONS,
    parse_sample_name,
    read_displacement_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "radius_config.xlsx"
DEFAULT_FORCE_CONFIG = PROJECT_ROOT / "config" / "force_config.xlsx"
DEFAULT_ALIGNMENT_MANIFEST = (
    PROJECT_ROOT / "outputs" / "force" / "fit_plots" / "reference" / "alignment_manifest.csv"
)
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "radius"
DEFAULT_FORCE_WORKBOOK = PROJECT_ROOT / "data" / "force" / "force_data.xlsx"
DEFAULT_LEGACY_ANNOTATION_ROOT = PROJECT_ROOT / "config" / "radius_config_temp"
DEFAULT_ANNOTATION_ROOT = PROJECT_ROOT / "config" / "radius_annotations"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "radius"

SELECTION_COLUMNS = [
    "sample",
    "video_file",
    "force_sheet",
    "force_fit_start_reference_mm",
    "force_fit_end_reference_mm",
    "manual_fit_start_mm",
    "manual_fit_end_mm",
    "point_count",
    "review_status",
    "notes",
]
OVERRIDE_COLUMNS = ["sample", "point_order", "replacement_frame", "reason"]
ALLOWED_REVIEW_STATUS = {"pending_review", "approved"}
CHINESE_SELECTION_COLUMNS = {
    "样品": "sample", "视频文件": "video_file", "力数据工作表": "force_sheet",
    "力拟合参考起点（mm）": "force_fit_start_reference_mm",
    "力拟合参考终点（mm）": "force_fit_end_reference_mm",
    "选帧位移起点（mm）": "manual_fit_start_mm",
    "选帧位移终点（mm）": "manual_fit_end_mm",
    "选帧数": "point_count", "标注状态": "review_status", "说明": "notes",
    "对齐事件": "信号", "视频对齐时刻（s）": "原视频时间",
    "力对齐名义时刻（s）": "力名义时间(大约)", "起点对齐方式": "新对齐",
}
ENGLISH_ALIGNMENT_COLUMNS = {
    "alignment_event": "信号",
    "video_alignment_time_s": "原视频时间",
    "force_alignment_nominal_time_s": "力名义时间(大约)",
    "alignment_mode": "新对齐",
}


def normalize_selection_ranges(rows: pd.DataFrame) -> pd.DataFrame:
    """Read English configuration fields and retain legacy label compatibility."""
    rows = rows.rename(columns={**CHINESE_SELECTION_COLUMNS, **ENGLISH_ALIGNMENT_COLUMNS}).copy()
    if "review_status" in rows:
        rows["review_status"] = rows["review_status"].replace({"已确认": "approved", "待确认": "pending_review"})
    for field in ["原视频时间", "力名义时间(大约)"]:
        if field in rows:
            rows[field] = rows[field].replace({"结束": "end"})
    if "新对齐" in rows:
        rows["新对齐"] = rows["新对齐"].replace({"从力拟合起点对齐": "begin"})
    return rows


MAX_DISPLACEMENT_BACKSTEP_MM = 0.001
SELECTED_COLUMNS = [
    "sample",
    "point_order",
    "frame",
    "time_s",
    "target_displacement_mm",
    "displacement_mm",
    "selection_error_mm",
    "image_path",
    "annotation_source",
    "selection_signature",
    "selection_status",
]
INPUT_CHECKSUM_COLUMNS = ["relative_path", "file_size", "sha256"]


@dataclass(frozen=True)
class RadiusConfig:
    selection_ranges: pd.DataFrame
    settings: dict[str, float | int | str]
    frame_overrides: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def _video_by_sample(video_dir: Path) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for path in sorted(video_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        normalized = parse_sample_name(path.name).normalized_name
        if normalized in videos:
            raise ValueError(f"duplicate video sample: {normalized}")
        videos[normalized] = path
    return videos


def build_initial_selection_ranges(
    force_config: pd.DataFrame,
    video_dir: Path,
) -> pd.DataFrame:
    required = {
        "test_id",
        "fit_curve_region_start_mm",
        "fit_curve_region_end_mm",
    }
    _require_columns(force_config, required, "force_config")
    expected_ids = [f"test_{index}" for index in range(1, len(FORCE_SAMPLE_ORDER) + 1)]
    if force_config["test_id"].astype(str).tolist() != expected_ids:
        raise ValueError("force_config test_id values must be ordered test_1 through test_53")
    if len(force_config) != 53 or len(FORCE_SAMPLE_ORDER) != 53:
        raise ValueError("radius initialization requires exactly 53 force rows and samples")

    video_by_sample = _video_by_sample(video_dir)
    if set(video_by_sample) != set(FORCE_SAMPLE_ORDER):
        missing = sorted(set(FORCE_SAMPLE_ORDER).difference(video_by_sample))
        extra = sorted(set(video_by_sample).difference(FORCE_SAMPLE_ORDER))
        raise ValueError(f"video/sample mismatch; missing={missing}, extra={extra}")

    rows: list[dict[str, object]] = []
    for index, sample in enumerate(FORCE_SAMPLE_ORDER):
        force_row = force_config.iloc[index]
        start = float(force_row["fit_curve_region_start_mm"])
        end = float(force_row["fit_curve_region_end_mm"])
        rows.append(
            {
                "sample": sample,
                "video_file": video_by_sample[sample].relative_to(video_dir).as_posix(),
                "force_sheet": str(force_row["test_id"]),
                "force_fit_start_reference_mm": start,
                "force_fit_end_reference_mm": end,
                "manual_fit_start_mm": start,
                "manual_fit_end_mm": end,
                "point_count": 10,
                "review_status": "pending_review",
                "notes": "",
            }
        )
    result = pd.DataFrame(rows, columns=SELECTION_COLUMNS)
    validate_selection_ranges(result, set(FORCE_SAMPLE_ORDER))
    return result


def validate_selection_ranges(
    rows: pd.DataFrame,
    expected_samples: set[str],
) -> None:
    _require_columns(rows, set(SELECTION_COLUMNS), "selection_ranges")
    if rows["sample"].astype(str).duplicated().any():
        raise ValueError("selection_ranges contains a duplicate sample")
    if rows["force_sheet"].astype(str).duplicated().any():
        raise ValueError("selection_ranges contains a duplicate force_sheet")
    actual_samples = set(rows["sample"].astype(str))
    if actual_samples != expected_samples:
        raise ValueError(
            "selection_ranges sample set mismatch; "
            f"missing={sorted(expected_samples - actual_samples)}, "
            f"extra={sorted(actual_samples - expected_samples)}"
        )
    statuses = set(rows["review_status"].astype(str))
    if not statuses.issubset(ALLOWED_REVIEW_STATUS):
        raise ValueError(
            f"invalid review_status values: {sorted(statuses - ALLOWED_REVIEW_STATUS)}"
        )

    for index, row in rows.iterrows():
        sample = str(row["sample"])
        numeric = {
            name: pd.to_numeric(row[name], errors="coerce")
            for name in (
                "force_fit_start_reference_mm",
                "force_fit_end_reference_mm",
                "manual_fit_start_mm",
                "manual_fit_end_mm",
                "point_count",
            )
        }
        if any(not math.isfinite(float(value)) for value in numeric.values()):
            raise ValueError(f"sample {sample} contains a non-finite configuration value")
        if not (
            float(numeric["force_fit_start_reference_mm"])
            < float(numeric["force_fit_end_reference_mm"])
        ):
            raise ValueError(f"sample {sample} has an invalid force reference range")
        if not (
            float(numeric["manual_fit_start_mm"])
            < float(numeric["manual_fit_end_mm"])
        ):
            raise ValueError(f"sample {sample} has an invalid manual fit range")
        point_count = float(numeric["point_count"])
        if point_count < 2 or not point_count.is_integer():
            raise ValueError(f"sample {sample} point_count must be an integer of at least 2")
        if not str(row["video_file"]).strip():
            raise ValueError(f"sample {sample} has an empty video_file")
        if not str(row["force_sheet"]).strip():
            raise ValueError(f"sample {sample} has an empty force_sheet")


def compute_selection_signature(
    sample: str,
    video_sha256: str,
    force_sha256: str,
    start_mm: float,
    end_mm: float,
    point_count: int,
    frames: Sequence[int],
    video_start_nominal_s: float,
    velocity_mm_min: float,
) -> str:
    payload = {
        "sample": sample,
        "video_sha256": video_sha256,
        "force_sha256": force_sha256,
        "start_mm": float(start_mm),
        "end_mm": float(end_mm),
        "point_count": int(point_count),
        "frames": [int(value) for value in frames],
        "video_start_nominal_s": float(video_start_nominal_s),
        "velocity_mm_min": float(velocity_mm_min),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _clean_displacement_table(displacement: pd.DataFrame) -> pd.DataFrame:
    _require_columns(displacement, {TIME_COL, DISPLACEMENT_COL}, "displacement table")
    clean = displacement[[TIME_COL, DISPLACEMENT_COL]].copy()
    clean[TIME_COL] = pd.to_numeric(clean[TIME_COL], errors="coerce")
    clean[DISPLACEMENT_COL] = pd.to_numeric(clean[DISPLACEMENT_COL], errors="coerce")
    clean = clean.dropna().sort_values(TIME_COL, kind="stable")
    clean = clean.drop_duplicates(TIME_COL, keep="first").reset_index(drop=True)
    if len(clean) < 2:
        raise ValueError("displacement table must contain at least two finite rows")
    if np.any(np.diff(clean[TIME_COL].to_numpy(dtype=float)) <= 0):
        raise ValueError("displacement table time values must be strictly increasing")
    displacement_steps = np.diff(clean[DISPLACEMENT_COL].to_numpy(dtype=float))
    if len(displacement_steps) and float(displacement_steps.min()) < -MAX_DISPLACEMENT_BACKSTEP_MM:
        raise ValueError(
            "displacement table contains a backstep larger than "
            f"{MAX_DISPLACEMENT_BACKSTEP_MM} mm"
        )
    return clean


def select_sample_frames(
    sample: str,
    start_mm: float,
    end_mm: float,
    point_count: int,
    displacement: pd.DataFrame,
    fps: float,
    frame_count: int,
    overrides: pd.DataFrame,
    video_sha256: str,
    force_sha256: str,
    video_start_nominal_s: float,
    velocity_mm_min: float,
) -> pd.DataFrame:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("video fps must be positive and finite")
    if frame_count < 1:
        raise ValueError("video frame_count must be positive")
    if point_count < 2:
        raise ValueError("point_count must be at least 2")
    if not math.isfinite(video_start_nominal_s):
        raise ValueError("video_start_nominal_s must be finite")
    if not math.isfinite(velocity_mm_min) or velocity_mm_min <= 0:
        raise ValueError("velocity_mm_min must be positive and finite")
    if not (math.isfinite(start_mm) and math.isfinite(end_mm) and start_mm < end_mm):
        raise ValueError("manual fit range must be finite and increasing")

    clean = _clean_displacement_table(displacement)
    displacement_values = clean[DISPLACEMENT_COL].to_numpy(dtype=float)
    monotone_displacement = np.maximum.accumulate(displacement_values)
    if start_mm < monotone_displacement[0] or end_mm > monotone_displacement[-1]:
        raise ValueError("manual fit range is outside the force displacement coverage")

    targets = np.linspace(float(start_mm), float(end_mm), int(point_count))
    start_video_s = float(start_mm) * 60.0 / float(velocity_mm_min) - float(
        video_start_nominal_s
    )
    end_video_s = float(end_mm) * 60.0 / float(velocity_mm_min) - float(
        video_start_nominal_s
    )
    video_end_exclusive_s = float(frame_count) / float(fps)
    timing_tolerance_s = 1.0 / float(fps)
    if (
        start_video_s < -timing_tolerance_s
        or end_video_s > video_end_exclusive_s + timing_tolerance_s
    ):
        raise ValueError(f"sample {sample} primary interval is outside the video coverage")

    # Keep the reviewed interval closed: choose the first real frame at or after
    # the start and the last real frame at or before the end.
    epsilon = 1e-9
    first_frame = max(0, int(math.ceil(start_video_s * float(fps) - epsilon)))
    last_frame = min(
        frame_count - 1,
        int(math.floor(end_video_s * float(fps) + epsilon)),
    )
    if last_frame - first_frame + 1 < point_count:
        raise ValueError(f"sample {sample} could not produce {point_count} unique frames")
    target_video_times = (
        targets * 60.0 / float(velocity_mm_min) - float(video_start_nominal_s)
    )
    frames = np.rint(target_video_times * float(fps)).astype(int)
    frames = np.clip(frames, first_frame, last_frame)

    _require_columns(overrides, set(OVERRIDE_COLUMNS), "frame_overrides")
    sample_overrides = overrides.loc[overrides["sample"].astype(str) == sample]
    if sample_overrides["point_order"].duplicated().any():
        raise ValueError(f"sample {sample} has duplicate frame override point_order")
    overridden_orders: set[int] = set()
    for _, override in sample_overrides.iterrows():
        order_value = float(override["point_order"])
        frame_value = float(override["replacement_frame"])
        if not order_value.is_integer() or not 1 <= int(order_value) <= point_count:
            raise ValueError(f"sample {sample} has invalid override point_order")
        if not frame_value.is_integer():
            raise ValueError(f"sample {sample} replacement_frame must be an integer")
        order = int(order_value)
        frames[order - 1] = int(frame_value)
        overridden_orders.add(order)

    if np.any(frames < first_frame) or np.any(frames > last_frame):
        raise ValueError(f"sample {sample} selected an out-of-range frame")
    if len(set(int(value) for value in frames)) != point_count:
        raise ValueError(f"sample {sample} could not produce {point_count} unique frames")

    frame_times = frames.astype(float) / float(fps)
    actual_displacement = (
        frame_times + float(video_start_nominal_s)
    ) * float(velocity_mm_min) / 60.0
    if np.any(np.diff(actual_displacement) <= 0):
        raise ValueError(f"sample {sample} selected displacement values are not increasing")
    for order in overridden_orders:
        value = float(actual_displacement[order - 1])
        if value < start_mm or value > end_mm:
            raise ValueError(f"sample {sample} override lies outside the manual fit range")

    signature = compute_selection_signature(
        sample,
        video_sha256,
        force_sha256,
        start_mm,
        end_mm,
        point_count,
        frames,
        video_start_nominal_s,
        velocity_mm_min,
    )
    rows: list[dict[str, object]] = []
    for index, (frame, time_s, target, actual) in enumerate(
        zip(frames, frame_times, targets, actual_displacement), start=1
    ):
        filename = (
            f"point_{index:02d}_frame_{int(frame):06d}_time_{float(time_s):.3f}s.jpg"
        )
        rows.append(
            {
                "sample": sample,
                "point_order": index,
                "frame": int(frame),
                "time_s": float(time_s),
                "target_displacement_mm": float(target),
                "displacement_mm": float(actual),
                "selection_error_mm": float(actual - target),
                "image_path": f"config/radius_annotations/{sample}/{filename}",
                "annotation_source": "new_export",
                "selection_signature": signature,
                "selection_status": "selected",
            }
        )
    return pd.DataFrame(rows, columns=SELECTED_COLUMNS)


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def load_selected_manifest(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    _require_columns(table, set(SELECTED_COLUMNS), "selected_frames")
    table = table.reindex(columns=SELECTED_COLUMNS)
    for column in (
        "point_order",
        "frame",
        "time_s",
        "target_displacement_mm",
        "displacement_mm",
        "selection_error_mm",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[
        [
            "point_order",
            "frame",
            "time_s",
            "target_displacement_mm",
            "displacement_mm",
            "selection_error_mm",
        ]
    ].isna().any().any():
        raise ValueError("selected_frames contains non-numeric point data")
    if table.duplicated(["sample", "point_order"]).any():
        raise ValueError("selected_frames contains duplicate sample/point_order values")
    if table.duplicated(["sample", "frame"]).any():
        raise ValueError("selected_frames contains duplicate sample/frame values")
    signatures_per_sample = table.groupby("sample")["selection_signature"].nunique()
    if (signatures_per_sample != 1).any():
        raise ValueError("selected_frames contains inconsistent selection signatures")
    return table


def resolve_legacy_annotation_dir(root: Path, sample: str) -> Path:
    exact = root / sample
    if exact.is_dir():
        return exact
    candidates: list[Path] = []
    if root.is_dir():
        for path in root.iterdir():
            if not path.is_dir():
                continue
            try:
                normalized = parse_sample_name(path.name).normalized_name
            except ValueError:
                continue
            if normalized == sample:
                candidates.append(path)
    if len(candidates) > 1:
        raise ValueError(f"sample {sample} has duplicate legacy annotation directories")
    return candidates[0] if candidates else exact


def _write_bytes_no_overwrite(destination: Path, payload: bytes) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if destination.exists():
        raise FileExistsError(f"will not overwrite existing annotation: {destination}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary annotation exists: {temporary}")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def export_selected_images(
    selection: pd.DataFrame,
    video_path: Path,
    annotation_dir: Path,
    legacy_annotation_dir: Path | None = None,
) -> pd.DataFrame:
    _require_columns(selection, set(SELECTED_COLUMNS), "selected frame selection")
    if selection.empty:
        raise ValueError("selected frame selection is empty")
    samples = set(selection["sample"].astype(str))
    signatures = set(selection["selection_signature"].astype(str))
    if len(samples) != 1 or len(signatures) != 1:
        raise ValueError("one export call must contain one sample and one signature")

    destination_by_index = {
        index: annotation_dir / Path(str(row["image_path"])).name
        for index, row in selection.iterrows()
    }
    signature_path = annotation_dir / "selection_signature.txt"
    calibration_source = (
        legacy_annotation_dir / "time_0.000s.jpg"
        if legacy_annotation_dir is not None
        else None
    )
    calibration_path = annotation_dir / "calibration_time_0.jpg"
    planned_destinations = list(destination_by_index.values()) + [signature_path]
    if calibration_source is not None and calibration_source.is_file():
        planned_destinations.append(calibration_path)
    existing = [path for path in planned_destinations if path.exists()]
    if existing:
        raise FileExistsError(f"will not overwrite existing annotation: {existing[0]}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"video has invalid fps: {video_path}")
    exported = selection.copy()
    payloads: dict[Path, bytes] = {}
    try:
        for index, row in selection.iterrows():
            frame = int(row["frame"])
            destination = destination_by_index[index]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"could not read frame {frame} from {video_path}")
            encoded_ok, encoded = cv2.imencode(".jpg", image)
            if not encoded_ok:
                raise ValueError(f"could not encode frame {frame} from {video_path}")
            payloads[destination] = encoded.tobytes()
            exported.loc[index, "annotation_source"] = "new_export"
    finally:
        capture.release()

    if calibration_source is not None and calibration_source.is_file():
        payloads[calibration_path] = calibration_source.read_bytes()
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for destination, payload in payloads.items():
        _write_bytes_no_overwrite(destination, payload)
    _write_bytes_no_overwrite(
        signature_path, (next(iter(signatures)) + "\n").encode("utf-8")
    )
    return exported.reindex(columns=SELECTED_COLUMNS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_input_checksum_table(data_root: Path, project_root: Path) -> pd.DataFrame:
    data_root_resolved = data_root.resolve()
    project_root_resolved = project_root.resolve()
    if not data_root_resolved.is_dir():
        raise FileNotFoundError(f"input data root does not exist: {data_root}")
    if not data_root_resolved.is_relative_to(project_root_resolved):
        raise ValueError("input data root must be inside the project root")
    rows = []
    for path in sorted(
        (candidate for candidate in data_root_resolved.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(project_root_resolved).as_posix(),
    ):
        rows.append(
            {
                "relative_path": path.relative_to(project_root_resolved).as_posix(),
                "file_size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return pd.DataFrame(rows, columns=INPUT_CHECKSUM_COLUMNS)


def verify_input_checksum_tables(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> pd.DataFrame:
    left = before.reindex(columns=INPUT_CHECKSUM_COLUMNS).sort_values(
        "relative_path", kind="stable"
    ).reset_index(drop=True)
    right = after.reindex(columns=INPUT_CHECKSUM_COLUMNS).sort_values(
        "relative_path", kind="stable"
    ).reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("input data changed while preparing radius outputs")
    verified = left.copy()
    verified["verification_status"] = "matched"
    return verified


def _video_metadata(video_path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or frame_count < 1:
        raise ValueError(f"video has invalid timing metadata: {video_path}")
    return fps, frame_count


def _load_velocity_by_test(force_config_path: Path) -> dict[str, float]:
    if not force_config_path.is_file():
        raise FileNotFoundError(f"force config does not exist: {force_config_path}")
    table = pd.read_excel(force_config_path, sheet_name="force_config")
    _require_columns(table, {"test_id", "velocity_mm_min"}, "force_config")
    if table["test_id"].astype(str).duplicated().any():
        raise ValueError("force_config contains a duplicate test_id")
    result: dict[str, float] = {}
    for _, row in table.iterrows():
        test_id = str(row["test_id"])
        velocity = float(pd.to_numeric(row["velocity_mm_min"], errors="coerce"))
        if not math.isfinite(velocity) or velocity <= 0:
            raise ValueError(f"{test_id} has an invalid velocity_mm_min")
        result[test_id] = velocity
    return result


def _load_alignment_by_test(alignment_manifest_path: Path) -> dict[str, pd.Series]:
    if not alignment_manifest_path.is_file():
        raise FileNotFoundError(
            f"force-video alignment manifest does not exist: {alignment_manifest_path}"
        )
    table = pd.read_csv(alignment_manifest_path)
    required = {"test_id", "sample", "video_start_nominal_s"}
    _require_columns(table, required, "force-video alignment manifest")
    if table["test_id"].astype(str).duplicated().any():
        raise ValueError("force-video alignment manifest contains a duplicate test_id")
    return {str(row["test_id"]): row for _, row in table.iterrows()}


def build_approved_manifest(
    config: RadiusConfig,
    video_dir: Path,
    force_workbook: Path,
    force_config_path: Path = DEFAULT_FORCE_CONFIG,
    alignment_manifest_path: Path = DEFAULT_ALIGNMENT_MANIFEST,
) -> pd.DataFrame:
    approved = config.selection_ranges.loc[
        config.selection_ranges["review_status"].astype(str) == "approved"
    ]
    if approved.empty:
        return pd.DataFrame(columns=SELECTED_COLUMNS)
    if not force_workbook.is_file():
        raise FileNotFoundError(f"force workbook does not exist: {force_workbook}")
    video_root = video_dir.resolve()
    force_hash = file_sha256(force_workbook)
    velocity_by_test = _load_velocity_by_test(force_config_path)
    alignment_by_test = _load_alignment_by_test(alignment_manifest_path)
    with pd.ExcelFile(force_workbook) as workbook:
        available_sheets = set(workbook.sheet_names)
        selections: list[pd.DataFrame] = []
        for _, row in approved.iterrows():
            sample = str(row["sample"])
            video_path = (video_dir / str(row["video_file"])).resolve()
            if not video_path.is_relative_to(video_root):
                raise ValueError(f"sample {sample} video_file escapes the video directory")
            if not video_path.is_file():
                raise FileNotFoundError(f"sample {sample} video does not exist: {video_path}")
            force_sheet = str(row["force_sheet"])
            if force_sheet not in available_sheets:
                raise ValueError(f"sample {sample} force sheet does not exist: {force_sheet}")
            if force_sheet not in velocity_by_test:
                raise ValueError(f"sample {sample} has no force-config velocity")
            if force_sheet not in alignment_by_test:
                raise ValueError(f"sample {sample} has no force-video alignment")
            alignment = alignment_by_test[force_sheet]
            if str(alignment["sample"]) != sample:
                raise ValueError(
                    f"sample {sample} does not match alignment row for {force_sheet}"
                )
            video_start_nominal_s = float(
                pd.to_numeric(alignment["video_start_nominal_s"], errors="coerce")
            )
            displacement = read_displacement_table(force_workbook, force_sheet)
            fps, frame_count = _video_metadata(video_path)
            selections.append(
                select_sample_frames(
                    sample=sample,
                    start_mm=float(row["manual_fit_start_mm"]),
                    end_mm=float(row["manual_fit_end_mm"]),
                    point_count=int(row["point_count"]),
                    displacement=displacement,
                    fps=fps,
                    frame_count=frame_count,
                    overrides=config.frame_overrides,
                    video_sha256=file_sha256(video_path),
                    force_sha256=force_hash,
                    video_start_nominal_s=video_start_nominal_s,
                    velocity_mm_min=velocity_by_test[force_sheet],
                )
            )
    return pd.concat(selections, ignore_index=True).reindex(columns=SELECTED_COLUMNS)


def validate_existing_annotations(
    manifest: pd.DataFrame,
    annotation_root: Path,
) -> None:
    """Validate packaged annotations without altering user-marked images."""
    annotation_root_resolved = annotation_root.resolve()
    for sample, rows in manifest.groupby("sample", sort=False):
        sample = str(sample)
        sample_dir = (annotation_root / sample).resolve()
        if not sample_dir.is_relative_to(annotation_root_resolved):
            raise ValueError(f"sample {sample} annotation path escapes annotation root")
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"sample {sample} annotation directory is missing")

        signatures = rows["selection_signature"].astype(str).unique().tolist()
        if len(signatures) != 1:
            raise ValueError(f"sample {sample} has inconsistent selection signatures")
        signature_path = sample_dir / "selection_signature.txt"
        if not signature_path.is_file():
            raise FileNotFoundError(f"sample {sample} selection signature is missing")
        if signature_path.read_text(encoding="utf-8").strip() != signatures[0]:
            raise ValueError(f"sample {sample} selection signature mismatch")
        if not (sample_dir / "calibration_time_0.jpg").is_file():
            raise FileNotFoundError(f"sample {sample} calibration image is missing")

        expected = {Path(str(value)).name for value in rows["image_path"]}
        if len(expected) != len(rows):
            raise ValueError(f"sample {sample} has duplicate canonical point filenames")
        actual = {
            path.name
            for path in sample_dir.glob("point_*.jpg")
            if path.is_file() and not path.stem.endswith("_temp")
        }
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        if missing or extra:
            raise ValueError(
                f"sample {sample} canonical point files mismatch; "
                f"missing={missing}, extra={extra}"
            )


def _atomic_write_json(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def prepare_approved_samples(
    config: RadiusConfig,
    video_dir: Path,
    force_workbook: Path,
    force_config_path: Path,
    alignment_manifest_path: Path,
    legacy_annotation_root: Path,
    annotation_root: Path,
    output_root: Path,
    config_path: Path | None = None,
    input_data_root: Path | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    input_checksums_before = None
    if input_data_root is not None:
        checksum_project_root = project_root or input_data_root.parent
        input_checksums_before = build_input_checksum_table(
            input_data_root, checksum_project_root
        )
    manifest = build_approved_manifest(
        config,
        video_dir,
        force_workbook,
        force_config_path=force_config_path,
        alignment_manifest_path=alignment_manifest_path,
    )
    if manifest.empty:
        return manifest

    samples = manifest["sample"].drop_duplicates().astype(str).tolist()
    existing_targets = [
        annotation_root / sample
        for sample in samples
        if (annotation_root / sample).exists()
    ]
    if existing_targets and len(existing_targets) != len(samples):
        raise FileExistsError(
            "annotation replay requires either all or none of the approved sample "
            f"directories; first existing directory: {existing_targets[0]}"
        )
    if existing_targets:
        validate_existing_annotations(manifest, annotation_root)
        exported = manifest.copy().reindex(columns=SELECTED_COLUMNS)
        annotation_mode = "existing_annotations_verified"
    else:
        legacy_by_sample = {
            sample: resolve_legacy_annotation_dir(legacy_annotation_root, sample)
            for sample in samples
        }
        missing_calibrations = [
            sample
            for sample, directory in legacy_by_sample.items()
            if not (directory / "time_0.000s.jpg").is_file()
        ]
        if missing_calibrations:
            raise FileNotFoundError(
                f"sample {missing_calibrations[0]} is missing the legacy calibration image"
            )
        staging_root = output_root / ".prepare_staging"
        if staging_root.exists():
            raise FileExistsError(f"stale preparation staging directory exists: {staging_root}")
        staging_root.mkdir(parents=True)

        exported_tables: list[pd.DataFrame] = []
        for sample in samples:
            sample_rows = manifest.loc[manifest["sample"].astype(str) == sample].copy()
            config_row = config.selection_ranges.loc[
                config.selection_ranges["sample"].astype(str) == sample
            ].iloc[0]
            legacy_annotation_dir = legacy_by_sample[sample]
            exported_tables.append(
                export_selected_images(
                    sample_rows,
                    video_dir / str(config_row["video_file"]),
                    staging_root / sample,
                    legacy_annotation_dir,
                )
            )
        annotation_root.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            staged = staging_root / sample
            target = annotation_root / sample
            if target.exists():
                raise FileExistsError(
                    f"will not overwrite existing annotation directory: {target}"
                )
            staged.replace(target)
        staging_root.rmdir()
        exported = pd.concat(exported_tables, ignore_index=True).reindex(
            columns=SELECTED_COLUMNS
        )
        annotation_mode = "fresh_frame_export"
    atomic_write_csv(exported, output_root / "selected_frames.csv")
    dependency_versions = {}
    for package in ("numpy", "pandas", "opencv-python-headless", "openpyxl"):
        try:
            dependency_versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            dependency_versions[package] = "not-installed"
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved_sample_count": len(samples),
        "pending_sample_count": int(
            (config.selection_ranges["review_status"] == "pending_review").sum()
        ),
        "selected_point_count": len(exported),
        "annotation_mode": annotation_mode,
        "force_workbook_sha256": file_sha256(force_workbook),
        "force_config_sha256": file_sha256(force_config_path),
        "alignment_manifest_sha256": file_sha256(alignment_manifest_path),
        "config_sha256": file_sha256(config_path) if config_path is not None else None,
        "radius_prepare_sha256": file_sha256(Path(__file__)),
        "dependency_versions": dependency_versions,
        "samples": samples,
        "failures": [],
    }
    _atomic_write_json(metadata, output_root / "processing_metadata.json")
    if input_checksums_before is not None:
        input_checksums = verify_input_checksum_tables(
            input_checksums_before,
            build_input_checksum_table(input_data_root, checksum_project_root),
        )
        atomic_write_csv(input_checksums, output_root / "input_checksums.csv")
    return exported


def write_radius_config(config: RadiusConfig, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"will not overwrite existing radius config: {path}")
    settings = pd.DataFrame(
        [{"key": key, "value": value} for key, value in config.settings.items()],
        columns=["key", "value"],
    )
    overrides = config.frame_overrides.reindex(columns=OVERRIDE_COLUMNS)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        config.selection_ranges.to_excel(writer, sheet_name="selection_ranges", index=False)
        settings.to_excel(writer, sheet_name="settings", index=False)
        overrides.to_excel(writer, sheet_name="frame_overrides", index=False)


def load_radius_config(path: Path) -> RadiusConfig:
    with pd.ExcelFile(path) as workbook:
        expected_sheets = ["selection_ranges", "settings", "frame_overrides"]
        if workbook.sheet_names != expected_sheets:
            raise ValueError(
                f"radius config sheets must be {expected_sheets}, got {workbook.sheet_names}"
            )
        selection_ranges = normalize_selection_ranges(pd.read_excel(workbook, sheet_name="selection_ranges"))
        settings_table = pd.read_excel(workbook, sheet_name="settings").rename(columns={"参数": "key", "值": "value"})
        frame_overrides = pd.read_excel(workbook, sheet_name="frame_overrides").rename(columns={
            "样品": "sample", "选帧顺序": "point_order", "替换帧号": "replacement_frame", "原因": "reason"})
    settings_table["key"] = settings_table["key"].replace({"膜直径（mm）": "film_diameter_mm", "默认选帧数": "default_point_count"})
    _require_columns(settings_table, {"key", "value"}, "settings")
    if settings_table["key"].astype(str).duplicated().any():
        raise ValueError("settings contains a duplicate key")
    settings = dict(zip(settings_table["key"].astype(str), settings_table["value"]))
    _require_columns(frame_overrides, set(OVERRIDE_COLUMNS), "frame_overrides")
    selection_ranges["notes"] = selection_ranges["notes"].fillna("").astype(str)
    return RadiusConfig(
        selection_ranges=selection_ranges,
        settings=settings,
        frame_overrides=frame_overrides.reindex(columns=OVERRIDE_COLUMNS),
    )


def initialize_config(force_config_path: Path, video_dir: Path, output: Path) -> None:
    force_config = pd.read_excel(force_config_path, sheet_name="force_config")
    selection_ranges = build_initial_selection_ranges(force_config, video_dir)
    config = RadiusConfig(
        selection_ranges=selection_ranges,
        settings={"film_diameter_mm": 45.0, "default_point_count": 10},
        frame_overrides=pd.DataFrame(columns=OVERRIDE_COLUMNS),
    )
    write_radius_config(config, output)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare user-selected frames for manual radius annotation."
    )
    parser.add_argument("--initialize-config", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force-config", type=Path, default=DEFAULT_FORCE_CONFIG)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--force-workbook", type=Path, default=DEFAULT_FORCE_WORKBOOK)
    parser.add_argument(
        "--alignment-manifest",
        type=Path,
        default=DEFAULT_ALIGNMENT_MANIFEST,
    )
    parser.add_argument(
        "--legacy-annotation-root",
        type=Path,
        default=DEFAULT_LEGACY_ANNOTATION_ROOT,
    )
    parser.add_argument(
        "--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--input-data-root", type=Path, default=PROJECT_ROOT / "data"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.initialize_config and args.validate_only:
        raise ValueError("choose only one of --initialize-config and --validate-only")
    if args.initialize_config:
        initialize_config(args.force_config, args.video_dir, args.config)
        print(f"Initialized pending-review radius config: {args.config}")
        return 0
    if args.validate_only:
        config = load_radius_config(args.config)
        validate_selection_ranges(config.selection_ranges, set(FORCE_SAMPLE_ORDER))
        required_settings = {"film_diameter_mm", "default_point_count"}
        missing_settings = required_settings.difference(config.settings)
        if missing_settings:
            raise ValueError(f"radius config is missing settings: {sorted(missing_settings)}")
        manifest = build_approved_manifest(
            config,
            args.video_dir,
            args.force_workbook,
            force_config_path=args.force_config,
            alignment_manifest_path=args.alignment_manifest,
        )
        pending = int((config.selection_ranges["review_status"] == "pending_review").sum())
        approved = int((config.selection_ranges["review_status"] == "approved").sum())
        print(
            f"Validated {len(config.selection_ranges)} samples: "
            f"pending_review={pending}, approved={approved}, "
            f"selected_points={len(manifest)}"
        )
        return 0
    config = load_radius_config(args.config)
    validate_selection_ranges(config.selection_ranges, set(FORCE_SAMPLE_ORDER))
    exported = prepare_approved_samples(
        config,
        video_dir=args.video_dir,
        force_workbook=args.force_workbook,
        force_config_path=args.force_config,
        alignment_manifest_path=args.alignment_manifest,
        legacy_annotation_root=args.legacy_annotation_root,
        annotation_root=args.annotation_root,
        output_root=args.output_root,
        config_path=args.config,
        input_data_root=args.input_data_root,
        project_root=PROJECT_ROOT,
    )
    print(
        f"Prepared {exported['sample'].nunique() if not exported.empty else 0} samples "
        f"and {len(exported)} selected frames."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
