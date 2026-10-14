from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "video"
DISPLACEMENT_WORKBOOK = ROOT.parent / "data" / "force" / "force_data.xlsx"
OUTPUT_ROOT = ROOT / "video_radius_outputs"
ANNOTATED_VIDEO_DIR = OUTPUT_ROOT / "标定视频"
SLOPE_OUTPUT_DIR = OUTPUT_ROOT / "斜率输出"

TIME_COL = "Time (s)"
DISPLACEMENT_COL = "Displacement (mm)"
TIME_COLUMN_ALIASES = (TIME_COL, "时间 (s)")
DISPLACEMENT_COLUMN_ALIASES = (DISPLACEMENT_COL, "位移 (mm)")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".m4v"}

# Sheet order in data/force/force_data.xlsx. The force workbook is the single source
# for both force processing and the time-displacement input used by radius processing.
FORCE_SAMPLE_ORDER = (
    "ethanol_25um_3h",
    "ethanol_50um_3h",
    "ethanol_100um_3h",
    "ethanol_150um_3h",
    "ipa_25um_3h",
    "ipa_50um_3h",
    "ipa_100um_3h",
    "ipa_150um_3h",
    "water_25um_3h",
    "water_50um_3h",
    "water_100um_3h",
    "water_150um_3h",
    "oil_150um_3h",
    "ethanol_25um_6h",
    "ethanol_50um_6h",
    "ethanol_100um_6h",
    "ethanol_150um_6h",
    "ipa_25um_6h",
    "ipa_50um_6h",
    "ipa_100um_6h",
    "ipa_150um_6h",
    "water_25um_6h",
    "water_50um_6h",
    "water_100um_6h",
    "water_150um_6h",
    "oil_25um_6h",
    "oil_150um_6h",
    "ethanol_25um_9h",
    "ethanol_50um_9h",
    "ethanol_100um_9h",
    "ethanol_150um_9h",
    "ipa_25um_9h",
    "ipa_50um_9h",
    "ipa_100um_9h",
    "ipa_150um_9h",
    "water_25um_9h",
    "water_50um_9h",
    "water_100um_9h",
    "water_150um_9h",
    "oil_50um_9h",
    "oil_100um_9h",
    "oil_150um_9h",
    "ethanol_25um_9h_1",
    "ethanol_100um_9h_1",
    "ipa_100um_9h_1",
    "water_100um_9h_1",
    "oil_100um_9h_1",
    "oil_100um_9h_2",
    "oil_100um_9h_3",
    "water_25um_6h_1",
    "water_100um_6h_1",
    "water_150um_6h_1",
    "water_100um_3h_1",
)
FORCE_SHEET_BY_SAMPLE = {
    sample: f"test_{index}"
    for index, sample in enumerate(FORCE_SAMPLE_ORDER, start=1)
}

FILM_DIAMETER_CM = 4.5
FILM_DIAMETER_MM = FILM_DIAMETER_CM * 10.0
MANUAL_ANNOTATION_FILE = ROOT / "manual_annotations.csv"

RADIUS_MIN_PX = 100
RADIUS_MAX_FRACTION = 0.48
RADIUS_SEARCH_MARGIN_PX = 50
RADIUS_SMOOTH_WINDOW = 5
FIT_DISPLACEMENT_RANGE_MM: tuple[float, float] | None = None
DETECTION_MAX_SIDE = 720
USE_RADIUS_CONTINUITY = True
ENFORCE_NONDECREASING_RADIUS = True
ZERO_RADIUS_AT_INITIAL_FRAME = True
BUBBLE_MAX_RADIUS_FRACTION_OF_FILM = 0.42
BUBBLE_MAX_RADIUS_FALLBACK_FRACTION = 0.22
BUBBLE_MAX_STEP_PX = 45.0


@dataclass(frozen=True)
class SampleName:
    raw_name: str
    liquid: str
    thickness_um: int
    time_h: int
    replicate: int | None
    normalized_name: str
    base_name: str


@dataclass(frozen=True)
class FitResult:
    slope_px_per_mm: float
    intercept_px: float
    r_squared: float
    points: int
    fit_start_mm: float
    fit_end_mm: float


@dataclass(frozen=True)
class FilmCalibration:
    center_x: float
    center_y: float
    radius_px: float
    diameter_cm: float = FILM_DIAMETER_CM

    @property
    def diameter_mm(self) -> float:
        return self.diameter_cm * 10.0

    @property
    def mm_per_px(self) -> float:
        return self.diameter_mm / (2.0 * self.radius_px)

    def radius_px_to_mm(self, radius_px: float) -> float:
        return float(radius_px) * self.mm_per_px


def parse_sample_name(name: str) -> SampleName:
    stem = Path(name).stem
    pattern = re.compile(
        r"^(?P<liquid>[A-Za-z]+)_(?P<thickness>\d+)u(?:m)?_(?P<time>\d+)h(?:_(?P<replicate>\d+))?$",
        re.IGNORECASE,
    )
    match = pattern.match(stem)
    if not match:
        raise ValueError(f"Cannot parse sample name: {name}")

    liquid = match.group("liquid").lower()
    thickness_um = int(match.group("thickness"))
    time_h = int(match.group("time"))
    replicate_text = match.group("replicate")
    replicate = int(replicate_text) if replicate_text is not None else None
    base_name = f"{liquid}_{thickness_um}um_{time_h}h"
    normalized_name = f"{base_name}_{replicate}" if replicate is not None else base_name
    return SampleName(
        raw_name=stem,
        liquid=liquid,
        thickness_um=thickness_um,
        time_h=time_h,
        replicate=replicate,
        normalized_name=normalized_name,
        base_name=base_name,
    )


def match_sheet_name(video_name: str, sheet_names: Iterable[str]) -> str | None:
    sheet_set = set(sheet_names)
    stem = Path(video_name).stem
    if stem in sheet_set:
        return stem

    try:
        sample = parse_sample_name(stem)
    except ValueError:
        return None

    for candidate in (sample.normalized_name, sample.base_name):
        if candidate in sheet_set:
            return candidate
        force_sheet = FORCE_SHEET_BY_SAMPLE.get(candidate)
        if force_sheet in sheet_set:
            return force_sheet
    return None


def interpolate_displacement(
    table: pd.DataFrame,
    frame_times_s: np.ndarray,
    time_col: str = TIME_COL,
    displacement_col: str = DISPLACEMENT_COL,
) -> np.ndarray:
    if time_col not in table.columns or displacement_col not in table.columns:
        raise ValueError(f"Table must contain {time_col!r} and {displacement_col!r}")

    data = table[[time_col, displacement_col]].copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data[displacement_col] = pd.to_numeric(data[displacement_col], errors="coerce")
    data = data.dropna().sort_values(time_col)
    data = data.drop_duplicates(subset=[time_col], keep="first")
    if data.empty:
        return np.full_like(frame_times_s, np.nan, dtype=float)

    time_s = data[time_col].to_numpy(dtype=float)
    displacement_mm = data[displacement_col].to_numpy(dtype=float)
    time_s = time_s - time_s[0]
    displacement_mm = displacement_mm - displacement_mm[0]
    return np.interp(frame_times_s, time_s, displacement_mm, left=displacement_mm[0], right=displacement_mm[-1])


def fit_radius_displacement(
    displacement_mm: Sequence[float],
    radius_px: Sequence[float],
    fit_range: tuple[float, float] | None = FIT_DISPLACEMENT_RANGE_MM,
) -> FitResult:
    x = np.asarray(displacement_mm, dtype=float)
    y = np.asarray(radius_px, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if fit_range is not None:
        lo, hi = sorted((float(fit_range[0]), float(fit_range[1])))
        mask &= (x >= lo) & (x <= hi)

    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size < 2:
        return FitResult(math.nan, math.nan, math.nan, int(x_fit.size), math.nan, math.nan)

    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    predicted = slope * x_fit + intercept
    ss_res = float(np.sum((y_fit - predicted) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = math.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return FitResult(
        slope_px_per_mm=float(slope),
        intercept_px=float(intercept),
        r_squared=float(r_squared),
        points=int(x_fit.size),
        fit_start_mm=float(np.min(x_fit)),
        fit_end_mm=float(np.max(x_fit)),
    )


def estimate_radius_from_gray(
    gray: np.ndarray,
    center: tuple[float, float],
    min_radius: int = RADIUS_MIN_PX,
    max_radius: int | None = None,
    expected_radius: float | None = None,
    search_margin: int = RADIUS_SEARCH_MARGIN_PX,
) -> float:
    if gray.ndim == 3:
        gray = np.mean(gray, axis=2)
    image = gray.astype(float)
    height, width = image.shape
    cx, cy = center

    if max_radius is None:
        max_radius = int(max(1, min(cx, cy, width - 1 - cx, height - 1 - cy)))
    max_radius = max(int(max_radius), min_radius + 1)

    if expected_radius is not None and math.isfinite(expected_radius):
        lo = max(min_radius, int(round(expected_radius - search_margin)))
        hi = min(max_radius, int(round(expected_radius + search_margin)))
    else:
        lo = min_radius
        hi = max_radius
    if hi <= lo:
        return math.nan

    gy, gx = np.gradient(image)
    gradient = np.hypot(gx, gy)
    yy, xx = np.indices(image.shape)
    radii = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    flat_radii = radii.ravel()
    flat_gradient = gradient.ravel()
    minlength = max_radius + 2
    sums = np.bincount(flat_radii, weights=flat_gradient, minlength=minlength)
    counts = np.bincount(flat_radii, minlength=minlength)
    scores = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)

    if hi + 1 > scores.size:
        hi = scores.size - 2
    if hi <= lo:
        return math.nan

    if hi - lo >= 5:
        kernel = np.ones(5, dtype=float) / 5.0
        scores = np.convolve(scores, kernel, mode="same")
    search_scores = scores[lo : hi + 1]
    if search_scores.size == 0 or not np.isfinite(search_scores).any() or float(np.nanmax(search_scores)) <= 0:
        return math.nan
    return float(lo + int(np.nanargmax(search_scores)))


def smooth_numeric_series(values: Sequence[float], window: int = RADIUS_SMOOTH_WINDOW) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if window <= 1 or array.size == 0:
        return array
    series = pd.Series(array)
    series = series.interpolate(limit_direction="both")
    return series.rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)


def postprocess_radius_series(
    values: Sequence[float],
    window: int = RADIUS_SMOOTH_WINDOW,
    enforce_nondecreasing: bool = ENFORCE_NONDECREASING_RADIUS,
    zero_at_initial: bool = ZERO_RADIUS_AT_INITIAL_FRAME,
) -> np.ndarray:
    smoothed = smooth_numeric_series(values, window=window)
    if smoothed.size:
        finite = np.isfinite(smoothed)
        if finite.any():
            filled = pd.Series(smoothed).interpolate(limit_direction="both").to_numpy(dtype=float)
            smoothed = filled
            if zero_at_initial:
                baseline = float(smoothed[0])
                smoothed = np.maximum(smoothed - baseline, 0.0)
            if enforce_nondecreasing:
                smoothed = np.maximum.accumulate(smoothed)
    return smoothed


def interpolate_manual_radius_series(
    annotations: pd.DataFrame,
    sample: str,
    frame_count: int,
    *,
    radius_col: str = "radius_px",
) -> np.ndarray | None:
    required = {"sample", "frame", radius_col}
    if annotations.empty or not required.issubset(annotations.columns):
        return None

    rows = annotations[annotations["sample"].astype(str) == sample].copy()
    if rows.empty:
        return None

    rows["frame"] = pd.to_numeric(rows["frame"], errors="coerce")
    rows[radius_col] = pd.to_numeric(rows[radius_col], errors="coerce")
    rows = rows.dropna(subset=["frame", radius_col]).sort_values("frame")
    if rows.empty:
        return None

    frames = np.clip(rows["frame"].to_numpy(dtype=float), 0, max(0, frame_count - 1))
    radius = rows[radius_col].to_numpy(dtype=float)
    unique_frames, unique_indices = np.unique(frames, return_index=True)
    unique_radius = radius[unique_indices]
    target_frames = np.arange(frame_count, dtype=float)
    interpolated = np.interp(target_frames, unique_frames, unique_radius, left=unique_radius[0], right=unique_radius[-1])
    return postprocess_radius_series(interpolated, window=1)


def load_manual_annotations(path: Path = MANUAL_ANNOTATION_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def estimate_growth_radius_from_diff(
    diff: np.ndarray,
    center: tuple[float, float],
    max_radius: int,
    *,
    min_radius: int = 3,
    expected_radius: float | None = None,
    max_step_px: float = BUBBLE_MAX_STEP_PX,
) -> float:
    if diff.ndim == 3:
        diff = np.mean(diff, axis=2)
    image = diff.astype(float)
    height, width = image.shape
    cx, cy = center
    max_radius = int(max(min_radius + 1, min(max_radius, cx, cy, width - 1 - cx, height - 1 - cy)))
    if max_radius <= min_radius:
        return 0.0

    yy, xx = np.indices(image.shape)
    radii = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    mask = radii <= max_radius
    roi_values = image[mask]
    if roi_values.size == 0:
        return 0.0

    median = float(np.median(roi_values))
    mad = float(np.median(np.abs(roi_values - median)))
    threshold = max(2.0, median + 4.0 * mad)
    high = image > threshold

    flat_radii = radii[mask].ravel()
    flat_values = image[mask].ravel()
    minlength = max_radius + 2
    sums = np.bincount(flat_radii, weights=flat_values, minlength=minlength)
    counts = np.bincount(flat_radii, minlength=minlength)
    means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)

    high_counts = np.bincount(radii[mask & high].ravel(), minlength=minlength)
    coverage = np.divide(high_counts, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)

    if max_radius - min_radius >= 5:
        mean_kernel = np.ones(7, dtype=float) / 7.0
        coverage_kernel = np.ones(5, dtype=float) / 5.0
        means = np.convolve(means, mean_kernel, mode="same")
        coverage = np.convolve(coverage, coverage_kernel, mode="same")

    active = (means > threshold * 0.55) & (coverage > 0.035)
    active[:min_radius] = False

    if expected_radius is not None and math.isfinite(expected_radius):
        hi = min(max_radius, int(math.ceil(expected_radius + max_step_px)))
        active[hi + 1 :] = False

    candidates = np.flatnonzero(active[: max_radius + 1])
    if candidates.size == 0:
        return float(expected_radius) if expected_radius is not None and math.isfinite(expected_radius) else 0.0

    valid: list[int] = []
    for radius in candidates:
        lo = max(min_radius, int(radius) - 12)
        if np.count_nonzero(active[lo : int(radius) + 1]) >= 4:
            valid.append(int(radius))
    if not valid:
        return float(candidates[-1])
    return float(max(valid))


def prepare_bubble_gray(cv2, frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def estimate_growth_radius_from_video_gray(
    cv2,
    gray: np.ndarray,
    baseline_gray: np.ndarray,
    center: tuple[float, float],
    max_radius: int,
    previous_growth_radius: float | None,
) -> float:
    scale = min(1.0, DETECTION_MAX_SIDE / float(max(gray.shape)))
    if scale < 1.0:
        resized_size = (
            max(1, int(round(gray.shape[1] * scale))),
            max(1, int(round(gray.shape[0] * scale))),
        )
        gray_small = cv2.resize(gray, resized_size, interpolation=cv2.INTER_AREA)
        baseline_small = cv2.resize(baseline_gray, resized_size, interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(gray_small, baseline_small)
        diff = cv2.GaussianBlur(diff, (7, 7), 0)
        detected = estimate_growth_radius_from_diff(
            diff,
            (center[0] * scale, center[1] * scale),
            max_radius=max(4, int(round(max_radius * scale))),
            expected_radius=previous_growth_radius * scale
            if previous_growth_radius is not None and math.isfinite(previous_growth_radius)
            else None,
            max_step_px=BUBBLE_MAX_STEP_PX * scale,
        )
        return detected / scale

    diff = cv2.absdiff(gray, baseline_gray)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)
    return estimate_growth_radius_from_diff(
        diff,
        center,
        max_radius=max_radius,
        expected_radius=previous_growth_radius,
    )


def circle_edge_support(gray: np.ndarray, x: float, y: float, radius: float) -> float:
    gy, gx = np.gradient(gray.astype(float))
    gradient = np.hypot(gx, gy)
    angles = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    xs = np.rint(x + radius * np.cos(angles)).astype(int)
    ys = np.rint(y + radius * np.sin(angles)).astype(int)
    valid = (xs >= 0) & (xs < gray.shape[1]) & (ys >= 0) & (ys < gray.shape[0])
    if np.count_nonzero(valid) < 60:
        return 0.0
    values = gradient[ys[valid], xs[valid]]
    return float(np.mean(values) * np.count_nonzero(valid) / len(valid))


def detect_film_calibration(cv2, frame: np.ndarray, needle_center: tuple[float, float]) -> FilmCalibration | None:
    height, width = frame.shape[:2]
    scale = min(1.0, 900.0 / float(max(height, width)))
    small = cv2.resize(frame, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.medianBlur(gray, 5)

    min_radius = int(min(gray.shape) * 0.25)
    max_radius = int(min(gray.shape) * 0.58)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(gray.shape) * 0.20,
        param1=80,
        param2=22,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    needle_x = needle_center[0] * scale
    needle_y = needle_center[1] * scale
    best_circle: tuple[float, float, float] | None = None
    best_score = -math.inf

    for x, y, radius in circles[0]:
        distance_to_needle = math.hypot(float(x) - needle_x, float(y) - needle_y)
        contains_needle = distance_to_needle <= float(radius) * 0.98
        support = circle_edge_support(gray, float(x), float(y), float(radius))
        center_penalty = distance_to_needle / max(float(radius), 1.0)
        score = support + float(radius) * 0.03 - center_penalty * 8.0
        if contains_needle:
            score += 18.0
        if score > best_score:
            best_score = score
            best_circle = (float(x), float(y), float(radius))

    if best_circle is None:
        return None

    x, y, radius = best_circle
    return FilmCalibration(center_x=x / scale, center_y=y / scale, radius_px=radius / scale)


def manual_film_calibration(annotations: pd.DataFrame, sample: str) -> FilmCalibration | None:
    required = {"sample", "film_center_x", "film_center_y", "film_radius_px"}
    if annotations.empty or not required.issubset(annotations.columns):
        return None
    rows = annotations[annotations["sample"].astype(str) == sample].copy()
    if rows.empty:
        return None
    for col in ["film_center_x", "film_center_y", "film_radius_px"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.dropna(subset=["film_center_x", "film_center_y", "film_radius_px"])
    if rows.empty:
        return None
    return FilmCalibration(
        center_x=float(rows["film_center_x"].median()),
        center_y=float(rows["film_center_y"].median()),
        radius_px=float(rows["film_radius_px"].median()),
    )


def estimate_radius_from_video_gray(
    cv2,
    gray: np.ndarray,
    center: tuple[float, float],
    max_radius: int,
    previous_radius: float | None,
) -> float:
    scale = min(1.0, DETECTION_MAX_SIDE / float(max(gray.shape)))
    if scale < 1.0:
        resized_width = max(1, int(round(gray.shape[1] * scale)))
        resized_height = max(1, int(round(gray.shape[0] * scale)))
        gray_for_detection = cv2.resize(gray, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        detection_center = (center[0] * scale, center[1] * scale)
        detection_min_radius = max(3, int(round(RADIUS_MIN_PX * scale)))
        detection_max_radius = max(detection_min_radius + 1, int(round(max_radius * scale)))
        detection_previous = previous_radius * scale if USE_RADIUS_CONTINUITY and previous_radius is not None else None
        detection_margin = max(4, int(round(RADIUS_SEARCH_MARGIN_PX * scale)))
        detected = estimate_radius_from_gray(
            gray_for_detection,
            detection_center,
            min_radius=detection_min_radius,
            max_radius=detection_max_radius,
            expected_radius=detection_previous,
            search_margin=detection_margin,
        )
        return detected / scale if math.isfinite(detected) else math.nan

    return estimate_radius_from_gray(
        gray,
        center,
        min_radius=RADIUS_MIN_PX,
        max_radius=max_radius,
        expected_radius=previous_radius if USE_RADIUS_CONTINUITY else None,
        search_margin=RADIUS_SEARCH_MARGIN_PX,
    )


def discover_videos(video_dir: Path) -> list[Path]:
    return sorted(path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def annotation_frame_indices(frame_count: int, count: int) -> list[int]:
    frame_count = max(0, int(frame_count))
    count = max(1, int(count))
    if frame_count == 0:
        return []
    if frame_count <= count:
        return list(range(frame_count))
    return sorted({int(round(value)) for value in np.linspace(0, frame_count - 1, count)})


def annotation_time_seconds(frame_index: int, fps: float) -> float:
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"FPS must be a positive finite number, got {fps!r}")
    return int(frame_index) / fps


def format_annotation_time(time_s: float) -> str:
    time_s = float(time_s)
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError(f"Annotation time must be a nonnegative finite number, got {time_s!r}")
    return f"{time_s:.3f}"


def annotation_image_filename(frame_index: int, fps: float) -> str:
    time_text = format_annotation_time(annotation_time_seconds(frame_index, fps))
    return f"time_{time_text}s.jpg"


def build_annotation_export_metadata(
    sample: str,
    frame_index: int,
    fps: float,
    sample_dir: Path,
) -> tuple[Path, dict[str, object]]:
    time_text = format_annotation_time(annotation_time_seconds(frame_index, fps))
    frame_file = sample_dir / annotation_image_filename(frame_index, fps)
    row: dict[str, object] = {
        "sample": sample,
        "frame": int(frame_index),
        "time_s": time_text,
        "radius_px": "",
        "film_center_x": "",
        "film_center_y": "",
        "film_radius_px": "",
        "image_path": str(frame_file),
        "notes": "",
    }
    return frame_file, row


def remove_legacy_annotation_images(sample_dir: Path) -> list[Path]:
    removed: list[Path] = []
    for path in sorted(sample_dir.glob("frame_*.jpg")):
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def write_image_unicode(cv2, path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def export_annotation_frames(
    video_dir: Path = VIDEO_DIR,
    output_root: Path = OUTPUT_ROOT,
    *,
    only: Sequence[str] | None = None,
    frame_count: int = 6,
) -> Path:
    cv2 = require_cv2()
    output_dir = output_root / "人工标注帧"
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(video_dir)
    if only:
        wanted = {Path(item).stem for item in only}
        videos = [video for video in videos if video.stem in wanted]
    rows: list[dict[str, object]] = []

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if not math.isfinite(fps) or fps <= 0:
            cap.release()
            print(f"Skipping {video_path.name}: invalid FPS {fps!r}")
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        center = (width / 2.0, height / 2.0)
        sample_dir = output_dir / video_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        removed_legacy = remove_legacy_annotation_images(sample_dir)
        if removed_legacy:
            print(f"Removed {len(removed_legacy)} legacy frame images: {sample_dir}")
        used_filenames: set[str] = set()
        for frame_index in annotation_frame_indices(total_frames, frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_file, row = build_annotation_export_metadata(
                video_path.stem,
                frame_index,
                fps,
                sample_dir,
            )
            if frame_file.name in used_filenames:
                cap.release()
                raise RuntimeError(f"Duplicate annotation time filename for {video_path.name}: {frame_file.name}")
            used_filenames.add(frame_file.name)
            time_text = str(row["time_s"])
            cx, cy = int(round(center[0])), int(round(center[1]))
            annotated = frame.copy()
            cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=34, thickness=2)
            cv2.putText(
                annotated,
                f"{video_path.stem} time={time_text}s",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            write_image_unicode(cv2, frame_file, annotated)
            rows.append(row)
        cap.release()

    template_path = output_dir / "manual_annotations_template.csv"
    pd.DataFrame(rows).to_csv(template_path, index=False, encoding="utf-8-sig")
    print(f"Annotation frames saved: {output_dir}")
    print(f"Template saved: {template_path}")
    return template_path


def read_displacement_table(workbook: Path, sheet_name: str) -> pd.DataFrame:
    table = pd.read_excel(workbook, sheet_name=sheet_name)
    rename_columns: dict[str, str] = {}
    for canonical, aliases in (
        (TIME_COL, TIME_COLUMN_ALIASES),
        (DISPLACEMENT_COL, DISPLACEMENT_COLUMN_ALIASES),
    ):
        if canonical in table.columns:
            continue
        source = next((alias for alias in aliases if alias in table.columns), None)
        if source is not None:
            rename_columns[source] = canonical
    if rename_columns:
        table = table.rename(columns=rename_columns)
    missing = [col for col in (TIME_COL, DISPLACEMENT_COL) if col not in table.columns]
    if missing:
        raise ValueError(f"Sheet {sheet_name!r} missing columns: {missing}")
    return table


def configure_matplotlib_fonts() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_radius_displacement(
    data: pd.DataFrame,
    fit: FitResult,
    output_path: Path,
    title: str,
    *,
    radius_col: str = "radius_px_smooth",
    y_label: str = "泡半径 (px)",
    slope_unit: str = "px/mm",
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib_fonts()
    plot_data = data.dropna(subset=["displacement_mm", radius_col])
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    ax.plot(
        plot_data["displacement_mm"],
        plot_data[radius_col],
        color="#2563EB",
        linewidth=1.8,
        label="radius",
    )
    ax.scatter(plot_data["displacement_mm"], plot_data[radius_col], s=8, color="#1D4ED8", alpha=0.45)

    if fit.points >= 2 and math.isfinite(fit.slope_px_per_mm):
        fit_x = np.linspace(fit.fit_start_mm, fit.fit_end_mm, 100)
        fit_y = fit.slope_px_per_mm * fit_x + fit.intercept_px
        label = f"fit: slope={fit.slope_px_per_mm:.4g} {slope_unit}, R2={fit.r_squared:.4g}"
        ax.plot(fit_x, fit_y, color="#DC2626", linewidth=2.2, label=label)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("位移 (mm)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: opencv-python. Install it with: python -m pip install opencv-python"
        ) from exc
    return cv2


def make_video_writer(cv2, output_path: Path, fps: float, width: int, height: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")
    return writer


def draw_radius_overlay(
    cv2,
    frame: np.ndarray,
    center: tuple[float, float],
    radius: float,
    radius_mm: float | None = None,
    film_calibration: FilmCalibration | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    if math.isfinite(radius):
        cx, cy = int(round(center[0])), int(round(center[1]))
        cv2.circle(annotated, (cx, cy), int(round(radius)), (0, 0, 255), 3)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        label = f"r={radius:.1f}px"
        if radius_mm is not None and math.isfinite(radius_mm):
            label += f" / {radius_mm:.2f}mm"
        cv2.putText(
            annotated,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if film_calibration is not None:
        cv2.circle(
            annotated,
            (int(round(film_calibration.center_x)), int(round(film_calibration.center_y))),
            int(round(film_calibration.radius_px)),
            (0, 180, 0),
            2,
        )
    return annotated


def write_annotated_video_file(
    cv2,
    input_video: Path,
    output_video: Path,
    radii_px: Sequence[float],
    fps: float,
    width: int,
    height: int,
    center: tuple[float, float],
    radii_mm: Sequence[float] | None = None,
    film_calibration: FilmCalibration | None = None,
) -> None:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video for annotation: {input_video}")

    writer = make_video_writer(cv2, output_video, fps, width, height)
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            radius = float(radii_px[frame_index]) if frame_index < len(radii_px) else math.nan
            radius_mm = float(radii_mm[frame_index]) if radii_mm is not None and frame_index < len(radii_mm) else None
            writer.write(draw_radius_overlay(cv2, frame, center, radius, radius_mm, film_calibration))
            frame_index += 1
    finally:
        cap.release()
        writer.release()


def analyze_video(
    video_path: Path,
    workbook: Path,
    sheet_names: Sequence[str],
    annotated_video_dir: Path,
    slope_output_dir: Path,
    *,
    write_annotated_video: bool = True,
    fit_range: tuple[float, float] | None = FIT_DISPLACEMENT_RANGE_MM,
    frame_step: int = 1,
    manual_annotations: pd.DataFrame | None = None,
) -> dict[str, object]:
    cv2 = require_cv2()
    stem = video_path.stem
    matched_sheet = match_sheet_name(stem, sheet_names)
    data_csv = slope_output_dir / f"{stem}_radius_displacement.csv"
    plot_png = slope_output_dir / f"{stem}_radius_displacement.png"
    annotated_video = annotated_video_dir / f"{stem}_red_circle.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video": video_path.name,
            "matched_sheet": matched_sheet,
            "status": "video_open_failed",
            "message": "Cannot open video file",
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0 or not math.isfinite(fps):
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    center = (width / 2.0, height / 2.0)

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        return {
            "video": video_path.name,
            "matched_sheet": matched_sheet,
            "status": "video_read_failed",
            "message": "Cannot read first video frame",
        }

    manual_annotations = manual_annotations if manual_annotations is not None else pd.DataFrame()
    film_calibration = manual_film_calibration(manual_annotations, stem)
    if film_calibration is None:
        mid_frame = first_frame
        if frame_count > 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
            ok_mid, maybe_mid = cap.read()
            if ok_mid:
                mid_frame = maybe_mid
        film_calibration = detect_film_calibration(cv2, mid_frame, center)

    if film_calibration is not None and math.isfinite(film_calibration.radius_px):
        max_growth_radius = int(max(20, film_calibration.radius_px * BUBBLE_MAX_RADIUS_FRACTION_OF_FILM))
    else:
        max_growth_radius = int(min(width, height) * BUBBLE_MAX_RADIUS_FALLBACK_FRACTION)

    baseline_gray = prepare_bubble_gray(cv2, first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows: list[dict[str, float | int]] = []
    previous_radius: float | None = None
    frame_index = 0
    frame_step = max(1, int(frame_step))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            radius = previous_radius
            if frame_index % frame_step == 0 or previous_radius is None:
                gray = prepare_bubble_gray(cv2, frame)
                radius = estimate_growth_radius_from_video_gray(
                    cv2,
                    gray,
                    baseline_gray,
                    center,
                    max_growth_radius,
                    previous_radius,
                )
                if math.isfinite(radius):
                    previous_radius = radius

            rows.append(
                {
                    "frame": frame_index,
                    "time_s": frame_index / fps,
                    "radius_px": float(radius) if radius is not None else math.nan,
                }
            )

            frame_index += 1
    finally:
        cap.release()

    data = pd.DataFrame(rows)
    manual_radius = interpolate_manual_radius_series(manual_annotations, stem, frame_count=len(data))
    if manual_radius is not None:
        data["radius_px_smooth"] = manual_radius
        radius_source = "manual"
    else:
        data["radius_px_smooth"] = postprocess_radius_series(data["radius_px"].to_numpy(dtype=float))
        radius_source = "auto_diff"

    if film_calibration is not None:
        data["radius_mm"] = data["radius_px"].astype(float) * film_calibration.mm_per_px
        data["radius_mm_smooth"] = data["radius_px_smooth"].astype(float) * film_calibration.mm_per_px
        data["radius_cm_smooth"] = data["radius_mm_smooth"] / 10.0
        data["film_center_x"] = film_calibration.center_x
        data["film_center_y"] = film_calibration.center_y
        data["film_radius_px"] = film_calibration.radius_px
        data["mm_per_px"] = film_calibration.mm_per_px
    else:
        data["radius_mm"] = math.nan
        data["radius_mm_smooth"] = math.nan
        data["radius_cm_smooth"] = math.nan
        data["film_center_x"] = math.nan
        data["film_center_y"] = math.nan
        data["film_radius_px"] = math.nan
        data["mm_per_px"] = math.nan

    status = "ok"
    message = ""
    fit_px = FitResult(math.nan, math.nan, math.nan, 0, math.nan, math.nan)
    fit_mm = FitResult(math.nan, math.nan, math.nan, 0, math.nan, math.nan)

    if matched_sheet is None:
        data["displacement_mm"] = math.nan
        status = "missing_sheet"
        message = "No matching displacement sheet"
    else:
        try:
            displacement_table = read_displacement_table(workbook, matched_sheet)
            data["displacement_mm"] = interpolate_displacement(displacement_table, data["time_s"].to_numpy(dtype=float))
            fit_px = fit_radius_displacement(data["displacement_mm"], data["radius_px_smooth"], fit_range=fit_range)
            if data["radius_mm_smooth"].notna().any():
                fit_mm = fit_radius_displacement(data["displacement_mm"], data["radius_mm_smooth"], fit_range=fit_range)
                plot_radius_displacement(
                    data,
                    fit_mm,
                    plot_png,
                    stem,
                    radius_col="radius_mm_smooth",
                    y_label="泡半径 (mm)",
                    slope_unit="mm/mm",
                )
            else:
                plot_radius_displacement(data, fit_px, plot_png, stem)
        except Exception as exc:  # keep processing the remaining videos
            data["displacement_mm"] = math.nan
            status = "displacement_error"
            message = str(exc)

    slope_output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_csv, index=False, encoding="utf-8-sig")

    if write_annotated_video:
        write_annotated_video_file(
            cv2,
            video_path,
            annotated_video,
            data["radius_px_smooth"].to_numpy(dtype=float),
            fps,
            width,
            height,
            center,
            radii_mm=data["radius_mm_smooth"].to_numpy(dtype=float)
            if data["radius_mm_smooth"].notna().any()
            else None,
            film_calibration=film_calibration,
        )

    return {
        "video": video_path.name,
        "sample": stem,
        "matched_sheet": matched_sheet,
        "status": status,
        "message": message,
        "frames": int(frame_index),
        "fps": fps,
        "width": width,
        "height": height,
        "center_x": center[0],
        "center_y": center[1],
        "radius_source": radius_source,
        "film_center_x": film_calibration.center_x if film_calibration is not None else math.nan,
        "film_center_y": film_calibration.center_y if film_calibration is not None else math.nan,
        "film_radius_px": film_calibration.radius_px if film_calibration is not None else math.nan,
        "mm_per_px": film_calibration.mm_per_px if film_calibration is not None else math.nan,
        "bubble_max_radius_px": max_growth_radius,
        "slope_px_per_mm": fit_px.slope_px_per_mm,
        "intercept_px": fit_px.intercept_px,
        "r_squared": fit_px.r_squared,
        "slope_radius_mm_per_displacement_mm": fit_mm.slope_px_per_mm,
        "intercept_radius_mm": fit_mm.intercept_px,
        "r_squared_radius_mm": fit_mm.r_squared,
        "fit_points": fit_px.points,
        "fit_start_mm": fit_px.fit_start_mm,
        "fit_end_mm": fit_px.fit_end_mm,
        "radius_displacement_csv": str(data_csv),
        "radius_displacement_plot": str(plot_png) if plot_png.exists() else "",
        "annotated_video": str(annotated_video) if write_annotated_video else "",
    }


def build_unmatched_sheet_rows(sheet_names: Sequence[str], matched_sheets: Iterable[str | None]) -> pd.DataFrame:
    matched = {sheet for sheet in matched_sheets if sheet}
    ignored = {"结果"}
    rows = [{"sheet": sheet} for sheet in sheet_names if sheet not in ignored and sheet not in matched]
    return pd.DataFrame(rows)


def parse_fit_range(values: Sequence[float] | None) -> tuple[float, float] | None:
    if not values:
        return None
    if len(values) != 2:
        raise ValueError("--fit-range needs exactly two numbers")
    return float(values[0]), float(values[1])


def run_batch(
    video_dir: Path = VIDEO_DIR,
    workbook: Path = DISPLACEMENT_WORKBOOK,
    output_root: Path = OUTPUT_ROOT,
    *,
    limit: int | None = None,
    only: Sequence[str] | None = None,
    write_annotated_video: bool = True,
    fit_range: tuple[float, float] | None = FIT_DISPLACEMENT_RANGE_MM,
    frame_step: int = 1,
    manual_annotation_file: Path = MANUAL_ANNOTATION_FILE,
) -> pd.DataFrame:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video folder not found: {video_dir}")
    if not workbook.exists():
        raise FileNotFoundError(f"Displacement workbook not found: {workbook}")

    annotated_video_dir = output_root / ANNOTATED_VIDEO_DIR.name
    slope_output_dir = output_root / SLOPE_OUTPUT_DIR.name
    annotated_video_dir.mkdir(parents=True, exist_ok=True)
    slope_output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(video_dir)
    if only:
        wanted = {Path(item).stem for item in only}
        videos = [video for video in videos if video.stem in wanted]
    if limit is not None:
        videos = videos[:limit]
    if not videos:
        raise ValueError("No videos selected for analysis")

    xls = pd.ExcelFile(workbook)
    sheet_names = list(xls.sheet_names)
    manual_annotations = load_manual_annotations(manual_annotation_file)
    results: list[dict[str, object]] = []

    print(f"Video folder: {video_dir}")
    print(f"Workbook: {workbook}")
    print(f"Output root: {output_root}")
    print(f"Selected videos: {len(videos)}")
    if not manual_annotations.empty:
        print(f"Manual annotations: {manual_annotation_file}")

    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path.name}")
        result = analyze_video(
            video_path,
            workbook,
            sheet_names,
            annotated_video_dir,
            slope_output_dir,
            write_annotated_video=write_annotated_video,
            fit_range=fit_range,
            frame_step=frame_step,
            manual_annotations=manual_annotations,
        )
        print(f"  status={result.get('status')} sheet={result.get('matched_sheet')}")
        results.append(result)

    summary = pd.DataFrame(results)
    summary_path = slope_output_dir / "radius_displacement_slope_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    unmatched = build_unmatched_sheet_rows(sheet_names, summary["matched_sheet"].tolist())
    unmatched_path = slope_output_dir / "unmatched_displacement_sheets.csv"
    unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

    print(f"Summary saved: {summary_path}")
    print(f"Unmatched sheets saved: {unmatched_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze video radius versus displacement and write red-circle videos.")
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--workbook", type=Path, default=DISPLACEMENT_WORKBOOK)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos.")
    parser.add_argument("--only", nargs="*", default=None, help="Process only these video stems or filenames.")
    parser.add_argument("--no-video", action="store_true", help="Skip writing annotated red-circle videos.")
    parser.add_argument("--fit-range", nargs=2, type=float, default=None, metavar=("START_MM", "END_MM"))
    parser.add_argument("--frame-step", type=int, default=1, help="Estimate radius every N frames and reuse between them.")
    parser.add_argument("--manual-annotations", type=Path, default=MANUAL_ANNOTATION_FILE)
    parser.add_argument("--export-annotation-frames", action="store_true")
    parser.add_argument("--annotation-frame-count", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.export_annotation_frames:
        export_annotation_frames(
            video_dir=args.video_dir,
            output_root=args.output_root,
            only=args.only,
            frame_count=args.annotation_frame_count,
        )
        return
    run_batch(
        video_dir=args.video_dir,
        workbook=args.workbook,
        output_root=args.output_root,
        limit=args.limit,
        only=args.only,
        write_annotated_video=not args.no_video,
        fit_range=parse_fit_range(args.fit_range),
        frame_step=args.frame_step,
        manual_annotation_file=args.manual_annotations,
    )


if __name__ == "__main__":
    main()
