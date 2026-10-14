from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from radius_prepare import (
    RadiusConfig,
    SELECTED_COLUMNS,
    load_radius_config,
    load_selected_manifest,
    validate_selection_ranges,
)
from video_radius_analysis import FORCE_SAMPLE_ORDER


POINT_COLUMNS = [
    "sample",
    "point_order",
    "frame",
    "time_s",
    "displacement_mm",
    "circle_center_x_px",
    "circle_center_y_px",
    "radius_px",
    "mm_per_px",
    "radius_mm",
    "red_pixel_count",
    "boundary_marker_count",
    "angular_coverage",
    "circle_rmse_px",
    "circle_rmse_fraction",
    "annotation_source",
    "used_in_fit",
    "point_status",
]
RESULT_COLUMNS = [
    "sample",
    "liquid",
    "thickness_um",
    "conditioning_time_h",
    "replicate",
    "status",
    "slope_mm_per_mm",
    "intercept_mm",
    "r_squared",
    "fit_point_count",
    "fit_start_displacement_mm",
    "fit_end_displacement_mm",
    "fit_start_frame",
    "fit_end_frame",
    "fit_start_time_s",
    "fit_end_time_s",
    "calibration_center_x_px",
    "calibration_center_y_px",
    "film_radius_px",
    "mm_per_px",
    "calibration_rmse_px",
    "calibration_angular_coverage_deg",
    "residual_rmse_mm",
    "slope_standard_error",
    "slope_ci95_normal_low",
    "slope_ci95_normal_high",
    "leave_one_out_max_relative_slope_change",
    "selection_signature",
    "failure_reason",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "radius_config.xlsx"
DEFAULT_FORCE_CONFIG = PROJECT_ROOT / "config" / "force_config.xlsx"
DEFAULT_SELECTION = PROJECT_ROOT / "outputs" / "radius" / "selected_frames.csv"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "radius"
DEFAULT_ANNOTATION_ROOT = PROJECT_ROOT / "config" / "radius_annotations"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "radius"
USER_EXCLUDED_SAMPLES = {"ipa_25um_6h"}


@dataclass(frozen=True)
class CircleFit:
    center_x_px: float
    center_y_px: float
    radius_px: float
    fit_rmse_px: float
    angular_coverage_deg: float
    pixel_count: int


@dataclass(frozen=True)
class FilmCalibration(CircleFit):
    mm_per_px: float


@dataclass(frozen=True)
class BoundaryFit(CircleFit):
    marker_count: int
    fit_rmse_fraction: float


@dataclass(frozen=True)
class RadiusFit:
    slope_mm_per_mm: float
    intercept_mm: float
    r_squared: float
    fit_point_count: int
    residual_rmse_mm: float
    slope_standard_error: float
    slope_ci95_normal_low: float
    slope_ci95_normal_high: float
    leave_one_out_max_relative_slope_change: float


def _require_matching_images(annotated: np.ndarray, raw: np.ndarray) -> None:
    if annotated.shape != raw.shape:
        raise ValueError("annotated and raw images must have identical shape")
    if annotated.ndim != 3 or annotated.shape[2] != 3:
        raise ValueError("annotated and raw images must have three BGR channels")


def build_added_color_mask(
    annotated: np.ndarray,
    raw: np.ndarray,
    channel: int,
) -> np.ndarray:
    _require_matching_images(annotated, raw)
    if channel not in (0, 1, 2):
        raise ValueError("BGR channel must be 0, 1, or 2")
    values = annotated.astype(np.int16)
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    other = [index for index in range(3) if index != channel]
    difference = np.max(np.abs(values - raw.astype(np.int16)), axis=2)
    return (
        (values[:, :, channel] >= 140)
        & (values[:, :, channel] - values[:, :, other[0]] >= 55)
        & (values[:, :, channel] - values[:, :, other[1]] >= 55)
        & (maximum - minimum >= 80)
        & (difference >= 30)
    )


def _fit_algebraic_circle(points: np.ndarray) -> tuple[float, float, float]:
    if len(points) < 3:
        raise ValueError("circle has too few points")
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack((x, y, np.ones(len(points))))
    coefficients, _, rank, _ = np.linalg.lstsq(
        design, -(x * x + y * y), rcond=None
    )
    if rank < 3:
        raise ValueError("points cannot define a circle")
    center_x = -float(coefficients[0]) / 2.0
    center_y = -float(coefficients[1]) / 2.0
    radius_squared = (
        center_x * center_x + center_y * center_y - float(coefficients[2])
    )
    if not math.isfinite(radius_squared) or radius_squared <= 0:
        raise ValueError("circle fit produced an invalid radius")
    return center_x, center_y, math.sqrt(radius_squared)


def _fit_geometric_circle(points: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
        raise ValueError("geometric circle fit requires at least three two-dimensional points")
    if not np.isfinite(values).all() or len(np.unique(values, axis=0)) < 3:
        raise ValueError("geometric circle fit requires finite, distinct points")

    origin = values.mean(axis=0)
    centered = values - origin
    singular_values = np.linalg.svd(centered, compute_uv=False)
    tolerance = np.finfo(float).eps
    if (
        singular_values[0] <= tolerance
        or singular_values[1] <= 1.0e-10 * singular_values[0]
    ):
        raise ValueError("points cannot define a well-conditioned circle")
    scale = float(math.sqrt(float(np.mean(np.sum(centered * centered, axis=1)))))
    if not math.isfinite(scale) or scale <= tolerance:
        raise ValueError("points cannot define a circle scale")

    normalized = centered / scale
    parameters = np.asarray(_fit_algebraic_circle(normalized), dtype=float)

    def residuals(current: np.ndarray) -> np.ndarray:
        return np.linalg.norm(current[:2] - normalized, axis=1) - current[2]

    def jacobian(current: np.ndarray) -> np.ndarray:
        offsets = current[:2] - normalized
        distances = np.linalg.norm(offsets, axis=1)
        if np.any(distances <= tolerance):
            raise ValueError("geometric circle center coincides with a boundary point")
        return np.column_stack(
            (offsets / distances[:, np.newaxis], -np.ones(len(values)))
        )

    converged = False
    for _ in range(100):
        current_residuals = residuals(parameters)
        current_jacobian = jacobian(parameters)
        gradient = current_jacobian.T @ current_residuals
        if float(np.linalg.norm(gradient, ord=np.inf)) <= 1.0e-10:
            converged = True
            break
        try:
            step = np.linalg.solve(
                current_jacobian.T @ current_jacobian, -gradient
            )
        except np.linalg.LinAlgError as error:
            raise ValueError("geometric circle fit is singular") from error
        cost = float(current_residuals @ current_residuals)
        accepted = False
        for power in range(30):
            candidate = parameters + step * (0.5**power)
            if not np.isfinite(candidate).all() or candidate[2] <= tolerance:
                continue
            candidate_residuals = residuals(candidate)
            candidate_cost = float(candidate_residuals @ candidate_residuals)
            rounding_tolerance = (
                10.0 * np.finfo(float).eps * max(1.0, cost)
            )
            if candidate_cost <= cost + rounding_tolerance:
                parameters = candidate
                accepted = True
                break
        if not accepted:
            raise ValueError("geometric circle fit line search failed")
    if not converged:
        final_gradient = jacobian(parameters).T @ residuals(parameters)
        if float(np.linalg.norm(final_gradient, ord=np.inf)) > 1.0e-8:
            raise ValueError("geometric circle fit did not converge")

    center_x = float(origin[0] + scale * parameters[0])
    center_y = float(origin[1] + scale * parameters[1])
    radius = float(scale * parameters[2])
    if not all(math.isfinite(value) for value in (center_x, center_y, radius)) or radius <= 0:
        raise ValueError("geometric circle fit produced invalid parameters")
    return center_x, center_y, radius


def _angular_coverage_deg(
    points: np.ndarray,
    center_x: float,
    center_y: float,
) -> float:
    angles = np.sort(
        np.mod(
            np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x),
            2.0 * np.pi,
        )
    )
    if len(angles) < 2:
        return 0.0
    gaps = np.diff(np.concatenate((angles, [angles[0] + 2.0 * np.pi])))
    return float(360.0 - math.degrees(float(gaps.max())))


def _fit_component(
    points: np.ndarray,
    min_pixels: int,
    min_coverage_deg: float,
    rmse_floor_px: float,
    rmse_fraction: float,
    min_radius_px: float,
) -> CircleFit | None:
    if len(points) < min_pixels:
        return None
    inliers = np.ones(len(points), dtype=bool)
    try:
        for _ in range(6):
            center_x, center_y, radius = _fit_algebraic_circle(points[inliers])
            residuals = (
                np.hypot(points[:, 0] - center_x, points[:, 1] - center_y) - radius
            )
            median = float(np.median(residuals[inliers]))
            mad = float(np.median(np.abs(residuals[inliers] - median)))
            updated = np.abs(residuals - median) <= max(rmse_floor_px, 3.0 * mad)
            if int(updated.sum()) < min_pixels:
                return None
            if np.array_equal(updated, inliers):
                break
            inliers = updated
        center_x, center_y, radius = _fit_algebraic_circle(points[inliers])
    except (ValueError, np.linalg.LinAlgError):
        return None
    residuals = (
        np.hypot(points[inliers, 0] - center_x, points[inliers, 1] - center_y)
        - radius
    )
    rmse = float(math.sqrt(float(np.mean(residuals * residuals))))
    coverage = _angular_coverage_deg(points[inliers], center_x, center_y)
    pixel_count = int(inliers.sum())
    if (
        pixel_count < min_pixels
        or radius <= min_radius_px
        or coverage < min_coverage_deg
        or rmse > max(rmse_floor_px, rmse_fraction * radius)
    ):
        return None
    return CircleFit(
        center_x_px=center_x,
        center_y_px=center_y,
        radius_px=radius,
        fit_rmse_px=rmse,
        angular_coverage_deg=coverage,
        pixel_count=pixel_count,
    )


def _component_fits(
    mask: np.ndarray,
    min_pixels: int,
    min_coverage_deg: float,
    rmse_floor_px: float,
    rmse_fraction: float,
    min_radius_px: float,
) -> list[CircleFit]:
    connected = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        connected, connectivity=8
    )
    fits: list[CircleFit] = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_pixels:
            continue
        y, x = np.nonzero(labels == label)
        fit = _fit_component(
            np.column_stack((x, y)).astype(float),
            min_pixels=min_pixels,
            min_coverage_deg=min_coverage_deg,
            rmse_floor_px=rmse_floor_px,
            rmse_fraction=rmse_fraction,
            min_radius_px=min_radius_px,
        )
        if fit is not None:
            fits.append(fit)
    return fits


def fit_green_calibration(
    annotated: np.ndarray,
    raw: np.ndarray,
    film_diameter_mm: float,
) -> FilmCalibration:
    if not math.isfinite(film_diameter_mm) or film_diameter_mm <= 0:
        raise ValueError("film_diameter_mm must be positive and finite")
    green = build_added_color_mask(annotated, raw, channel=1)
    connected = cv2.morphologyEx(
        green.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    y, x = np.nonzero(connected)
    fit = _fit_component(
        np.column_stack((x, y)).astype(float),
        min_pixels=120,
        min_coverage_deg=120.0,
        rmse_floor_px=3.0,
        rmse_fraction=0.012,
        min_radius_px=5.0,
    )
    if fit is None:
        raise ValueError("green film calibration must contain valid arcs from one circle")
    return FilmCalibration(
        center_x_px=fit.center_x_px,
        center_y_px=fit.center_y_px,
        radius_px=fit.radius_px,
        fit_rmse_px=fit.fit_rmse_px,
        angular_coverage_deg=fit.angular_coverage_deg,
        pixel_count=fit.pixel_count,
        mm_per_px=float(film_diameter_mm) / (2.0 * fit.radius_px),
    )


def fit_red_annotation(
    annotated: np.ndarray,
    raw: np.ndarray,
    system_red_mask: np.ndarray | None = None,
) -> BoundaryFit:
    red = build_added_color_mask(annotated, raw, channel=2)
    title_end = int(math.ceil(annotated.shape[0] * 0.18))
    red[:title_end] = False
    if system_red_mask is not None:
        if system_red_mask.shape != red.shape:
            raise ValueError("system red mask must match image height and width")
        expanded = cv2.dilate(
            system_red_mask.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        red[expanded != 0] = False
    # Keep nearby hand-drawn dots separate. A closing operation can bridge the
    # one- or two-pixel JPEG gaps between adjacent markers and undercount them.
    connected = red.astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        connected, connectivity=8
    )
    marker_centers: list[tuple[float, float]] = []
    marker_shapes: list[tuple[int, int, int]] = []
    red_pixel_count = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > 1500:
            continue
        if max(width, height) / min(width, height) > 4.0:
            continue
        marker_centers.append(
            (float(centroids[label, 0]), float(centroids[label, 1]))
        )
        marker_shapes.append((area, width, height))
        red_pixel_count += area

    if len(marker_centers) == 1:
        area, width, height = marker_shapes[0]
        if area <= 250 and width <= 20 and height <= 20:
            center_x, center_y = marker_centers[0]
            return BoundaryFit(
                center_x_px=center_x,
                center_y_px=center_y,
                radius_px=0.0,
                fit_rmse_px=0.0,
                angular_coverage_deg=0.0,
                pixel_count=red_pixel_count,
                marker_count=1,
                fit_rmse_fraction=0.0,
            )

    if not 5 <= len(marker_centers) <= 40:
        raise ValueError(
            "expected one zero-radius marker or 5 to 40 separate red boundary markers"
        )
    points = np.asarray(marker_centers, dtype=float)
    try:
        center_x, center_y, radius = _fit_geometric_circle(points)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError("red boundary markers cannot define a circle") from error
    coverage = _angular_coverage_deg(points, center_x, center_y)
    if radius <= 5.0 or coverage < 180.0:
        raise ValueError(
            "red boundary markers must span at least 180 degrees around the boundary"
        )
    residuals = np.hypot(points[:, 0] - center_x, points[:, 1] - center_y) - radius
    rmse = float(math.sqrt(float(np.mean(residuals * residuals))))
    return BoundaryFit(
        center_x_px=center_x,
        center_y_px=center_y,
        radius_px=radius,
        fit_rmse_px=rmse,
        angular_coverage_deg=coverage,
        pixel_count=red_pixel_count,
        marker_count=len(marker_centers),
        fit_rmse_fraction=rmse / radius,
    )


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((x, np.ones(len(x))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2:
        raise ValueError("displacement values cannot define a linear fit")
    return float(coefficients[0]), float(coefficients[1])


def fit_radius_displacement(
    displacement_mm: Sequence[float],
    radius_mm: Sequence[float],
) -> RadiusFit:
    x = np.asarray(displacement_mm, dtype=float)
    y = np.asarray(radius_mm, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("displacement and radius must be equal-length one-dimensional data")
    if len(x) < 3:
        raise ValueError("radius-displacement fit requires at least three points")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("radius-displacement fit requires finite values")
    if len(np.unique(x)) < 2:
        raise ValueError("radius-displacement fit requires varying displacement")

    slope, intercept = _ols_slope_intercept(x, y)
    predicted = slope * x + intercept
    residuals = y - predicted
    sse = float(np.sum(residuals * residuals))
    centered = y - float(y.mean())
    sst = float(np.sum(centered * centered))
    r_squared = 1.0 if sst == 0.0 and sse == 0.0 else 1.0 - sse / sst
    residual_rmse = math.sqrt(sse / len(x))
    sxx = float(np.sum((x - float(x.mean())) ** 2))
    slope_standard_error = math.sqrt((sse / (len(x) - 2)) / sxx)
    ci_half_width = 1.96 * slope_standard_error

    leave_one_out_slopes: list[float] = []
    for omitted in range(len(x)):
        keep = np.arange(len(x)) != omitted
        leave_one_out_slope, _ = _ols_slope_intercept(x[keep], y[keep])
        leave_one_out_slopes.append(leave_one_out_slope)
    denominator = max(abs(slope), np.finfo(float).eps)
    sensitivity = max(abs(value - slope) / denominator for value in leave_one_out_slopes)
    return RadiusFit(
        slope_mm_per_mm=slope,
        intercept_mm=intercept,
        r_squared=r_squared,
        fit_point_count=len(x),
        residual_rmse_mm=residual_rmse,
        slope_standard_error=slope_standard_error,
        slope_ci95_normal_low=slope - ci_half_width,
        slope_ci95_normal_high=slope + ci_half_width,
        leave_one_out_max_relative_slope_change=float(sensitivity),
    )


def _read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def _read_capture_frame(capture: cv2.VideoCapture, frame: int, video_path: Path) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = capture.read()
    if not ok or image is None:
        raise ValueError(f"could not read frame {frame} from {video_path}")
    return image


def _base_result(
    sample_row: pd.Series,
    force_metadata: pd.Series,
    status: str,
    failure_reason: str,
) -> dict[str, object]:
    return {
        "sample": str(sample_row["sample"]),
        "liquid": force_metadata.get("liquid", np.nan),
        "thickness_um": force_metadata.get("thickness_um", np.nan),
        "conditioning_time_h": force_metadata.get("ideal_dry_time_h", np.nan),
        "replicate": force_metadata.get("repeat_id", np.nan),
        "status": status,
        "slope_mm_per_mm": np.nan,
        "intercept_mm": np.nan,
        "r_squared": np.nan,
        "fit_point_count": 0,
        "fit_start_displacement_mm": np.nan,
        "fit_end_displacement_mm": np.nan,
        "fit_start_frame": np.nan,
        "fit_end_frame": np.nan,
        "fit_start_time_s": np.nan,
        "fit_end_time_s": np.nan,
        "calibration_center_x_px": np.nan,
        "calibration_center_y_px": np.nan,
        "film_radius_px": np.nan,
        "mm_per_px": np.nan,
        "calibration_rmse_px": np.nan,
        "calibration_angular_coverage_deg": np.nan,
        "residual_rmse_mm": np.nan,
        "slope_standard_error": np.nan,
        "slope_ci95_normal_low": np.nan,
        "slope_ci95_normal_high": np.nan,
        "leave_one_out_max_relative_slope_change": np.nan,
        "selection_signature": "",
        "failure_reason": failure_reason,
    }


def _empty_point_row(sample: str, row: pd.Series) -> dict[str, object]:
    return {
        "sample": sample,
        "point_order": int(row["point_order"]),
        "frame": int(row["frame"]),
        "time_s": float(row["time_s"]),
        "displacement_mm": float(row["displacement_mm"]),
        "circle_center_x_px": np.nan,
        "circle_center_y_px": np.nan,
        "radius_px": np.nan,
        "mm_per_px": np.nan,
        "radius_mm": np.nan,
        "red_pixel_count": np.nan,
        "boundary_marker_count": np.nan,
        "angular_coverage": np.nan,
        "circle_rmse_px": np.nan,
        "circle_rmse_fraction": np.nan,
        "annotation_source": str(row["annotation_source"]),
        "used_in_fit": False,
        "point_status": "not_processed",
    }


def process_sample(
    sample_row: pd.Series,
    selection: pd.DataFrame,
    force_metadata: pd.Series,
    video_dir: Path,
    annotation_root: Path,
    film_diameter_mm: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    sample = str(sample_row["sample"])
    if sample in USER_EXCLUDED_SAMPLES:
        return (
            _base_result(
                sample_row,
                force_metadata,
                "excluded_by_user",
                "temporarily excluded from radius fitting by user",
            ),
            pd.DataFrame(columns=POINT_COLUMNS),
        )
    if str(sample_row.get("review_status", "")) != "approved":
        return (
            _base_result(sample_row, force_metadata, "pending_review", "fit range not approved"),
            pd.DataFrame(columns=POINT_COLUMNS),
        )
    expected_count = int(sample_row["point_count"])
    selected = selection.loc[selection["sample"].astype(str) == sample].copy()
    selected = selected.sort_values("point_order", kind="stable").reset_index(drop=True)
    if (
        len(selected) != expected_count
        or selected["point_order"].astype(int).tolist() != list(range(1, expected_count + 1))
        or not selected["frame"].is_unique
        or selected["selection_signature"].nunique() != 1
    ):
        return (
            _base_result(
                sample_row,
                force_metadata,
                "selection_failed",
                f"expected {expected_count} unique selected frames",
            ),
            pd.DataFrame(columns=POINT_COLUMNS),
        )
    if not selected["annotation_source"].astype(str).eq("new_export").all():
        return (
            _base_result(
                sample_row,
                force_metadata,
                "stale_annotation_protocol",
                "point-marker analysis requires frames freshly exported from the raw video",
            ),
            pd.DataFrame(columns=POINT_COLUMNS),
        )

    signature = str(selected.loc[0, "selection_signature"])
    sample_dir = annotation_root / sample
    signature_path = sample_dir / "selection_signature.txt"
    if not signature_path.is_file() or signature_path.read_text(encoding="utf-8").strip() != signature:
        result = _base_result(
            sample_row, force_metadata, "stale_selection", "selection signature mismatch"
        )
        result["selection_signature"] = signature
        return result, pd.DataFrame(columns=POINT_COLUMNS)

    point_rows = [_empty_point_row(sample, row) for _, row in selected.iterrows()]
    points = pd.DataFrame(point_rows, columns=POINT_COLUMNS)
    video_path = video_dir / str(sample_row["video_file"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        result = _base_result(sample_row, force_metadata, "fit_failed", "video could not be opened")
        result["selection_signature"] = signature
        return result, points
    try:
        raw_zero = _read_capture_frame(capture, 0, video_path)
        calibration_image = _read_image(sample_dir / "calibration_time_0.jpg")
        calibration = fit_green_calibration(
            calibration_image, raw_zero, film_diameter_mm=film_diameter_mm
        )
    except Exception as exc:
        capture.release()
        result = _base_result(
            sample_row, force_metadata, "calibration_failed", str(exc)
        )
        result["selection_signature"] = signature
        return result, points

    missing = False
    invalid = False
    try:
        for index, selected_row in selected.iterrows():
            annotation_path = sample_dir / Path(str(selected_row["image_path"])).name
            if not annotation_path.is_file():
                points.loc[index, "point_status"] = "missing_annotation"
                missing = True
                continue
            try:
                annotated = _read_image(annotation_path)
                raw = _read_capture_frame(capture, int(selected_row["frame"]), video_path)
                red_fit = fit_red_annotation(
                    annotated,
                    raw,
                )
            except Exception:
                points.loc[index, "point_status"] = "invalid_circle"
                invalid = True
                continue
            points.loc[index, "circle_center_x_px"] = red_fit.center_x_px
            points.loc[index, "circle_center_y_px"] = red_fit.center_y_px
            points.loc[index, "radius_px"] = red_fit.radius_px
            points.loc[index, "mm_per_px"] = calibration.mm_per_px
            points.loc[index, "radius_mm"] = red_fit.radius_px * calibration.mm_per_px
            points.loc[index, "red_pixel_count"] = red_fit.pixel_count
            points.loc[index, "boundary_marker_count"] = red_fit.marker_count
            points.loc[index, "angular_coverage"] = red_fit.angular_coverage_deg
            points.loc[index, "circle_rmse_px"] = red_fit.fit_rmse_px
            points.loc[index, "circle_rmse_fraction"] = red_fit.fit_rmse_fraction
            if red_fit.marker_count == 1:
                points.loc[index, "point_status"] = "zero_radius"
            else:
                points.loc[index, "point_status"] = (
                    "valid"
                    if red_fit.fit_rmse_fraction <= 0.05
                    else "non_circular_boundary"
                )
    finally:
        capture.release()

    non_circular = points["point_status"].eq("non_circular_boundary").any()
    if missing or invalid:
        if missing:
            status = "incomplete_annotation"
            reason = f"expected {expected_count} annotations; at least one image is missing"
        elif invalid:
            status = "invalid_annotation"
            reason = "at least one image does not contain valid red boundary markers"
        result = _base_result(sample_row, force_metadata, status, reason)
        result.update(
            {
                "selection_signature": signature,
                "calibration_center_x_px": calibration.center_x_px,
                "calibration_center_y_px": calibration.center_y_px,
                "film_radius_px": calibration.radius_px,
                "mm_per_px": calibration.mm_per_px,
                "calibration_rmse_px": calibration.fit_rmse_px,
                "calibration_angular_coverage_deg": calibration.angular_coverage_deg,
            }
        )
        points["used_in_fit"] = False
        return result, points.reindex(columns=POINT_COLUMNS)

    try:
        fit = fit_radius_displacement(
            points["displacement_mm"].to_numpy(dtype=float),
            points["radius_mm"].to_numpy(dtype=float),
        )
    except Exception as exc:
        result = _base_result(sample_row, force_metadata, "fit_failed", str(exc))
        result["selection_signature"] = signature
        return result, points.reindex(columns=POINT_COLUMNS)

    points["used_in_fit"] = True
    result = _base_result(
        sample_row,
        force_metadata,
        "ok_with_qc_warning" if non_circular else "ok",
        (
            "at least one fitted boundary exceeds the 0.05 normalized "
            "circle-residual QC threshold; all selected points retained"
            if non_circular
            else ""
        ),
    )
    result.update(
        {
            "slope_mm_per_mm": fit.slope_mm_per_mm,
            "intercept_mm": fit.intercept_mm,
            "r_squared": fit.r_squared,
            "fit_point_count": fit.fit_point_count,
            "fit_start_displacement_mm": float(points.iloc[0]["displacement_mm"]),
            "fit_end_displacement_mm": float(points.iloc[-1]["displacement_mm"]),
            "fit_start_frame": int(points.iloc[0]["frame"]),
            "fit_end_frame": int(points.iloc[-1]["frame"]),
            "fit_start_time_s": float(points.iloc[0]["time_s"]),
            "fit_end_time_s": float(points.iloc[-1]["time_s"]),
            "calibration_center_x_px": calibration.center_x_px,
            "calibration_center_y_px": calibration.center_y_px,
            "film_radius_px": calibration.radius_px,
            "mm_per_px": calibration.mm_per_px,
            "calibration_rmse_px": calibration.fit_rmse_px,
            "calibration_angular_coverage_deg": calibration.angular_coverage_deg,
            "residual_rmse_mm": fit.residual_rmse_mm,
            "slope_standard_error": fit.slope_standard_error,
            "slope_ci95_normal_low": fit.slope_ci95_normal_low,
            "slope_ci95_normal_high": fit.slope_ci95_normal_high,
            "leave_one_out_max_relative_slope_change": (
                fit.leave_one_out_max_relative_slope_change
            ),
            "selection_signature": signature,
        }
    )
    return result, points.reindex(columns=POINT_COLUMNS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def _atomic_write_result_workbook(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="radius_results", index=False)
    temporary.replace(destination)


def _atomic_write_metadata(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def run_batch(
    config: RadiusConfig,
    force_config: pd.DataFrame,
    selection: pd.DataFrame,
    video_dir: Path,
    annotation_root: Path,
    output_root: Path,
    write_outputs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_force_columns = {
        "test_id",
        "liquid",
        "thickness_um",
        "ideal_dry_time_h",
        "repeat_id",
    }
    missing = required_force_columns.difference(force_config.columns)
    if missing:
        raise ValueError(f"force_config is missing columns: {sorted(missing)}")
    if force_config["test_id"].astype(str).duplicated().any():
        raise ValueError("force_config contains duplicate test_id values")
    force_by_id = force_config.set_index(force_config["test_id"].astype(str), drop=False)
    film_diameter_mm = float(config.settings["film_diameter_mm"])

    result_rows: list[dict[str, object]] = []
    point_tables: list[pd.DataFrame] = []
    for _, sample_row in config.selection_ranges.iterrows():
        force_sheet = str(sample_row["force_sheet"])
        if force_sheet not in force_by_id.index:
            raise ValueError(
                f"sample {sample_row['sample']} has no force metadata row {force_sheet}"
            )
        result, points = process_sample(
            sample_row,
            selection,
            force_by_id.loc[force_sheet],
            video_dir,
            annotation_root,
            film_diameter_mm,
        )
        result_rows.append(result)
        if not points.empty:
            point_tables.append(points)

    results = pd.DataFrame(result_rows).reindex(columns=RESULT_COLUMNS)
    points = (
        pd.concat(point_tables, ignore_index=True).reindex(columns=POINT_COLUMNS)
        if point_tables
        else pd.DataFrame(columns=POINT_COLUMNS)
    )
    if not write_outputs:
        return results, points

    _atomic_write_csv(points, output_root / "manual_radius_points.csv")
    _atomic_write_result_workbook(results, output_root / "radius_results.xlsx")
    metadata_path = output_root / "processing_metadata.json"
    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["radius_processing"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(results),
        "point_row_count": len(points),
        "status_counts": {
            str(key): int(value)
            for key, value in results["status"].value_counts(dropna=False).items()
        },
        "radius_process_sha256": _sha256(Path(__file__)),
    }
    _atomic_write_metadata(metadata, metadata_path)
    return results, points


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure manually annotated radii and fit radius-displacement slopes."
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force-config", type=Path, default=DEFAULT_FORCE_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument(
        "--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = load_radius_config(args.config)
    validate_selection_ranges(config.selection_ranges, set(FORCE_SAMPLE_ORDER))
    force_config = pd.read_excel(args.force_config, sheet_name="force_config")
    selection = (
        load_selected_manifest(args.selection)
        if args.selection.is_file()
        else pd.DataFrame(columns=SELECTED_COLUMNS)
    )
    results, points = run_batch(
        config=config,
        force_config=force_config,
        selection=selection,
        video_dir=args.video_dir,
        annotation_root=args.annotation_root,
        output_root=args.output_root,
        write_outputs=not args.validate_only,
    )
    counts = results["status"].value_counts(dropna=False).to_dict()
    print(
        f"Validated {len(results)} samples and {len(points)} point rows; "
        f"status_counts={counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
