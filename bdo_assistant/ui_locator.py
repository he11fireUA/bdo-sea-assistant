from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    def normalized(self, frame_width: int, frame_height: int) -> dict[str, float]:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        return {
            "x": self.x / frame_width,
            "y": self.y / frame_height,
            "width": self.width / frame_width,
            "height": self.height / frame_height,
        }

    @classmethod
    def from_normalized(
        cls,
        values: Mapping[str, float],
        frame_width: int,
        frame_height: int,
    ) -> "Rect":
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        return cls(
            x=round(values["x"] * frame_width),
            y=round(values["y"] * frame_height),
            width=round(values["width"] * frame_width),
            height=round(values["height"] * frame_height),
        )


@dataclass(frozen=True)
class ShipBarsDetection:
    hp: Rect
    provisions: Rect
    confidence: float


@dataclass(frozen=True)
class CalibrationResult:
    minimap: ElementDetection
    bars: ShipBarsDetection


class UILocatorError(RuntimeError):
    """Raised when a UI element cannot be located confidently."""


@dataclass(frozen=True)
class ElementDetection:
    rect: Rect
    confidence: float
    scale: float


def locate_minimap(
    image: np.ndarray,
    anchor_template: np.ndarray,
    *,
    minimum_confidence: float = 0.60,
) -> ElementDetection:
    """Locate the minimap from its stable 2D badge at multiple UI scales."""
    if image is None or image.ndim != 3:
        raise ValueError("image must be a BGR color frame")
    if anchor_template is None or anchor_template.ndim != 3:
        raise ValueError("anchor_template must be a BGR image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(anchor_template, cv2.COLOR_BGR2GRAY)
    best: tuple[float, tuple[int, int], float] | None = None
    for raw_scale in np.arange(0.60, 1.81, 0.05):
        scale = round(float(raw_scale), 2)
        width = max(8, round(template_gray.shape[1] * float(scale)))
        height = max(8, round(template_gray.shape[0] * float(scale)))
        if width >= gray.shape[1] or height >= gray.shape[0]:
            continue
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        candidate = cv2.resize(template_gray, (width, height), interpolation=interpolation)
        response = cv2.matchTemplate(gray, candidate, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, location = cv2.minMaxLoc(response)
        item = (float(confidence), location, scale)
        if best is None or item[0] > best[0]:
            best = item

    if best is None or best[0] < minimum_confidence:
        confidence = 0.0 if best is None else best[0]
        raise UILocatorError(f"minimap 2D anchor confidence too low: {confidence:.3f}")

    confidence, (anchor_x, anchor_y), scale = best
    reference_width, reference_height = 300, 260
    anchor_offset_x, anchor_offset_y = 10, 3
    x = max(0, anchor_x - round(anchor_offset_x * scale))
    y = max(0, anchor_y - round(anchor_offset_y * scale))
    width = min(round(reference_width * scale), image.shape[1] - x)
    height = min(round(reference_height * scale), image.shape[0] - y)
    return ElementDetection(
        rect=Rect(x=x, y=y, width=width, height=height),
        confidence=confidence,
        scale=scale,
    )


def _components(mask: np.ndarray) -> list[Rect]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    result: list[Rect] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < 18 or width < 25 or height < 2 or height > 35:
            continue
        if width / height < 3.0:
            continue
        result.append(Rect(int(x), int(y), int(width), int(height)))
    return result


def _dense_color_band(mask: np.ndarray, component: Rect) -> tuple[int, int]:
    x1 = max(0, component.x)
    x2 = min(mask.shape[1], component.x + component.width)
    y1 = max(0, component.y)
    y2 = min(mask.shape[0], component.y + component.height)
    row_counts = mask[y1:y2, x1:x2].sum(axis=1)
    required = max(3, round(component.width * 0.40))
    rows = np.flatnonzero(row_counts >= required)
    if rows.size == 0:
        return component.y, component.y + component.height
    return y1 + int(rows[0]), y1 + int(rows[-1]) + 1


def _track_edges(
    image: np.ndarray,
    red: Rect,
    yellow: Rect,
    expected_width: int | None = None,
) -> tuple[int, int]:
    dy = yellow.y - red.y
    expected_width = max(60, round(dy * 11.5)) if expected_width is None else expected_width
    left_guess = round((red.x + yellow.x) / 2)
    y1 = max(0, red.y - round(dy * 0.6))
    y2 = min(image.shape[0], yellow.y + yellow.height + round(dy * 0.6))
    edges = cv2.Canny(cv2.cvtColor(image[y1:y2], cv2.COLOR_BGR2GRAY), 40, 100)
    scores = edges.sum(axis=0) / 255.0

    def strongest(start: int, end: int, target: int) -> int:
        start = max(0, start)
        end = min(len(scores), end)
        if start >= end:
            return target
        candidates = list(range(start, end))
        return max(candidates, key=lambda x: (scores[x], -abs(x - target)))

    left = strongest(left_guess - round(dy * 0.7), left_guess + round(dy * 0.4) + 1, left_guess)
    # Use the paired vertical spacing as the coarse UI-scale estimate, then
    # refine only near that expected track edge. The narrow window excludes
    # stronger decorative HUD edges 20–32 px beyond the real track.
    expected_right = left + expected_width
    margin = max(3, round(expected_width * 0.04))
    right = strongest(
        expected_right - margin,
        expected_right + margin + 1,
        expected_right,
    )
    frame_width = image.shape[1]
    left = min(max(0, left), frame_width - 1)
    right = min(max(left + 1, right), frame_width)
    return left, right


def locate_ship_bars(
    image: np.ndarray,
    *,
    expected_scale: float | None = None,
) -> ShipBarsDetection:
    """Locate the paired ship HP (red) and provisions (yellow) tracks."""
    if image is None or image.ndim != 3:
        raise ValueError("image must be a BGR color frame")
    if expected_scale is not None:
        if isinstance(expected_scale, bool) or not isinstance(expected_scale, Real):
            raise ValueError("expected_scale must be a finite number")
        expected_scale = float(expected_scale)
        if not isfinite(expected_scale):
            raise ValueError("expected_scale must be a finite number")
        tolerance = 1e-9
        if not 0.60 - tolerance <= expected_scale <= 1.80 + tolerance:
            raise ValueError("expected_scale must be between 0.60 and 1.80")
        expected_scale = min(1.80, max(0.60, expected_scale))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red_mask = (((hue <= 10) | (hue >= 170)) & (saturation >= 95) & (value >= 75))
    yellow_mask = ((hue >= 15) & (hue <= 40) & (saturation >= 95) & (value >= 75))

    red_parts = _components(red_mask)
    yellow_parts = _components(yellow_mask)
    pairs: list[tuple[float, Rect, Rect]] = []
    for red in red_parts:
        for yellow in yellow_parts:
            dy = yellow.y - red.y
            if not 7 <= dy <= 70:
                continue
            if expected_scale is not None:
                expected_dy = 16.5 * expected_scale
                dy_tolerance = max(5.0, expected_dy * 0.45)
                expected_track_width = round(191 * expected_scale)
                if abs(dy - expected_dy) > dy_tolerance:
                    continue
                if max(red.width, yellow.width) > expected_track_width * 1.05:
                    continue
            alignment = abs(red.x - yellow.x)
            if alignment > max(14, round(dy * 0.8)):
                continue
            if max(red.width, yellow.width) < dy * 3:
                continue
            score = (
                1.0
                - min(0.35, alignment / max(1, dy) * 0.20)
                - min(0.20, abs(red.height - yellow.height) / max(red.height, yellow.height) * 0.10)
            )
            pairs.append((score, red, yellow))

    if not pairs:
        raise UILocatorError("ship HP/provisions bar pair was not found")

    confidence, red, yellow = max(pairs, key=lambda item: item[0])
    expected_width = None if expected_scale is None else round(191 * expected_scale)
    left, right = _track_edges(image, red, yellow, expected_width)
    if expected_width is not None and right - left < expected_width * 0.90:
        raise UILocatorError("projected ship tracks extend outside the frame")
    red_top, red_bottom = _dense_color_band(red_mask, red)
    yellow_top, yellow_bottom = _dense_color_band(yellow_mask, yellow)
    return ShipBarsDetection(
        hp=Rect(left, red_top, right - left, red_bottom - red_top),
        provisions=Rect(left, yellow_top, right - left, yellow_bottom - yellow_top),
        confidence=max(0.0, min(1.0, confidence)),
    )


def calibrate_frame(image: np.ndarray, minimap_anchor: np.ndarray) -> CalibrationResult:
    minimap = locate_minimap(image, minimap_anchor)
    return CalibrationResult(
        minimap=minimap,
        bars=locate_ship_bars(image, expected_scale=minimap.scale),
    )


def render_calibration_preview(
    image: np.ndarray,
    result: CalibrationResult,
) -> np.ndarray:
    preview = image.copy()
    items = [
        (result.minimap.rect, "MINIMAP", (255, 180, 0)),
        (result.bars.hp, "SHIP HP", (0, 0, 255)),
        (result.bars.provisions, "PROVISIONS", (0, 220, 255)),
    ]
    line_width = max(2, round(min(image.shape[:2]) / 500))
    for rect, label, color in items:
        cv2.rectangle(
            preview,
            (rect.x, rect.y),
            (rect.x + rect.width, rect.y + rect.height),
            color,
            line_width,
        )
        label_y = max(24, rect.y - 8)
        cv2.putText(
            preview,
            label,
            (rect.x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            line_width,
            cv2.LINE_AA,
        )
    return preview


def _region_config(
    rect: Rect,
    confidence: float,
    frame_width: int,
    frame_height: int,
    scale: float,
) -> dict[str, Any]:
    return {
        "absolute": {
            "x": rect.x,
            "y": rect.y,
            "width": rect.width,
            "height": rect.height,
        },
        "normalized": rect.normalized(frame_width, frame_height),
        "confidence": float(confidence),
        "ui_scale": float(scale),
    }


def build_layout(
    frame_width: int,
    frame_height: int,
    minimap: ElementDetection,
    bars: ShipBarsDetection,
) -> dict[str, Any]:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    bar_scale = bars.hp.width / 191.0
    return {
        "schema_version": 1,
        "capture_size": {"width": frame_width, "height": frame_height},
        "regions": {
            "minimap": _region_config(
                minimap.rect, minimap.confidence, frame_width, frame_height, minimap.scale
            ),
            "ship_hp_bar": _region_config(
                bars.hp, bars.confidence, frame_width, frame_height, bar_scale
            ),
            "ship_rations_bar": _region_config(
                bars.provisions, bars.confidence, frame_width, frame_height, bar_scale
            ),
        },
        "thresholds": {
            "ship_hp_warning_percent": 60,
            "ship_rations_warning_percent": 10,
        },
    }


def save_layout(path: str | Path, layout: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(layout, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
