from __future__ import annotations

from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .ui_locator import (
    CalibrationResult,
    ElementDetection,
    Rect,
    ShipBarsDetection,
    build_layout,
    calibrate_frame,
    render_calibration_preview,
    save_layout,
)


class CalibrationFlowState(str, Enum):
    IDLE = "idle"
    CAPTURED = "captured"
    PREVIEW_READY = "preview_ready"
    SAVED = "saved"
    ERROR = "error"


def _validate_image(value: object, label: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 3
        or value.shape[2] != 3
        or value.size == 0
        or value.dtype != np.uint8
    ):
        raise ValueError(f"{label} must be a non-empty BGR uint8 image")
    return value


class CalibrationFlowController:
    def __init__(
        self,
        *,
        detector: Callable,
        preview_renderer: Callable,
        layout_builder: Callable,
        saver: Callable,
    ) -> None:
        for callback, name in (
            (detector, "detector"),
            (preview_renderer, "preview_renderer"),
            (layout_builder, "layout_builder"),
            (saver, "saver"),
        ):
            if not callable(callback):
                raise ValueError(f"{name} must be callable")
        self._detector = detector
        self._preview_renderer = preview_renderer
        self._layout_builder = layout_builder
        self._saver = saver
        self.state = CalibrationFlowState.IDLE
        self.frame: np.ndarray | None = None
        self.preview: np.ndarray | None = None
        self._layout_snapshot: Mapping | None = None
        self.detection = None

    @property
    def layout(self) -> Mapping | None:
        return deepcopy(self._layout_snapshot)

    def capture(self, frame: np.ndarray) -> None:
        image = _validate_image(frame, "frame")
        frozen = image.copy()
        frozen.setflags(write=False)
        self.frame = frozen
        self.preview = None
        self._layout_snapshot = None
        self.detection = None
        self.state = CalibrationFlowState.CAPTURED

    def detect(self, anchor: np.ndarray) -> None:
        if self.state is not CalibrationFlowState.CAPTURED or self.frame is None:
            raise RuntimeError("capture is required before detection")
        anchor_image = _validate_image(anchor, "anchor")
        detection = self._detector(self.frame, anchor_image)
        preview = _validate_image(
            self._preview_renderer(self.frame, detection),
            "preview",
        )
        if preview.shape != self.frame.shape:
            raise ValueError("preview must match captured frame dimensions")
        height, width = self.frame.shape[:2]
        layout = self._layout_builder(width, height, detection)
        if not isinstance(layout, Mapping):
            raise ValueError("layout_builder must return a mapping")
        layout_snapshot = deepcopy(layout)
        frozen_preview = preview.copy()
        frozen_preview.setflags(write=False)
        self.detection = detection
        self.preview = frozen_preview
        self._layout_snapshot = layout_snapshot
        self.state = CalibrationFlowState.PREVIEW_READY

    def confirm_and_save(self, destination: str | Path) -> None:
        if self.state is not CalibrationFlowState.PREVIEW_READY or self._layout_snapshot is None:
            raise RuntimeError("calibration preview must be reviewed before confirm")
        path = Path(destination)
        self._saver(path, deepcopy(self._layout_snapshot))
        self.state = CalibrationFlowState.SAVED

    def apply_manual_regions(self, regions: Mapping[str, Rect]) -> None:
        if (
            self.state is not CalibrationFlowState.PREVIEW_READY
            or self.frame is None
            or type(self.detection) is not CalibrationResult
        ):
            raise RuntimeError("preview is required before manual correction")
        required = {"minimap", "ship_hp_bar", "ship_rations_bar"}
        if not isinstance(regions, Mapping) or set(regions) != required:
            raise ValueError("manual regions must contain minimap and ship bars")
        height, width = self.frame.shape[:2]
        for name, rect in regions.items():
            if type(rect) is not Rect:
                raise ValueError(f"manual region must be a Rect: {name}")
            if any(
                type(value) is not int
                for value in (rect.x, rect.y, rect.width, rect.height)
            ):
                raise ValueError(f"manual region values must be exact integers: {name}")
            if (
                rect.x < 0
                or rect.y < 0
                or rect.width <= 0
                or rect.height <= 0
                or rect.x + rect.width > width
                or rect.y + rect.height > height
            ):
                raise ValueError(f"manual region leaves frame: {name}")
        corrected = CalibrationResult(
            minimap=ElementDetection(
                regions["minimap"],
                self.detection.minimap.confidence,
                self.detection.minimap.scale,
            ),
            bars=ShipBarsDetection(
                regions["ship_hp_bar"],
                regions["ship_rations_bar"],
                self.detection.bars.confidence,
            ),
        )
        preview = _validate_image(
            self._preview_renderer(self.frame, corrected),
            "preview",
        )
        if preview.shape != self.frame.shape:
            raise ValueError("preview must match captured frame dimensions")
        layout = self._layout_builder(width, height, corrected)
        if not isinstance(layout, Mapping):
            raise ValueError("layout_builder must return a mapping")
        layout_snapshot = deepcopy(layout)
        frozen_preview = preview.copy()
        frozen_preview.setflags(write=False)
        self.detection = corrected
        self.preview = frozen_preview
        self._layout_snapshot = layout_snapshot

    def retry(self) -> None:
        self.frame = None
        self.preview = None
        self._layout_snapshot = None
        self.detection = None
        self.state = CalibrationFlowState.IDLE


def build_default_calibration_flow() -> CalibrationFlowController:
    return CalibrationFlowController(
        detector=calibrate_frame,
        preview_renderer=render_calibration_preview,
        layout_builder=lambda width, height, result: build_layout(
            width,
            height,
            result.minimap,
            result.bars,
        ),
        saver=save_layout,
    )
