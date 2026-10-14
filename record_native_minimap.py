from __future__ import annotations

import argparse
import json
from math import isfinite
from numbers import Real
from pathlib import Path
import signal
import sys
import time
from typing import Callable

from bdo_assistant.calibration_store import JsonCalibrationStore
from bdo_assistant.minimap_replay_recorder import (
    MANIFEST_TIME_PRECISION_SECONDS,
    NativeMinimapReplayRecorder,
    NativeMinimapReplaySession,
)
from bdo_assistant.runtime import WgcWindowFrameSource
from bdo_assistant.ui_locator import Rect


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _recording_interval(value: str) -> float:
    parsed = _positive_float(value)
    if parsed < MANIFEST_TIME_PRECISION_SECONDS:
        raise argparse.ArgumentTypeError("must be at least 0.000001 seconds for manifest precision")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explicit bounded recorder for a confirmed native minimap ROI."
    )
    parser.add_argument(
        "--record", action="store_true", required=True,
        help="Confirm this intentional local minimap recording command.",
    )
    parser.add_argument("--hwnd", type=_positive_int, required=True, help="Selected window HWND")
    parser.add_argument(
        "--calibration", type=Path, required=True,
        help="Confirmed UI calibration JSON file",
    )
    parser.add_argument(
        "--destination", type=Path, required=True,
        help="New replay directory; it must not already exist",
    )
    parser.add_argument(
        "--duration", type=_positive_float, default=60.0,
        help="Maximum capture duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--interval", type=_recording_interval, default=0.2,
        help="Minimum seconds between persisted ROI frames (default: 0.2)",
    )
    parser.add_argument(
        "--max-frames", type=_positive_int, default=300,
        help="Maximum persisted ROI frames (default: 300)",
    )
    return parser.parse_args(argv)


def _validated_recording(args: argparse.Namespace) -> tuple[Rect, tuple[int, int]]:
    if args.record is not True:
        raise ValueError("--record is required before capture")
    if type(args.hwnd) is not int or args.hwnd <= 0:
        raise ValueError("hwnd must be a positive integer")
    if not isinstance(args.calibration, Path):
        raise ValueError("calibration must be an explicit path")
    if not isinstance(args.destination, Path):
        raise ValueError("destination must be an explicit path")
    if args.destination.exists():
        raise FileExistsError(f"destination already exists: {args.destination}")
    for name in ("duration", "interval"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    if args.interval < MANIFEST_TIME_PRECISION_SECONDS:
        raise ValueError("interval must be at least 0.000001 seconds for manifest precision")
    if type(args.max_frames) is not int or args.max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")

    readiness = JsonCalibrationStore(args.calibration).inspect()
    if not readiness.ready or readiness.layout is None:
        raise ValueError(f"confirmed calibration is required: {readiness.diagnostic}")
    absolute = readiness.layout["regions"]["minimap"]["absolute"]
    capture_size = readiness.layout["capture_size"]
    return (
        Rect(absolute["x"], absolute["y"], absolute["width"], absolute["height"]),
        (capture_size["width"], capture_size["height"]),
    )


def _print_scope(rect: Rect, calibration: Path, destination: Path) -> None:
    print("SCOPE: raw-local-minimap-roi")
    print(f"CALIBRATION: {calibration}")
    print(f"ROI: x={rect.x} y={rect.y} width={rect.width} height={rect.height}")
    print(f"DESTINATION: {destination}")


def run(
    args: argparse.Namespace,
    *,
    source_factory: Callable[[int], object] = WgcWindowFrameSource,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Validate every capture boundary before constructing the native frame source."""
    if not callable(source_factory) or not callable(sleep):
        raise ValueError("source_factory and sleep must be callable")
    rect, expected_capture_size = _validated_recording(args)
    _print_scope(rect, args.calibration, args.destination)

    recorder = NativeMinimapReplayRecorder(
        args.destination,
        minimap=rect,
        interval_seconds=float(args.interval),
        max_frames=args.max_frames,
        expected_capture_size=expected_capture_size,
    )
    source = source_factory(args.hwnd)
    session = NativeMinimapReplaySession(
        source, recorder, deadline_seconds=float(args.duration)
    )
    previous_handlers: dict[int, object] = {}

    def interrupt(_signum, _frame) -> None:
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupt)
        session.start()
        while session.lifecycle_state != "TERMINAL":
            sleep(0.05)
    except KeyboardInterrupt:
        manifest = session.stop()
    else:
        manifest = session.stop()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    cleanup = "complete"
    try:
        session.retry_cleanup()
    except Exception as exc:
        cleanup = f"retry required ({type(exc).__name__}: {exc})"
    details = json.loads(manifest.read_text(encoding="utf-8"))
    status = details["status"]
    print(f"MANIFEST: {manifest}")
    print(f"STATUS: {status}")
    print(f"CLEANUP: {cleanup}")
    if status != "completed":
        print(f"RECORDING INCOMPLETE ({status}): {details['reason']}")
    return 0 if cleanup == "complete" and status == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"RECORDING REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
