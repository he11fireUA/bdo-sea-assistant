from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from windows_capture import Frame, InternalCaptureControl, WindowsCapture

from list_bdo_windows import list_windows


def find_bdo_window() -> tuple[int, str]:
    matches = [
        (hwnd, title)
        for hwnd, _pid, exe, title in list_windows()
        if exe.lower() == "blackdesert64.exe"
    ]
    if not matches:
        raise RuntimeError("Visible BlackDesert64.exe window was not found")
    return matches[0]


def relative_crop(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    height, width = image.shape[:2]
    return image[int(y1 * height):int(y2 * height), int(x1 * width):int(x2 * width)]


def mean_diff(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None or previous.shape != current.shape:
        return 0.0
    return float(np.mean(cv2.absdiff(previous, current)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Passive BDO journey recorder")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=1800)
    parser.add_argument("--preview-width", type=int, default=1702)
    parser.add_argument("--keyframe-every", type=int, default=15)
    args = parser.parse_args()

    hwnd, title = find_bdo_window()
    session_name = datetime.now().strftime("journey_%Y%m%d_%H%M%S")
    session_dir = Path("journeys") / session_name
    preview_dir = session_dir / "previews"
    keyframe_dir = session_dir / "keyframes"
    preview_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    Path("latest_journey.txt").write_text(str(session_dir.resolve()), encoding="utf-8")

    metadata = {
        "session": session_name,
        "window_title": title,
        "hwnd": hwnd,
        "started_local": datetime.now().isoformat(timespec="seconds"),
        "interval_seconds": args.interval,
        "max_samples": args.max_samples,
        "preview_width": args.preview_width,
        "keyframe_every": args.keyframe_every,
        "capture_method": "Windows Graphics Capture (window HWND)",
        "input_sent": False,
    }
    (session_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_file = (session_dir / "metrics.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "sequence", "slot", "local_time", "elapsed_seconds", "width", "height",
            "full_diff", "minimap_diff", "ship_hud_diff", "preview_file", "keyframe_file",
        ],
    )
    writer.writeheader()
    csv_file.flush()

    started = time.monotonic()
    last_sample = -args.interval
    sequence = 0
    previous_preview: np.ndarray | None = None
    previous_minimap: np.ndarray | None = None
    previous_ship_hud: np.ndarray | None = None
    stop_requested = False
    callback_error: str | None = None

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    capture = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=hwnd)

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl) -> None:
        nonlocal last_sample, sequence, previous_preview, previous_minimap, previous_ship_hud, callback_error
        try:
            elapsed = time.monotonic() - started
            if stop_requested:
                capture_control.stop()
                return
            if elapsed - last_sample < args.interval:
                return

            image = frame.convert_to_bgr().frame_buffer
            if image is None or image.size == 0:
                raise RuntimeError("Empty frame")

            height, width = image.shape[:2]
            preview_height = max(1, round(height * args.preview_width / width))
            preview = cv2.resize(image, (args.preview_width, preview_height), interpolation=cv2.INTER_AREA)

            # Generous crop around the minimap on the reference 5106x1440 layout.
            # Product detection does not rely on this estimate; ui_locator calibrates it dynamically.
            minimap = relative_crop(preview, 0.92, 0.00, 1.00, 0.25)
            # Generous crop around the ship HP/provisions HUD on the reference 5120x1440 layout.
            ship_hud = relative_crop(preview, 0.44, 0.80, 0.56, 0.95)

            slot = sequence % args.max_samples
            preview_file = preview_dir / f"preview_{slot:04d}.jpg"
            cv2.imwrite(str(preview_file), preview, [cv2.IMWRITE_JPEG_QUALITY, 76])

            keyframe_file = ""
            if sequence % args.keyframe_every == 0:
                key_slot = (sequence // args.keyframe_every) % max(1, args.max_samples // args.keyframe_every)
                key_path = keyframe_dir / f"keyframe_{key_slot:04d}.jpg"
                cv2.imwrite(str(key_path), image, [cv2.IMWRITE_JPEG_QUALITY, 82])
                keyframe_file = str(key_path.resolve())

            writer.writerow(
                {
                    "sequence": sequence,
                    "slot": slot,
                    "local_time": datetime.now().isoformat(timespec="milliseconds"),
                    "elapsed_seconds": round(elapsed, 3),
                    "width": width,
                    "height": height,
                    "full_diff": round(mean_diff(previous_preview, preview), 3),
                    "minimap_diff": round(mean_diff(previous_minimap, minimap), 3),
                    "ship_hud_diff": round(mean_diff(previous_ship_hud, ship_hud), 3),
                    "preview_file": str(preview_file.resolve()),
                    "keyframe_file": keyframe_file,
                }
            )
            csv_file.flush()

            previous_preview = preview.copy()
            previous_minimap = minimap.copy()
            previous_ship_hud = ship_hud.copy()
            sequence += 1
            last_sample = elapsed
        except Exception as exc:
            callback_error = repr(exc)
            capture_control.stop()

    @capture.event
    def on_closed() -> None:
        pass

    print(f"SESSION_DIR={session_dir.resolve()}", flush=True)
    print(f"WINDOW={title!r} HWND={hwnd}", flush=True)
    print("RECORDING_PASSIVE_WGC=1", flush=True)
    capture.start()

    csv_file.close()
    summary = {
        **metadata,
        "ended_local": datetime.now().isoformat(timespec="seconds"),
        "samples_recorded": sequence,
        "callback_error": callback_error,
    }
    (session_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if callback_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
