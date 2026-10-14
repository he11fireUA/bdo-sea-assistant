from __future__ import annotations

import argparse
import json
import time
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only WGC frame liveness probe")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--label", default="foreground")
    args = parser.parse_args()

    hwnd, title = find_bdo_window()
    output_dir = Path("captures") / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    last_sample = -args.interval
    samples: list[np.ndarray] = []
    sample_stats: list[dict[str, float | int | str]] = []
    callback_errors: list[str] = []

    capture = WindowsCapture(
        cursor_capture=False,
        draw_border=False,
        window_hwnd=hwnd,
    )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl) -> None:
        nonlocal last_sample
        try:
            elapsed = time.monotonic() - started
            if elapsed - last_sample >= args.interval:
                converted = frame.convert_to_bgr()
                image = converted.frame_buffer
                if image is None or image.size == 0:
                    raise RuntimeError("Frame conversion returned an empty image")
                index = len(samples)
                mean = float(np.mean(image))
                std = float(np.std(image))
                diff = (
                    float(np.mean(cv2.absdiff(samples[-1], image)))
                    if samples and samples[-1].shape == image.shape
                    else 0.0
                )
                filename = output_dir / f"frame_{index:03d}.jpg"
                cv2.imwrite(str(filename), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                samples.append(image.copy())
                sample_stats.append(
                    {
                        "index": index,
                        "elapsed_seconds": round(elapsed, 3),
                        "width": int(image.shape[1]),
                        "height": int(image.shape[0]),
                        "mean_brightness": round(mean, 3),
                        "pixel_stddev": round(std, 3),
                        "mean_abs_diff_from_previous": round(diff, 3),
                        "file": str(filename.resolve()),
                    }
                )
                last_sample = elapsed
            if elapsed >= args.duration:
                capture_control.stop()
        except Exception as exc:
            callback_errors.append(repr(exc))
            capture_control.stop()

    @capture.event
    def on_closed() -> None:
        pass

    capture.start()

    diffs = [float(s["mean_abs_diff_from_previous"]) for s in sample_stats[1:]]
    report = {
        "label": args.label,
        "window_title": title,
        "hwnd": hwnd,
        "requested_duration_seconds": args.duration,
        "sample_count": len(sample_stats),
        "callback_errors": callback_errors,
        "non_black": bool(sample_stats and max(float(s["pixel_stddev"]) for s in sample_stats) > 2.0),
        "updating": bool(diffs and max(diffs) > 0.5),
        "max_mean_abs_diff": round(max(diffs), 3) if diffs else 0.0,
        "samples": sample_stats,
    }
    report_file = output_dir / "report.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if sample_stats and not callback_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
