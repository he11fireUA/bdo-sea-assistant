from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from bdo_assistant.ui_locator import (
    UILocatorError,
    build_layout,
    calibrate_frame,
    render_calibration_preview,
    save_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate BDO minimap and ship resource bars in a captured frame."
    )
    parser.add_argument("--image", type=Path, required=True, help="Captured BDO frame")
    parser.add_argument("--anchor", type=Path, required=True, help="Minimap 2D badge template")
    parser.add_argument("--preview", type=Path, required=True, help="Annotated preview output")
    parser.add_argument("--config", type=Path, required=True, help="UI layout JSON output")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Write the layout config after generating the preview",
    )
    return parser.parse_args()


def load_color_image(path: Path, label: str):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {label}: {path}")
    return image


def describe_rect(label: str, rect, confidence: float) -> str:
    return (
        f"{label}: x={rect.x} y={rect.y} width={rect.width} "
        f"height={rect.height} confidence={confidence:.3f}"
    )


def run(args: argparse.Namespace) -> int:
    frame = load_color_image(args.image, "capture image")
    anchor = load_color_image(args.anchor, "minimap anchor")
    result = calibrate_frame(frame, anchor)

    preview = render_calibration_preview(frame, result)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.preview), preview):
        raise OSError(f"failed to write preview: {args.preview}")

    print(describe_rect("MINIMAP", result.minimap.rect, result.minimap.confidence))
    print(describe_rect("SHIP HP", result.bars.hp, result.bars.confidence))
    print(describe_rect("PROVISIONS", result.bars.provisions, result.bars.confidence))
    print(f"PREVIEW SAVED: {args.preview}")

    if not args.confirm:
        print("CONFIG NOT SAVED: review the preview and rerun with --confirm")
        return 0

    height, width = frame.shape[:2]
    layout = build_layout(width, height, result.minimap, result.bars)
    save_layout(args.config, layout)
    print(f"CONFIG SAVED: {args.config}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (ValueError, OSError, UILocatorError) as error:
        print(f"CALIBRATION FAILED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
