from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

image = cv2.imread(sys.argv[1])
if image is None:
    raise SystemExit(f"Could not read {sys.argv[1]}")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

masks = {
    "red": (((h <= 8) | (h >= 172)) & (s >= 130) & (v >= 100)).astype("uint8") * 255,
    "yellow": ((h >= 15) & (h <= 38) & (s >= 130) & (v >= 110)).astype("uint8") * 255,
}

for name, mask in masks.items():
    mask[: int(image.shape[0] * 0.55), :] = 0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for x, y, width, height, area in stats[1:count]:
        if area >= 20:
            boxes.append((int(area), int(x), int(y), int(width), int(height)))
    boxes.sort(reverse=True)
    print(name)
    for area, x, y, width, height in boxes[:30]:
        print(f"  area={area:5d} box=({x},{y})-({x+width},{y+height}) size={width}x{height}")
