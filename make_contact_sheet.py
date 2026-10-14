from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("session_dir")
parser.add_argument("start", type=int)
parser.add_argument("end", type=int)
parser.add_argument("output")
args = parser.parse_args()

root = Path(args.session_dir)
metrics = (root / "metrics.csv").read_text(encoding="utf-8").splitlines()
# Recorder has not wrapped in this short session, so sequence equals preview slot.
frames = []
for seq in range(args.start, args.end + 1):
    path = root / "previews" / f"preview_{seq:04d}.jpg"
    image = cv2.imread(str(path))
    if image is None:
        continue
    image = cv2.resize(image, (851, 240), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (190, 28), (0, 0, 0), -1)
    cv2.putText(image, f"seq {seq}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    frames.append(image)

columns = 4
rows = (len(frames) + columns - 1) // columns
blank = np.zeros_like(frames[0])
while len(frames) < rows * columns:
    frames.append(blank.copy())
sheet_rows = [np.hstack(frames[r * columns:(r + 1) * columns]) for r in range(rows)]
sheet = np.vstack(sheet_rows)
cv2.imwrite(args.output, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(Path(args.output).resolve())
