from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pytest

import record_native_minimap as cli


def _layout() -> dict[str, object]:
    return {
        "schema_version": 1,
        "capture_size": {"width": 1000, "height": 500},
        "regions": {
            "minimap": {
                "absolute": {"x": 700, "y": 20, "width": 250, "height": 200},
                "normalized": {"x": 0.7, "y": 0.04, "width": 0.25, "height": 0.4},
                "confidence": 0.91,
                "ui_scale": 1.0,
            },
            "ship_hp_bar": {
                "absolute": {"x": 50, "y": 400, "width": 190, "height": 8},
                "normalized": {"x": 0.05, "y": 0.8, "width": 0.19, "height": 0.016},
                "confidence": 0.88,
                "ui_scale": 1.0,
            },
            "ship_rations_bar": {
                "absolute": {"x": 50, "y": 420, "width": 190, "height": 8},
                "normalized": {"x": 0.05, "y": 0.84, "width": 0.19, "height": 0.016},
                "confidence": 0.87,
                "ui_scale": 1.0,
            },
        },
        "thresholds": {
            "ship_hp_warning_percent": 60,
            "ship_rations_warning_percent": 10,
        },
    }


def _args(tmp_path, *, record: bool = True, **overrides) -> argparse.Namespace:
    calibration = tmp_path / "ui_layout.json"
    calibration.parent.mkdir(parents=True, exist_ok=True)
    calibration.write_text(json.dumps(_layout()), encoding="utf-8")
    values = {
        "record": record,
        "hwnd": 123,
        "calibration": calibration,
        "destination": tmp_path / "replay",
        "duration": 1.0,
        "interval": 0.1,
        "max_frames": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parse_requires_explicit_record_and_has_no_capture_side_effect(tmp_path) -> None:
    with pytest.raises(SystemExit) as error:
        cli.parse_args(
            [
                "--hwnd", "123", "--calibration", str(tmp_path / "missing.json"),
                "--destination", str(tmp_path / "replay"),
            ]
        )

    assert error.value.code == 2


def test_parse_rejects_interval_below_native_manifest_time_precision(tmp_path) -> None:
    with pytest.raises(SystemExit) as error:
        cli.parse_args(
            [
                "--record", "--hwnd", "123", "--calibration", str(tmp_path / "layout.json"),
                "--destination", str(tmp_path / "replay"), "--interval", "0.0000005",
            ]
        )

    assert error.value.code == 2


def test_run_refuses_unapproved_or_invalid_inputs_before_constructing_source(tmp_path) -> None:
    constructed: list[int] = []

    def source_factory(hwnd: int):
        constructed.append(hwnd)
        raise AssertionError("native source must not be constructed")

    invalid_calibration = tmp_path / "not-confirmed.json"
    clipped = _args(tmp_path / "clipped")
    clipped_layout = _layout()
    clipped_layout["regions"]["minimap"]["absolute"]["x"] = 900
    clipped.calibration.write_text(json.dumps(clipped_layout), encoding="utf-8")
    for args in (
        _args(tmp_path / "unapproved", record=False),
        _args(tmp_path / "invalid_hwnd", hwnd=0),
        _args(tmp_path / "invalid_calibration", calibration=invalid_calibration),
        clipped,
    ):
        with pytest.raises(ValueError):
            cli.run(args, source_factory=source_factory)

    assert constructed == []


def test_run_refuses_existing_destination_before_constructing_source(tmp_path) -> None:
    destination = tmp_path / "replay"
    destination.mkdir()
    constructed: list[int] = []
    args = _args(tmp_path, destination=destination)

    with pytest.raises(FileExistsError, match="destination already exists"):
        cli.run(args, source_factory=lambda hwnd: constructed.append(hwnd))

    assert constructed == []


def test_run_records_confirmed_minimap_and_reports_manifest_and_cleanup(tmp_path, capsys) -> None:
    sources = []

    class Source:
        def start(self, on_frame, on_closed) -> None:
            on_frame(np.zeros((500, 1000, 3), dtype=np.uint8))

        def stop(self) -> None:
            self.stopped = True

    def source_factory(hwnd: int) -> Source:
        assert hwnd == 123
        source = Source()
        sources.append(source)
        return source

    result = cli.run(_args(tmp_path), source_factory=source_factory)

    output = capsys.readouterr().out
    assert result == 0
    assert "SCOPE: raw-local-minimap-roi" in output
    assert "ROI: x=700 y=20 width=250 height=200" in output
    assert "MANIFEST:" in output
    assert "STATUS: completed" in output
    assert "CLEANUP: complete" in output
    assert sources[0].stopped
    manifest = json.loads((tmp_path / "replay" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["minimap_rect"] == {"x": 700, "y": 20, "width": 250, "height": 200}


def test_run_reports_nonzero_when_source_swallows_recording_callback_failure(tmp_path, capsys) -> None:
    class Source:
        def start(self, on_frame, on_closed) -> None:
            try:
                on_frame(np.zeros((600, 1200, 3), dtype=np.uint8))
            except ValueError:
                pass

        def stop(self) -> None:
            self.stopped = True

    result = cli.run(_args(tmp_path), source_factory=lambda _hwnd: Source())

    output = capsys.readouterr().out
    manifest = json.loads((tmp_path / "replay" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert result != 0
    assert "RECORDING INCOMPLETE (failed): recording failed: capture size does not match calibrated capture size" in output


def test_run_terminalizes_without_writing_frames_when_live_size_differs_from_calibration(tmp_path) -> None:
    sources = []

    class Source:
        def start(self, on_frame, on_closed) -> None:
            on_frame(np.zeros((600, 1200, 3), dtype=np.uint8))

        def stop(self) -> None:
            self.stopped = True

    def source_factory(_hwnd: int) -> Source:
        source = Source()
        sources.append(source)
        return source

    with pytest.raises(ValueError, match="capture size does not match calibrated capture size"):
        cli.run(_args(tmp_path), source_factory=source_factory)

    manifest = json.loads((tmp_path / "replay" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "capture size does not match calibrated capture size" in manifest["reason"]
    assert manifest["frames"] == []
    assert not list((tmp_path / "replay" / "frames").iterdir())
    assert sources[0].stopped


def test_run_interrupt_aborts_and_cleans_up_source(tmp_path) -> None:
    sources = []

    class Source:
        def start(self, on_frame, on_closed) -> None:
            pass

        def stop(self) -> None:
            self.stopped = True

    def source_factory(hwnd: int) -> Source:
        source = Source()
        sources.append(source)
        return source

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    result = cli.run(_args(tmp_path), source_factory=source_factory, sleep=interrupt)

    assert result == 1
    assert sources[0].stopped
    manifest = json.loads((tmp_path / "replay" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"
    assert manifest["reason"] == "stopped by caller"


def test_run_deadline_aborts_and_cleans_up_source(tmp_path) -> None:
    sources = []

    class Source:
        def start(self, on_frame, on_closed) -> None:
            pass

        def stop(self) -> None:
            self.stopped = True

    def source_factory(hwnd: int) -> Source:
        source = Source()
        sources.append(source)
        return source

    result = cli.run(
        _args(tmp_path, duration=0.01), source_factory=source_factory, sleep=time.sleep
    )

    assert result == 1
    assert sources[0].stopped
    manifest = json.loads((tmp_path / "replay" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"
    assert manifest["reason"] == "deadline reached"
