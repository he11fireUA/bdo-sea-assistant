from pathlib import Path
import subprocess
import sys


def test_cli_writes_preview_and_confirmed_config(tmp_path):
    project = Path(__file__).parents[1]
    frame = project / "tests" / "fixtures" / "full_frame_arehaza.jpg"
    anchor = project / "tests" / "fixtures" / "minimap_2d_anchor.png"
    config = tmp_path / "ui_layout.json"
    preview = tmp_path / "ui_preview.jpg"

    completed = subprocess.run(
        [
            sys.executable,
            str(project / "calibrate_ui.py"),
            "--image",
            str(frame),
            "--anchor",
            str(anchor),
            "--config",
            str(config),
            "--preview",
            str(preview),
            "--confirm",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert preview.exists()
    assert config.exists()
    assert "MINIMAP" in completed.stdout
    assert "CONFIG SAVED" in completed.stdout


def test_cli_without_confirmation_does_not_write_config(tmp_path):
    project = Path(__file__).parents[1]
    frame = project / "tests" / "fixtures" / "full_frame_arehaza.jpg"
    anchor = project / "tests" / "fixtures" / "minimap_2d_anchor.png"
    config = tmp_path / "ui_layout.json"
    preview = tmp_path / "ui_preview.jpg"

    completed = subprocess.run(
        [
            sys.executable,
            str(project / "calibrate_ui.py"),
            "--image",
            str(frame),
            "--anchor",
            str(anchor),
            "--config",
            str(config),
            "--preview",
            str(preview),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert preview.exists()
    assert not config.exists()
    assert "CONFIG NOT SAVED" in completed.stdout
