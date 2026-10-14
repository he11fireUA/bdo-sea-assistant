import os
from pathlib import Path

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication

from bdo_assistant.app import AlphaAppSettings
from bdo_assistant.modern_window import (
    ModernMainWindow,
    application_icon,
    application_icon_path,
    configure_application,
)


class FakeController:
    settings = AlphaAppSettings()


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_windows_icon_asset_is_multi_resolution_and_openable() -> None:
    path = application_icon_path()

    assert path == Path(__file__).parents[1] / "assets" / "app_icon.ico"
    assert path.is_file()
    with Image.open(path) as image:
        assert image.format == "ICO"
        assert set(image.ico.sizes()) == {
            (16, 16),
            (20, 20),
            (24, 24),
            (32, 32),
            (40, 40),
            (48, 48),
            (64, 64),
            (128, 128),
            (256, 256),
        }
        image.load()


def test_configured_application_and_main_window_use_non_null_app_icon() -> None:
    application = configure_application(_app())
    window = ModernMainWindow(FakeController())

    assert not application.windowIcon().isNull()
    assert not application_icon().isNull()
    assert not window.windowIcon().isNull()
    assert window.windowIcon().pixmap(QSize(32, 32)).toImage() == application.windowIcon().pixmap(QSize(32, 32)).toImage()
    window.close()
