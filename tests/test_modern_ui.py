from bdo_assistant.modern_ui import (
    APP_PAGES,
    DashboardSnapshot,
    ModernUiModel,
    PageId,
)


def test_modern_ui_exposes_complete_product_navigation() -> None:
    assert [page.page_id for page in APP_PAGES] == [
        PageId.DASHBOARD,
        PageId.CAPTURE,
        PageId.CALIBRATION,
        PageId.ALERTS,
        PageId.SETTINGS,
    ]
    assert all(page.title_uk and page.title_en and page.icon for page in APP_PAGES)


def test_dashboard_defaults_fail_closed_without_runtime_evidence() -> None:
    model = ModernUiModel(locale="uk")

    snapshot = model.dashboard

    assert snapshot == DashboardSnapshot()
    assert snapshot.monitoring == "stopped"
    assert snapshot.capture == "unknown"
    assert snapshot.calibration == "unknown"
    assert snapshot.minimap == "unknown"
    assert snapshot.navigation == "unknown"
    assert snapshot.hp_percent is None
    assert snapshot.rations_percent is None


def test_model_localizes_page_titles_without_changing_page_identity() -> None:
    model = ModernUiModel(locale="uk")
    ukrainian = model.navigation_items()

    model.set_locale("en")
    english = model.navigation_items()

    assert [item.page_id for item in ukrainian] == [item.page_id for item in english]
    assert ukrainian[0].title == "Огляд"
    assert english[0].title == "Dashboard"


def test_model_rejects_unsupported_locale_and_snapshot_type() -> None:
    model = ModernUiModel()

    for invalid in ("de", "", None, True):
        try:
            model.set_locale(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid locale: {invalid!r}")

    try:
        model.update_dashboard(object())
    except ValueError:
        pass
    else:
        raise AssertionError("accepted invalid dashboard snapshot")


def test_dashboard_snapshot_validates_percentages_and_states() -> None:
    valid = DashboardSnapshot(
        monitoring="running",
        capture="active",
        calibration="valid",
        minimap="visible",
        navigation="moving",
        hp_percent=78.5,
        rations_percent=42.0,
    )
    valid.validate()

    invalid_values = [
        {"hp_percent": float("nan")},
        {"rations_percent": 101},
        {"hp_percent": True},
        {"monitoring": "pretending"},
        {"capture": "connected-ish"},
    ]
    for values in invalid_values:
        try:
            DashboardSnapshot(**values).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid snapshot: {values!r}")
