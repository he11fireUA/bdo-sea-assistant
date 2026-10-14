from bdo_assistant.window_discovery import (
    RawWindow,
    WindowCandidate,
    select_bdo_windows,
)


def test_selects_only_visible_blackdesert64_top_level_windows() -> None:
    windows = (
        RawWindow(101, 11, "BLACK DESERT - 526388", True, False, 10, 20, 5106, 1440),
        RawWindow(102, 12, "Launcher", True, False, 0, 0, 900, 700),
        RawWindow(103, 11, "", True, False, 0, 0, 100, 100),
        RawWindow(104, 11, "BLACK DESERT hidden", False, False, 0, 0, 5106, 1440),
    )

    selected = select_bdo_windows(windows, {11: "BlackDesert64.exe", 12: "patcher.exe"})

    assert selected == (
        WindowCandidate(
            hwnd=101,
            pid=11,
            title="BLACK DESERT - 526388",
            executable="BlackDesert64.exe",
            width=5106,
            height=1440,
            minimized=False,
        ),
    )


def test_discovery_is_case_insensitive_and_preserves_minimized_state() -> None:
    windows = (
        RawWindow(200, 21, "BLACK DESERT", True, True, 0, 0, 2560, 1440),
    )

    selected = select_bdo_windows(windows, {21: "blackdesert64.EXE"})

    assert selected[0].minimized is True


def test_discovery_rejects_invalid_window_and_process_inputs() -> None:
    valid_window = RawWindow(1, 2, "BDO", True, False, 0, 0, 100, 100)
    invalid_calls = (
        ((object(),), {2: "BlackDesert64.exe"}),
        ((valid_window,), {True: "BlackDesert64.exe"}),
        ((valid_window,), {2: ""}),
    )
    for windows, processes in invalid_calls:
        try:
            select_bdo_windows(windows, processes)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid discovery input")


def test_raw_window_and_candidate_reject_invalid_geometry_and_identifiers() -> None:
    invalid_windows = (
        lambda: RawWindow(True, 2, "BDO", True, False, 0, 0, 100, 100),
        lambda: RawWindow(1, 0, "BDO", True, False, 0, 0, 100, 100),
        lambda: RawWindow(1, 2, "BDO", True, False, 0, 0, 0, 100),
        lambda: RawWindow(1, 2, "BDO", 1, False, 0, 0, 100, 100),
    )
    for make in invalid_windows:
        try:
            make()
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid raw window")
