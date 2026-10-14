from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RawWindow:
    hwnd: int
    pid: int
    title: str
    visible: bool
    minimized: bool
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if type(self.hwnd) is not int or self.hwnd <= 0:
            raise ValueError("hwnd must be a positive integer")
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("pid must be a positive integer")
        if type(self.title) is not str:
            raise ValueError("title must be a string")
        if type(self.visible) is not bool or type(self.minimized) is not bool:
            raise ValueError("window visibility flags must be boolean")
        if any(type(value) is not int for value in (self.x, self.y, self.width, self.height)):
            raise ValueError("window geometry must use integers")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("window dimensions must be positive")


@dataclass(frozen=True)
class WindowCandidate:
    hwnd: int
    pid: int
    title: str
    executable: str
    width: int
    height: int
    minimized: bool

    def __post_init__(self) -> None:
        if type(self.hwnd) is not int or self.hwnd <= 0:
            raise ValueError("candidate hwnd must be a positive integer")
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("candidate pid must be a positive integer")
        if type(self.title) is not str or not self.title.strip():
            raise ValueError("candidate title must be non-empty")
        if type(self.executable) is not str or not self.executable.strip():
            raise ValueError("candidate executable must be non-empty")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("candidate width must be a positive integer")
        if type(self.height) is not int or self.height <= 0:
            raise ValueError("candidate height must be a positive integer")
        if type(self.minimized) is not bool:
            raise ValueError("candidate minimized flag must be boolean")


def select_bdo_windows(
    windows: Sequence[RawWindow],
    process_names: Mapping[int, str],
) -> tuple[WindowCandidate, ...]:
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        raise ValueError("windows must be a sequence")
    if not isinstance(process_names, Mapping):
        raise ValueError("process_names must be a mapping")
    for window in windows:
        if type(window) is not RawWindow:
            raise ValueError("windows must contain RawWindow values")
    for pid, executable in process_names.items():
        if type(pid) is not int or pid <= 0:
            raise ValueError("process ids must be positive integers")
        if type(executable) is not str or not executable.strip():
            raise ValueError("process names must be non-empty strings")

    selected = []
    for window in windows:
        executable = process_names.get(window.pid)
        if (
            not window.visible
            or not window.title.strip()
            or executable is None
            or executable.casefold() != "blackdesert64.exe"
        ):
            continue
        selected.append(
            WindowCandidate(
                hwnd=window.hwnd,
                pid=window.pid,
                title=window.title.strip(),
                executable=executable,
                width=window.width,
                height=window.height,
                minimized=window.minimized,
            )
        )
    return tuple(sorted(selected, key=lambda candidate: candidate.hwnd))


def discover_bdo_windows() -> tuple[WindowCandidate, ...]:
    return select_bdo_windows(_enumerate_top_level_windows(), _enumerate_process_names())


def _enumerate_process_names() -> dict[int, str]:
    import ctypes
    from ctypes import wintypes

    max_path = 260
    th32cs_snapprocess = 0x00000002
    invalid_handle = ctypes.c_void_p(-1).value

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * max_path),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    create_snapshot.restype = wintypes.HANDLE
    process_first = kernel32.Process32FirstW
    process_first.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W))
    process_first.restype = wintypes.BOOL
    process_next = kernel32.Process32NextW
    process_next.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W))
    process_next.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    snapshot = create_snapshot(th32cs_snapprocess, 0)
    if snapshot == invalid_handle:
        raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")
    processes: dict[int, str] = {}
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        success = process_first(snapshot, ctypes.byref(entry))
        while success:
            pid = int(entry.th32ProcessID)
            executable = str(entry.szExeFile)
            if pid > 0 and executable:
                processes[pid] = executable
            success = process_next(snapshot, ctypes.byref(entry))
    finally:
        close_handle(snapshot)
    return processes


def _enumerate_top_level_windows() -> tuple[RawWindow, ...]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    windows: list[RawWindow] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def callback(hwnd, lparam):
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if int(pid.value) <= 0 or width <= 0 or height <= 0:
            return True
        windows.append(
            RawWindow(
                hwnd=int(hwnd),
                pid=int(pid.value),
                title=title_buffer.value,
                visible=bool(user32.IsWindowVisible(hwnd)),
                minimized=bool(user32.IsIconic(hwnd)),
                x=int(rect.left),
                y=int(rect.top),
                width=width,
                height=height,
            )
        )
        return True

    if not user32.EnumWindows(callback, 0):
        raise OSError(ctypes.get_last_error(), "EnumWindows failed")
    return tuple(windows)
