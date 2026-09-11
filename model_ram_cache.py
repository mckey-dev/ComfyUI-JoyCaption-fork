from __future__ import annotations

import atexit
import ctypes
import mmap
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _PinnedFile:
    path: Path
    handle: object
    mapping: mmap.mmap
    locked: bool


_pins: dict[str, _PinnedFile] = {}


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.c_void_p),
        ("strides", ctypes.c_void_p),
        ("suboffsets", ctypes.c_void_p),
        ("internal", ctypes.c_void_p),
    ]


_PyObject_GetBuffer = ctypes.pythonapi.PyObject_GetBuffer
_PyObject_GetBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_PyObject_GetBuffer.restype = ctypes.c_int
_PyBuffer_Release = ctypes.pythonapi.PyBuffer_Release
_PyBuffer_Release.argtypes = [ctypes.POINTER(_PyBuffer)]
_PyBUF_SIMPLE = 0

if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _kernel32.VirtualLock.restype = ctypes.c_int
    _kernel32.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _kernel32.VirtualUnlock.restype = ctypes.c_int
    _libc = None
else:
    _kernel32 = None
    _libc = ctypes.CDLL(None, use_errno=True)
    _libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libc.mlock.restype = ctypes.c_int
    _libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libc.munlock.restype = ctypes.c_int


def _mapping_address(mapping: mmap.mmap) -> int:
    view = _PyBuffer()
    if _PyObject_GetBuffer(mapping, ctypes.byref(view), _PyBUF_SIMPLE) != 0:
        raise OSError("Could not read mmap buffer address")
    try:
        if not view.buf:
            raise OSError("mmap buffer address is null")
        return int(view.buf)
    finally:
        _PyBuffer_Release(ctypes.byref(view))


def _virtual_lock(address: int, length: int) -> bool:
    return _kernel32.VirtualLock(ctypes.c_void_p(address), ctypes.c_size_t(length)) != 0


def _mlock(address: int, length: int) -> bool:
    return _libc.mlock(ctypes.c_void_p(address), ctypes.c_size_t(length)) == 0


def _try_lock(mapping: mmap.mmap) -> bool:
    length = len(mapping)
    if length == 0:
        return True
    address = _mapping_address(mapping)
    if os.name == "nt":
        return _virtual_lock(address, length)
    return _mlock(address, length)


def _touch_pages(mapping: mmap.mmap) -> None:
    length = len(mapping)
    if length == 0:
        return
    page = mmap.PAGESIZE
    offset = 0
    while offset < length:
        mapping[offset]
        offset += page
    mapping[length - 1]


def _try_unlock(mapping: mmap.mmap) -> None:
    length = len(mapping)
    if length == 0:
        return
    address = _mapping_address(mapping)
    if os.name == "nt":
        _kernel32.VirtualUnlock(ctypes.c_void_p(address), ctypes.c_size_t(length))
        return
    _libc.munlock(ctypes.c_void_p(address), ctypes.c_size_t(length))


def _release(pin: _PinnedFile) -> None:
    if pin.locked:
        try:
            _try_unlock(pin.mapping)
        except (OSError, ValueError):
            pass
    try:
        pin.mapping.close()
    except (BufferError, ValueError, OSError):
        pass
    try:
        pin.handle.close()
    except OSError:
        pass


def _pin_one(path: Path) -> _PinnedFile:
    handle = path.open("rb")
    try:
        size = os.fstat(handle.fileno()).st_size
        mapping = mmap.mmap(handle.fileno(), size, access=mmap.ACCESS_READ)
    except BaseException:
        handle.close()
        raise
    _touch_pages(mapping)
    locked = False
    try:
        locked = _try_lock(mapping)
    except OSError:
        locked = False
    if not locked:
        print(
            "[LLM Text Processor] "
            f"Could not lock {path.name} in RAM; pages may be reclaimed under memory pressure."
        )
    return _PinnedFile(path=path, handle=handle, mapping=mapping, locked=locked)


def pin_files(paths: list[Path | None] | tuple[Path | None, ...]) -> None:
    wanted: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        wanted.append(resolved)

    wanted_keys = {str(path) for path in wanted}
    for key in list(_pins):
        if key not in wanted_keys:
            _release(_pins.pop(key))

    for path in wanted:
        key = str(path)
        if key in _pins:
            continue
        _pins[key] = _pin_one(path)


def unpin() -> None:
    pins = list(_pins.values())
    _pins.clear()
    for pin in pins:
        _release(pin)


def pinned_paths() -> tuple[Path, ...]:
    return tuple(pin.path for pin in _pins.values())


atexit.register(unpin)
