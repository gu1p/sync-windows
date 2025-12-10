import os
import subprocess
import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
import platform

win32file = None
if platform.system() == "Windows":
    import win32file


    # =========================
    #  Basic config / constants
    # =========================

    FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
    FILE_ATTRIBUTE_RECALL_ON_OPEN        = 0x00040000

    # For GetCompressedFileSizeW
    _GetCompressedFileSizeW = ctypes.windll.kernel32.GetCompressedFileSizeW
    _GetCompressedFileSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    _GetCompressedFileSizeW.restype = wintypes.DWORD


# =========================
#  Enums & dataclasses
# =========================

class SyncStatus(Enum):
    ONLINE_ONLY = auto()   # Cloud only (placeholder)
    LOCAL       = auto()   # Data present locally
    MIXED       = auto()   # Folder: mix of statuses in children
    IGNORED     = auto()   # Not considered (e.g. $Temp, Trash)
    ERROR       = auto()   # Failed to inspect


@dataclass
class SizeInfo:
    logical_bytes: int = 0  # “Tamanho” – logical size
    local_bytes:   int = 0  # “Tamanho em disco” – on-disk allocation

    @property
    def cloud_bytes(self) -> int:
        """Bytes that exist only in the cloud (logical - local, never negative)."""
        diff = self.logical_bytes - self.local_bytes
        return diff if diff > 0 else 0


@dataclass
class CloudFile:
    path: Path
    status: SyncStatus
    size: SizeInfo
    error: Optional[str] = None

    # ---- ergonomic API ----

    def is_synced(self) -> bool:
        """True if this file is cloud-only (no local copy)."""
        return self.status == SyncStatus.ONLINE_ONLY

    def free_local(self) -> None:
        """Make this file online-only (like 'Free up space')."""
        if not self.path.exists():
            return
        if self.status == SyncStatus.IGNORED:
            return
        _run_attrib_free(self.path)

    def sync_local(self) -> None:
        """
        Ensure file is kept locally (like 'Always keep on this device').

        Note: CFAPI providers may or may not honor this immediately / at all.
        """
        if not self.path.exists():
            return
        if self.status == SyncStatus.IGNORED:
            return
        _run_attrib_pin(self.path)


@dataclass
class CloudFolder:
    path: Path
    status: SyncStatus
    size: SizeInfo
    children: List["CloudNode"] = field(default_factory=list)
    error: Optional[str] = None
    truncated: bool = False

    # ---- ergonomic API ----

    def files(self, recursive: bool = False) -> Iterable[CloudFile]:
        """List files directly (or recursively) under this folder."""
        for child in self.children:
            if isinstance(child, CloudFile):
                yield child
            elif recursive and isinstance(child, CloudFolder):
                yield from child.files(recursive=True)

    def folders(self, recursive: bool = False) -> Iterable["CloudFolder"]:
        """List child folders directly (or recursively)."""
        for child in self.children:
            if isinstance(child, CloudFolder):
                yield child
                if recursive:
                    yield from child.folders(recursive=True)

    def is_fully_synced(self) -> bool:
        """
        True if all non-ignored children (recursively) are ONLINE_ONLY.

        IGNORED nodes do not block sync.
        """
        if self.status == SyncStatus.ERROR:
            return False

        for node in self.walk():
            if node.status in (SyncStatus.LOCAL, SyncStatus.MIXED):
                return False
        return True

    def walk(self) -> Iterable["CloudNode"]:
        """Yield self + all descendants."""
        yield self
        for child in self.children:
            if isinstance(child, CloudFolder):
                yield from child.walk()
            else:
                yield child

    def free_local(self) -> None:
        """Free local copies for all files in this folder (recursive)."""
        for node in self.walk():
            if isinstance(node, CloudFile):
                node.free_local()

    def sync_local(self) -> None:
        """Ensure all files in this folder are kept locally (recursive)."""
        for node in self.walk():
            if isinstance(node, CloudFile):
                node.sync_local()


CloudNode = CloudFile | CloudFolder


class _TraversalBudget:
    """Bound traversal to avoid retaining arbitrarily large trees."""

    def __init__(self, max_entries: Optional[int], max_depth: Optional[int]) -> None:
        self.max_entries = max_entries if max_entries and max_entries > 0 else None
        self.max_depth = max_depth if max_depth is not None and max_depth >= 0 else None
        self.visited = 0
        self.truncated = False

    def consume(self) -> bool:
        """Count a visited node; return False if over budget."""
        self.visited += 1
        if self.max_entries is not None and self.visited > self.max_entries:
            self.truncated = True
            return False
        return True

    def can_descend(self, depth: int) -> bool:
        """
        True if we are allowed to traverse a folder at `depth`.

        Depth is root=0; children=1; grandchildren=2, etc.
        """
        if self.max_depth is not None and depth > self.max_depth:
            self.truncated = True
            return False
        if self.max_entries is not None and self.visited >= self.max_entries:
            self.truncated = True
            return False
        return True


# =========================
#  Ignore rules
# =========================

def should_ignore(path: Path) -> bool:
    """
    Paths to ignore in sync decisions.

    You can tweak this: currently ignores any segment containing '$Temp' or named 'Trash'.
    """
    lowered_parts = [p.lower() for p in path.parts]
    return any("$temp" in part or part == "trash" for part in lowered_parts)


# =========================
#  Low-level helpers (tiny)
# =========================

def _get_attributes(path: Path) -> int:
    return win32file.GetFileAttributes(str(path))


def _is_online_only(path: Path) -> bool:
    attrs = _get_attributes(path)
    return bool(attrs & (FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS | FILE_ATTRIBUTE_RECALL_ON_OPEN))


def _get_logical_size(path: Path) -> int:
    """Logical size in bytes (“Tamanho”)."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _get_local_size(path: Path) -> int:
    """
    Allocated size on disk (“Tamanho em disco”).

    Uses GetCompressedFileSizeW to approximate allocation (works for CFAPI placeholders, compressed files, etc.).
    """
    high = wintypes.DWORD(0)
    low = _GetCompressedFileSizeW(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF:
        # Error or >4GB, both cases: combine anyway.
        err = ctypes.GetLastError()
        if err != 0:
            return 0  # fallback
    return (high.value << 32) + low


def _get_file_size_info(path: Path) -> SizeInfo:
    logical = _get_logical_size(path)
    local = _get_local_size(path)
    return SizeInfo(logical_bytes=logical, local_bytes=local)


def _run_attrib_free(path: Path) -> None:
    """attrib +U -P: mark as unpinned, allow eviction (cloud-only)."""
    subprocess.run(
        ["attrib", "+U", "-P", str(path)],
        capture_output=True,
        text=True,
        shell=False,
        check=False,
    )


def _run_attrib_pin(path: Path) -> None:
    """attrib -U +P: pin / keep on this device (where supported)."""
    subprocess.run(
        ["attrib", "-U", "+P", str(path)],
        capture_output=True,
        text=True,
        shell=False,
        check=False,
    )


# =========================
#  Tree building & status
# =========================

def _build_node(path: Path) -> CloudNode:
    """Create a CloudFile or CloudFolder for a single path, without children."""
    if should_ignore(path):
        if path.is_dir():
            return CloudFolder(path=path, status=SyncStatus.IGNORED, size=SizeInfo())
        else:
            return CloudFile(path=path, status=SyncStatus.IGNORED, size=SizeInfo())

    if not path.exists():
        if path.is_dir():
            return CloudFolder(path=path, status=SyncStatus.ERROR, size=SizeInfo(), error="Path does not exist")
        else:
            return CloudFile(path=path, status=SyncStatus.ERROR, size=SizeInfo(), error="Path does not exist")

    if path.is_dir():
        # Folder, we'll fill children later
        return CloudFolder(path=path, status=SyncStatus.LOCAL, size=SizeInfo())
    else:
        return _build_file_node(path)


def _build_file_node(path: Path) -> CloudFile:
    """Create a CloudFile with status and size populated."""
    try:
        online_only = _is_online_only(path)
        status = SyncStatus.ONLINE_ONLY if online_only else SyncStatus.LOCAL
        size = _get_file_size_info(path)
        return CloudFile(path=path, status=status, size=size)
    except Exception as e:
        return CloudFile(path=path, status=SyncStatus.ERROR, size=SizeInfo(), error=str(e))


def _populate_folder_children(folder: CloudFolder, *, budget: _TraversalBudget, depth: int) -> None:
    """Populate children recursively and compute folder size/status with traversal limits."""
    if folder.status == SyncStatus.IGNORED:
        return
    if not budget.can_descend(depth):
        folder.truncated = True
        if folder.error is None:
            folder.error = "Traversal truncated (depth limit)"
        return

    children: List[CloudNode] = []

    try:
        with os.scandir(folder.path) as it:
            for entry in it:
                if not budget.consume():
                    folder.truncated = True
                    if folder.error is None:
                        folder.error = "Traversal truncated (entry limit reached)"
                    break

                child_path = Path(entry.path)
                child_node = _build_node(child_path)
                if isinstance(child_node, CloudFolder) and child_node.status != SyncStatus.ERROR:
                    _populate_folder_children(child_node, budget=budget, depth=depth + 1)
                children.append(child_node)

                if budget.truncated:
                    folder.truncated = True
                    if folder.error is None:
                        folder.error = "Traversal truncated (entry limit reached)"
                    break
    except Exception as e:
        folder.status = SyncStatus.ERROR
        folder.error = f"scandir failed: {e}"
        return

    folder.children = children
    folder.size = _aggregate_folder_size(folder)
    folder.status = _aggregate_folder_status(folder)
    folder.truncated = folder.truncated or any(isinstance(child, CloudFolder) and child.truncated for child in children)


def _aggregate_folder_size(folder: CloudFolder) -> SizeInfo:
    """Sum sizes from all descendants (ignoring IGNORED/ERROR)."""
    logical = 0
    local = 0

    for node in folder.children:
        if node.status in (SyncStatus.IGNORED, SyncStatus.ERROR):
            continue
        if isinstance(node, CloudFile):
            logical += node.size.logical_bytes
            local += node.size.local_bytes
        elif isinstance(node, CloudFolder):
            logical += node.size.logical_bytes
            local += node.size.local_bytes

    return SizeInfo(logical_bytes=logical, local_bytes=local)


def _aggregate_folder_status(folder: CloudFolder) -> SyncStatus:
    """Compute folder sync status from children."""
    effective = [
        node.status
        for node in folder.children
        if node.status not in (SyncStatus.IGNORED, SyncStatus.ERROR)
    ]

    if not effective:
        # Only ignored/error children → don't block parents.
        return SyncStatus.IGNORED

    unique = set(effective)

    if unique == {SyncStatus.ONLINE_ONLY}:
        return SyncStatus.ONLINE_ONLY
    if unique == {SyncStatus.LOCAL}:
        return SyncStatus.LOCAL
    return SyncStatus.MIXED


# =========================
#  Public API
# =========================

def build_sync_tree(
    root: Path,
    *,
    max_entries: Optional[int] = 50_000,
    max_depth: Optional[int] = None,
) -> CloudFolder:
    """
    Build a CloudFolder tree starting from `root`, with traversal caps to avoid OOM.

    Use `walk_cloud_files` for large trees when you only need streaming status.
    Set `max_entries=None` to disable the cap (not recommended for huge trees).
    """
    budget = _TraversalBudget(max_entries, max_depth)
    root_node = _build_node(root)
    if not isinstance(root_node, CloudFolder):
        raise ValueError(f"Root must be a folder: {root}")

    if not budget.consume():
        root_node.truncated = True
        root_node.error = root_node.error or "Traversal truncated (entry limit reached)"
        return root_node

    _populate_folder_children(root_node, budget=budget, depth=0)
    root_node.truncated = root_node.truncated or budget.truncated
    if root_node.truncated and root_node.error is None:
        root_node.error = "Traversal truncated; prefer walk_cloud_files() for streaming status."
    return root_node


def walk_cloud_files(
    root: Path,
    *,
    max_entries: Optional[int] = 50_000,
    max_depth: Optional[int] = None,
) -> Iterator[CloudFile]:
    """
    Stream CloudFile nodes without retaining the full tree in memory.

    Depth is relative to `root` (root=0). Set `max_entries=None` to disable the cap.
    """
    if not root.exists():
        return

    budget = _TraversalBudget(max_entries, max_depth)
    stack: list[tuple[Path, int]] = [(root, 0)]

    while stack:
        current, depth = stack.pop()
        if should_ignore(current):
            continue
        if budget.max_depth is not None and depth > budget.max_depth:
            continue

        try:
            with os.scandir(current) as it:
                for entry in it:
                    entry_path = Path(entry.path)
                    if should_ignore(entry_path):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if budget.can_descend(depth + 1):
                                stack.append((entry_path, depth + 1))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        continue

                    if not budget.consume():
                        return
                    yield _build_file_node(entry_path)
                    if budget.truncated:
                        return
        except OSError:
            continue


def probe_cloud_file(path: Path) -> CloudFile:
    """Lightweight status probe for a single file without walking the whole tree."""
    node = _build_node(path)
    if isinstance(node, CloudFile):
        return node
    return CloudFile(
        path=path,
        status=SyncStatus.ERROR,
        size=SizeInfo(),
        error="Path is a folder",
    )


# =========================
#  Lightweight usage probe
# =========================

def sample_local_usage(
    root: Path,
    *,
    max_entries: int = 5000,
    timeout_s: float = 1.5,
    max_dirs: Optional[int] = None,
) -> tuple[int, bool]:
    """
    Estimate local allocation under `root` by walking a bounded number of entries.

    Returns (local_bytes, complete) where `complete` indicates whether the traversal
    finished before hitting limits/timeouts.
    """
    if not root.exists():
        return 0, True

    start = time.time()
    time_budget = timeout_s if timeout_s and timeout_s > 0 else None
    entry_budget = max_entries if max_entries and max_entries > 0 else None
    dir_budget = max_dirs if max_dirs is not None and max_dirs > 0 else entry_budget
    entries_seen = 0
    dirs_seen = 0
    local_bytes = 0
    stack = [root]

    while stack:
        current = stack.pop()
        if should_ignore(current):
            continue
        if time_budget is not None and (time.time() - start) >= time_budget:
            return local_bytes, False
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if time_budget is not None and (time.time() - start) >= time_budget:
                        return local_bytes, False
                    entry_path = Path(entry.path)
                    if should_ignore(entry_path):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs_seen += 1
                            if dir_budget is not None and dirs_seen > dir_budget:
                                return local_bytes, False
                            stack.append(entry_path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        entries_seen += 1
                        local_bytes += _get_local_size(entry_path)
                    except OSError:
                        continue
                    if entry_budget is not None and entries_seen >= entry_budget:
                        return local_bytes, False
        except OSError:
            continue

    return local_bytes, True
