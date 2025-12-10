from __future__ import annotations

import math
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class MigrationConfig:
    """Configuration for the migration."""

    source_root: Path
    sync_root: Path
    migration_subdir: str
    max_local_bytes: int
    high_watermark: float = 0.90
    low_watermark: float = 0.60
    usage_refresh_interval: float = 3.0
    usage_reconcile_interval: float = 30.0  # how often to touch disk for actual usage
    usage_reconcile_timeout: float = 1.5  # max seconds spent during a reconcile sweep
    usage_reconcile_entry_limit: int = 5000  # max files visited during reconciliation
    chunk_size_bytes: int = 4 * 1024 * 1024  # 4 MiB
    manifest_path: Optional[Path] = None
    eager_free_after_copy: bool = False  # if True, free everything at the end
    scan_workers: int = 8  # concurrent directory scanners
    scan_queue_size: int = 2000  # results buffer between scanner and copier
    pending_window_size: int = 4000  # max SourceFiles held in memory at once
    event_queue_size: int = 5000  # max events buffered between engine and UI
    copy_headroom_bytes: int = 1 * 1024 * 1024 * 1024  # required free bytes before starting next batch
    space_wait_seconds: float = 5.0  # base wait when space is insufficient
    space_wait_max_seconds: float = 60.0  # cap for wait backoff when freeing is slow
    copy_progress_bytes_step: int = 64 * 1024 * 1024  # throttle progress events
    copy_progress_interval: float = 1.0  # seconds between progress events even if bytes step not hit
    scan_status_fast_interval: float = 0.5  # heartbeat when scanning/backpressure active
    scan_status_slow_interval: float = 2.0  # heartbeat when idle to reduce chatter
    free_batch_size: int = 512  # how many manifest entries to consider per free pass
    eager_free_interval: float = 10.0  # seconds between eager free attempts
    eager_free_pending_threshold: float = 0.5  # skip eager free when pending backlog is above this fraction of the window

    @property
    def migration_root(self) -> Path:
        return self.sync_root / self.migration_subdir

    def resolve_manifest_path(self) -> Path:
        if self.manifest_path is not None:
            return self.manifest_path
        # Default: stable temp file derived from source path (keeps out of sync tree).
        temp_dir = Path(tempfile.gettempdir())
        digest = hashlib.sha256(str(self.source_root.resolve()).encode("utf-8")).hexdigest()[:12]
        return temp_dir / f"migration_inventory_{digest}.sqlite"


@dataclass
class SourceFile:
    """Represents a file on the source (external) drive."""

    src: Path
    src_rel: Path
    dest_rel: Path
    size: int
    mtime: float


@dataclass
class SourceScanResult:
    """Lightweight scan result produced by concurrent scanners."""

    path: Path
    rel: Path
    size: int
    mtime: float


@dataclass
class ManifestEntry:
    """State for a single file in the migration inventory."""

    rel: str  # POSIX-style relative path in destination (normalized)
    source_rel: str  # Original source relative path (posix)
    logical_bytes: int
    mtime: float
    state: str  # "pending" | "copied" | "freed" | "error"


@dataclass
class MigrationEvent:
    """Event passed from the engine thread to the Textual UI."""

    type: str
    payload: dict


def same_file(entry: ManifestEntry, size: int, mtime: float) -> bool:
    """Return True if manifest entry matches given size/mtime."""
    return entry.logical_bytes == size and math.isclose(entry.mtime, mtime)


INVALID_CHARS = set('<>:"/\\|?*')
RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
}


def normalize_windows_component(name: str) -> Tuple[str, bool]:
    changed = False
    cleaned_chars = []
    for ch in name:
        if ch in INVALID_CHARS or ord(ch) < 32:
            cleaned_chars.append("_")
            changed = True
        else:
            cleaned_chars.append(ch)
    sanitized = "".join(cleaned_chars)
    stripped = sanitized.rstrip(" .")
    if stripped != sanitized:
        changed = True
        sanitized = stripped
    if not sanitized:
        sanitized = "_"
        changed = True

    parts = sanitized.split(".")
    base = parts[0]
    if base.lower() in RESERVED_NAMES:
        base = base + "_"
        changed = True
    if len(parts) > 1:
        sanitized = ".".join([base] + parts[1:])
    else:
        sanitized = base

    return sanitized, changed


def normalize_windows_path(rel: Path) -> Tuple[Path, bool]:
    """Normalize a relative path to be Windows-safe (component-wise)."""
    new_parts = []
    changed_any = False
    for part in rel.parts:
        new_part, changed = normalize_windows_component(part)
        new_parts.append(new_part)
        changed_any = changed_any or changed
    return Path(*new_parts), changed_any
