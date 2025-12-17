from __future__ import annotations

import queue
import threading
import sys
import traceback
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from old import models as migration_models
from engine.models import Tree
from engine.events import EventType
from old.sync_api import probe_cloud_file, SyncStatus  # type: ignore


class MigrationEngine:
    """
    Thin bridge:
    - Runs the new sharded Tree copier (all FS ops live there).
    - Subscribes to its events and relays them to the UI queue (non-blocking).
    - Marks copied files online-only in a background worker fed by copy_done events.
    - No extra persistence beyond the new ProgressTracker JSON.
    """

    def __init__(self, config: migration_models.MigrationConfig, events: "queue.Queue[migration_models.MigrationEvent]"):
        self.config = config
        self.events = events

        configured_max = getattr(events, "maxsize", 0) or 0
        self._event_queue_maxsize: int = configured_max or max(0, self.config.event_queue_size)
        self._event_soft_limit: int = int(self._event_queue_maxsize * 0.8) if self._event_queue_maxsize else 0
        self._low_priority_events = {EventType.COPY_PROGRESS, EventType.SCAN_STATUS, EventType.SCAN_PROGRESS}

        self.total_bytes: int = 0
        self.copied_bytes: int = 0
        self.freed_bytes: int = 0
        self.freed_files: int = 0
        self.last_freed: str = ""
        self.local_usage_bytes: int = 0

        self._stop_flag = False
        self._online_queue: "queue.Queue[tuple[str, int]]" = queue.Queue(maxsize=512)
        self._online_thread: Optional[threading.Thread] = None

    # ------------------ lifecycle ------------------

    @property
    def migration_root(self) -> Path:
        return self.config.migration_root

    def run(self) -> None:
        """Blocking main loop: run the sharded copier and forward events."""
        self.total_bytes = 0
        self.copied_bytes = 0
        self.freed_bytes = 0
        self.freed_files = 0
        self.last_freed = ""
        self.local_usage_bytes = 0
        self._emit_status("Starting migration...")
        self._emit_usage()
        self._start_online_only_worker()
        tree = Tree(
            self.config.source_root,
            event_cb=self._handle_event_from_tree,
        )
        try:
            tree.copy_to(dst=self.migration_root, progress_dir=None)
        except Exception as exc:  # noqa: BLE001
            self._emit_engine_exception(exc)
        finally:
            self.stop()

    def _emit_engine_exception(self, exc: BaseException) -> None:
        """Log a fatal error with stack trace to the UI and stderr, then signal finish."""
        # Print to stdout so the traceback is visible after the TUI exits.
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stdout)

        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        self._write_error_file(tb_lines)
        self._emit(EventType.LOG, level="error", message="Engine error encountered; shutting down.")
        for line in tb_lines:
            for segment in line.rstrip("\n").splitlines():
                self._emit(EventType.LOG, level="error", message=segment)
        self._emit(EventType.PHASE, stage="error", message="Engine stopped due to an error")
        self._emit(EventType.FINISHED, message="Engine stopped due to an error.", error=True)

    def _write_error_file(self, tb_lines: list[str]) -> None:
        """Append stack trace to sync_errors.log for visibility in terminals that hide stdout."""
        log_path = Path("sync_errors.log").resolve()
        timestamp = datetime.now().isoformat(timespec="seconds")
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] Engine error\n")
                for line in tb_lines:
                    f.write(line)
                f.write("\n")
        except Exception as log_exc:  # noqa: BLE001
            print(f"Failed to write {log_path}: {log_exc}", file=sys.stdout)

    def stop(self) -> None:
        self._stop_flag = True
        self._stop_online_only_worker(wait=True)

    # ------------------ event bridge ------------------

    def _handle_event_from_tree(self, type_: Union[EventType, str], payload: dict) -> None:
        """Translate tree events to UI queue; enqueue copy_done for online-only."""
        normalized = self._normalize_event_type(type_)

        if normalized == EventType.COPY_START:
            self.total_bytes += int(payload.get("size", 0) or 0)
            self._emit_status()
        if normalized == EventType.COPY_DONE:
            self.copied_bytes += int(payload.get("bytes_total", 0) or 0)
            self._emit_status()
            self._emit_usage()
            rel = payload.get("rel")
            size = int(payload.get("bytes_total", 0) or 0)
            if rel:
                self._enqueue_online_only(rel, size)
        if normalized == EventType.FINISHED:
            self._emit_status(payload.get("message", ""))
            self._emit_usage()
        self._emit(type_, **payload)

    def _emit(self, type_: Union[EventType, str], **payload) -> None:
        type_name = type_.value if isinstance(type_, EventType) else str(type_)
        maxsize = self._event_queue_maxsize
        if maxsize:
            try:
                backlog = self.events.qsize()
            except Exception:
                backlog = 0
            if backlog >= self._event_soft_limit and self._should_drop_low_priority(type_):
                return

        event = migration_models.MigrationEvent(type=type_name, payload=payload)
        try:
            self.events.put_nowait(event)
        except queue.Full:
            if not self._should_drop_low_priority(type_):
                try:
                    self.events.put(event, timeout=0.05)
                except queue.Full:
                    pass

    @staticmethod
    def _normalize_event_type(type_: Union[EventType, str]) -> Optional[EventType]:
        if isinstance(type_, EventType):
            return type_
        try:
            return EventType(str(type_))
        except Exception:
            return None

    def _should_drop_low_priority(self, type_: Union[EventType, str]) -> bool:
        try:
            normalized = type_ if isinstance(type_, EventType) else EventType(str(type_))
        except Exception:
            return False
        if normalized == EventType.LOG:
            return False
        return normalized in self._low_priority_events

    # ------------------ online-only worker ------------------

    def _start_online_only_worker(self) -> None:
        if self._online_thread and self._online_thread.is_alive():
            return

        def _worker() -> None:
            while not self._stop_flag or not self._online_queue.empty():
                try:
                    rel, logical_bytes = self._online_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    self._mark_online_only(rel, logical_bytes)
                finally:
                    self._online_queue.task_done()

        self._online_thread = threading.Thread(target=_worker, daemon=True)
        self._online_thread.start()

    def _stop_online_only_worker(self, wait: bool = False, timeout: float = 5.0) -> None:
        if self._online_thread is None:
            return
        self._stop_flag = True
        if wait:
            self._online_thread.join(timeout=timeout)
        if not self._online_thread.is_alive():
            self._online_thread = None

    def _enqueue_online_only(self, rel: str, logical_bytes: int) -> None:
        try:
            self._online_queue.put_nowait((rel, logical_bytes))
        except queue.Full:
            self._emit(EventType.LOG, level="warning", message="Online-only queue full; skipping free-local enqueue.")

    def _mark_online_only(self, rel: str, logical_bytes: int) -> None:
        path = self.migration_root / rel
        self._emit(EventType.FREE_START, rel=rel)
        attempts = 0
        max_attempts = 2
        delay = 0.5
        while attempts <= max_attempts and not self._stop_flag:
            cf = probe_cloud_file(path)
            if cf.status == SyncStatus.ONLINE_ONLY:
                self._record_freed(rel, logical_bytes)
                return
            if cf.status in (SyncStatus.LOCAL, SyncStatus.MIXED):
                try:
                    cf.free_local()
                except Exception as exc:  # noqa: BLE001
                    self._emit(
                        EventType.LOG,
                        level="warning",
                        message=f"Error freeing {rel} (attempt {attempts + 1}/{max_attempts + 1}): {exc!r}",
                    )
            attempts += 1
            if attempts <= max_attempts:
                import time as _t
                _t.sleep(delay)

        cf = probe_cloud_file(path)
        if cf.status == SyncStatus.ONLINE_ONLY:
            self._record_freed(rel, logical_bytes)
        elif cf.status == SyncStatus.ERROR:
            self._emit(
                EventType.LOG,
                level="warning",
                message=f"Could not verify cloud-only for {rel}: {cf.error or 'unknown error'}",
            )

    def _record_freed(self, rel: str, logical_bytes: int) -> None:
        self.freed_bytes += logical_bytes
        self.freed_files += 1
        self.last_freed = rel
        self._emit(
            EventType.FREE_DONE,
            rel=rel,
            freed_bytes=logical_bytes,
            freed_files=self.freed_files,
            last_freed=rel,
        )
        self._emit_status()
        self._emit_usage()

    def _emit_status(self, message: str = "") -> None:
        self._emit(
            EventType.STATUS,
            message=message,
            total_bytes=self.total_bytes,
            copied_bytes=self.copied_bytes,
            freed_bytes=self.freed_bytes,
            freed_files=self.freed_files,
            last_freed=self.last_freed,
        )

    def _emit_usage(self) -> None:
        self.local_usage_bytes = max(self.copied_bytes - self.freed_bytes, 0)
        self._emit(
            EventType.USAGE,
            local_bytes=self.local_usage_bytes,
            max_local_bytes=self.config.max_local_bytes,
        )
