from __future__ import annotations

import queue
import threading
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Log, ProgressBar, Static

from old.engine import MigrationEngine
from old.models import MigrationEvent


class MigrationApp(App[None]):
    """Textual UI to display migration progress."""

    LOG_MAX_LINES = 500
    LOG_BACKPRESSURE_THRESHOLD = 2000
    MAX_EVENTS_PER_DRAIN = 500

    CSS = """
    Screen {
        layout: vertical;
    }

    #top {
        height: 8;
        padding: 1;
    }

    #stats {
        padding: 1;
    }

    #log {
        height: 1fr;
        border: solid $accent;
    }

    .label {
        text-style: bold;
    }

    ProgressBar {
        height: 1;
    }
    """

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, engine: MigrationEngine) -> None:
        super().__init__()
        self.engine = engine
        self.engine_thread: Optional[threading.Thread] = None

        self.total_bytes: int = 0
        self.copied_bytes: int = 0
        self.freed_bytes: int = 0
        self.freed_files: int = 0
        self.last_freed: str = ""
        self.local_bytes: int = 0
        self.max_local_bytes: int = self.engine.config.max_local_bytes
        self.phase_message: str = "Starting..."
        self.scan_files: int = 0
        self.scan_bytes: int = 0
        self._dropped_log_events: int = 0
        self._log_backpressure_threshold: int = self.LOG_BACKPRESSURE_THRESHOLD
        max_event_queue = getattr(self.engine.events, "maxsize", 0) or self.engine.config.event_queue_size
        if max_event_queue:
            self._log_backpressure_threshold = max(100, int(max_event_queue * 0.6))

    # ------------------ layout ------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="top"):
            src = str(self.engine.config.source_root)
            dst = str(self.engine.migration_root)
            yield Static(f"Source:      {src}", id="src_label")
            yield Static(f"Destination: {dst}", id="dst_label")
            yield Static(f"Phase: {self.phase_message}", id="phase_label")
            yield Static("Scan: starting...", id="scan_label")
            yield Static("Scan dir: (pending...)", id="scan_dir_label")

        with Vertical(id="stats"):
            yield Static("Copy to sync folder", classes="label")
            yield ProgressBar(id="copy_overall")

            yield Static("Cloud / freed (online-only)", classes="label")
            yield ProgressBar(id="cloud_overall")
            yield Static("Freed files: 0", id="freed_count_label")
            yield Static("Latest freed: -", id="latest_freed_label")

            yield Static("Local usage vs max", classes="label")
            with Horizontal():
                yield ProgressBar(id="local_usage")
                yield Static("", id="local_usage_text")

            yield Static("Current file", classes="label")
            yield Static("", id="current_file_label")
            yield ProgressBar(id="current_file_progress")

        yield Log(id="log", max_lines=self.LOG_MAX_LINES)
        yield Footer()

    # ------------------ lifecycle ------------------

    def on_mount(self) -> None:
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=True)
        self.engine_thread.start()
        self.set_interval(0.1, self._drain_events)

    def action_toggle_dark(self) -> None:
        self.theme = "textual-dark" if self.theme == "textual-light" else "textual-light"

    def action_quit(self) -> None:
        self.engine.stop()
        self.exit()

    # ------------------ event handling ------------------

    def _event_backlog(self) -> int:
        try:
            return self.engine.events.qsize()
        except Exception:
            return 0

    def _drain_events(self) -> None:
        processed = 0

        while processed < self.MAX_EVENTS_PER_DRAIN:
            try:
                event = self.engine.events.get_nowait()
            except queue.Empty:
                break
            drop_logs = self._event_backlog() > self._log_backpressure_threshold
            if drop_logs and event.type == "log":
                self._dropped_log_events += 1
                processed += 1
                continue
            self._handle_event(event)
            processed += 1

        if self._dropped_log_events and self._event_backlog() <= self._log_backpressure_threshold:
            self._log_line(
                f"Suppressed {self._dropped_log_events} log events to keep the UI responsive",
                "warning",
            )
            self._dropped_log_events = 0

    def _handle_event(self, event: MigrationEvent) -> None:
        t = event.type
        p = event.payload

        if t == "log":
            self._log_line(p.get("message", ""), p.get("level", "info"))
        elif t == "phase":
            self._update_phase(p.get("stage", ""), p.get("message", ""))
        elif t == "scan_progress":
            self._update_scan_progress(p.get("files", 0), p.get("bytes", 0))
        elif t == "scan_status":
            self._update_scan_status(
                p.get("current_dir", "(pending...)"),
                p.get("scan_queue", 0),
                p.get("pending_files", 0),
                p.get("pending_window_limit"),
            )
        elif t == "shard_start":
            self._handle_shard_event(p, done=False)
        elif t == "shard_done":
            self._handle_shard_event(p, done=True)
        elif t == "status":
            self.total_bytes = p.get("total_bytes", 0)
            self.copied_bytes = p.get("copied_bytes", 0)
            self.freed_bytes = p.get("freed_bytes", 0)
            self.freed_files = p.get("freed_files", self.freed_files)
            self.last_freed = p.get("last_freed", self.last_freed)
            msg = p.get("message", "")
            if msg:
                self._log_line(msg, "info")
            self._update_overall_bars()
        elif t == "usage":
            self.local_bytes = p.get("local_bytes", 0)
            self.max_local_bytes = p.get("max_local_bytes", self.max_local_bytes)
            self._update_local_usage()
        elif t == "copy_start":
            rel = p.get("rel", "")
            size = p.get("size", 0)
            self._log_line(f"Copying {rel} ({self._fmt_bytes(size)})", "info")
            self._update_current_file(rel, 0, size)
        elif t == "copy_progress":
            self._update_current_file(p.get("rel", ""), p.get("bytes_done", 0), p.get("bytes_total", 0))
        elif t == "copy_done":
            rel = p.get("rel", "")
            self._log_line(f"Finished copying {rel}", "success")
            self._update_current_file(rel, 0, 0)
            self._update_overall_bars()
        elif t == "free_start":
            rel = p.get("rel", "")
            self._log_line(f"Freeing local copy of {rel}", "info")
        elif t == "free_done":
            rel = p.get("rel", "")
            freed_bytes = p.get("freed_bytes", 0)
            self._log_line(f"Freed {rel} ({self._fmt_bytes(freed_bytes)} locally)", "success")
            self.freed_files = p.get("freed_files", self.freed_files)
            self.last_freed = p.get("last_freed", rel)
            self._update_overall_bars()
            self._update_freed_labels()
        elif t == "finished":
            msg = p.get("message", "Finished.")
            is_error = bool(p.get("error"))
            self._log_line(msg, "error" if is_error else "success")
            self._update_overall_bars()
            self._update_local_usage()
            if is_error:
                self.exit()

    # ------------------ UI helpers ------------------

    def _log_line(self, message: str, level: str = "info") -> None:
        log = self.query_one("#log", Log)
        prefix = {
            "info": "[INFO] ",
            "warning": "[WARN] ",
            "error": "[ERROR]",
            "success": "[OK]  ",
        }.get(level, "")
        log.write_line(f"{prefix} {message}")

    def _update_phase(self, stage: str, message: str) -> None:
        label = self.query_one("#phase_label", Static)
        text = message or stage or "..."
        self.phase_message = text
        label.update(f"Phase: {text}")

    def _update_scan_progress(self, files: int, bytes_count: int) -> None:
        label = self.query_one("#scan_label", Static)
        self.scan_files = files
        self.scan_bytes = bytes_count
        label.update(f"Scan: {files} files ({self._fmt_bytes(bytes_count)})")

    def _update_scan_status(
        self, current_dir: str, scan_queue: int, pending_files: int, pending_limit: Optional[int] = None
    ) -> None:
        label = self.query_one("#scan_dir_label", Static)
        safe_dir = current_dir or "(pending...)"
        queue_val = scan_queue if isinstance(scan_queue, int) else 0
        pending_val = pending_files if isinstance(pending_files, int) else 0
        limit_val = pending_limit if isinstance(pending_limit, int) and pending_limit > 0 else None
        window_text = (
            f"pending_window={pending_val}/{limit_val}"
            if limit_val is not None
            else f"pending_window={pending_val}"
        )
        label.update(f"Scan dir: {safe_dir} | scan_queue={queue_val} | {window_text}")

    def _handle_shard_event(self, payload: dict, done: bool) -> None:
        prefix = payload.get("prefix", "")
        parent = payload.get("parent", "")
        total = payload.get("total")
        remaining = payload.get("remaining")
        if remaining is None and isinstance(total, int):
            copied = payload.get("copied")
            if isinstance(copied, int):
                remaining = max(total - copied, 0)
        status_dir = parent or "."
        state_label = "done" if done else "active"
        self._update_scan_status(
            f"{status_dir} | shard={prefix or '-'} | {state_label}",
            scan_queue=0,
            pending_files=remaining if isinstance(remaining, int) else 0,
            pending_limit=total if isinstance(total, int) and total > 0 else None,
        )

    def _update_overall_bars(self) -> None:
        copy_bar = self.query_one("#copy_overall", ProgressBar)
        cloud_bar = self.query_one("#cloud_overall", ProgressBar)

        total = max(self.total_bytes, 1)

        copy_bar.update(total=total, progress=min(self.copied_bytes, total))
        cloud_bar.update(total=total, progress=min(self.freed_bytes, total))
        self._update_freed_labels()

    def _update_freed_labels(self) -> None:
        count_label = self.query_one("#freed_count_label", Static)
        latest_label = self.query_one("#latest_freed_label", Static)
        count_label.update(f"Freed files: {self.freed_files}")
        latest = self.last_freed or "-"
        latest_label.update(f"Latest freed: {latest}")

    def _update_local_usage(self) -> None:
        usage_bar = self.query_one("#local_usage", ProgressBar)
        usage_text = self.query_one("#local_usage_text", Static)

        total = max(self.max_local_bytes, 1)
        usage_bar.update(total=total, progress=min(self.local_bytes, total))
        percent = (self.local_bytes / total) * 100.0
        usage_text.update(f"{self._fmt_bytes(self.local_bytes)} / {self._fmt_bytes(total)} ({percent:4.1f}%)")

    def _update_current_file(self, rel: str, bytes_done: int, bytes_total: int) -> None:
        label = self.query_one("#current_file_label", Static)
        bar = self.query_one("#current_file_progress", ProgressBar)

        if not rel or bytes_total == 0:
            label.update("")
            bar.update(total=1, progress=0)
            return

        label.update(f"{rel} ({self._fmt_bytes(bytes_done)} / {self._fmt_bytes(bytes_total)})")
        bar.update(total=bytes_total, progress=bytes_done)

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        value = float(n)
        for unit in units:
            if value < 1024.0 or unit == units[-1]:
                return f"{value:0.1f} {unit}"
            value /= 1024.0
