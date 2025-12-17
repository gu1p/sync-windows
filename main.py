"""
HD → Cloud migration helper.

Copies files from a (big) external drive into a cloud sync folder
in controlled batches, keeping local usage below a configured limit.

Features
--------
- Typer CLI
- Textual TUI with progress bars and live stats
- Concurrent scan + copy pipeline
- Chunked copy with per-file progress
- "Cloud/offload" progress (bytes freed from local disk)
- Local usage tracking via your sync_api tree
- Resumable via a SQLite inventory (idempotent)
"""

from __future__ import annotations

import json
import queue
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import typer

from old.config_factory import DerivationDetails, derive_migration_config
from old.engine import MigrationEngine
from old import models as migration_models
from old.ui import MigrationApp


cli = typer.Typer(
    help="External-drive to cloud sync migration with a minimal CLI; advanced overrides via SYNC_MIGRATION_OVERRIDES."
)


@cli.command()
def run(
    origin: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Root folder on the external HD.",
    ),
    dst: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Root folder of the cloud provider's sync folder.",
    ),
    max_local_gb: float = typer.Option(
        200.0,
        "--max-local-gb",
        help="Hard limit (GiB) for local storage used by the migration subtree.",
    ),
    subdir: str = typer.Option(
        "external_migration",
        "--subdir",
        help="Subfolder inside dst to receive migrated files.",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        "--print-config",
        help="Print derived config and exit without launching the UI/engine.",
    ),
) -> None:
    """
    Run the migration with a Textual TUI.

    CLI surface: origin, dst, and optional --max-local-gb / --subdir / --explain (--print-config alias).
    Advanced tuning is only available through SYNC_MIGRATION_OVERRIDES (JSON/TOML/YAML mapping to MigrationConfig).

    This will:
      - scan ORIGIN while copying
      - feed files into DST/subdir
      - keep local usage below the configured limit
      - free already-migrated files when needed
      - resume based on inventory SQLite DB
    """

    config, details = derive_migration_config(
        source_root=origin,
        sync_root=dst,
        migration_subdir=subdir,
        max_local_gb=max_local_gb,
    )

    summary_lines = _render_derivation_summary(config, details)

    _log_overrides(details)

    if explain:
        _print_derivation(config, details, summary_lines)
        raise typer.Exit(code=0)

    events: "queue.Queue[migration_models.MigrationEvent]" = queue.Queue(maxsize=config.event_queue_size)
    _enqueue_startup_summary(events, summary_lines, details)
    engine = MigrationEngine(config=config, events=events)

    app = MigrationApp(engine)
    app.run()


def _config_as_dict(config: migration_models.MigrationConfig) -> dict:
    return {f.name: getattr(config, f.name) for f in fields(migration_models.MigrationConfig)}


def _log_overrides(details: DerivationDetails) -> None:
    if details.applied_overrides:
        sources = ", ".join(details.override_sources) if details.override_sources else "overrides"
        typer.echo(f"Applied overrides from {sources}: {details.applied_overrides}")
    if details.override_errors:
        for err in details.override_errors:
            typer.echo(f"Override error: {err}", err=True)


def _print_derivation(config: migration_models.MigrationConfig, details: DerivationDetails, summary: Iterable[str]) -> None:
    payload = {
        "summary": list(summary),
        "config": _config_as_dict(config),
        "derivation": details.as_dict(),
    }
    typer.echo(json.dumps(payload, indent=2, default=str))


def _enqueue_startup_summary(
    events: "queue.Queue[migration_models.MigrationEvent]", summary: Iterable[str], details: DerivationDetails
) -> None:
    for line in summary:
        events.put(
            migration_models.MigrationEvent(
                type="log",
                payload={"level": "info", "message": line},
            )
        )

    if details.notes:
        for note in details.notes:
            events.put(
                migration_models.MigrationEvent(
                    type="log",
                    payload={"level": "info", "message": f"Derivation note: {note}"},
                )
            )

    if details.applied_overrides:
        applied = ", ".join(f"{k}={_fmt_override_value(v)}" for k, v in details.applied_overrides.items())
        source = f" from {', '.join(details.override_sources)}" if details.override_sources else ""
        events.put(
            migration_models.MigrationEvent(
                type="log",
                payload={"level": "info", "message": f"Overrides{source}: {applied}"},
            )
        )

    if details.override_errors:
        for err in details.override_errors:
            events.put(
                migration_models.MigrationEvent(
                    type="log",
                    payload={"level": "warning", "message": f"Override error: {err}"},
                )
            )


def _render_derivation_summary(config: migration_models.MigrationConfig, details: DerivationDetails) -> list[str]:
    summary = [
        f"Source: {config.source_root}",
        f"Destination: {config.migration_root}",
        f"Limits: max_local={_fmt_gib(config.max_local_bytes)}",
        f"Events: event_queue_size={config.event_queue_size}",
    ]
    if details.override_sources:
        summary.append(f"Overrides applied from: {', '.join(details.override_sources)}")
    return summary


def _fmt_gib(n: int) -> str:
    return f"{n / (1024 ** 3):0.1f} GiB"


def _fmt_override_value(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


if __name__ == "__main__":
    cli()
