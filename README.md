# Sync migration CLI

Tooling to move data from an external drive into a cloud sync root while keeping local usage under a cap. Runs a Textual TUI on top of a Typer CLI.

## Commands
- `run`: main flow. Positional `origin` (external/source root) and `dst` (provider sync root). Optional `--max-local-gb` (default 200 GiB), `--subdir` (default `external_migration`), and `--explain/--print-config` to dump the derived config and exit. No other public flags.
- `bench`: synthetic load/perf harness; unchanged from earlier revisions.

## Quick start
- Install deps (`typer`, `textual`, platform extras) in your environment, then run:
  - `python -m main run /Volumes/External /path/to/CloudSync --max-local-gb 200 --subdir external_migration`
- See the derived plan without running the engine:
  - `python -m main run /Volumes/External /path/to/CloudSync --max-local-gb 150 --explain`

## Overrides (escape hatch)
- Advanced tuning is hidden from `--help` and only available via the `SYNC_MIGRATION_OVERRIDES` env var.
- The env var can point to a JSON/TOML/YAML file or contain inline JSON. Only a few inputs are still honored by the new engine: `max_local_gb`, `max_local_bytes` (takes precedence over `max_local_gb`), `migration_subdir`, and `event_queue_size` (UI/engine queue size).
- Examples:
  - Inline JSON: `export SYNC_MIGRATION_OVERRIDES='{"max_local_gb":250,"event_queue_size":10000}'`
  - File-based: `SYNC_MIGRATION_OVERRIDES=/path/to/overrides.toml` with:
    ```
    max_local_bytes = 175000000000
    migration_subdir = "external_migration"
    event_queue_size = 0  # unbounded queue
    ```
- Overrides are logged at startup and surfaced in `--explain` output (including parse errors), so you can verify what took effect.

## Deprecated surface
- Previous per-flag tuning options (headroom, pending window, manifest path, etc.) no longer map to the new engine; only the limited overrides above are respected now.
