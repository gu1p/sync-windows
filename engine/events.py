from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    LOG = "log"
    PHASE = "phase"
    SCAN_PROGRESS = "scan_progress"
    SCAN_STATUS = "scan_status"
    STATUS = "status"
    USAGE = "usage"
    COPY_START = "copy_start"
    COPY_PROGRESS = "copy_progress"
    COPY_DONE = "copy_done"
    FREE_START = "free_start"
    FREE_DONE = "free_done"
    FINISHED = "finished"
    SHARD_START = "shard_start"
    SHARD_DONE = "shard_done"

