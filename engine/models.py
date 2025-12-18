from __future__ import annotations

import abc
import dataclasses
import json
import os
import pathlib
import random
import shutil
import string
import time
import functools
import os
from enum import Enum
from typing import Optional, Iterable, Any, Callable

from engine.sharding import new_shard, hash_str
from engine.events import EventType

_CHARACTERS = string.ascii_letters + string.digits

def random_string(length: int) -> str:
    return ''.join(random.choice(_CHARACTERS) for _ in range(length))


_MAX_NUMBER_NODES = 50_000
_MAX_DIRECTORY_DEPTH = 1_000


def _retry_permission_errors(attempts: int = 5, base_delay: float = 0.2):
    """Decorator to retry on PermissionError with backoff (e.g., Windows file locks)."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Optional[PermissionError] = None
            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except PermissionError as exc:
                    last_exc = exc
                    if attempt == attempts - 1:
                        raise
                    time.sleep(base_delay * (attempt + 1))
            if last_exc:
                raise last_exc
            return None

        return wrapper
    return decorator

def _skip_missing_folder(fn):
    """Decorator to keep copying even if a folder vanishes or is inaccessible."""
    @functools.wraps(fn)
    def wrapper(self, folder: "Folder", dst_root: pathlib.Path, tracker: ProgressTracker, depth: int):
        try:
            return fn(self, folder, dst_root, tracker, depth)
        except (FileNotFoundError, NotADirectoryError) as exc:
            try:
                folder_path = folder.path()
            except Exception as e:
                folder_path = "(unknown)"


            self._emit(EventType.LOG, level="warning", message=f"Skipping missing/inaccessible folder {folder_path}: {exc}")
            tracker.mark_dir_completed(folder_path)
            return None
    return wrapper

def _fs_path(path: str) -> str:
    """Return a filesystem-safe path (adds Windows long-path prefix when needed)."""
    if os.name != "nt":
        return path
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path

def _strip_long_path_prefix(path: str) -> str:
    """Remove Windows long-path prefix so relative paths can be computed."""
    if os.name != "nt":
        return path
    unc_prefix = "\\\\?\\UNC\\"
    if path.startswith(unc_prefix):
        return "\\\\" + path[len(unc_prefix):]
    standard_prefix = "\\\\?\\"
    if path.startswith(standard_prefix):
        return path[len(standard_prefix):]
    return path

def _safe_scandir(path: str):
    """scandir that retries with long-path prefix on Windows and tolerates path-length errors."""
    candidates = [path]
    if os.name == "nt":
        lp = _fs_path(path)
        if lp != path:
            candidates.append(lp)

    last_exc = None
    for candidate in candidates:
        try:
            yield from os.scandir(candidate)
            return
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as exc:
            last_exc = exc
            continue

    if last_exc:
        raise last_exc

def _path_variants(path: str) -> list[pathlib.Path]:
    """
    Return the plain path first, then the long-path variant if it differs.
    Some providers (e.g., VMware shared folders) reject '\\\\?\\' prefixes,
    so we try both when needed.
    """
    plain = pathlib.Path(path)
    if os.name != "nt":
        return [plain]
    long_variant = pathlib.Path(_fs_path(path))
    if long_variant == plain:
        return [plain]
    return [plain, long_variant]

class NodeKind(Enum):
    FILE = "FILE"
    FOLDER = "FOLDER"
    # A subset of nodes
    SHARD = "SHARD"

class Status(Enum):
    DONE = "DONE"
    NOT_DONE = "NOT_DONE"


@dataclasses.dataclass(slots=True)
class _Metadata:
    depth: int = 0
    status: Status = Status.NOT_DONE


@dataclasses.dataclass(slots=True)
class _FolderState:
    prefixes: set[bytes] = dataclasses.field(default_factory=set)
    active_prefix: Optional[bytes] = None
    copied_nodes: set[str] = dataclasses.field(default_factory=set)
    status: Status = Status.NOT_DONE

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "_FolderState":
        raw_status = data.get("status", Status.NOT_DONE.value)
        try:
            status = Status(raw_status)
        except ValueError:
            status = Status.NOT_DONE

        prefixes: set[bytes] = set()
        for hex_prefix in data.get("copied_prefixes", []):
            if not isinstance(hex_prefix, str):
                continue
            try:
                prefixes.add(bytes.fromhex(hex_prefix))
            except ValueError:
                continue

        active_prefix = data.get("active_prefix")
        if isinstance(active_prefix, str):
            try:
                active_prefix = bytes.fromhex(active_prefix)
            except ValueError:
                active_prefix = None
        else:
            active_prefix = None

        nodes: set[str] = set()
        for node_name in data.get("copied_nodes", []):
            if node_name is None:
                continue
            nodes.add(str(node_name))

        if active_prefix is None:
            nodes.clear()

        return cls(prefixes=prefixes, active_prefix=active_prefix, copied_nodes=nodes, status=status)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "status": self.status.value,
            "copied_prefixes": sorted(prefix.hex() for prefix in self.prefixes),
        }
        if self.active_prefix:
            payload["active_prefix"] = self.active_prefix.hex()
            payload["copied_nodes"] = sorted(self.copied_nodes)
        return payload

class Node(abc.ABC):
    @abc.abstractmethod
    def kind(self) -> NodeKind: ...

    @abc.abstractmethod
    def iterate_children(self, known_prefixes: Optional[set[bytes]] = None) -> Iterable["Node"]: ...

    @abc.abstractmethod
    def parent(self) -> Optional[str]: ...

    @abc.abstractmethod
    def __hash__(self) -> int: ...

    @abc.abstractmethod
    def __eq__(self, other: "Node") -> bool: ...

    @abc.abstractmethod
    def path(self) -> str: ...

    @abc.abstractmethod
    def copy(self, dst: pathlib.Path): ...

    @abc.abstractmethod
    def metadata(self) -> Optional[_Metadata]: ...

class Sync:
    def __init__(self, origin: Node, dst: Node):
        self.origin = origin
        self.dst = dst

    def sync(self):
        pass

class ProgressTracker:
    """
    Tracks directories, per-folder ignored prefixes, and the in-progress shard
    (at most one prefix plus copied child names) in a single JSON file. Entries
    are keyed by relative directory paths; when a folder completes, its
    descendant entries are dropped to keep the file compact. State is rewritten
    atomically on each persist.
    """
    def __init__(self, root: pathlib.Path, dst: pathlib.Path, state_dir: Optional[pathlib.Path] = None):
        self._root = root.resolve()
        self._dst = dst.resolve()
        default_state_dir = state_dir if state_dir is not None else pathlib.Path.cwd() / ".sync_state"
        self._state_dir = default_state_dir.resolve()
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / "state.json"
        self._state_version = 2
        self._dirs: dict[str, _FolderState] = {}
        self._load()
        self._ensure_folder_entry("")

    def _load(self) -> None:
        if not self._state_file.exists():
            return

        try:
            with self._state_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        stored_root = data.get("root")
        if stored_root:
            try:
                stored_root_path = pathlib.Path(stored_root).resolve()
            except OSError:
                stored_root_path = None
            if stored_root_path and stored_root_path != self._root:
                # Ignore stale state from a different root.
                return

        dirs_data = data.get("dirs", {})
        if isinstance(dirs_data, dict):
            for rel, payload in dirs_data.items():
                if not isinstance(rel, str) or not isinstance(payload, dict):
                    continue
                self._dirs[rel] = _FolderState.from_json(payload)

    def _rel(self, path: str) -> str:
        target = self._root if not path else pathlib.Path(path).resolve()
        try:
            rel = os.path.relpath(target, self._root)
        except ValueError:
            # Different drives/mounts (Windows UNC vs local) — fall back to absolute.
            return str(target)
        if rel == ".":
            return ""
        return rel

    def _persist(self) -> None:
        dirs_json = {}
        for rel in sorted(self._dirs):
            dirs_json[rel] = self._dirs[rel].to_json()

        payload = {
            "version": self._state_version,
            "root": str(self._root),
            "dst": str(self._dst),
            "dirs": dirs_json,
        }
        tmp = self._state_file.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")
        try:
            self._replace_state_file(tmp)
        except PermissionError:
            try:
                tmp.unlink(missing_ok=True)  # cleanup best-effort
            except Exception:
                pass
            print(f"Warning: could not persist tracker state to {self._state_file}", flush=True)

    @_retry_permission_errors()
    def _replace_state_file(self, tmp: pathlib.Path) -> None:
        tmp.replace(self._state_file)

    def _ensure_folder_entry(self, path: str) -> _FolderState:
        rel = self._rel(path)
        state = self._dirs.get(rel)
        if state is None:
            state = _FolderState()
            self._dirs[rel] = state
        return state

    def _drop_descendants(self, rel: str) -> None:
        prefix = rel + os.sep if rel else ""
        to_delete = [entry for entry in self._dirs if entry.startswith(prefix) and entry != rel]
        for entry in to_delete:
            del self._dirs[entry]

    def ignored_prefixes(self, path: str) -> set[bytes]:
        return set(self._ensure_folder_entry(path).prefixes)

    def add_ignored_prefix(self, path: str, prefix: bytes) -> None:
        if not prefix:
            return
        state = self._ensure_folder_entry(path)
        if prefix in state.prefixes:
            return
        state.prefixes.add(prefix)
        self._persist()

    def start_active_shard(self, path: str, prefix: bytes) -> None:
        if not prefix:
            return
        state = self._ensure_folder_entry(path)
        state.active_prefix = prefix
        state.copied_nodes.clear()
        self._persist()

    def record_copied_node(self, path: str, node_name: str) -> None:
        if not node_name:
            return
        state = self._ensure_folder_entry(path)
        if state.active_prefix is None:
            return
        if node_name in state.copied_nodes:
            return
        state.copied_nodes.add(node_name)
        self._persist()

    def finish_active_shard(self, path: str, prefix: bytes) -> None:
        if not prefix:
            return
        state = self._ensure_folder_entry(path)
        if state.active_prefix and state.active_prefix != prefix:
            return
        state.prefixes.add(prefix)
        state.active_prefix = None
        state.copied_nodes.clear()
        self._persist()

    def clear_ignored_prefixes(self, path: str) -> None:
        state = self._ensure_folder_entry(path)
        if not state.prefixes:
            return
        state.prefixes.clear()
        self._persist()

    def active_shard(self, path: str) -> Optional[tuple[bytes, set[str]]]:
        state = self._ensure_folder_entry(path)
        if state.active_prefix is None:
            return None
        return state.active_prefix, set(state.copied_nodes)

    def is_dir_completed(self, path: str) -> bool:
        rel = self._rel(path)
        probe = rel
        while True:
            state = self._dirs.get(probe)
            if state and state.status == Status.DONE:
                return True
            if probe == "":
                break
            probe = os.path.dirname(probe)
        return False

    def mark_dir_completed(self, path: str) -> None:
        rel = self._rel(path)
        state = self._ensure_folder_entry(path)
        state.status = Status.DONE
        state.prefixes.clear()
        state.active_prefix = None
        state.copied_nodes.clear()
        self._drop_descendants(rel)
        self._persist()

class Tree:
    def __init__(
        self,
        path: pathlib.Path,
        *,
        event_cb: Optional[Callable[[EventType, dict], None]] = None,
    ):
        assert path.is_dir()
        resolved = path.resolve()
        self._root_path = resolved
        self._root = Folder(_path=str(resolved))

        self._event_cb = event_cb
        self._stop_flag = False

        self.total_bytes: int = 0
        self.total_files: int = 0
        self.copied_bytes: int = 0
        self.copied_files: int = 0

    # ------------------ event helpers ------------------

    def _emit(self, type_: EventType, **payload: Any) -> None:
        if not self._event_cb:
            return
        try:
            self._event_cb(type_, payload)
        except Exception:
            # Never block or propagate errors to the copier.
            pass

    def copy_to(self, dst: pathlib.Path, progress_dir: Optional[pathlib.Path] = None) -> None:
        self._stop_flag = False
        dst_root = dst.resolve()
        dst_root.mkdir(parents=True, exist_ok=True)
        tracker = ProgressTracker(root=self._root_path, dst=dst_root, state_dir=progress_dir)
        self._emit(EventType.PHASE, stage="copy_loop", message="Scanning + copying via shards")
        self._emit(EventType.SCAN_PROGRESS, files=0, bytes=0)
        self._copy_folder(self._root, dst_root, tracker, depth=0)
        self._emit(EventType.PHASE, stage="finished", message="Migration complete")
        self._emit(EventType.FINISHED, message="Migration complete.")

    @_skip_missing_folder
    def _copy_folder(self, folder: "Folder", dst_root: pathlib.Path, tracker: ProgressTracker, depth: int) -> None:
        if self._stop_flag:
            return
        if depth > _MAX_DIRECTORY_DEPTH:
            raise RecursionError("Maximum directory depth exceeded")

        if tracker.is_dir_completed(folder.path()):
            return

        folder_path = folder.path()
        active = tracker.active_shard(folder_path)
        active_prefix = None
        copied_nodes: set[str] = set()
        if active:
            active_prefix, copied_nodes = active

        for shard in folder.iterate_children(known_prefixes=tracker.ignored_prefixes(folder_path)):
            if self._stop_flag:
                break

            if active_prefix is not None and shard._prefix != active_prefix:
                continue

            if active_prefix is None:
                tracker.start_active_shard(folder_path, shard._prefix)
                active_prefix = shard._prefix
                copied_nodes = set()

            self._copy_shard(shard, dst_root, tracker, depth=depth, copied_nodes=copied_nodes)
            tracker.finish_active_shard(folder_path, shard._prefix)
            active_prefix = None
            copied_nodes = set()

        if active_prefix is None and not self._stop_flag:
            tracker.mark_dir_completed(folder_path)

    def _copy_shard(self, shard: "Shard", dst_root: pathlib.Path, tracker: ProgressTracker, depth: int, copied_nodes: set[str]) -> None:
        parent_path = shard.parent()
        shard_total = shard.size() if hasattr(shard, "size") else getattr(shard, "_size", 0) or 0
        processed_in_shard = len(copied_nodes)

        def emit_scan_status(remaining: int) -> None:
            current_dir = parent_path or str(self._root_path)
            try:
                current_dir = os.path.relpath(current_dir, self._root_path)
            except Exception:
                pass
            self._emit(
                EventType.SCAN_STATUS,
                current_dir=current_dir or ".",
                scan_queue=0,
                pending_files=max(remaining, 0),
                pending_limit=shard_total or None,
            )

        remaining = max(shard_total - processed_in_shard, 0)
        self._emit(
            EventType.SHARD_START,
            prefix=shard._prefix.hex(),
            parent=parent_path or "",
            total=shard_total,
            remaining=remaining,
        )
        emit_scan_status(remaining)

        for node in shard.iterate_children():
            if self._stop_flag:
                break
            if node.kind() == NodeKind.FILE:
                node: File
                name = os.path.basename(node.path())
                if name in copied_nodes:
                    processed_in_shard += 1
                    emit_scan_status(max(shard_total - processed_in_shard, 0))
                    continue
                self.total_files += 1
                try:
                    node_size = getattr(node, "_size", 0) or 0
                    self.total_bytes += int(node_size)
                except Exception:
                    node_size = 0
                self._emit(EventType.SCAN_PROGRESS, files=self.total_files, bytes=self.total_bytes)
                self._copy_file(node, dst_root, size_hint=node_size)
                copied_nodes.add(name)
                if parent_path:
                    tracker.record_copied_node(parent_path, name)
                processed_in_shard += 1
                emit_scan_status(max(shard_total - processed_in_shard, 0))
            elif node.kind() == NodeKind.FOLDER:
                node: Folder
                self._copy_folder(node, dst_root, tracker, depth=depth + 1)
                processed_in_shard += 1
                emit_scan_status(max(shard_total - processed_in_shard, 0))
            elif node.kind() == NodeKind.SHARD:
                node: Shard
                self._copy_shard(node, dst_root, tracker, depth=depth)
                processed_in_shard += 1
                emit_scan_status(max(shard_total - processed_in_shard, 0))
        shard.set_scan_status(Status.DONE)
        remaining = max(shard_total - processed_in_shard, 0)
        self._emit(
            EventType.SHARD_DONE,
            prefix=shard._prefix.hex(),
            parent=parent_path or "",
            total=shard_total,
            remaining=remaining,
        )
        emit_scan_status(remaining)

    def _copy_file(self, file_node: File, dst_root: pathlib.Path, size_hint: int = 0) -> None:
        if self._stop_flag:
            return
        try:
            from old import models as legacy_models  # Lazy import for path normalization
        except Exception:
            legacy_models = None

        src_path = pathlib.Path(file_node.path())
        src_for_rel = pathlib.Path(_strip_long_path_prefix(str(src_path)))
        root_for_rel = pathlib.Path(_strip_long_path_prefix(str(self._root_path)))
        try:
            rel = src_for_rel.relative_to(root_for_rel)
        except ValueError:
            rel = pathlib.Path(os.path.relpath(str(src_for_rel), str(root_for_rel)))
        if legacy_models:
            rel, _ = legacy_models.normalize_windows_path(rel)
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_path.with_name(dst_path.name + f".tmp.{random_string(8)}")
        size = size_hint or 0
        if size == 0:
            size_exc: Optional[Exception] = None
            for candidate in _path_variants(str(src_path)):
                try:
                    size = candidate.stat().st_size
                    size_exc = None
                    break
                except OSError as exc:
                    size_exc = exc
                    continue
            if size_exc:
                raise size_exc
        rel_key = rel.as_posix()

        src_variants = _path_variants(str(src_path))
        dst_variants = _path_variants(str(dst_path))
        tmp_variants = _path_variants(str(tmp_path))

        self._emit(EventType.COPY_START, rel=rel_key, size=size, src=str(src_path), dst=str(dst_path))
        progress_step = max(size // 10, 1)  # coarse progress; fine-grained left to old layer if needed
        min_progress_interval = 0.2

        def _copy_once(src_fs: pathlib.Path, dst_fs: pathlib.Path, tmp_fs: pathlib.Path) -> None:
            bytes_done = 0
            last_progress_emit = 0
            last_progress_time = time.time()

            dst_fs.parent.mkdir(parents=True, exist_ok=True)
            tmp_fs.parent.mkdir(parents=True, exist_ok=True)

            with src_fs.open("rb") as f_src, tmp_fs.open("wb") as f_dst:
                buffer = bytearray(1024 * 1024)
                view = memoryview(buffer)
                while True:
                    read = f_src.readinto(buffer)
                    if not read:
                        break
                    f_dst.write(view[:read])
                    bytes_done += read
                    now = time.time()
                    if (
                        bytes_done == size
                        or bytes_done - last_progress_emit >= progress_step
                        or (now - last_progress_time) >= min_progress_interval
                    ):
                        self._emit(EventType.COPY_PROGRESS, rel=rel_key, bytes_done=bytes_done, bytes_total=size)
                        last_progress_emit = bytes_done
                        last_progress_time = now

            try:
                shutil.copystat(src_fs, tmp_fs)
            except OSError:
                pass

            tmp_fs.replace(dst_fs)

        last_attempt: Optional[tuple[pathlib.Path, pathlib.Path, pathlib.Path]] = None
        last_exc: Optional[Exception] = None
        for src_fs in src_variants:
            for dst_fs in dst_variants:
                for tmp_fs in tmp_variants:
                    try:
                        _copy_once(src_fs, dst_fs, tmp_fs)
                        file_node.metadata().status = Status.DONE
                        self.copied_bytes += size
                        self.copied_files += 1
                        self._emit(EventType.COPY_DONE, rel=rel_key, bytes_total=size)
                        return
                    except Exception as exc:  # noqa: BLE001
                        last_attempt = (src_fs, dst_fs, tmp_fs)
                        last_exc = exc
                        try:
                            if tmp_fs.exists():
                                tmp_fs.unlink()
                        except Exception:
                            pass

        if last_exc:
            details = ""
            if last_attempt:
                details = f" (src={last_attempt[0]}, dst={last_attempt[1]}, tmp={last_attempt[2]})"
            self._emit(EventType.LOG, level="error", message=f"Error copying {rel_key}: {last_exc!r}{details}")


# avoid allocating new objects
_EMPTY_CHILDREN = ()

@dataclasses.dataclass(slots=True)
class File(Node):
    _path: str
    _size: int
    _parent: str
    _hash: bytes = dataclasses.field(init=False)
    _metadata: _Metadata = dataclasses.field(default_factory=_Metadata)

    def __post_init__(self):
        self._hash = hash_str(self._path)

    def path(self) -> str:
        return self._path

    def kind(self) -> NodeKind:
        return NodeKind.FILE

    def iterate_children(self, known_prefixes: Optional[set[bytes]] = None) -> Iterable[Node]:
        return _EMPTY_CHILDREN

    def __hash__(self) -> int:
        return int.from_bytes(self._hash, "big")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Node):
            return self._path == other.path()
        return NotImplemented

    def parent(self) -> Optional[str]:
        return self._parent

    def copy(self, dst: pathlib.Path):
        dst.mkdir(exist_ok=True, parents=True)
        src_variants = _path_variants(str(self._path))
        dst_variants = _path_variants(str(dst))
        last_exc: Optional[Exception] = None
        for src_fs in src_variants:
            for dst_fs in dst_variants:
                try:
                    shutil.copy(src_fs, dst_fs)
                    self._metadata.status = Status.DONE
                    return
                except Exception as exc:
                    last_exc = exc
                    continue
        if last_exc:
            raise last_exc

    def metadata(self) -> Optional[_Metadata]:
        return self._metadata

@dataclasses.dataclass(slots=True)
class Folder(Node):
    _path: str
    _parent: Optional[str] = None
    _metadata: _Metadata = dataclasses.field(default_factory=_Metadata)

    def path(self) -> str:
        return self._path

    def kind(self) -> NodeKind:
        return NodeKind.FOLDER

    def iterate_children(self, known_prefixes: Optional[set[bytes]] = None) -> Iterable["Node"]:
        def _list_path():
            self._metadata = _Metadata(depth=self._metadata.depth)
            for path in _safe_scandir(self._path):
                if path.is_symlink():
                    continue


                entry_path = os.path.join(self._path, path.name)
                if path.is_dir():
                    yield Folder(
                        _path=entry_path,
                        _parent=self._path,
                    )
                else:
                    yield File(
                        _path=entry_path,
                        _size=path.stat().st_size,
                        _parent=self._path
                    )
        seen_prefixes = set(known_prefixes or ())
        keep_scanning = True
        while keep_scanning:
            shard = new_shard(
                nodes_iter_factory=_list_path,
                known_prefixes=seen_prefixes,
                cap=_MAX_NUMBER_NODES,
            )
            if shard is None:
                break

            if shard.shard_nodes_count == 0 or shard.seen_nodes == 0:
                break

            if shard.ignored_nodes < shard.seen_nodes and shard.prefix:
                seen_prefixes.add(shard.prefix)
            else:
                keep_scanning = False

            yield Shard(
                _parent=self._path,
                _prefix=shard.prefix,
                _nodes=shard.__iter__(),
                _size=shard.shard_nodes_count,
            )

    def parent(self) -> Optional[str]: return self._parent

    def metadata(self) -> Optional[_Metadata]: return self._metadata

    def __hash__(self): return hash(self._path)

    def __eq__(self, other: Node) -> bool: return self._path == other.path()

    def copy(self, dst: pathlib.Path):
        pass

@dataclasses.dataclass(slots=True)
class Shard(Node):
    _parent: Optional[str]
    _nodes: Iterable[Node]
    _prefix: bytes
    _size: int = 0
    _path: str = dataclasses.field(init=False)
    _metadata: _Metadata = dataclasses.field(default_factory=_Metadata)

    def set_scan_status(self, status: Status):
        self._metadata.status = status

    def __post_init__(self):
        shard_name = self._prefix.hex()
        full = shard_name if self._parent is None else os.path.join(self._parent, shard_name)
        object.__setattr__(self, "_path", full)

    def __hash__(self) -> int:
        return int.from_bytes(self._prefix, "big", signed=False)

    def __eq__(self, other: "Node") -> bool:
        return self.path() == other.path()

    def path(self) -> str:
        return self._path

    def kind(self) -> NodeKind:
        return NodeKind.SHARD

    def size(self) -> int:
        return self._size

    def iterate_children(self, known_prefixes: Optional[set[bytes]] = None) -> Iterable["Node"]:
        yield from self._nodes

    def parent(self) -> Optional[str]:
        return self._parent

    def copy(self, dst: pathlib.Path):
        for node in self._nodes:
            node.copy(dst)
        self._metadata.status = Status.DONE

    def metadata(self) -> Optional[_Metadata]:
        return self._metadata
