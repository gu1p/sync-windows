from __future__ import annotations

import dataclasses
import hashlib
from typing import Callable, Iterator, Iterable, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.models import Node


def hash_bytes(data: bytes) -> bytes:
    return hashlib.blake2b(data).digest()


def hash_str(s: str) -> bytes:
    return hash_bytes(s.encode("utf-8"))


def hash_fn(node: "Node") -> bytes:
    return hash_str(str(node.path()))


@dataclasses.dataclass(slots=True, frozen=True)
class ShardView:
    prefix: bytes
    shard_nodes_count: int
    seen_nodes: int
    ignored_nodes: int
    _nodes: list["Node"]

    def __iter__(self) -> Iterator["Node"]:
        for i in range(self.shard_nodes_count):
            yield self._nodes[i]

    def __len__(self) -> int:
        return self.shard_nodes_count


def _starts_with_known_prefix(h: bytes, prefixes: Set[bytes]) -> bool:
    for p in prefixes:
        if h.startswith(p):
            return True
    return False


def _find_min_hash(
    nodes_iter_factory: Callable[[], Iterable["Node"]],
    known_prefixes: Set[bytes],
) -> Optional[bytes]:
    minimum: Optional[bytes] = None
    for node in nodes_iter_factory():
        h = hash_fn(node)
        if _starts_with_known_prefix(h, known_prefixes):
            continue
        if minimum is None or h < minimum:
            minimum = h
    return minimum


def _count_with_prefix(
    nodes_iter_factory: Callable[[], Iterable["Node"]],
    prefix: bytes,
    known_prefixes: Set[bytes],
) -> int:
    count = 0
    for node in nodes_iter_factory():
        h = hash_fn(node)
        if _starts_with_known_prefix(h, known_prefixes):
            continue
        if h.startswith(prefix):
            count += 1
    return count


def new_shard(
    nodes_iter_factory: Callable[[], Iterable["Node"]],
    known_prefixes: Set[bytes],
    cap: int,
) -> Optional[ShardView]:
    if b"" in known_prefixes:
        raise ValueError("prefixes must be non-empty")

    minimum_hash = _find_min_hash(nodes_iter_factory, known_prefixes)
    if minimum_hash is None:
        return None

    # Grow the prefix until the shard fits under the cap.
    prefix_len = 1
    while True:
        prefix = minimum_hash[:prefix_len]
        count = _count_with_prefix(nodes_iter_factory, prefix, known_prefixes)
        if count == 0:
            return None
        if count <= cap or prefix_len >= len(minimum_hash):
            break
        prefix_len += 1

    shard_nodes: list["Node"] = []
    seen = 0
    ignored = 0
    for node in nodes_iter_factory():
        seen += 1
        h = hash_fn(node)
        if _starts_with_known_prefix(h, known_prefixes):
            ignored += 1
            continue
        if h.startswith(prefix):
            shard_nodes.append(node)
        else:
            ignored += 1

    return ShardView(
        prefix=prefix,
        shard_nodes_count=len(shard_nodes),
        seen_nodes=seen,
        ignored_nodes=ignored,
        _nodes=shard_nodes,
    )
