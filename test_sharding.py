import os
import pathlib
import tempfile
import types
import unittest
from unittest import mock

from engine import models, sharding
from engine.models import Node, NodeKind


class FakeNode(Node):
    """Minimal Node implementation for sharding tests."""

    __slots__ = ("_path", "_hash")

    def __init__(self, path: str, h: bytes):
        self._path = path
        self._hash = h

    def kind(self) -> NodeKind:  # pragma: no cover - not used by sharding logic
        return NodeKind.FILE

    def iterate_children(self, known_prefixes=None):
        return ()

    def parent(self):
        return None

    def __hash__(self):
        return hash(self._path)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.path() == other.path()
        return False

    def path(self) -> str:
        return self._path

    def copy(self, dst: pathlib.Path):  # pragma: no cover - unused here
        raise NotImplementedError

    def metadata(self):
        return None


def _patch_hash_fn(target_module, func):
    """Temporarily patch sharding.hash_fn and restore after."""
    original = target_module.hash_fn
    target_module.hash_fn = func
    return original


class ShardingDeterminismTests(unittest.TestCase):
    def test_deterministic_prefix_independent_of_iteration_order(self):
        """Same set of nodes yields same shard prefix even if iteration order differs."""
        nodes = [
            ("b", b"\x01\xff"),  # larger second byte
            ("c", b"\x01\x10"),  # smallest hash overall
            ("a", b"\x02\x00"),
        ]

        def run_with_order(order):
            def factory():
                for name in order:
                    h = dict(nodes)[name]
                    yield FakeNode(f"/root/{name}", h)

            original = _patch_hash_fn(sharding, lambda n: n._hash)
            try:
                return sharding.new_shard(factory, known_prefixes=set(), cap=2)
            finally:
                sharding.hash_fn = original

        shard1 = run_with_order(["a", "b", "c"])
        shard2 = run_with_order(["c", "b", "a"])

        self.assertIsNotNone(shard1)
        self.assertIsNotNone(shard2)
        self.assertEqual(shard1.prefix, shard2.prefix)
        self.assertEqual(shard1.prefix, b"\x01")
        self.assertCountEqual([n.path() for n in shard1], ["/root/b", "/root/c"])
        self.assertCountEqual([n.path() for n in shard2], ["/root/b", "/root/c"])

    def test_prefix_grows_until_under_cap(self):
        """Prefix expands beyond one byte when density would exceed cap."""
        nodes = [
            ("x0", b"\x01\x00"),
            ("x1", b"\x01\x01"),
            ("y", b"\x02\x00"),
        ]

        def factory():
            for name, h in nodes:
                yield FakeNode(f"/root/{name}", h)

        original = _patch_hash_fn(sharding, lambda n: n._hash)
        try:
            shard = sharding.new_shard(factory, known_prefixes=set(), cap=1)
        finally:
            sharding.hash_fn = original

        self.assertIsNotNone(shard)
        # With cap=1 and two hashes starting with 0x01, we refine to 0x0100.
        self.assertEqual(shard.prefix, b"\x01\x00")
        self.assertEqual(shard.shard_nodes_count, 1)
        self.assertEqual([n.path() for n in shard], ["/root/x0"])

    def test_skips_known_prefixes(self):
        """Already-copied prefixes are ignored and the next shard is chosen."""
        nodes = [
            ("a", b"\x00\x00"),
            ("b", b"\x01\x00"),
        ]

        def factory():
            for name, h in nodes:
                yield FakeNode(f"/root/{name}", h)

        original = _patch_hash_fn(sharding, lambda n: n._hash)
        try:
            shard = sharding.new_shard(factory, known_prefixes={b"\x00"}, cap=10)
        finally:
            sharding.hash_fn = original

        self.assertIsNotNone(shard)
        self.assertEqual(shard.prefix, b"\x01")
        self.assertEqual([n.path() for n in shard], ["/root/b"])


class ShardingScaleTests(unittest.TestCase):
    def test_partition_million_like_dataset_with_cap(self):
        """
        Simulate sharding ~1,000,000 files without touching the filesystem.
        Hashes are grouped so each prefix corresponds to ~cap items, exercising
        multiple passes per shard while keeping memory bounded.
        """
        total_nodes = 1_000_000
        cap = 100_000  # 10 shards expected
        group_size = 100_000

        def make_hash(i: int) -> bytes:
            group = i // group_size
            return bytes(
                (
                    group,  # coarse bucket drives prefix selection
                    (i >> 16) & 0xFF,
                    (i >> 8) & 0xFF,
                    i & 0xFF,
                )
            )

        def factory():
            for i in range(total_nodes):
                yield FakeNode(f"/root/file{i}", make_hash(i))

        original_hash = _patch_hash_fn(sharding, lambda n: n._hash)
        try:
            known = set()
            shards = []
            while True:
                shard = sharding.new_shard(factory, known_prefixes=known, cap=cap)
                if shard is None:
                    break
                shards.append(shard)
                known.add(shard.prefix)
        finally:
            sharding.hash_fn = original_hash

        self.assertEqual(len(shards), 10)
        self.assertEqual({bytes([i]) for i in range(10)}, {s.prefix for s in shards})
        self.assertTrue(all(s.shard_nodes_count <= cap for s in shards))
        self.assertEqual(
            total_nodes,
            sum(s.shard_nodes_count for s in shards),
            "All nodes should be covered exactly once",
        )


class TreeCopyTests(unittest.TestCase):
    def test_copy_to_with_mocked_scandir_and_copy(self):
        """
        Simulate copying a folder tree without touching the real filesystem.
        Verifies that files are copied to matching destination paths and shards
        follow deterministic prefixes.
        """
        src_root = tempfile.mkdtemp()
        dst_root = tempfile.mkdtemp()

        files = ["f1.txt", "f2.txt", "f3.txt"]
        for name in files:
            pathlib.Path(src_root, name).write_text("data")

        class FakeDirEntry:
            def __init__(self, name: str, is_dir=False, is_symlink=False, size=1):
                self.name = name
                self._is_dir = is_dir
                self._is_symlink = is_symlink
                self._size = size
                self.path = os.path.join(src_root, name)

            def is_dir(self):
                return self._is_dir

            def is_symlink(self):
                return self._is_symlink

            def stat(self):
                st = types.SimpleNamespace()
                st.st_size = self._size
                return st

        # Patch scandir to yield fake entries for the root folder only.
        def fake_scandir(path):
            self.assertEqual(os.path.realpath(path), os.path.realpath(src_root))
            for name in files:
                yield FakeDirEntry(name)

        copied = []
        replaced: list[str] = []
        orig_replace = pathlib.Path.replace

        def fake_copy(src, dst):
            copied.append((os.path.abspath(src), os.path.abspath(dst)))

        def fake_replace(self, target):
            target_abs = os.path.abspath(target)
            # Ignore internal progress tracker files.
            if "state.json" in target_abs:
                if pathlib.Path(self).exists():
                    return orig_replace(self, target)
                return pathlib.Path(target)
            if pathlib.Path(self).exists():
                orig_replace(self, target)
            replaced.append(target_abs)
            return pathlib.Path(target)

        with mock.patch("os.scandir", fake_scandir), mock.patch("shutil.copy", fake_copy), mock.patch.object(pathlib.Path, "replace", fake_replace):
            tree = models.Tree(pathlib.Path(src_root))
            tree.copy_to(dst=pathlib.Path(dst_root), progress_dir=pathlib.Path(tempfile.mkdtemp()))

        expected_dsts = {os.path.realpath(os.path.join(dst_root, name)) for name in files}
        replaced_filtered = {os.path.realpath(p) for p in replaced if "state.json" not in p}
        self.assertEqual(replaced_filtered, expected_dsts)
        self.assertEqual(len(replaced_filtered), len(files))

    def test_resume_skips_already_copied_shard_from_state(self):
        """
        Simulate a crash after the first shard is copied; the second run should
        skip the persisted prefix and only copy the remaining files.
        """
        src_root = tempfile.mkdtemp()
        dst_root = tempfile.mkdtemp()
        state_dir = pathlib.Path(tempfile.mkdtemp())
        files = ["f1.txt", "f2.txt", "f3.txt"]
        for name in files:
            pathlib.Path(src_root, name).write_text("data")

        class FakeDirEntry:
            def __init__(self, name: str, size: int = 1):
                self.name = name
                self.path = os.path.join(src_root, name)
                self._size = size

            def is_dir(self):
                return False

            def is_symlink(self):
                return False

            def stat(self):
                st = types.SimpleNamespace()
                st.st_size = self._size
                return st

        def fake_scandir(path):
            self.assertEqual(os.path.realpath(path), os.path.realpath(src_root))
            for name in files:
                yield FakeDirEntry(name)

        copies_phase1: list[tuple[str, str]] = []
        replaced_phase1: list[str] = []
        replaced_phase2: list[str] = []
        orig_replace = pathlib.Path.replace
        current_phase = {"name": "phase1"}

        original_cap = models._MAX_NUMBER_NODES
        original_hash = sharding.hash_fn
        models._MAX_NUMBER_NODES = 1  # force one node per shard
        sharding.hash_fn = lambda n: os.path.basename(n.path()).encode("utf-8")

        def fake_replace(self, target):
            target_abs = os.path.abspath(target)
            if "state.json" in target_abs:
                if pathlib.Path(self).exists():
                    return orig_replace(self, target)
                return pathlib.Path(target)
            dest_list = replaced_phase1 if current_phase["name"] == "phase1" else replaced_phase2
            if pathlib.Path(self).exists():
                orig_replace(self, target)
            dest_list.append(target_abs)
            return pathlib.Path(target)

        try:
            # Phase 1: copy only the first shard, then "crash" before finishing.
            with mock.patch("os.scandir", fake_scandir), mock.patch("shutil.copy", lambda src, dst: copies_phase1.append((src, dst))), mock.patch.object(pathlib.Path, "replace", fake_replace):
                tree = models.Tree(pathlib.Path(src_root))
                tracker = models.ProgressTracker(root=pathlib.Path(src_root), dst=pathlib.Path(dst_root), state_dir=state_dir)
                folder = tree._root
                shard_iter = folder.iterate_children(known_prefixes=tracker.ignored_prefixes(folder.path()))
                first_shard = next(shard_iter)
                tracker.start_active_shard(folder.path(), first_shard._prefix)
                tree._copy_shard(first_shard, pathlib.Path(dst_root), tracker, depth=0, copied_nodes=set())
                tracker.finish_active_shard(folder.path(), first_shard._prefix)
                # simulate crash: do not process remaining shards or mark folder completed

            # Phase 2: resume; persisted prefix should prevent re-copy of first shard.
            current_phase["name"] = "phase2"
            with mock.patch("os.scandir", fake_scandir), mock.patch("shutil.copy", lambda src, dst: copies_phase2.append((src, dst))), mock.patch.object(pathlib.Path, "replace", fake_replace):
                tree.copy_to(dst=pathlib.Path(dst_root), progress_dir=state_dir)
        finally:
            models._MAX_NUMBER_NODES = original_cap
            sharding.hash_fn = original_hash

        # First shard copied once; remaining files copied on resume.
        self.assertEqual(len(replaced_phase1), 1)
        self.assertEqual(len(replaced_phase2), len(files) - 1)
        phase1_files = {os.path.basename(dst).split(".tmp.", 1)[0] for dst in replaced_phase1}
        phase2_files = {os.path.basename(dst) for dst in replaced_phase2}
        all_copied = set(phase2_files) | set(phase1_files)
        self.assertEqual(all_copied, set(files))
        # No duplicate copies of the first shard's file during resume.
        self.assertTrue(phase1_files.isdisjoint(phase2_files))

    def test_resume_partial_shard_uses_copied_nodes(self):
        """
        Persist an in-progress shard with one file already copied; resume should
        skip that file, finish the shard, and proceed to remaining shards.
        """
        src_root = tempfile.mkdtemp()
        dst_root = tempfile.mkdtemp()
        state_dir = pathlib.Path(tempfile.mkdtemp())
        files = ["a1.txt", "a2.txt", "b1.txt"]
        for name in files:
            pathlib.Path(src_root, name).write_text("data")

        class FakeDirEntry:
            def __init__(self, name: str, size: int = 1):
                self.name = name
                self.path = os.path.join(src_root, name)
                self._size = size

            def is_dir(self):
                return False

            def is_symlink(self):
                return False

            def stat(self):
                st = types.SimpleNamespace()
                st.st_size = self._size
                return st

        def fake_scandir(path):
            self.assertEqual(os.path.realpath(path), os.path.realpath(src_root))
            for name in files:
                yield FakeDirEntry(name)

        copies: list[tuple[str, str]] = []
        replaced: list[str] = []
        orig_replace = pathlib.Path.replace

        original_cap = models._MAX_NUMBER_NODES
        original_hash = sharding.hash_fn
        models._MAX_NUMBER_NODES = 2  # first shard will hold two items
        sharding.hash_fn = lambda n: bytes([ord(os.path.basename(n.path())[0])])

        def fake_replace(self, target):
            target_abs = os.path.abspath(target)
            if "state.json" in target_abs:
                if pathlib.Path(self).exists():
                    return orig_replace(self, target)
                return pathlib.Path(target)
            if pathlib.Path(self).exists():
                orig_replace(self, target)
            replaced.append(target_abs)
            return pathlib.Path(target)

        try:
            with mock.patch("os.scandir", fake_scandir):
                tree = models.Tree(pathlib.Path(src_root))
                tracker = models.ProgressTracker(root=pathlib.Path(src_root), dst=pathlib.Path(dst_root), state_dir=state_dir)
                folder = tree._root
                shard_iter = folder.iterate_children(known_prefixes=tracker.ignored_prefixes(folder.path()))
                first_shard = next(shard_iter)
                tracker.start_active_shard(folder.path(), first_shard._prefix)
                tracker.record_copied_node(folder.path(), "a1.txt")

            with mock.patch("os.scandir", fake_scandir), mock.patch("shutil.copy", lambda src, dst: copies.append((src, dst))), mock.patch.object(pathlib.Path, "replace", fake_replace):
                tree.copy_to(dst=pathlib.Path(dst_root), progress_dir=state_dir)
        finally:
            models._MAX_NUMBER_NODES = original_cap
            sharding.hash_fn = original_hash

        copied_basenames = {os.path.basename(dst) for dst in replaced if "state.json" not in dst}
        self.assertEqual(copied_basenames, {"a2.txt", "b1.txt"})

    def test_deep_nested_tree_sharded_and_copied_completely(self):
        """
        Build a deep, mixed tree with enough files per folder to force multiple
        shards. Uses fake scandir/copy to avoid real IO while ensuring every
        file across nested folders is copied exactly once and to the right
        relative destination.
        """
        src_root = pathlib.Path(tempfile.mkdtemp())
        dst_root = pathlib.Path(tempfile.mkdtemp())

        # Structure: root has 8 files and 3 dirs; each dir has multiple files;
        # dirB also contains a nested dir with files.
        structure = {
            "": {
                "files": [f"root{i}.txt" for i in range(8)],
                "dirs": ["dirA", "dirB", "dirC"],
            },
            "dirA": {
                "files": [f"a{i}.dat" for i in range(6)],
                "dirs": [],
            },
            "dirB": {
                "files": [f"b{i}.bin" for i in range(7)],
                "dirs": ["deep"],
            },
            os.path.join("dirB", "deep"): {
                "files": [f"deep{i}.log" for i in range(5)],
                "dirs": [],
            },
            "dirC": {
                "files": [f"c{i}.cfg" for i in range(9)],
                "dirs": [],
            },
        }

        class FakeDirEntry:
            def __init__(self, base: pathlib.Path, name: str, is_dir: bool):
                self.name = name
                self.path = str(base / name)
                self._is_dir = is_dir

            def is_dir(self):
                return self._is_dir

            def is_symlink(self):
                return False

            def stat(self):
                st = types.SimpleNamespace()
                st.st_size = 1
                return st

        # Build lookup of absolute path -> entries
        tree_entries: dict[str, list[FakeDirEntry]] = {}
        for rel, payload in structure.items():
            base = (src_root / rel).resolve()
            base.mkdir(parents=True, exist_ok=True)
            entries: list[FakeDirEntry] = []
            for fname in payload["files"]:
                (base / fname).write_text("data")
                entries.append(FakeDirEntry(base, fname, is_dir=False))
            for dname in payload["dirs"]:
                entries.append(FakeDirEntry(base, dname, is_dir=True))
            tree_entries[str(base)] = entries

        def fake_scandir(path):
            key = os.path.realpath(path)
            for entry in tree_entries.get(key, []):
                yield entry

        copies: list[tuple[str, str]] = []
        replaced: list[str] = []

        original_cap = models._MAX_NUMBER_NODES
        models._MAX_NUMBER_NODES = 3  # force multiple shards per folder
        orig_replace = pathlib.Path.replace
        def fake_replace(self, target):
            target_abs = os.path.abspath(target)
            if "state.json" in target_abs:
                if pathlib.Path(self).exists():
                    return orig_replace(self, target)
                return pathlib.Path(target)
            if pathlib.Path(self).exists():
                orig_replace(self, target)
            replaced.append(target_abs)
            return pathlib.Path(target)
        try:
            with mock.patch("os.scandir", fake_scandir), mock.patch("shutil.copy", lambda src, dst: copies.append((src, dst))), mock.patch.object(pathlib.Path, "replace", fake_replace):
                tree = models.Tree(src_root)
                tree.copy_to(dst_root, progress_dir=pathlib.Path(tempfile.mkdtemp()))
        finally:
            models._MAX_NUMBER_NODES = original_cap

        # Verify every file was copied exactly once with correct relative path.
        expected_files = []
        for rel, payload in structure.items():
            for fname in payload["files"]:
                expected_files.append(os.path.join(rel, fname))

        copied_rel = []
        for dst in replaced:
            if "state.json" in dst:
                continue
            rel = pathlib.Path(dst).resolve().relative_to(dst_root.resolve())
            copied_rel.append(str(rel))

        self.assertEqual(set(copied_rel), set(expected_files))
        self.assertEqual(len(copied_rel), len(expected_files))

    def test_recursion_depth_guard_raises_on_excessive_depth(self):
        """
        Force a deep chain of directories to hit the depth guard and raise
        RecursionError instead of recursing indefinitely.
        """
        src_root = pathlib.Path(tempfile.mkdtemp())
        dst_root = pathlib.Path(tempfile.mkdtemp())

        depth_limit = 3
        chain = [f"level{i}" for i in range(depth_limit + 2)]  # ensure we exceed limit

        class FakeDirEntry:
            def __init__(self, base: pathlib.Path, name: str, is_dir: bool):
                self.name = name
                self.path = str(base / name)
                self._is_dir = is_dir

            def is_dir(self):
                return self._is_dir

            def is_symlink(self):
                return False

            def stat(self):
                st = types.SimpleNamespace()
                st.st_size = 1
                return st

        tree_entries: dict[str, list[FakeDirEntry]] = {}
        # Build a chain: root -> level0 -> level1 -> ... -> leaf (file)
        current = src_root.resolve()
        current.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(chain):
            next_path = current / name
            next_path.mkdir(parents=True, exist_ok=True)
            tree_entries[str(current)] = [FakeDirEntry(current, name, is_dir=True)]
            current = next_path.resolve()
        # leaf directory contains a file to stop traversal if depth were allowed
        (current / "leaf.txt").parent.mkdir(parents=True, exist_ok=True)
        (current / "leaf.txt").write_text("data")
        tree_entries[str(current)] = [FakeDirEntry(current, "leaf.txt", is_dir=False)]

        def fake_scandir(path):
            key = os.path.realpath(path)
            for entry in tree_entries.get(key, []):
                yield entry

        original_depth = models._MAX_DIRECTORY_DEPTH
        models._MAX_DIRECTORY_DEPTH = depth_limit
        try:
            with mock.patch("os.scandir", fake_scandir), mock.patch("shutil.copy", lambda src, dst: None):
                tree = models.Tree(src_root)
                with self.assertRaises(RecursionError):
                    tree.copy_to(dst_root, progress_dir=pathlib.Path(tempfile.mkdtemp()))
        finally:
            models._MAX_DIRECTORY_DEPTH = original_depth


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
