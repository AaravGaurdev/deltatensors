"""
Cache of verified hashes, so verify="cached" skips re-hashing unchanged files.

Entries live in ``$DELTATENSORS_CACHE_DIR`` (default ``~/.cache/deltatensors``),
one small JSON file per key. A key fingerprints every file it covers by
absolute path, size and mtime (ns), so editing, replacing or touching any shard
invalidates it. This trusts that a file whose size and mtime are unchanged has
unchanged content; use verify="full" where that assumption doesn't hold.
"""

from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Union

_SCHEMA = 1


def cache_dir() -> Path:
    env = os.environ.get("DELTATENSORS_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "deltatensors"


def _fingerprint(paths: Iterable[Union[str, Path]]) -> List[list]:
    fp = []
    for p in paths:
        p = Path(p).resolve()
        st = p.stat()
        fp.append([str(p), st.st_size, st.st_mtime_ns])
    return sorted(fp)


def make_key(kind: str, paths: Iterable[Union[str, Path]], **extra) -> str:
    """Key for a check of ``kind`` over ``paths``; ``extra`` must be JSON-able."""
    blob = json.dumps({"schema": _SCHEMA, "kind": kind, "files": _fingerprint(paths), **extra},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(key: str) -> Optional[str]:
    try:
        with open(cache_dir() / f"{key}.json", encoding="utf-8") as f:
            return json.load(f)["value"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def put(key: str, value: str) -> None:
    """Best effort: an unwritable cache only costs a re-hash next time."""
    try:
        d = cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"value": value}, f)
        os.replace(tmp, d / f"{key}.json")
    except OSError:
        pass


def keys_digest(names: Iterable[str]) -> str:
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
