from __future__ import annotations

import shutil
from pathlib import Path


def safe_remove_traininghub_path(path_value: str | Path, data_root: Path) -> list[str]:
    path = Path(path_value)
    root = data_root.resolve()
    target = path.resolve(strict=False)
    if target == root or not _is_relative_to(target, root):
        return []
    if not target.exists():
        return []
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
        _remove_empty_parents(target.parent, root)
    return [str(target)]


def _remove_empty_parents(path: Path, stop_at: Path) -> None:
    current = path
    while current != stop_at and _is_relative_to(current, stop_at):
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
