from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class ModelIntrospectionError(ValueError):
    pass


def transformer_layers(model: Any) -> Sequence[Any]:
    for path in (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("layers",),
    ):
        current = model
        for name in path:
            current = getattr(current, name, None)
            if current is None:
                break
        if current is not None and hasattr(current, "__len__") and hasattr(current, "__getitem__"):
            return current
    raise ModelIntrospectionError("Unable to locate transformer blocks for this architecture.")


def layer_count(model: Any) -> int:
    return len(transformer_layers(model))


def normalize_layer_targets(raw: str | list[int], total_layers: int) -> list[int]:
    if total_layers <= 0:
        return []
    if raw == "all":
        return list(range(total_layers))
    if raw == "last":
        return [total_layers - 1]
    if raw == "every-4":
        indexes = list(range(0, total_layers, 4))
        if indexes[-1] != total_layers - 1:
            indexes.append(total_layers - 1)
        return indexes
    if isinstance(raw, list):
        return sorted({index for index in raw if 0 <= index < total_layers})
    return list(range(total_layers))


def proportional_layer_pairs(source_layers: int, target_layers: int, source_targets: list[int]) -> list[list[int]]:
    if source_layers <= 0 or target_layers <= 0:
        return []
    pairs: list[list[int]] = []
    for source_index in source_targets:
        if source_layers == 1:
            target_index = target_layers - 1
        else:
            target_index = round(source_index * (target_layers - 1) / (source_layers - 1))
        pairs.append([source_index, min(max(target_index, 0), target_layers - 1)])
    return pairs
