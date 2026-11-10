"""The damage taxonomy supported by the baseline.

The six model classes follow the CarDD category list. ``other_uncertain`` is a
*reviewer-only* label: a model checkpoint must never emit it, and the service
refuses to load a checkpoint whose class names are not exactly the six model
classes below.
"""

from __future__ import annotations

import re

#: Classes a checkpoint is allowed (and required) to expose.
MODEL_DAMAGE_TYPES: tuple[str, ...] = (
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "lamp_broken",
    "tire_flat",
)

#: Extra label a human reviewer may choose. Never produced by a model.
REVIEW_ONLY_DAMAGE_TYPE = "other_uncertain"

#: Everything a finding may be labelled with after human review.
REVIEW_DAMAGE_TYPES: tuple[str, ...] = (*MODEL_DAMAGE_TYPES, REVIEW_ONLY_DAMAGE_TYPE)

_SEPARATORS = re.compile(r"[\s\-_]+")


def normalize_class_name(name: str) -> str:
    """Canonical form of a class name: lower case, runs of space/hyphen/underscore -> ``_``.

    ``"Glass Shatter"``, ``"glass-shatter"`` and ``"glass_shatter"`` all become
    ``"glass_shatter"``. Nothing else is rewritten: an unknown name stays unknown.
    """
    return _SEPARATORS.sub("_", str(name).strip().lower()).strip("_")


def class_mapping_problem(names: list[str] | tuple[str, ...]) -> str | None:
    """Return a human-readable problem if ``names`` is not exactly the supported taxonomy.

    The comparison is by *name*, not by index, so a checkpoint may order its
    classes differently. Returns ``None`` when the mapping is acceptable.
    """
    normalized = [normalize_class_name(n) for n in names]
    if len(set(normalized)) != len(normalized):
        return f"checkpoint has duplicate class names after normalization: {normalized}"
    expected, actual = set(MODEL_DAMAGE_TYPES), set(normalized)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        shown = unexpected[:8] + (["..."] if len(unexpected) > 8 else [])
        return (
            f"checkpoint classes do not match the supported taxonomy "
            f"(missing={missing}, unexpected={shown}, total={len(actual)})"
        )
    return None
