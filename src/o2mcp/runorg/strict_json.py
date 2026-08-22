"""Duplicate-key-rejecting JSON and exact-type control-record helpers.

Execution receipts authorize scheduler retries, downstream launches, registry
rewrites, and destructive lifecycle operations.  Python's permissive JSON and
``str``/``int`` coercions are therefore inappropriate: every accepted wire byte
must have one unambiguous typed meaning across all readers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one object while rejecting repeated member names."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in JSON object: {key!r}")
        result[key] = value
    return result


def strict_json_value(text: str, label: str) -> Any:
    """Decode one RFC-style JSON value with duplicate/non-finite rejection."""

    if not isinstance(text, str):
        raise ValueError(f"{label} must be UTF-8 text")
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON number: {token}")),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        # Preserve the duplicate/non-finite reason for audit logs and parity
        # tests while still attaching the caller's control-record label.
        raise ValueError(f"{label} is malformed JSON: {exc}") from exc


def strict_json_object(text: str, label: str) -> dict[str, Any]:
    """Decode one JSON object and reject scalar/array top-level values."""

    value = strict_json_value(text, label)
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def exact_object(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    """Require an exact object field set without subclasses or extras."""

    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} has unsupported fields")
    return value


def exact_str(value: Any, label: str) -> str:
    """Require a JSON string without coercion."""

    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    return value


def exact_int(value: Any, label: str) -> int:
    """Require a JSON integer while excluding booleans and integral floats."""

    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def exact_bool(value: Any, label: str) -> bool:
    """Require a JSON boolean without accepting integers."""

    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def exact_list(value: Any, label: str) -> list[Any]:
    """Require a plain JSON array."""

    if type(value) is not list:
        raise ValueError(f"{label} must be an array")
    return value


__all__ = [
    "exact_bool",
    "exact_int",
    "exact_list",
    "exact_object",
    "exact_str",
    "strict_json_object",
    "strict_json_value",
]
