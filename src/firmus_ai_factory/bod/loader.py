"""BoD file loader.

Accepts either JSON or YAML on disk and returns a validated ``BasisOfDesign``.

Design notes
------------

* YAML is optional - if PyYAML isn't installed we still handle JSON.
* We validate with Pydantic v2 which gives structured, path-qualified errors
  ("cooling.loops[0].return_temp_c: return_temp_c must exceed supply_temp_c")
  rather than the usual "invalid config".
* The JSON Schema is derivable via ``BasisOfDesign.model_json_schema()`` so
  ETAP / Ansys / OneDrive can validate against the same contract without
  importing Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from firmus_ai_factory.bod.schema import BasisOfDesign


PathLike = Union[str, Path]


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as fh:
        return fh.read()


def _parse(path: Path) -> Dict[str, Any]:
    """Parse a BoD file to a dict, dispatching on suffix."""
    suffix = path.suffix.lower()
    text = _read_text(path)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "PyYAML is required to load .yaml BoD files. "
                "Install it with `pip install pyyaml`, or convert to JSON."
            ) from exc
        loaded = yaml.safe_load(text)
    elif suffix == ".json":
        loaded = json.loads(text)
    else:
        raise ValueError(
            f"Unsupported BoD file extension {suffix!r} (expected .yaml/.yml/.json)"
        )

    if not isinstance(loaded, dict):
        raise ValueError(f"BoD root must be a mapping, got {type(loaded).__name__}")
    return loaded


def load_bod(path: PathLike) -> BasisOfDesign:
    """Load a BoD from a JSON or YAML file on disk.

    Raises
    ------
    pydantic.ValidationError
        If the document fails schema validation.  The error message contains
        the exact dotted path of every failure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BoD file not found: {p}")
    data = _parse(p)
    return BasisOfDesign.model_validate(data)


def load_bod_dict(data: Dict[str, Any]) -> BasisOfDesign:
    """Validate an already-parsed dict as a BoD.

    Useful when the BoD is generated in-memory (e.g. by the optimizer) rather
    than read from disk.
    """
    return BasisOfDesign.model_validate(data)


def dump_bod(bod: BasisOfDesign, path: PathLike, *, indent: int = 2) -> Path:
    """Serialise a BoD back to disk as JSON or YAML.

    The written file round-trips through ``load_bod`` cleanly.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    payload = bod.model_dump(mode="json")

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "PyYAML is required to write .yaml BoD files."
            ) from exc
        p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    elif suffix == ".json":
        p.write_text(json.dumps(payload, indent=indent, default=str), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported BoD file extension {suffix!r} (expected .yaml/.yml/.json)"
        )
    return p


def export_json_schema(path: PathLike) -> Path:
    """Write the canonical JSON Schema for the BoD to disk."""
    p = Path(path)
    schema = BasisOfDesign.model_json_schema()
    p.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return p
