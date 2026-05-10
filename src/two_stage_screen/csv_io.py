from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from pathlib import Path


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read CSV as list of dicts; preserve column order (handles multiline quoted fields)."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], []
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
        return fieldnames, rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extend_fieldnames(existing: list[str], new_cols: Iterable[str]) -> list[str]:
    seen = dict.fromkeys(existing)
    out = list(existing)
    for c in new_cols:
        if c not in seen:
            seen[c] = None
            out.append(c)
    return out


def stable_row_hash(title: str, abstract: str) -> str:
    h = hashlib.sha256(f"{title}\n|||{abstract}".encode("utf-8")).hexdigest()
    return f"h_{h[:16]}"


def get_record_id(row: dict[str, str], id_column: str, title_column: str, abstract_column: str) -> str:
    raw_id = (row.get(id_column) or "").strip()
    if raw_id:
        return raw_id
    return stable_row_hash(row.get(title_column, ""), row.get(abstract_column, ""))


def write_jsonl(path: Path, payload: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
