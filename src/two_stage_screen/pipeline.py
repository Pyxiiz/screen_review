from __future__ import annotations

import json
from dataclasses import dataclass

from two_stage_screen import PROMPT_VERSION
from two_stage_screen.csv_io import extend_fieldnames, get_record_id, write_jsonl
from two_stage_screen.llm import LLMClient
from two_stage_screen.models import (
    ChunkMapResponse,
    ScreenConfig,
    Stage1ModelResponse,
    Stage2ModelResponse,
)
from two_stage_screen import prompts


@dataclass
class Stage1RowResult:
    record_id: str
    structured: Stage1ModelResponse | None
    error: str | None
    raw_json: str | None


@dataclass
class Stage2RowResult:
    record_id: str
    structured: Stage2ModelResponse | None
    error: str | None
    raw_json: str | None
    skipped_reason: str | None = None


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(cleaned[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def should_run_stage2(cfg: ScreenConfig, stage1_decision: str | None) -> tuple[bool, str | None]:
    if cfg.stage2_policy == "all":
        return True, None
    if stage1_decision is None:
        return False, "missing_stage1_decision"
    d = stage1_decision.strip().lower()
    if cfg.stage2_policy == "include_and_unclear":
        return d in {"include", "unclear"}, f"skipped_stage2_policy({cfg.stage2_policy})"
    if cfg.stage2_policy == "include_only":
        return d == "include", f"skipped_stage2_policy({cfg.stage2_policy})"
    return False, "unknown_policy"


def run_stage1_on_row(llm: LLMClient, cfg: ScreenConfig, row: dict[str, str]) -> Stage1RowResult:
    cmap = cfg.columns
    rid = get_record_id(row, cmap.id, cmap.title, cmap.abstract)
    title = row.get(cmap.title, "") or ""
    abstract = row.get(cmap.abstract, "") or ""
    try:
        sys_m = prompts.stage1_system_message()
        usr_m = prompts.stage1_user_message(cfg, title, abstract)
        parsed, raw = llm.complete_json_schema(system=sys_m, user=usr_m, schema_model=Stage1ModelResponse)
        return Stage1RowResult(record_id=rid, structured=parsed, error=None, raw_json=raw)
    except Exception as e:  # noqa: BLE001
        return Stage1RowResult(record_id=rid, structured=None, error=str(e), raw_json=None)


def run_stage2_on_row(llm: LLMClient, cfg: ScreenConfig, row: dict[str, str]) -> Stage2RowResult:
    cmap = cfg.columns
    rid = get_record_id(row, cmap.id, cmap.title, cmap.abstract)
    title = row.get(cmap.title, "") or ""
    abstract = row.get(cmap.abstract, "") or ""
    full_text = row.get(cmap.full_text, "") or ""

    if not full_text.strip():
        return Stage2RowResult(
            record_id=rid,
            structured=None,
            error="empty_full_text",
            raw_json=None,
            skipped_reason="empty_full_text",
        )

    chk = cfg.chunking
    try:
        if len(full_text) <= chk.max_chars_single_pass:
            sys_m = prompts.stage2_system_message()
            usr_m = prompts.stage2_user_message(cfg, title, abstract, full_text)
            parsed, raw = llm.complete_json_schema(system=sys_m, user=usr_m, schema_model=Stage2ModelResponse)
            return Stage2RowResult(record_id=rid, structured=parsed, error=None, raw_json=raw)
        chunks = _chunk_text(full_text, chk.chunk_size, chk.chunk_overlap)
        all_notes: list[str] = []
        for idx, excerpt in enumerate(chunks):
            sys_m = prompts.chunk_map_system_message()
            usr_m = prompts.chunk_map_user_message(cfg, idx, len(chunks), excerpt)
            chunk_parsed, _ = llm.complete_json_schema(
                system=sys_m,
                user=usr_m,
                schema_model=ChunkMapResponse,
            )
            all_notes.extend(chunk_parsed.notes)
        aggregated = "\n".join(f"- {n}" for n in all_notes) if all_notes else "(no eligibility-relevant notes extracted)"
        sys_r = prompts.chunked_reduce_system_message()
        usr_r = prompts.chunked_reduce_user_message(cfg, title, abstract, aggregated)
        parsed, raw = llm.complete_json_schema(system=sys_r, user=usr_r, schema_model=Stage2ModelResponse)
        return Stage2RowResult(record_id=rid, structured=parsed, error=None, raw_json=raw)
    except Exception as e:  # noqa: BLE001
        return Stage2RowResult(record_id=rid, structured=None, error=str(e), raw_json=None)


def append_stage1_audit(path, cfg: ScreenConfig, result: Stage1RowResult, title_len: int, abstract_len: int) -> None:
    payload = {
        "stage": 1,
        "prompt_version": PROMPT_VERSION,
        "model": cfg.llm.model,
        "record_id": result.record_id,
        "title_chars": title_len,
        "abstract_chars": abstract_len,
        "error": result.error,
        "response": json.loads(result.raw_json) if result.raw_json else None,
    }
    write_jsonl(path, payload)


def append_stage2_audit(path, cfg: ScreenConfig, result: Stage2RowResult, ft_len: int) -> None:
    payload = {
        "stage": 2,
        "prompt_version": PROMPT_VERSION,
        "model": cfg.llm.model,
        "record_id": result.record_id,
        "full_text_chars": ft_len,
        "skipped_reason": result.skipped_reason,
        "error": result.error,
        "response": json.loads(result.raw_json) if result.raw_json else None,
    }
    write_jsonl(path, payload)


def merge_stage1_columns(fieldnames: list[str], rows: list[dict[str, str]]) -> list[str]:
    extra = [
        "stage1_decision",
        "stage1_rationale",
        "stage1_triggered_criteria_json",
        "stage1_error",
        "screen_model",
        "prompt_version",
    ]
    return extend_fieldnames(fieldnames, extra)


def apply_stage1_to_row(cfg: ScreenConfig, row: dict[str, str], result: Stage1RowResult) -> dict[str, str]:
    row = dict(row)
    row["prompt_version"] = PROMPT_VERSION
    row["screen_model"] = cfg.llm.model
    if result.structured:
        row["stage1_decision"] = result.structured.decision
        row["stage1_rationale"] = result.structured.rationale
        row["stage1_triggered_criteria_json"] = json.dumps(result.structured.triggered_criteria)
        row["stage1_error"] = ""
    else:
        row["stage1_decision"] = ""
        row["stage1_rationale"] = ""
        row["stage1_triggered_criteria_json"] = ""
        row["stage1_error"] = result.error or "unknown_error"
    return row


def merge_stage2_columns(fieldnames: list[str]) -> list[str]:
    extra = [
        "stage2_decision",
        "stage2_rationale",
        "stage2_supporting_excerpts_json",
        "stage2_error",
        "stage2_skipped_reason",
    ]
    return extend_fieldnames(fieldnames, extra)


def apply_stage2_to_row(cfg: ScreenConfig, row: dict[str, str], result: Stage2RowResult) -> dict[str, str]:
    row = dict(row)
    row["stage2_skipped_reason"] = result.skipped_reason or ""
    if result.skipped_reason:
        row.setdefault("stage2_decision", "")
        row.setdefault("stage2_rationale", "")
        row.setdefault("stage2_supporting_excerpts_json", "")
        row.setdefault("stage2_error", "")
    if result.error and not result.skipped_reason:
        row["stage2_decision"] = ""
        row["stage2_rationale"] = ""
        row["stage2_supporting_excerpts_json"] = ""
        row["stage2_error"] = result.error
        return row
    if result.skipped_reason:
        row["stage2_error"] = result.error or ""
        return row
    if result.structured:
        row["stage2_decision"] = result.structured.decision
        row["stage2_rationale"] = result.structured.rationale
        row["stage2_supporting_excerpts_json"] = json.dumps(result.structured.supporting_excerpts)
        row["stage2_error"] = ""
    return row


def carry_forward_exclude_stage2(cfg: ScreenConfig, row: dict[str, str]) -> dict[str, str]:
    """When skipping stage 2 due to policy, propagate stage 1 if appropriate."""
    row = dict(row)
    s1 = (row.get("stage1_decision") or "").strip().lower()
    reason = row.get("stage2_skipped_reason") or ""
    if "skipped_stage2_policy" in reason and s1:
        row["stage2_decision"] = s1
        row["stage2_rationale"] = "Carried forward from stage 1; full-text LLM skipped by policy."
        row["stage2_supporting_excerpts_json"] = json.dumps([])
        row["stage2_error"] = ""
    return row
