from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx
from two_stage_screen.config_loader import load_screen_config
from two_stage_screen.env_loader import load_project_dotenv
from two_stage_screen.csv_io import extend_fieldnames, get_record_id, read_csv_rows, write_csv_rows
from two_stage_screen.api_keys import resolve_ncbi_api_key
from two_stage_screen.pubmed_enrich import enrich_row
from two_stage_screen.eval_metrics import compare_gold_predictions
from two_stage_screen.llm import LLMClient
from two_stage_screen.pipeline import (
    Stage2RowResult,
    apply_stage1_to_row,
    apply_stage2_to_row,
    append_stage1_audit,
    append_stage2_audit,
    carry_forward_exclude_stage2,
    merge_stage1_columns,
    merge_stage2_columns,
    run_stage1_on_row,
    run_stage2_on_row,
    should_run_stage2,
)
from two_stage_screen.models import ScreenConfig
from two_stage_screen.quick_screen import build_keyword_screen_config, parse_comma_separated_keywords


def _truncate(path: Path) -> None:
    if path.exists():
        path.unlink()


def cmd_stage1(args: argparse.Namespace) -> int:
    cfg = load_screen_config(Path(args.config))
    in_path = Path(args.input)
    jsonl_path = Path(args.jsonl_out)
    csv_out = Path(args.csv_out)
    if not args.append:
        _truncate(jsonl_path)
    fieldnames, rows = read_csv_rows(in_path)
    if not fieldnames:
        print("ERROR: CSV has no header row.", file=sys.stderr)
        return 2

    cmap = cfg.columns
    llm = LLMClient(cfg.llm, api_key=getattr(args, "openai_api_key", None))
    out_rows: list[dict[str, str]] = []

    lim = args.limit if args.limit is not None else len(rows)
    for idx, row in enumerate(rows[:lim]):
        title = row.get(cmap.title, "") or ""
        abstract = row.get(cmap.abstract, "") or ""
        rid = get_record_id(row, cmap.id, cmap.title, cmap.abstract)
        res = run_stage1_on_row(llm, cfg, row)
        append_stage1_audit(jsonl_path, cfg, res, len(title), len(abstract))
        merged = apply_stage1_to_row(cfg, row, res)
        out_rows.append(merged)
        d = res.structured.decision if res.structured else "ERROR"
        print(f"[stage1] {idx + 1}/{lim} id={rid} decision={d}")

    merged_fields = merge_stage1_columns(fieldnames, out_rows)
    write_csv_rows(csv_out, merged_fields, _pad_rows(out_rows, merged_fields))
    print(f"Wrote {csv_out} ({len(out_rows)} rows). Audit log: {jsonl_path}")
    return 0


def _pad_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict[str, str]]:
    return [{k: r.get(k, "") for k in fieldnames} for r in rows]


def cmd_stage2(args: argparse.Namespace) -> int:
    cfg = load_screen_config(Path(args.config))
    in_path = Path(args.input)
    jsonl_path = Path(args.jsonl_out)
    csv_out = Path(args.csv_out)
    if not args.append:
        _truncate(jsonl_path)
    fieldnames, rows = read_csv_rows(in_path)
    if not fieldnames:
        print("ERROR: CSV has no header row.", file=sys.stderr)
        return 2

    cmap = cfg.columns
    llm = LLMClient(cfg.llm, api_key=getattr(args, "openai_api_key", None))
    out_rows: list[dict[str, str]] = []

    lim = args.limit if args.limit is not None else len(rows)
    for idx, row in enumerate(rows[:lim]):
        rid = get_record_id(row, cmap.id, cmap.title, cmap.abstract)
        s1 = row.get("stage1_decision")
        ft = row.get(cmap.full_text, "") or ""

        ok, skip_reason = should_run_stage2(cfg, s1)
        if ok:
            res = run_stage2_on_row(llm, cfg, row)
            append_stage2_audit(jsonl_path, cfg, res, len(ft))
            merged_row = apply_stage2_to_row(cfg, row, res)
        else:
            res = Stage2RowResult(
                record_id=rid,
                structured=None,
                error=None,
                raw_json=None,
                skipped_reason=skip_reason,
            )
            append_stage2_audit(jsonl_path, cfg, res, len(ft))
            merged_row = apply_stage2_to_row(cfg, row, res)
            merged_row = carry_forward_exclude_stage2(cfg, merged_row)

        out_rows.append(merged_row)
        outcome = merged_row.get("stage2_decision") or merged_row.get("stage2_skipped_reason") or "?"
        print(f"[stage2] {idx + 1}/{lim} id={rid} outcome={outcome}")

    canonical_fields = list(
        dict.fromkeys(extend_fieldnames(merge_stage2_columns(fieldnames), [k for r in out_rows for k in r])),
    )
    padded = _pad_rows(out_rows, canonical_fields)
    write_csv_rows(csv_out, canonical_fields, padded)
    print(f"Wrote {csv_out} ({len(out_rows)} rows). Audit log: {jsonl_path}")
    return 0


def _execute_two_stage_run(
    cfg: ScreenConfig,
    args: argparse.Namespace,
) -> int:
    """Stage 1 + stage 2 in memory; writes csv_out and jsonl_out from args."""
    fieldnames, rows = read_csv_rows(Path(args.input))
    if not fieldnames:
        print("ERROR: CSV has no header row.", file=sys.stderr)
        return 2

    cmap = cfg.columns
    llm = LLMClient(cfg.llm, api_key=getattr(args, "openai_api_key", None))

    jsonl_stage1_arg = getattr(args, "jsonl_stage1", None)
    if jsonl_stage1_arg:
        stage1_jsonl = Path(jsonl_stage1_arg)
        keep_stage1_log = True
    else:
        fd, tmp_s1 = tempfile.mkstemp(prefix="audit_s1_", suffix=".jsonl")
        os.close(fd)
        stage1_jsonl = Path(tmp_s1)
        keep_stage1_log = False

    stage2_jsonl = Path(args.jsonl_out)

    _truncate(stage1_jsonl)
    _truncate(stage2_jsonl)

    lim = args.limit if args.limit is not None else len(rows)
    slice_rows = rows[:lim]

    s1_rows: list[dict[str, str]] = []
    for idx, row in enumerate(slice_rows):
        title = row.get(cmap.title, "") or ""
        abstract_row = row.get(cmap.abstract, "") or ""
        rid = get_record_id(row, cmap.id, cmap.title, cmap.abstract)
        res = run_stage1_on_row(llm, cfg, row)
        append_stage1_audit(stage1_jsonl, cfg, res, len(title), len(abstract_row))
        merged = apply_stage1_to_row(cfg, row, res)
        s1_rows.append(merged)
        d = res.structured.decision if res.structured else "ERROR"
        print(f"[run/stage1] {idx + 1}/{lim} id={rid} decision={d}")

    s2_out: list[dict[str, str]] = []
    for idx, row in enumerate(s1_rows):
        rid = get_record_id(row, cmap.id, cmap.title, cmap.abstract)
        s1 = row.get("stage1_decision")
        ft = row.get(cmap.full_text, "") or ""

        ok, skip_reason = should_run_stage2(cfg, s1)
        if ok:
            res = run_stage2_on_row(llm, cfg, row)
            append_stage2_audit(stage2_jsonl, cfg, res, len(ft))
            merged_row = apply_stage2_to_row(cfg, row, res)
        else:
            res = Stage2RowResult(
                record_id=rid,
                structured=None,
                error=None,
                raw_json=None,
                skipped_reason=skip_reason,
            )
            append_stage2_audit(stage2_jsonl, cfg, res, len(ft))
            merged_row = apply_stage2_to_row(cfg, row, res)
            merged_row = carry_forward_exclude_stage2(cfg, merged_row)

        s2_out.append(merged_row)
        outcome = merged_row.get("stage2_decision") or merged_row.get("stage2_skipped_reason") or "?"
        print(f"[run/stage2] {idx + 1}/{lim} id={rid} outcome={outcome}")

    merged_fields_s1 = merge_stage1_columns(fieldnames, s1_rows)
    final_fields = merge_stage2_columns(merged_fields_s1)
    canonical_fields = list(dict.fromkeys(extend_fieldnames(final_fields, [k for r in s2_out for k in r])))
    write_csv_rows(Path(args.csv_out), canonical_fields, _pad_rows(s2_out, canonical_fields))

    if not keep_stage1_log:
        try:
            stage1_jsonl.unlink(missing_ok=True)
        except OSError:
            pass
    print(f"Wrote {args.csv_out} ({len(s2_out)} rows). Stage2 audit: {stage2_jsonl}")
    if keep_stage1_log:
        print(f"Stage1 audit: {stage1_jsonl}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_screen_config(Path(args.config))
    return _execute_two_stage_run(cfg, args)


def cmd_screen(args: argparse.Namespace) -> int:
    """Screen CSV using comma-separated include/exclude keywords (minimal LLM interface)."""
    inc = parse_comma_separated_keywords(args.include)
    exc = parse_comma_separated_keywords(args.exclude)
    if not inc and not exc:
        print(
            "ERROR: provide at least one keyword via --include and/or --exclude (comma-separated).",
            file=sys.stderr,
        )
        return 2
    base: ScreenConfig | None = None
    if getattr(args, "config", None):
        base = load_screen_config(Path(args.config))
    cfg = build_keyword_screen_config(inc, exc, base=base)
    return _execute_two_stage_run(cfg, args)


def cmd_enrich_pubmed(args: argparse.Namespace) -> int:
    api_key = resolve_ncbi_api_key(env_var_name=args.api_key_env, override=args.ncbi_api_key)
    in_path = Path(args.input)
    csv_out = Path(args.csv_out)
    fieldnames, rows = read_csv_rows(in_path)
    if not fieldnames:
        print("ERROR: CSV has no header row.", file=sys.stderr)
        return 2

    pmid_c = args.pmid_column
    title_c = args.title_column
    pmcid_c = args.pmcid_column
    missing = [c for c in (pmid_c, title_c) if c not in fieldnames]
    if missing:
        print(f"ERROR: CSV missing required column(s): {', '.join(missing)}", file=sys.stderr)
        return 2
    has_pmcid_col = pmcid_c in fieldnames

    extra_cols = [
        "id",
        "title",
        "abstract",
        "full_text",
        "enrich_abstract_ok",
        "enrich_fulltext_ok",
        "enrich_message",
    ]
    out_rows: list[dict[str, str]] = []
    lim = args.limit if args.limit is not None else len(rows)

    with httpx.Client() as client:
        for idx, row in enumerate(rows[:lim]):
            pmid = (row.get(pmid_c) or "").strip()
            title = row.get(title_c) or ""
            if has_pmcid_col:
                pmcid = (row.get(pmcid_c) or "").strip() or None
            else:
                pmcid = None
            try:
                enriched = enrich_row(client, pmid, title, pmcid, api_key)
            except httpx.HTTPError as e:
                merged = dict(row)
                merged.update(
                    {
                        "id": pmid,
                        "title": title,
                        "abstract": "",
                        "full_text": "",
                        "enrich_abstract_ok": "0",
                        "enrich_fulltext_ok": "0",
                        "enrich_message": f"abstract:http_error:{e}; fulltext:not_attempted",
                    },
                )
                out_rows.append(merged)
                print(f"[enrich-pubmed] {idx + 1}/{lim} id={pmid or '?'} ERROR")
                continue
            merged = dict(row)
            merged.update(enriched)
            out_rows.append(merged)
            a_ok = enriched.get("enrich_abstract_ok")
            f_ok = enriched.get("enrich_fulltext_ok")
            print(f"[enrich-pubmed] {idx + 1}/{lim} id={pmid} abstract_ok={a_ok} fulltext_ok={f_ok}")

    base_extended = extend_fieldnames(fieldnames, extra_cols)
    canonical_fields = list(dict.fromkeys(extend_fieldnames(base_extended, [k for r in out_rows for k in r])))
    write_csv_rows(csv_out, canonical_fields, _pad_rows(out_rows, canonical_fields))
    print(f"Wrote {csv_out} ({len(out_rows)} rows).")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    rep = compare_gold_predictions(
        Path(args.gold),
        Path(args.predictions),
        id_column=args.id_column,
        gold_column=args.gold_column,
        pred_column=args.pred_column,
    )
    print(json.dumps(rep, indent=2))
    return 0


def _add_openai_api_key_arg(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--openai-api-key",
        default=None,
        dest="openai_api_key",
        help="API key for this run; overrides the env var from config llm.api_key_env (default OPENAI_API_KEY).",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Two-stage LLM-assisted eligibility screening from CSV.")
    sub = p.add_subparsers(dest="command", required=True)

    s1 = sub.add_parser("stage1", help="Screen using title and abstract only.")
    s1.add_argument("--config", required=True, help="Path to YAML config (see config.example.yaml).")
    s1.add_argument("--input", required=True)
    s1.add_argument("--jsonl-out", required=True)
    s1.add_argument("--csv-out", required=True)
    s1.add_argument("--append", action="store_true")
    s1.add_argument("--limit", type=int, default=None)
    _add_openai_api_key_arg(s1)
    s1.set_defaults(func=cmd_stage1)

    s2 = sub.add_parser("stage2", help="Screen using full text (typically after stage1).")
    s2.add_argument("--config", required=True, help="Path to YAML config (see config.example.yaml).")
    s2.add_argument("--input", required=True)
    s2.add_argument("--jsonl-out", required=True)
    s2.add_argument("--csv-out", required=True)
    s2.add_argument("--append", action="store_true")
    s2.add_argument("--limit", type=int, default=None)
    _add_openai_api_key_arg(s2)
    s2.set_defaults(func=cmd_stage2)

    r = sub.add_parser("run", help="Run stage1 then stage2 in one invocation (in-memory).")
    r.add_argument("--config", required=True, help="Path to YAML config (see config.example.yaml).")
    r.add_argument("--input", required=True)
    r.add_argument("--csv-out", required=True)
    r.add_argument("--jsonl-out", required=True, help="Stage 2 audit JSONL path.")
    r.add_argument("--jsonl-stage1", default=None, help="Optional Stage 1 audit path (omit to discard temp file).")
    r.add_argument("--limit", type=int, default=None)
    _add_openai_api_key_arg(r)
    r.set_defaults(func=cmd_run)

    scr = sub.add_parser(
        "screen",
        help="Screen a CSV with comma-separated --include / --exclude keywords (full YAML optional).",
    )
    scr.add_argument("--input", required=True, help="Input CSV (id, title, abstract; optional full_text).")
    scr.add_argument(
        "--include",
        default="",
        metavar="KEYWORDS",
        help='Inclusion keywords, comma-separated (e.g. "rain,soil,water").',
    )
    scr.add_argument(
        "--exclude",
        default="",
        metavar="KEYWORDS",
        help='Exclusion keywords, comma-separated (e.g. "cows,forests").',
    )
    scr.add_argument("--csv-out", required=True)
    scr.add_argument("--jsonl-out", required=True, help="Stage 2 audit JSONL path.")
    scr.add_argument(
        "--config",
        default=None,
        help="Optional YAML: reuse llm, chunking, columns, stage2_policy; rubric still from --include/--exclude.",
    )
    scr.add_argument("--jsonl-stage1", default=None, help="Optional Stage 1 audit JSONL path.")
    scr.add_argument("--limit", type=int, default=None)
    _add_openai_api_key_arg(scr)
    scr.set_defaults(func=cmd_screen)

    ev = sub.add_parser("eval", help="Compare gold labels to predictions (include|exclude|unclear).")
    ev.add_argument("--gold", required=True)
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--id-column", default="id")
    ev.add_argument("--gold-column", default="gold_label")
    ev.add_argument("--pred-column", default="stage1_decision")
    ev.set_defaults(func=cmd_eval)

    en = sub.add_parser(
        "enrich-pubmed",
        help="Fetch abstract (PubMed) and full text (PMC when PMCID present) via NCBI efetch; adds screening columns.",
    )
    en.add_argument("--input", required=True)
    en.add_argument("--csv-out", required=True)
    en.add_argument("--pmid-column", default="PMID")
    en.add_argument("--title-column", default="Title")
    en.add_argument("--pmcid-column", default="PMCID")
    en.add_argument(
        "--api-key-env",
        default="NCBI_API_KEY",
        help="Environment variable name to read for the optional NCBI API key.",
    )
    en.add_argument(
        "--ncbi-api-key",
        default=None,
        dest="ncbi_api_key",
        help="NCBI API key for this run; overrides the value loaded from the env var named by --api-key-env.",
    )
    en.add_argument("--limit", type=int, default=None)
    en.set_defaults(func=cmd_enrich_pubmed)

    return p


def main(argv: list[str] | None = None) -> int:
    load_project_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    return rc if isinstance(rc, int) else 0


def launch_screen(argv: list[str] | None = None) -> int:
    """
    Console entry for the keyword screening interface: same as
    ``two-stage-screen screen ...`` without typing the ``screen`` subcommand.
    """
    import sys

    if argv is None:
        argv = sys.argv[1:]
    return main(["screen", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
