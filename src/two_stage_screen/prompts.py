from __future__ import annotations

from two_stage_screen import PROMPT_VERSION
from two_stage_screen.models import ScreenConfig


def keywords_block(cfg: ScreenConfig) -> str:
    inc = "\n".join(f"- {k}" for k in cfg.inclusion_keywords) or "(none)"
    exc = "\n".join(f"- {k}" for k in cfg.exclusion_keywords) or "(none)"
    return (
        "Keyword hints (not sufficient alone; semantic criteria above take precedence).\n"
        f"Inclusion-oriented terms:\n{inc}\n"
        f"Exclusion-oriented terms:\n{exc}\n"
    )


def stage1_system_message() -> str:
    return (
        "You are assisting with title and abstract screening for a systematic rapid review. "
        "Respond with a single JSON object only. No prose outside JSON. "
        f"prompt_version={PROMPT_VERSION}."
    )


def stage1_user_message(cfg: ScreenConfig, title: str, abstract: str) -> str:
    return (
        f"PRIMARY RESEARCH QUESTION:\n{cfg.research_question.strip()}\n\n"
        f"ELIGIBILITY RUBRIC:\n{cfg.eligibility_text.strip()}\n\n"
        f"{keywords_block(cfg)}"
        "TASK: Based ONLY on the title and abstract below, classify the record.\n"
        'Return JSON keys: decision (include | exclude | unclear), rationale (string), '
        "triggered_criteria (array of short strings).\n"
        "Use \"unclear\" when the abstract does not supply enough evidence to confirm inclusion or exclusion.\n\n"
        f"TITLE:\n{title.strip()}\n\n"
        f"ABSTRACT:\n{abstract.strip()}\n"
    )


def stage2_system_message() -> str:
    return (
        "You are assisting with full-text screening for a systematic rapid review. "
        "Respond with a single JSON object only. "
        f"prompt_version={PROMPT_VERSION}."
    )


def stage2_user_message(cfg: ScreenConfig, title: str, abstract: str, full_text: str) -> str:
    return (
        f"PRIMARY RESEARCH QUESTION:\n{cfg.research_question.strip()}\n\n"
        f"ELIGIBILITY RUBRIC:\n{cfg.eligibility_text.strip()}\n\n"
        f"{keywords_block(cfg)}"
        "TASK: Use the full article text plus the title and abstract to classify the record.\n"
        'Return JSON keys: decision (include | exclude | unclear), rationale (string), '
        "supporting_excerpts (array of short verbatim excerpts from the full text; keep each excerpt under ~280 chars).\n\n"
        f"TITLE:\n{title.strip()}\n\n"
        f"ABSTRACT:\n{abstract.strip()}\n\n"
        f"FULL TEXT:\n{full_text.strip()}\n"
    )


def chunk_map_system_message() -> str:
    return (
        "You extract factual notes from a contiguous excerpt of a research article "
        "to support eligibility screening. Return JSON only. "
        f"prompt_version={PROMPT_VERSION}."
    )


def chunk_map_user_message(cfg: ScreenConfig, chunk_index: int, chunk_total: int, excerpt: str) -> str:
    return (
        f"PRIMARY RESEARCH QUESTION:\n{cfg.research_question.strip()}\n\n"
        f"ELIGIBILITY RUBRIC:\n{cfg.eligibility_text.strip()}\n\n"
        "From the excerpt below, list concise bullet-point notes relevant to deciding inclusion versus exclusion "
        '(actual AI use in health libraries versus teaching-only, theoretical-only, wrong library type, non-English, wrong publication type, etc.). '
        'If nothing relevant appears, return an empty notes array.\n'
        'Return JSON: {"notes": ["...", ...]}.\n\n'
        f"EXCERPT {chunk_index + 1} of {chunk_total}:\n{excerpt}\n"
    )


def chunked_reduce_system_message() -> str:
    return stage2_system_message()


def chunked_reduce_user_message(cfg: ScreenConfig, title: str, abstract: str, aggregated_notes: str) -> str:
    return (
        f"PRIMARY RESEARCH QUESTION:\n{cfg.research_question.strip()}\n\n"
        f"ELIGIBILITY RUBRIC:\n{cfg.eligibility_text.strip()}\n\n"
        f"{keywords_block(cfg)}"
        "The full article was too long to send at once. Below are extracted notes from sequential excerpts.\n"
        'TASK: Decide eligibility using title, abstract, and the notes.\n'
        'Return JSON keys: decision (include | exclude | unclear), rationale (string), '
        "supporting_excerpts (short phrases drawn from or paraphrased closely from the notes; quote verbatim only when clearly from the article).\n\n"
        f"TITLE:\n{title.strip()}\n\n"
        f"ABSTRACT:\n{abstract.strip()}\n\n"
        f"AGGREGATED NOTES FROM FULL TEXT:\n{aggregated_notes.strip()}\n"
    )
