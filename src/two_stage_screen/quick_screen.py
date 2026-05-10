"""Build screening config from comma-separated include/exclude keyword lists (no YAML required)."""

from __future__ import annotations

from two_stage_screen.models import ScreenConfig


def parse_comma_separated_keywords(value: str | None) -> list[str]:
    """Split on commas; trim whitespace; drop empty tokens."""
    if value is None or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def build_keyword_screen_config(
    inclusion: list[str],
    exclusion: list[str],
    *,
    base: ScreenConfig | None = None,
) -> ScreenConfig:
    """
    Build a full ScreenConfig from keyword lists. If ``base`` is provided (e.g. from YAML),
    only research_question, eligibility_text, inclusion_keywords, and exclusion_keywords
    are replaced; llm, chunking, columns, and stage2_policy are kept from ``base``.
    """
    rq, el_text = _rubric_from_keywords(inclusion, exclusion)
    if base is None:
        return ScreenConfig(
            research_question=rq,
            eligibility_text=el_text,
            inclusion_keywords=inclusion,
            exclusion_keywords=exclusion,
        )
    return base.model_copy(
        update={
            "research_question": rq,
            "eligibility_text": el_text,
            "inclusion_keywords": inclusion,
            "exclusion_keywords": exclusion,
        },
    )


def _rubric_from_keywords(inclusion: list[str], exclusion: list[str]) -> tuple[str, str]:
    inc_joined = ", ".join(inclusion) if inclusion else "(none supplied)"
    exc_joined = ", ".join(exclusion) if exclusion else "(none supplied)"
    research_question = (
        "Screen each citation for eligibility. "
        f"Inclusion-oriented themes: {inc_joined}. "
        f"Exclusion-oriented themes: {exc_joined}."
    )
    inc_lines = "\n".join(f"  - {k}" for k in inclusion) if inclusion else "  - (no inclusion keywords; use title/abstract/full text and keyword hints only when listed)."
    exc_lines = "\n".join(f"  - {k}" for k in exclusion) if exclusion else "  - (no exclusion keywords supplied.)"
    eligibility_text = f"""Eligibility (decision: include | exclude | unclear):

Include:
{inc_lines}
Prefer include when the record substantively aligns with one or more inclusion themes.

Exclude:
{exc_lines}
Exclude when the record is primarily or substantially about an exclusion theme, or clearly incompatible with inclusion.

Unclear:
Use unclear when the title and abstract (stage 1), or full text when run (stage 2), do not provide enough evidence."""
    return research_question.strip(), eligibility_text.strip()
