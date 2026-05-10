from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Decision = Literal["include", "exclude", "unclear"]


class Stage1ModelResponse(BaseModel):
    decision: Decision
    rationale: str = Field(..., description="Brief justification tied to eligibility criteria.")
    triggered_criteria: list[str] = Field(
        default_factory=list,
        description="Short labels for which bullets or themes drove the decision.",
    )


class Stage2ModelResponse(BaseModel):
    decision: Decision
    rationale: str
    supporting_excerpts: list[str] = Field(
        default_factory=list,
        description="Short verbatim excerpts from the full text that support the decision (each under ~280 characters).",
    )


class ColumnMap(BaseModel):
    id: str = "id"
    title: str = "title"
    abstract: str = "abstract"
    full_text: str = "full_text"


class LLMConfig(BaseModel):
    provider: str = "openai"
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 1200


class ChunkingConfig(BaseModel):
    max_chars_single_pass: int = 100_000
    chunk_size: int = 28_000
    chunk_overlap: int = 400


class ScreenConfig(BaseModel):
    columns: ColumnMap = Field(default_factory=ColumnMap)
    research_question: str
    eligibility_text: str
    inclusion_keywords: list[str] = Field(default_factory=list)
    exclusion_keywords: list[str] = Field(default_factory=list)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    stage2_policy: Literal["include_and_unclear", "include_only", "all"] = "include_and_unclear"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class ChunkMapResponse(BaseModel):
    notes: list[str] = Field(
        ...,
        description="Bullet notes of facts relevant to eligibility from this excerpt only.",
    )
