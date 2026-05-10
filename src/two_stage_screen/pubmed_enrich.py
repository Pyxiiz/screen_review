from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Without API key NCBI recommends <=3 req/s; with key <=10 req/s.
DELAY_NO_KEY_SEC = 0.35
DELAY_WITH_KEY_SEC = 0.12

MAX_RETRIES = 4
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def localname(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def normalize_pmcid(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    m = re.match(r"PMC?(\d+)$", s, re.IGNORECASE)
    if m:
        return m.group(1)
    if s.isdigit():
        return s
    return None


def _first_by_local_name(root: ET.Element, name: str) -> ET.Element | None:
    for el in root.iter():
        if localname(el.tag) == name:
            return el
    return None


def parse_pubmed_abstract_xml(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    abstract_el = _first_by_local_name(root, "Abstract")
    if abstract_el is None:
        return ""
    parts: list[str] = []
    for child in abstract_el:
        if localname(child.tag) != "AbstractText":
            continue
        label = (child.get("Label") or "").strip()
        piece = "".join(child.itertext()).strip()
        if not piece:
            continue
        if label:
            parts.append(f"{label}: {piece}")
        else:
            parts.append(piece)
    if parts:
        return "\n\n".join(parts)
    return "".join(abstract_el.itertext()).strip()


def _first_body(root: ET.Element) -> ET.Element | None:
    for el in root.iter():
        if localname(el.tag) == "body":
            return el
    return None


def element_to_plain_text(elem: ET.Element) -> str:
    return " ".join(t.strip() for t in elem.itertext() if t and t.strip())


def parse_pmc_body_xml(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    body = _first_body(root)
    if body is None:
        return ""
    return element_to_plain_text(body)


def _efetch_params(db: str, id_str: str, api_key: str | None) -> dict[str, str]:
    p: dict[str, str] = {"db": db, "id": id_str, "retmode": "xml"}
    if api_key:
        p["api_key"] = api_key
    return p


def efetch_get(
    client: httpx.Client,
    db: str,
    id_str: str,
    api_key: str | None,
) -> tuple[bytes, int | None]:
    """GET efetch; returns (body_bytes, http_status_or_none_on_success)."""
    params = _efetch_params(db, id_str, api_key)
    url = f"{EFETCH_URL}?{urlencode(params)}"
    last_status: int | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = client.get(url, timeout=120.0)
            last_status = r.status_code
            if r.status_code == 200:
                return r.content, None
            if r.status_code not in RETRYABLE_STATUS or attempt == MAX_RETRIES - 1:
                return r.content, r.status_code
        except httpx.HTTPError:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 1.0 * (2**attempt)
            time.sleep(wait)
            continue
        wait = 1.0 * (2**attempt)
        time.sleep(wait)
    return b"", last_status


@dataclass
class FetchOutcome:
    text: str
    ok: bool
    message: str


def fetch_pubmed_abstract(
    client: httpx.Client,
    pmid: str,
    api_key: str | None,
) -> FetchOutcome:
    pmid = pmid.strip()
    if not pmid or not pmid.isdigit():
        return FetchOutcome("", False, "invalid_pmid")
    time.sleep(DELAY_WITH_KEY_SEC if api_key else DELAY_NO_KEY_SEC)
    try:
        data, err_status = efetch_get(client, "pubmed", pmid, api_key)
    except httpx.HTTPError as e:
        return FetchOutcome("", False, f"http_error:{e}")
    if err_status is not None:
        return FetchOutcome("", False, f"http_status:{err_status}")
    try:
        abstract = parse_pubmed_abstract_xml(data)
    except ET.ParseError as e:
        return FetchOutcome("", False, f"parse_error:{e}")
    if not abstract.strip():
        return FetchOutcome("", False, "empty_abstract")
    return FetchOutcome(abstract.strip(), True, "")


def fetch_pmc_fulltext(
    client: httpx.Client,
    pmcid_raw: str | None,
    api_key: str | None,
) -> FetchOutcome:
    numeric = normalize_pmcid(pmcid_raw)
    if not numeric:
        return FetchOutcome("", False, "no_pmcid")
    time.sleep(DELAY_WITH_KEY_SEC if api_key else DELAY_NO_KEY_SEC)
    try:
        data, err_status = efetch_get(client, "pmc", numeric, api_key)
    except httpx.HTTPError as e:
        return FetchOutcome("", False, f"http_error:{e}")
    if err_status is not None:
        return FetchOutcome("", False, f"http_status:{err_status}")
    try:
        text = parse_pmc_body_xml(data)
    except ET.ParseError as e:
        return FetchOutcome("", False, f"parse_error:{e}")
    if not text.strip():
        return FetchOutcome("", False, "empty_body")
    return FetchOutcome(text.strip(), True, "")


def enrich_row(
    client: httpx.Client,
    pmid: str,
    title: str,
    pmcid: str | None,
    api_key: str | None,
) -> dict[str, str]:
    """Build id, title, abstract, full_text, and enrich_* status columns."""
    ab = fetch_pubmed_abstract(client, pmid, api_key)
    ft = fetch_pmc_fulltext(client, pmcid, api_key)
    messages: list[str] = []
    if not ab.ok:
        messages.append(f"abstract:{ab.message}" if ab.message else "abstract:failed")
    if not ft.ok:
        messages.append(f"fulltext:{ft.message}" if ft.message else "fulltext:skipped")
    msg = "; ".join(messages) if messages else ""
    return {
        "id": pmid.strip(),
        "title": title or "",
        "abstract": ab.text,
        "full_text": ft.text,
        "enrich_abstract_ok": "1" if ab.ok else "0",
        "enrich_fulltext_ok": "1" if ft.ok else "0",
        "enrich_message": msg,
    }
