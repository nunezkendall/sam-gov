import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from pypdf import PdfReader


DEFAULT_CONFIG = {
    "sam": {
        "base_url": "https://api.sam.gov/opportunities/v2/search",
        "naics_codes": ["541511", "541512", "541513", "541519", "518210"],
        "set_aside_codes": ["SBA", "SBP"],
        "min_amount": 5000,
        "max_amount": 150000,
        "require_amount": False,
        "posted_days_back": 3,
        "updated_days_back": 30,
        "page_limit": 500,
        "status": "active",
        "exclude_expired": True,
        "allow_onsite_states": ["AZ", "Arizona"],
        "exclude_brandname": True,
        "exclude_massive": True,
        "scope_indicator_threshold": 2,
        "filter_non_az_onsite": True,
    },
    "keywords": ["AWS", "Terraform", "DevOps", "automation", "Python", "Linux"],
    "output": {
        "raw": "output/opportunities_raw.json",
        "filtered": "output/opportunities_filtered.json",
        "scored": "output/opportunities_scored.json",
        "top10": "output/top10.json",
        "openai_raw": "output/openai_raw.txt",
        "state": "data/state.json",
    },
    "openai": {
        "model": "gpt-5.2-chat-latest",
        "max_items": 200,
        "timeout_seconds": 120,
        "include_description_full": True,
        "include_attachments": True,
        "attachment_max_files": 5,
        "attachment_max_bytes": 5_000_000,
        "attachment_text_limit": 8000,
    },
}


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_config(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(json.dumps(DEFAULT_CONFIG))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def format_mmddyyyy(dt: datetime) -> str:
    return dt.strftime("%m/%d/%Y")


def load_state(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def normalize_text(value: str) -> str:
    return value.lower()


def keyword_matches(text: str, keywords: Iterable[str]) -> List[str]:
    if not text:
        return []
    hay = normalize_text(text)
    hits = []
    for kw in keywords:
        if normalize_text(kw) in hay:
            hits.append(kw)
    return hits


def find_remote_indicators(text: str) -> List[str]:
    if not text:
        return []
    hay = normalize_text(text)
    indicators = [
        "remote",
        "telework",
        "virtual",
        "offsite",
        "work from home",
        "wfh",
        "distributed",
        "location independent",
    ]
    hits = []
    for term in indicators:
        if term in hay:
            hits.append(term)
    return hits


def find_onsite_indicators(text: str) -> List[str]:
    if not text:
        return []
    hay = normalize_text(text)
    indicators = [
        "on-site",
        "onsite",
        "in person",
        "in-person",
        "on site",
        "must be on-site",
        "must be onsite",
        "no telework",
        "no remote",
        "not remote",
        "requires onsite",
        "requires on-site",
        "requires in-person",
    ]
    hits = []
    for term in indicators:
        if term in hay:
            hits.append(term)
    return hits


def find_brandname_indicators(text: str) -> List[str]:
    if not text:
        return []
    hay = normalize_text(text)
    indicators = [
        "brand name",
        "brand-name",
        "sole source",
        "sole-source",
        "only one responsible source",
        "intent to award",
        "notice of intent",
        "unique qualifications",
        "proprietary",
        "noncompetitive",
        "no substitute",
        "only source",
        "exclusive",
    ]
    hits = []
    for term in indicators:
        if term in hay:
            hits.append(term)
    return hits


def find_scope_indicators(text: str) -> List[str]:
    if not text:
        return []
    hay = normalize_text(text)
    indicators = [
        "global",
        "worldwide",
        "enterprise",
        "enterprise-wide",
        "enterprise wide",
        "24x7",
        "24x6",
        "24/7",
        "24/6",
        "soc",
        "security operations center",
        "noc",
        "network operations center",
        "multi-region",
        "multi region",
        "multiple regions",
        "multiple sites",
        "multi-site",
        "multi site",
        "large scale",
        "large-scale",
        "major program",
        "program management office",
        "pmo",
        "integrated support across",
        "end-to-end",
    ]
    hits = []
    for term in indicators:
        if term in hay:
            hits.append(term)
    return hits


def normalize_state(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.strip().lower()


def is_az_state(value: Optional[str]) -> bool:
    norm = normalize_state(value)
    return norm in ("az", "arizona")


def fetch_description_text(url: Optional[str], api_key: str, timeout_seconds: int) -> str:
    if not url:
        return ""
    params = {}
    if "api_key=" not in url:
        params["api_key"] = api_key
    try:
        resp = requests.get(url, params=params, timeout=timeout_seconds)
        if resp.status_code != 200:
            return ""
        return resp.text or ""
    except requests.RequestException:
        return ""

def fetch_attachment_text(
    url: str,
    api_key: str,
    timeout_seconds: int,
    max_bytes: int,
) -> Tuple[str, Optional[str], int]:
    params = {}
    if "api_key=" not in url:
        params["api_key"] = api_key
    try:
        resp = requests.get(url, params=params, timeout=timeout_seconds)
        if resp.status_code != 200:
            return "", resp.headers.get("Content-Type"), 0
        content = resp.content or b""
        if max_bytes and len(content) > max_bytes:
            return "", resp.headers.get("Content-Type"), len(content)
        content_type = resp.headers.get("Content-Type")
        text = ""
        if (content_type and "pdf" in content_type.lower()) or url.lower().endswith(".pdf"):
            try:
                reader = PdfReader(io.BytesIO(content))
                parts = []
                for page in reader.pages:
                    parts.append(page.extract_text() or "")
                text = "\n".join(parts)
            except Exception:
                text = ""
        return text, content_type, len(content)
    except requests.RequestException:
        return "", None, 0


def make_snippet(text: str, limit: int = 800) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def pick_amount(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    award = record.get("award") or {}
    candidates = [
        ("data.award.amount", record.get("data", {}).get("award", {}).get("amount")),
        ("award.amount", award.get("amount")),
        ("estimatedTotalValue", record.get("estimatedTotalValue")),
        ("estimatedValue", record.get("estimatedValue")),
        ("baseAndAllOptionsValue", record.get("baseAndAllOptionsValue")),
        ("fundingCeiling", record.get("fundingCeiling")),
    ]
    for label, value in candidates:
        if value is None:
            continue
        try:
            return float(value), label
        except (TypeError, ValueError):
            continue
    return None, None


def extract_updated_date(record: Dict[str, Any]) -> Optional[datetime]:
    for key in ("updatedDate", "updatedDateTime", "lastModifiedDate", "modifiedDate"):
        value = record.get(key)
        if isinstance(value, str):
            parsed = parse_date(value)
            if parsed:
                return parsed
    return None


def fetch_opportunities(
    base_url: str,
    api_key: str,
    posted_from: str,
    posted_to: str,
    naics_code: str,
    set_aside_code: str,
    page_limit: int,
    status: Optional[str],
    timeout_seconds: int,
    max_retries: int = 5,
) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    offset = 0

    while True:
        params = {
            "api_key": api_key,
            "postedFrom": posted_from,
            "postedTo": posted_to,
            "ncode": naics_code,
            "typeOfSetAside": set_aside_code,
            "limit": page_limit,
            "offset": offset,
        }
        if status:
            params["status"] = status

        attempt = 0
        while True:
            resp = requests.get(base_url, params=params, timeout=timeout_seconds)
            if resp.status_code == 200:
                break
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                delay = min(2 ** attempt, 30)
                time.sleep(delay)
                attempt += 1
                continue
            raise RuntimeError(f"SAM.gov API error {resp.status_code}: {resp.text[:500]}")

        payload = resp.json()

        page = payload.get("opportunitiesData") or payload.get("data") or payload.get("items") or []
        total = payload.get("totalRecords")

        if isinstance(page, list):
            collected.extend(page)
        else:
            raise RuntimeError("Unexpected response format: missing opportunitiesData array.")

        if total is None:
            # If totalRecords missing, stop once we get less than limit.
            if len(page) < page_limit:
                break
            offset += page_limit
            continue

        offset += page_limit
        if offset >= int(total):
            break

        time.sleep(0.2)

    return collected


def build_clean_record(
    record: Dict[str, Any],
    keywords: List[str],
    min_amount: float,
    max_amount: float,
    require_amount: bool,
    sam_api_key: str,
    timeout_seconds: int,
    allow_onsite_states: List[str],
    scope_threshold: int,
    include_description_full: bool,
    include_attachments: bool,
    attachment_max_files: int,
    attachment_max_bytes: int,
    attachment_text_limit: int,
) -> Optional[Dict[str, Any]]:
    title = (record.get("title") or "").strip()
    notice_id = (record.get("noticeId") or "").strip()
    if not notice_id:
        return None

    naics = str(record.get("naicsCode") or "").strip()
    set_aside_code = (record.get("setAsideCode") or "").strip()
    set_aside = (record.get("setAside") or "").strip()
    posted = (record.get("postedDate") or "").strip()
    response_deadline = (record.get("responseDeadLine") or record.get("reponseDeadLine") or "").strip()
    agency = (record.get("fullParentPathName") or record.get("department") or "").strip()
    office = (record.get("office") or "").strip()
    place = record.get("placeOfPerformance") or {}
    place_city = place.get("city", {}).get("name") if isinstance(place.get("city"), dict) else place.get("city")
    place_state = place.get("state", {}).get("name") if isinstance(place.get("state"), dict) else place.get("state")
    place_zip = place.get("zip")
    updated = extract_updated_date(record)

    amount, amount_source = pick_amount(record)
    amount_in_range = None
    if amount is not None:
        amount_in_range = min_amount <= amount <= max_amount
        if not amount_in_range:
            return None
    elif require_amount:
        return None

    title_hits = keyword_matches(title, keywords)
    title_remote = find_remote_indicators(title)
    title_onsite = find_onsite_indicators(title)
    desc_hits: List[str] = []
    desc_remote: List[str] = []
    desc_onsite: List[str] = []
    desc_brand: List[str] = []
    desc_scope: List[str] = []
    desc_text = fetch_description_text(record.get("description"), sam_api_key, timeout_seconds)
    if desc_text:
        desc_hits = keyword_matches(desc_text, keywords)
        desc_remote = find_remote_indicators(desc_text)
        desc_onsite = find_onsite_indicators(desc_text)
        desc_brand = find_brandname_indicators(desc_text)
        desc_scope = find_scope_indicators(desc_text)

    attachments_summary = []
    attachments_texts = []
    if include_attachments:
        links = record.get("resourceLinks") or []
        if isinstance(links, list):
            for link in links[:attachment_max_files]:
                if not isinstance(link, str) or not link:
                    continue
                text, content_type, size = fetch_attachment_text(
                    link,
                    sam_api_key,
                    timeout_seconds,
                    attachment_max_bytes,
                )
                if text:
                    attachments_texts.append(text)
                attachments_summary.append(
                    {
                        "url": link,
                        "contentType": content_type,
                        "byteSize": size,
                        "textSnippet": make_snippet(text, 800) if text else "",
                    }
                )
    keyword_hits = title_hits or desc_hits
    if not keyword_hits:
        return None

    pop_state = place_state if isinstance(place_state, str) else ""
    allow_onsite = any(is_az_state(s) for s in [pop_state] + allow_onsite_states)
    onsite_indicators = title_onsite + desc_onsite
    brand_indicators = desc_brand
    scope_indicators = desc_scope
    onsite_required = bool(onsite_indicators)
    remote_possible = bool(title_remote or desc_remote)
    onsite_possible = allow_onsite and not remote_possible
    if onsite_required and not allow_onsite:
        onsite_possible = False

    deadline_dt = parse_date(response_deadline) if response_deadline else None
    days_to_deadline = None
    if deadline_dt:
        days_to_deadline = (deadline_dt - utc_now()).total_seconds() / 86400.0

    clean = {
        "noticeId": notice_id,
        "title": title,
        "solicitationNumber": (record.get("solicitationNumber") or "").strip(),
        "type": (record.get("type") or "").strip(),
        "baseType": (record.get("baseType") or "").strip(),
        "naicsCode": naics,
        "setAsideCode": set_aside_code,
        "setAside": set_aside,
        "postedDate": posted,
        "responseDeadline": response_deadline,
        "agency": agency,
        "office": office,
        "placeOfPerformance": {
            "city": place_city,
            "state": place_state,
            "zip": place_zip,
        },
        "links": {
            "uiLink": record.get("uiLink"),
            "description": record.get("description"),
            "additionalInfoLink": record.get("additionalInfoLink"),
            "resourceLinks": record.get("resourceLinks"),
        },
        "descriptionSnippet": make_snippet(desc_text, 800) if desc_text else "",
        "descriptionFull": desc_text if include_description_full else "",
        "attachments": attachments_summary,
        "attachmentsTextCombined": make_snippet(
            " ".join(attachments_texts),
            attachment_text_limit,
        )
        if attachments_texts
        else "",
        "amountEstimate": amount,
        "amountSource": amount_source,
        "amountInRange": amount_in_range,
        "keywordsMatched": keyword_hits,
        "keywordsMatchedIn": {
            "title": title_hits,
            "description": desc_hits,
        },
        "remoteIndicators": {
            "title": title_remote,
            "description": desc_remote,
        },
        "onsiteIndicators": {
            "title": title_onsite,
            "description": desc_onsite,
        },
        "remotePossible": remote_possible,
        "onsitePossible": onsite_possible,
        "brandNameLikely": bool(brand_indicators),
        "brandNameIndicators": brand_indicators,
        "scopeIndicators": scope_indicators,
        "scopeRisk": len(scope_indicators) >= scope_threshold,
        "workModeNotes": {
            "onsiteRequired": onsite_required,
            "onsiteAllowedStates": allow_onsite_states,
            "placeOfPerformanceState": pop_state or None,
        },
        "timeToDeadlineDays": days_to_deadline,
        "updatedDate": updated.isoformat().replace("+00:00", "Z") if updated else None,
    }
    return clean


def filter_recent(
    records: List[Dict[str, Any]],
    last_run: Optional[datetime],
    updated_days_back: int,
    exclude_expired: bool,
) -> List[Dict[str, Any]]:
    if last_run is None:
        cutoff = utc_now() - timedelta(days=updated_days_back)
    else:
        cutoff = last_run

    now = utc_now()
    filtered = []
    for rec in records:
        posted = parse_date(rec.get("postedDate", "")) or None
        updated = None
        if rec.get("updatedDate"):
            updated = parse_date(rec.get("updatedDate"))
        if updated is None:
            updated = posted
        if exclude_expired:
            deadline = parse_date(rec.get("responseDeadline", ""))
            if deadline and deadline < now:
                continue
        if updated and updated >= cutoff:
            filtered.append(rec)
    return filtered


def openai_score(
    api_key: str,
    model: str,
    opportunities: List[Dict[str, Any]],
    timeout_seconds: int,
    raw_output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    prompt = {
        "role": "system",
        "content": (
            "You score federal contract opportunities for a small IT services firm. "
            "Return only JSON. No markdown. "
            "For each opportunity, provide relevance (0-100) and ease (0-100). "
            "Relevance means fit to AWS, Terraform, DevOps, automation, Python, Linux. "
            "Ease means likely low complexity and within a small team capacity. "
            "Use title, descriptionFull, and attachmentsTextCombined if present. "
            "Do not invent missing facts. If info missing, score conservatively. "
            "Also return a ranked top10 list of noticeIds with brief reason."
        ),
    }

    user = {
        "role": "user",
        "content": json.dumps(
            {
                "opportunities": opportunities,
                "scoring_instructions": {
                    "relevance_weight": 0.6,
                    "ease_weight": 0.4,
                    "overall_score": "round(0.6*relevance + 0.4*ease)",
                },
                "output_schema": {
                    "scored": [
                        {
                            "noticeId": "string",
                            "relevance": 0,
                            "ease": 0,
                            "overall": 0,
                            "rationale": "short sentence",
                        }
                    ],
                    "top10": [
                        {"noticeId": "string", "reason": "short sentence"}
                    ],
                },
            },
            ensure_ascii=True,
        ),
    }

    payload = {
        "model": model,
        "input": [prompt, user],
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload,
        timeout=timeout_seconds,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    text_parts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text"):
                    text_parts.append(content.get("text", ""))
    if not text_parts and "output_text" in data:
        text_parts.append(data["output_text"])

    text = "\n".join(text_parts).strip()
    if not text:
        raise RuntimeError("OpenAI API returned no text output.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if raw_output_path:
            ensure_parent(raw_output_path)
            with raw_output_path.open("w", encoding="utf-8") as f:
                f.write(text)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily SAM.gov opportunities fetcher + scorer")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--no-score", action="store_true", help="Skip OpenAI scoring")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    config = load_config(config_path)

    sam_cfg = config["sam"]
    output_cfg = config["output"]

    sam_key = os.getenv("SAM_API_KEY")
    if not sam_key:
        eprint("Missing SAM_API_KEY environment variable.")
        return 2

    openai_key = os.getenv("OPENAI_API_KEY")
    do_score = (not args.no_score) and bool(openai_key)

    posted_days_back = int(sam_cfg.get("posted_days_back", 3))
    updated_days_back = int(sam_cfg.get("updated_days_back", 30))
    today = utc_now().date()
    posted_from = format_mmddyyyy(datetime.combine(today - timedelta(days=posted_days_back), datetime.min.time()))
    posted_to = format_mmddyyyy(datetime.combine(today, datetime.min.time()))

    def resolve_output_path(value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = base_dir / path
        return path

    state_path = resolve_output_path(output_cfg["state"])
    state = load_state(state_path)
    last_run = parse_date(state.get("last_run_utc", "")) if state else None

    all_records: Dict[str, Dict[str, Any]] = {}

    for naics in sam_cfg["naics_codes"]:
        for set_aside in sam_cfg["set_aside_codes"]:
            eprint(f"Fetching NAICS {naics} / set-aside {set_aside} from {posted_from} to {posted_to}")
            records = fetch_opportunities(
                sam_cfg["base_url"],
                sam_key,
                posted_from,
                posted_to,
                naics,
                set_aside,
                int(sam_cfg.get("page_limit", 1000)),
                sam_cfg.get("status"),
                int(config.get("openai", {}).get("timeout_seconds", 120)),
            )
            for rec in records:
                notice_id = rec.get("noticeId")
                if notice_id:
                    all_records[str(notice_id)] = rec

    raw_list = list(all_records.values())
    raw_path = resolve_output_path(output_cfg["raw"])
    ensure_parent(raw_path)
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(raw_list, f, indent=2, ensure_ascii=True)

    clean_records: List[Dict[str, Any]] = []
    for rec in raw_list:
        clean = build_clean_record(
            rec,
            config.get("keywords", []),
            float(sam_cfg.get("min_amount", 0)),
            float(sam_cfg.get("max_amount", 0)),
            bool(sam_cfg.get("require_amount", False)),
            sam_key,
            int(config.get("openai", {}).get("timeout_seconds", 120)),
            sam_cfg.get("allow_onsite_states", ["AZ", "Arizona"]),
            int(sam_cfg.get("scope_indicator_threshold", 2)),
            bool(config.get("openai", {}).get("include_description_full", True)),
            bool(config.get("openai", {}).get("include_attachments", True)),
            int(config.get("openai", {}).get("attachment_max_files", 5)),
            int(config.get("openai", {}).get("attachment_max_bytes", 5_000_000)),
            int(config.get("openai", {}).get("attachment_text_limit", 8000)),
        )
        if clean:
            clean_records.append(clean)

    clean_records = filter_recent(
        clean_records,
        last_run,
        updated_days_back,
        bool(sam_cfg.get("exclude_expired", True)),
    )
    if sam_cfg.get("exclude_brandname", True):
        clean_records = [rec for rec in clean_records if not rec.get("brandNameLikely")]
    if sam_cfg.get("exclude_massive", True):
        clean_records = [rec for rec in clean_records if not rec.get("scopeRisk")]
    if sam_cfg.get("filter_non_az_onsite", True):
        clean_records = [
            rec
            for rec in clean_records
            if rec.get("remotePossible") or rec.get("onsitePossible")
        ]

    filtered_path = resolve_output_path(output_cfg["filtered"])
    ensure_parent(filtered_path)
    with filtered_path.open("w", encoding="utf-8") as f:
        json.dump(clean_records, f, indent=2, ensure_ascii=True)

    scored = None
    if do_score:
        max_items = int(config.get("openai", {}).get("max_items", 200))
        score_input = clean_records[:max_items]
        if score_input:
            scored = openai_score(
                openai_key,
                config.get("openai", {}).get("model", "gpt-5.2-chat-latest"),
                score_input,
                int(config.get("openai", {}).get("timeout_seconds", 120)),
                resolve_output_path(output_cfg.get("openai_raw", "output/openai_raw.txt")),
            )
            scored_path = resolve_output_path(output_cfg["scored"])
            ensure_parent(scored_path)
            with scored_path.open("w", encoding="utf-8") as f:
                json.dump(scored, f, indent=2, ensure_ascii=True)

            top10_path = resolve_output_path(output_cfg["top10"])
            ensure_parent(top10_path)
            by_id = {rec.get("noticeId"): rec for rec in clean_records}
            enriched_top10 = []
            for item in scored.get("top10", []):
                notice_id = item.get("noticeId")
                base = by_id.get(notice_id, {})
                enriched_top10.append(
                    {
                        "noticeId": notice_id,
                        "reason": item.get("reason"),
                        "title": base.get("title"),
                        "uiLink": (base.get("links") or {}).get("uiLink"),
                        "descriptionSnippet": base.get("descriptionSnippet"),
                        "responseDeadline": base.get("responseDeadline"),
                        "timeToDeadlineDays": base.get("timeToDeadlineDays"),
                        "awardEstimate": base.get("amountEstimate"),
                    }
                )
            with top10_path.open("w", encoding="utf-8") as f:
                json.dump(enriched_top10, f, indent=2, ensure_ascii=True)
        else:
            eprint("No records to score.")
    elif not openai_key and not args.no_score:
        eprint("OPENAI_API_KEY not set; skipping scoring.")

    state["last_run_utc"] = utc_now().isoformat().replace("+00:00", "Z")
    save_state(state_path, state)

    eprint(f"Raw records: {len(raw_list)}")
    eprint(f"Filtered records: {len(clean_records)}")
    if scored:
        eprint("Scoring complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
