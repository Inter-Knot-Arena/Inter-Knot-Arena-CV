from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

ROSTER_PATH = Path(__file__).resolve().parent / "contracts" / "agent-roster.json"


@lru_cache(maxsize=1)
def _load_roster() -> List[dict]:
    with ROSTER_PATH.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    agents = payload.get("agents")
    if not isinstance(agents, list):
        raise ValueError(f"Invalid roster payload: {ROSTER_PATH}")
    return [agent for agent in agents if isinstance(agent, dict)]


def current_agent_ids() -> List[str]:
    return [str(agent["agentId"]) for agent in _load_roster() if str(agent.get("status") or "") == "current"]


def upcoming_agent_ids() -> List[str]:
    return [str(agent["agentId"]) for agent in _load_roster() if str(agent.get("status") or "") == "upcoming"]


def roster_agent_ids(include_upcoming: bool = False) -> List[str]:
    output = list(current_agent_ids())
    if include_upcoming:
        output.extend(upcoming_agent_ids())
    return output


def agent_display_names(include_upcoming: bool = False) -> Dict[str, str]:
    valid_status = {"current", "upcoming"} if include_upcoming else {"current"}
    output: Dict[str, str] = {}
    for agent in _load_roster():
        status = str(agent.get("status") or "")
        if status not in valid_status:
            continue
        canonical = str(agent.get("agentId") or "").strip()
        display_name = str(agent.get("displayName") or "").strip()
        if canonical and display_name:
            output[canonical] = display_name
    return output


def canonical_alias_map(include_upcoming: bool = False) -> Dict[str, str]:
    valid_status = {"current", "upcoming"} if include_upcoming else {"current"}
    output: Dict[str, str] = {}
    for agent in _load_roster():
        status = str(agent.get("status") or "")
        if status not in valid_status:
            continue
        canonical = str(agent.get("agentId") or "").strip()
        if not canonical:
            continue
        output[canonical] = canonical
        aliases = agent.get("aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    output[alias.strip()] = canonical
    return output


def canonicalize_agent_label(label: str, include_upcoming: bool = False) -> str:
    value = str(label or "").strip()
    if not value:
        return ""
    if value == "unknown":
        return "unknown"
    return canonical_alias_map(include_upcoming=include_upcoming).get(value, "")


def valid_agent_labels(include_upcoming: bool = False) -> List[str]:
    labels = sorted(canonical_alias_map(include_upcoming=include_upcoming).keys())
    labels.append("unknown")
    return labels


def _slugify_focus_token(value: str) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    token = token.replace(" - ", " ")
    token = token.replace("&", " ")
    token = token.replace(".", "")
    token = token.replace("'", "")
    token = token.replace("/", " ")
    token = token.replace("-", " ")
    token = "_".join(part for part in token.split() if part)
    if not token:
        return ""
    if token.startswith("agent_"):
        return token
    return f"agent_{token}"


def _focus_candidates(raw: str) -> List[str]:
    value = str(raw or "").strip()
    if not value:
        return []
    output: List[str] = [value]
    slug = _slugify_focus_token(value)
    if slug and slug not in output:
        output.append(slug)
    return output


def _extract_focus_note_values(note: str) -> List[str]:
    text = str(note or "")
    if "focus=" not in text:
        return []
    value = text.split("focus=", 1)[1].split(";", 1)[0].strip()
    if not value:
        return []
    raw_values = [value]
    for separator in (",", "|", "/"):
        exploded: List[str] = []
        for item in raw_values:
            exploded.extend(part.strip() for part in item.split(separator))
        raw_values = exploded
    return [item for item in raw_values if item]


def source_focus_agent_ids(source: dict) -> List[str]:
    alias_map = canonical_alias_map(include_upcoming=True)
    display_names = agent_display_names(include_upcoming=True)
    display_index = {name.lower(): agent_id for agent_id, name in display_names.items()}
    output: List[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        for candidate in _focus_candidates(raw):
            canonical = canonicalize_agent_label(candidate, include_upcoming=True)
            if not canonical:
                canonical = display_index.get(candidate.lower(), "")
            if not canonical:
                canonical = alias_map.get(candidate, "")
            if canonical and canonical not in seen:
                seen.add(canonical)
                output.append(canonical)

    explicit = source.get("focusAgentId")
    if isinstance(explicit, str) and explicit.strip():
        add(explicit.strip())

    for key in ("focusAgentIds", "focusAgents"):
        values = source.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, str) and item.strip():
                add(item.strip())

    for raw in _extract_focus_note_values(str(source.get("licenseNote") or "")):
        add(raw)

    return output


def focus_agents_from_sources(sources: Iterable[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for canonical in source_focus_agent_ids(source):
            counts[canonical] = counts.get(canonical, 0) + 1
    return counts
