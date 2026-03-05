from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from manifest_lib import ensure_manifest_defaults, load_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roster_taxonomy import agent_display_names, current_agent_ids, focus_agents_from_sources


DEFAULT_QUERY_TEMPLATE = "Zenless Zone Zero {display_name} gameplay 2026"


def _run_command(command: list[str]) -> tuple[int, str]:
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return process.returncode, process.stdout


def _resolve_ytdlp_command() -> list[str]:
    ytdlp = shutil.which("yt-dlp")
    if ytdlp:
        return [ytdlp]
    raise RuntimeError("yt-dlp not found in PATH")


def _search_youtube(query: str, max_results: int) -> List[Dict[str, Any]]:
    command = _resolve_ytdlp_command() + [
        "--no-warnings",
        "--print",
        "%(upload_date)s|%(id)s|%(title)s",
        f"ytsearch{max(1, max_results)}:{query}",
    ]
    code, output = _run_command(command)
    if code != 0 and not output.strip():
        raise RuntimeError(output.strip()[:4000])
    entries: List[Dict[str, Any]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        upload_date, video_id, title = (part.strip() for part in line.split("|", 2))
        if not video_id:
            continue
        entries.append(
            {
                "upload_date": upload_date,
                "id": video_id,
                "title": title,
                "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
            }
        )
    return entries


def _upload_date_iso(value: str) -> str:
    text = str(value or "").strip()
    if len(text) != 8 or not text.isdigit():
        return ""
    return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")


def _pick_entry(entries: List[Dict[str, Any]], min_upload_date: str, existing_urls: set[str]) -> Dict[str, Any] | None:
    threshold = str(min_upload_date or "").strip()
    for entry in entries:
        url = str(entry.get("webpage_url") or entry.get("url") or "").strip()
        upload_date = str(entry.get("upload_date") or "").strip()
        if not url or url in existing_urls:
            continue
        if threshold and (not upload_date or upload_date < threshold):
            continue
        return entry
    return None


def _candidate_queries(display_name: str, agent_id: str, primary_template: str) -> List[str]:
    return [
        primary_template.format(agent_id=agent_id, display_name=display_name),
        f"ZZZ {display_name} gameplay 2026",
        f"Zenless Zone Zero {display_name} guide 2026",
        f"ZZZ {display_name} build 2025",
        f"Zenless Zone Zero {display_name} 2.5 gameplay",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover fresh public gameplay sources for missing roster agents.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-file", default="docs/public_sources.discovery.seed.json")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--min-upload-date", default="20250101")
    parser.add_argument("--query-template", default=DEFAULT_QUERY_TEMPLATE)
    parser.add_argument("--agent-ids", default="", help="Optional comma-separated subset of agent ids.")
    args = parser.parse_args()

    manifest = ensure_manifest_defaults(load_manifest(Path(args.manifest).resolve()))
    sources = manifest.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("manifest.sources must be an array")

    focus_counts = focus_agents_from_sources(sources)
    display_names = agent_display_names(include_upcoming=False)
    requested = [item.strip() for item in str(args.agent_ids).split(",") if item.strip()]
    if requested:
        target_agents = requested
    else:
        target_agents = [agent_id for agent_id in current_agent_ids() if focus_counts.get(agent_id, 0) <= 0]

    existing_urls = {
        str(source.get("url") or "").strip()
        for source in sources
        if isinstance(source, dict) and str(source.get("url") or "").strip()
    }

    discovered: List[Dict[str, Any]] = []
    for index, agent_id in enumerate(target_agents, start=1):
        display_name = display_names.get(agent_id, agent_id.replace("agent_", "").replace("_", " ").title())
        picked = None
        for query in _candidate_queries(display_name=display_name, agent_id=agent_id, primary_template=args.query_template):
            entries = _search_youtube(query=query, max_results=max(1, int(args.max_results)))
            picked = _pick_entry(entries, min_upload_date=str(args.min_upload_date), existing_urls=existing_urls)
            if picked is not None:
                break
        if picked is None:
            continue
        url = str(picked.get("webpage_url") or picked.get("url") or "").strip()
        existing_urls.add(url)
        discovered.append(
            {
                "sourceId": f"auto_{agent_id}_{index:03d}",
                "url": url,
                "captureDate": _upload_date_iso(str(picked.get("upload_date") or "")) or datetime.utcnow().strftime("%Y-%m-%d"),
                "licenseNote": f"public gameplay video; focus={display_name}",
                "focusAgentId": agent_id,
                "locale": "unknown",
                "resolution": "unknown",
                "gamePatch": "unknown",
                "collector": "public-web-discovery",
                "sourceType": "public",
                "sourceTags": [
                    "auto-discovered",
                    str(picked.get("title") or "").strip(),
                ],
            }
        )

    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(discovered, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"targets": len(target_agents), "discovered": len(discovered), "output": str(output_path)}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
