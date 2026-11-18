#!/usr/bin/env python3
"""Monitor public publication-search leads and aggregate GitHub activity.

This script does not identify individual downloaders and cannot establish
that a publication used Coldscatter or that a citation was omitted.
"""
from __future__ import annotations

import json
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO = os.environ.get("COLDSCATTER_GITHUB_REPO", "mahdiyeh-ghasemi/Coldscatter")
STATE_PATH = Path(os.environ.get(
    "COLDSCATTER_STATE",
    ".coldscatter_monitor_state.json",
))
USER_AGENT = "Coldscatter-publication-monitor/1.0 (repository-maintainer tool)"


def fetch_json(url, token=None):
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))


def publication_leads():
    leads = {}

    openalex_url = "https://api.openalex.org/works?" + urlencode({
        "search": "Coldscatter",
        "per-page": 20,
        "select": "id,doi,title,publication_date,authorships,primary_location",
    })
    data = fetch_json(openalex_url)
    for work in data.get("results", []):
        key = work.get("doi") or work.get("id")
        if not key:
            continue
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])
        ]
        source = (work.get("primary_location") or {}).get("source") or {}
        leads[key] = {
            "title": work.get("title", ""),
            "date": work.get("publication_date", ""),
            "authors": authors,
            "source": source.get("display_name", ""),
            "url": work.get("doi") or work.get("id", ""),
            "index": "OpenAlex",
        }

    crossref_url = "https://api.crossref.org/works?" + urlencode({
        "query.title": "Coldscatter",
        "rows": 20,
    })
    data = fetch_json(crossref_url)
    for work in data.get("message", {}).get("items", []):
        title_list = work.get("title") or []
        title = title_list[0] if title_list else ""
        doi = work.get("DOI")
        key = f"https://doi.org/{doi}" if doi else title
        if not key:
            continue
        authors = [
            " ".join(x for x in (a.get("given"), a.get("family")) if x)
            for a in work.get("author", [])
        ]
        leads.setdefault(key, {
            "title": title,
            "date": (work.get("published", {}).get("date-parts") or [[""]])[0][0],
            "authors": authors,
            "source": (work.get("container-title") or [""])[0],
            "url": key,
            "index": "Crossref",
        })
    return leads


def github_activity():
    token = os.environ.get("COLDSCATTER_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    base = f"https://api.github.com/repos/{REPO}"
    report = []

    if not token:
        report.append(
            "GitHub traffic statistics skipped: set COLDSCATTER_GITHUB_TOKEN "
            "with repository traffic-read permission."
        )
    else:
        for endpoint in ("views", "clones"):
            try:
                data = fetch_json(f"{base}/traffic/{endpoint}", token)
                total = data.get("count", 0)
                unique = data.get("uniques", 0)
                report.append(
                    f"GitHub {endpoint} (rolling 14-day window): "
                    f"{total} total, {unique} unique."
                )
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                report.append(f"GitHub {endpoint} endpoint unavailable: {exc}")

    try:
        releases = fetch_json(f"{base}/releases?per_page=100", token)
        downloads = sum(
            asset.get("download_count", 0)
            for release in releases
            for asset in release.get("assets", [])
        )
        report.append(
            f"GitHub release assets: {downloads} aggregate downloads "
            f"across the latest {len(releases)} listed releases."
        )
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        report.append(f"GitHub release data unavailable: {exc}")

    return report


def send_email(subject, body):
    host = os.environ.get("COLDSCATTER_SMTP_HOST")
    recipient = os.environ.get("COLDSCATTER_ALERT_TO")
    sender = os.environ.get("COLDSCATTER_ALERT_FROM")
    if not all((host, recipient, sender)):
        print("Email alert not sent: SMTP host, ALERT_TO, or ALERT_FROM is unset.")
        return

    port = int(os.environ.get("COLDSCATTER_SMTP_PORT", "587"))
    username = os.environ.get("COLDSCATTER_SMTP_USER")
    password = os.environ.get("COLDSCATTER_SMTP_PASSWORD")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        if username and password:
            server.login(username, password)
        server.send_message(message)


def main():
    try:
        leads = publication_leads()
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        print(f"Publication search failed: {exc}", file=sys.stderr)
        return 2

    old_state = {}
    establish_baseline = not STATE_PATH.exists()

    if STATE_PATH.exists():
        try:
            loaded_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded_state, dict):
                old_state = loaded_state
            else:
                establish_baseline = True
        except (OSError, json.JSONDecodeError):
            establish_baseline = True

    stored_ids = old_state.get("seen_publication_ids")
    if not isinstance(stored_ids, list):
        establish_baseline = True
        stored_ids = []

    seen = {item for item in stored_ids if isinstance(item, str)}

    if establish_baseline:
        seen.update(leads)
        new_ids = []
        print("Initial baseline established; existing matches will not trigger alerts.")
    else:
        new_ids = sorted(set(leads) - seen)

    activity = github_activity()

    print("=== Coldscatter monitoring report ===")
    print(f"Repository: {REPO}")
    print(f"Publication-search matches: {len(leads)}")
    print("These are search leads, not verified software-use records.")

    for key in new_ids:
        print(json.dumps(leads[key], ensure_ascii=False, indent=2))
    for line in activity:
        print(line)

    if new_ids:
        body = [
            "New publication-search matches for Coldscatter were found.",
            "",
            "These records are leads only. Manually inspect each paper to",
            "determine whether it actually used Coldscatter and whether a",
            "software citation is present. Search matches do not establish",
            "use, omission, infringement, or misconduct.",
            "",
        ]
        for key in new_ids:
            body.append(json.dumps(leads[key], ensure_ascii=False, indent=2))
            body.append("")
        body.extend(activity)
        send_email(
            f"Coldscatter: {len(new_ids)} new publication-search lead(s)",
            "\\n".join(body),
        )

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_state = STATE_PATH.with_name(STATE_PATH.name + ".tmp")
    temporary_state.write_text(
        json.dumps({
            "seen_publication_ids": sorted(seen.union(leads)),
            "last_match_count": len(leads),
        }, indent=2),
        encoding="utf-8",
    )
    temporary_state.replace(STATE_PATH)
    print(f"State saved to {STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
