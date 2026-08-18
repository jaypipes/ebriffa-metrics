"""custom_metrics.py

Fills the three gaps left by the `github-community-projects/contributors`
and `github-community-projects/issue-metrics` actions:

  1. Issues created / PRs raised in a date window, broken out by whether the
     author was a first-time contributor to the repo as of that window.
  2. Staleness: how many currently-open issues have had no activity for at
     least N days, for a configurable set of day thresholds.
  3. Mean time from an issue's creation to the merge of the pull request
     that closes it.

Writes a markdown fragment to CUSTOM_METRICS_OUTPUT so it can be
concatenated with the output of the other two actions into one report.

Environment variables:
  GH_TOKEN               GitHub token with read access to the repo. (required)
  REPOSITORY             "owner/repo".                               (required)
  START_DATE             YYYY-MM-DD, inclusive start of the window.  (required)
  END_DATE               YYYY-MM-DD, inclusive end of the window.    (required)
  STALE_THRESHOLDS       Comma-separated day counts. Default "3,7,14,30".
  CUSTOM_METRICS_OUTPUT  Output markdown filename. Default "custom_metrics.md".

Notes / known limitations:
  - "New contributor" is defined the same way the `contributors` action
    defines it: an author whose earliest issue/PR/commit-adjacent activity
    in the repo falls on or after START_DATE. This costs one extra search
    call per unique author in the window (cached).
  - Staleness uses `updatedAt`, which is the same signal GitHub's own UI
    uses for "last activity" (comments, labels, edits, etc. all bump it).
  - Issue-to-PR merge time relies on GitHub's automatic issue/PR linking
    (closing keywords like "Fixes #123", or the sidebar link). PRs that
    close an issue without one of those linkages will not be found.
  - GitHub's search API rate limit is much lower than the REST/GraphQL
    core limit (30 req/min authenticated), so this script backs off on
    403/secondary-rate-limit responses. For very large repos/windows,
    consider raising the per-call delay.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from statistics import mean, median

import requests

GITHUB_API = "https://api.github.com"
GITHUB_GRAPHQL = "https://api.github.com/graphql"


def env(name: str, default: str | None = None, required: bool = False) -> str:
    val = os.environ.get(name, default)
    if required and not val:
        print(f"::error::Missing required environment variable {name}", file=sys.stderr)
        sys.exit(1)
    return val


def make_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return session


def request_with_backoff(fn, *args, max_retries: int = 5, **kwargs):
    """Run a request function, retrying on secondary rate limits / abuse detection."""
    for attempt in range(max_retries):
        resp = fn(*args, **kwargs)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            wait = int(resp.headers.get("Retry-After", 10)) * (attempt + 1)
            print(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("Exceeded retries due to rate limiting")


def rest_search(session: requests.Session, query: str, sort: str | None = None, order: str | None = None, limit: int | None = None) -> list[dict]:
    """Page through the /search/issues endpoint (covers both issues and PRs)."""
    items: list[dict] = []
    page = 1
    while True:
        params = {"q": query, "per_page": 100, "page": page}
        if sort:
            params["sort"] = sort
        if order:
            params["order"] = order
        resp = request_with_backoff(session.get, f"{GITHUB_API}/search/issues", params=params)
        batch = resp.json().get("items", [])
        items.extend(batch)
        time.sleep(1)  # search API is rate limited more aggressively than core
        if limit and len(items) >= limit:
            return items[:limit]
        if len(batch) < 100:
            break
        page += 1
    return items


def graphql(session: requests.Session, query: str, variables: dict) -> dict:
    resp = request_with_backoff(session.post, GITHUB_GRAPHQL, json={"query": query, "variables": variables})
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# --- Metric 1: new vs. existing contributor breakdown -----------------------

def first_contribution_date(session: requests.Session, repo: str, author: str, cache: dict) -> str | None:
    if author in cache:
        return cache[author]
    items = rest_search(session, f"repo:{repo} author:{author}", sort="created", order="asc", limit=1)
    first = items[0]["created_at"] if items else None
    cache[author] = first
    return first


def build_new_vs_existing(session: requests.Session, repo: str, start: str, end: str) -> dict:
    issues = rest_search(session, f"repo:{repo} is:issue created:{start}..{end}")
    prs = rest_search(session, f"repo:{repo} is:pr created:{start}..{end}")
    author_first_seen: dict[str, str | None] = {}

    def tally(items: list[dict]) -> tuple[int, int]:
        new_count = existing_count = 0
        for item in items:
            user = item.get("user")
            if not user:
                continue
            login = user["login"]
            first = first_contribution_date(session, repo, login, author_first_seen)
            if first and first[:10] >= start:
                new_count += 1
            else:
                existing_count += 1
        return new_count, existing_count

    issues_new, issues_existing = tally(issues)
    prs_new, prs_existing = tally(prs)
    new_contributors = sum(
        1 for first in author_first_seen.values() if first and first[:10] >= start
    )

    return {
        "new_contributor_count": new_contributors,
        "issues_new": issues_new,
        "issues_existing": issues_existing,
        "prs_new": prs_new,
        "prs_existing": prs_existing,
    }


# --- Metric 2: staleness buckets --------------------------------------------

STALE_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    issues(states: OPEN, first: 100, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes { number updatedAt }
    }
  }
}
"""


def open_issue_ages_days(session: requests.Session, repo: str) -> list[int]:
    owner, name = repo.split("/")
    cursor = None
    now = datetime.now(timezone.utc)
    ages = []
    while True:
        data = graphql(session, STALE_QUERY, {"owner": owner, "name": name, "cursor": cursor})
        conn = data["repository"]["issues"]
        for node in conn["nodes"]:
            updated = parse_dt(node["updatedAt"])
            ages.append((now - updated).days)
        if not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]
    return ages


def staleness_buckets(ages: list[int], thresholds: list[int]) -> dict[int, int]:
    return {t: sum(1 for a in ages if a >= t) for t in thresholds}


# --- Metric 3: issue creation -> linked PR merge time -----------------------

PR_LINKED_ISSUES_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      mergedAt
      closingIssuesReferences(first: 10) { nodes { number createdAt } }
    }
  }
}
"""


def merged_pr_numbers(session: requests.Session, repo: str, start: str, end: str) -> list[int]:
    items = rest_search(session, f"repo:{repo} is:pr is:merged merged:{start}..{end}")
    return [item["number"] for item in items]


def issue_to_merge_durations_seconds(session: requests.Session, repo: str, start: str, end: str) -> list[float]:
    owner, name = repo.split("/")
    durations = []
    for number in merged_pr_numbers(session, repo, start, end):
        data = graphql(session, PR_LINKED_ISSUES_QUERY, {"owner": owner, "name": name, "number": number})
        pr = data["repository"]["pullRequest"]
        if not pr or not pr["mergedAt"]:
            continue
        merged_at = parse_dt(pr["mergedAt"])
        for issue in pr["closingIssuesReferences"]["nodes"]:
            created_at = parse_dt(issue["createdAt"])
            durations.append((merged_at - created_at).total_seconds())
    return durations


# --- Formatting ---------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    days = seconds / 86400
    if days >= 1:
        return f"{days:.1f} days"
    hours = seconds / 3600
    return f"{hours:.1f} hours"


def build_markdown(repo: str, start: str, end: str, contrib: dict, ages: list[int], buckets: dict[int, int], merge_durations: list[float]) -> str:
    lines = [f"## Custom metrics ({start} to {end})", ""]

    lines += [
        "### New vs. existing contributors (by issue/PR authorship)",
        "",
        "_Counts below are based on who opened an issue or PR, not who committed "
        "code — see the Contributors section above for commit-based counts. "
        "The two won't match: someone can file issues without ever committing, "
        "or vice versa._",
        "",
        f"- New issue/PR authors this period: **{contrib['new_contributor_count']}**",
        "",
        "| | New (issue/PR authors) | Existing (issue/PR authors) |",
        "|---|---|---|",
        f"| Issues created | {contrib['issues_new']} | {contrib['issues_existing']} |",
        f"| PRs raised | {contrib['prs_new']} | {contrib['prs_existing']} |",
        "",
    ]

    lines += ["### Open issue staleness", ""]
    lines += ["| No activity for at least | Open issue count |", "|---|---|"]
    for threshold, count in sorted(buckets.items()):
        lines.append(f"| {threshold} days | {count} |")
    lines.append(f"\n_Based on {len(ages)} currently open issues._\n")

    lines += ["### Issue creation \u2192 linked PR merge time", ""]
    if merge_durations:
        lines += [
            f"- Sample size: {len(merge_durations)} issue/PR pairs",
            f"- Mean: {fmt_duration(mean(merge_durations))}",
            f"- Median: {fmt_duration(median(merge_durations))}",
            "",
            "_Only counts PRs merged in this window that used a closing "
            "keyword (e.g. \"Fixes #123\") or the linked-issue sidebar._",
        ]
    else:
        lines.append("_No merged PRs with linked issues found in this window._")

    return "\n".join(lines) + "\n"


def main() -> None:
    token = env("GH_TOKEN", required=True)
    repo = env("REPOSITORY", required=True)
    start = env("START_DATE", required=True)
    end = env("END_DATE", required=True)
    thresholds = [int(x) for x in env("STALE_THRESHOLDS", "3,7,14,30").split(",")]
    output_path = env("CUSTOM_METRICS_OUTPUT", "custom_metrics.md")

    session = make_session(token)

    print(f"Computing new vs. existing contributor breakdown for {repo} ({start}..{end})...")
    contrib = build_new_vs_existing(session, repo, start, end)

    print("Computing open issue staleness buckets...")
    ages = open_issue_ages_days(session, repo)
    buckets = staleness_buckets(ages, thresholds)

    print("Computing issue-to-PR merge time...")
    merge_durations = issue_to_merge_durations_seconds(session, repo, start, end)

    markdown = build_markdown(repo, start, end, contrib, ages, buckets, merge_durations)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
