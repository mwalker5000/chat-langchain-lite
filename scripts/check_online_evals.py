"""Check that the project's online evaluators are actually scoring traces.

An online evaluator whose run rule errors still writes an `auto_eval` feedback
row — but with n=0, errors=1 and avg=0, which in the UI is easy to misread as a
genuine score of 0. This script keeps the two apart: for each evaluator key
registered by `scripts.setup` it counts scored rows and error rows on recent
root runs, and exits 1 when a key has error rows but nothing scored.

See the "Online Evaluators" section of README.md for what to do when a key comes
back broken.

Usage:
    python -m scripts.check_online_evals                 # last 48h
    python -m scripts.check_online_evals --hours 6
    python -m scripts.check_online_evals --limit 500
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv(override=True)

from scripts.setup import EVALUATORS
PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "chat-lc-lite")

EVALUATOR_KEYS = [ev["feedback_key"] for ev in EVALUATORS]


def collect_counts(hours: int, limit: int) -> tuple[dict, int]:
    """Count scored and errored feedback rows per evaluator key on recent root runs."""
    from langsmith import Client

    client = Client()
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    counts = {key: {"scored": 0, "errors": 0} for key in EVALUATOR_KEYS}

    runs = 0
    for run in client.list_runs(
        project_name=PROJECT_NAME,
        start_time=start_time,
        is_root=True,
        select=["id", "start_time", "feedback_stats"],
        limit=limit,
    ):
        runs += 1
        for key, stats in (run.feedback_stats or {}).items():
            if key not in counts:
                continue
            counts[key]["scored"] += stats.get("n") or 0
            counts[key]["errors"] += stats.get("errors") or 0

    return counts, runs


def report(counts: dict, runs: int, hours: int) -> list:
    """Print a per-key summary and return the keys that errored without ever scoring."""
    print(f"\nOnline evaluators on '{PROJECT_NAME}' — {runs} root run(s) in the last {hours}h\n")

    broken = []
    for key in EVALUATOR_KEYS:
        scored = counts[key]["scored"]
        errors = counts[key]["errors"]
        if scored == 0 and errors > 0:
            broken.append(key)
            status = "❌ ERRORED, never scored"
        elif scored == 0:
            status = "—  no feedback rows"
        elif errors > 0:
            status = "⚠️  scoring, some errors"
        else:
            status = "✅ scoring"
        print(f"  {key:<22} scored={scored:<5} errors={errors:<5} {status}")

    return broken


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hours",
        type=int,
        default=48,
        help="How far back to look for scored traces (default: 48).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of root runs to inspect (default: 200).",
    )
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        print("Error: LANGSMITH_API_KEY not set.")
        sys.exit(1)

    counts, runs = collect_counts(args.hours, args.limit)
    broken = report(counts, runs, args.hours)

    if broken:
        print(
            f"\n{len(broken)} evaluator(s) are erroring instead of scoring: "
            f"{', '.join(broken)}."
        )
        print("  Open each run rule in LangSmith → Evaluators and read the evaluator")
        print("  run error — a judge-model 403 or an unresolvable variable_mapping key")
        print("  produces exactly this signature. See README.md → Online Evaluators.")
        sys.exit(1)

    print("\nAll online evaluators are producing scores.")


if __name__ == "__main__":
    main()
