"""
main.py — Madagascar IPC Food Security Pipeline Orchestrator
============================================================
Runs the full pipeline end-to-end or individual stages.

Usage:
    python main.py                        # full pipeline
    python main.py --steps ingest         # ingestion only
    python main.py --steps preprocess     # preprocessing only
    python main.py --steps features       # feature engineering only
    python main.py --steps train          # training only
    python main.py --steps evaluate       # evaluation only
    python main.py --steps ingest preprocess features train evaluate
    python main.py --skip ingest          # full pipeline minus ingestion
"""

import argparse
import sys
import os
import time
from datetime import datetime, timezone

# ── make sure repo root is on the path ───────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_ingestion  import run_ingestion
from src.preprocessing   import run_preprocessing
from src.features        import run_features
from src.train           import run_training
from src.evaluate        import run_evaluation


# ── pipeline stages (ordered) ─────────────────────────────────────────────────

STAGES = ["ingest", "preprocess", "features", "train", "evaluate"]

STAGE_FNS = {
    "ingest":     run_ingestion,
    "preprocess": run_preprocessing,
    "features":   lambda: run_features("cs"),
    "train":      run_training,
    "evaluate":   run_evaluation,
}

STAGE_DESCRIPTIONS = {
    "ingest":     "Fetch raw data from FEWS NET API",
    "preprocess": "Clean, harmonise and split by scenario",
    "features":   "Lag/rolling/unit features + encoding (CS scenario)",
    "train":      "Train binary + multiclass XGBoost classifiers",
    "evaluate":   "Evaluate models, save metrics and plots",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _separator(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _success(stage: str, elapsed: float):
    print(f"\n  ✓  [{stage}] completed in {elapsed:.1f}s")


def _failure(stage: str, err: Exception):
    print(f"\n  ✗  [{stage}] FAILED: {err}")


def run_pipeline(steps: list[str]):
    started_at = datetime.now(timezone.utc)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Madagascar IPC Food Security — Pipeline Orchestrator   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Started : {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Stages  : {' → '.join(steps)}")

    results = {}

    for stage in steps:
        _separator(f"STAGE: {stage.upper()}  —  {STAGE_DESCRIPTIONS[stage]}")
        t0 = time.time()
        try:
            STAGE_FNS[stage]()
            elapsed = time.time() - t0
            _success(stage, elapsed)
            results[stage] = {"status": "ok", "elapsed_s": round(elapsed, 1)}
        except Exception as e:
            elapsed = time.time() - t0
            _failure(stage, e)
            results[stage] = {"status": "failed", "error": str(e), "elapsed_s": round(elapsed, 1)}
            print("\n  Pipeline halted. Fix the error above and re-run.")
            print(f"  To resume from this stage:  python main.py --steps {stage}")
            _print_summary(results, started_at)
            sys.exit(1)

    _print_summary(results, started_at)


def _print_summary(results: dict, started_at: datetime):
    total = time.time() - started_at.timestamp()
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                     PIPELINE SUMMARY                    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for stage, info in results.items():
        icon   = "✓" if info["status"] == "ok" else "✗"
        status = info["status"].upper()
        secs   = info["elapsed_s"]
        desc   = STAGE_DESCRIPTIONS.get(stage, "")
        print(f"║  {icon}  {stage:<12} {status:<8} {secs:>6.1f}s   {desc[:28]:<28} ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Total elapsed: {total:>6.1f}s" + " " * 38 + "║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    failed = [s for s, r in results.items() if r["status"] == "failed"]
    if not failed:
        print("  Pipeline completed successfully.")
        print("  Next step: serve the API")
        print("    source venv/bin/activate")
        print("    PYTHONPATH=$(pwd) uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload\n")
    else:
        print(f"  Pipeline failed at stage(s): {', '.join(failed)}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Madagascar IPC Food Security Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--steps", nargs="+", choices=STAGES, metavar="STAGE",
        help=f"Run only these stages (choices: {', '.join(STAGES)})",
    )
    group.add_argument(
        "--skip", nargs="+", choices=STAGES, metavar="STAGE",
        help="Run full pipeline except these stages",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available stages and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable pipeline stages (in order):\n")
        for s in STAGES:
            print(f"  {s:<14} {STAGE_DESCRIPTIONS[s]}")
        print()
        return

    if args.steps:
        # preserve canonical order even if user specifies out of order
        steps = [s for s in STAGES if s in args.steps]
    elif args.skip:
        steps = [s for s in STAGES if s not in args.skip]
    else:
        steps = STAGES  # full pipeline

    if not steps:
        print("No stages selected. Run with --list to see options.")
        sys.exit(1)

    run_pipeline(steps)


if __name__ == "__main__":
    main()