#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reset_participant.py — wipe a single participant's data for a specific phase.

Lets the experimenter re-run a phase from scratch without touching other
participants or other phases.

Usage:
    python reset_participant.py --participant P01 --phase 1
    python reset_participant.py --participant P01 --phase 2
    python reset_participant.py --participant P01 --phase 3
    python reset_participant.py --participant P01 --phase 3 --condition in_context
    python reset_participant.py --participant P01 --all

Add --yes to skip the interactive confirmation.

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent

CONDITIONS = ("in_context", "retrieval", "parametric")


# ───────────────────────── path helpers ─────────────────────────

def participant_dir(pid: str) -> Path:
    return REPO_ROOT / "participants" / pid


def status_path(pid: str) -> Path:
    return participant_dir(pid) / "status.json"


def phase1_targets(pid: str) -> list[Path]:
    """Phase 1 conversation logs live under participants/<pid>/phase1/."""
    return [participant_dir(pid) / "phase1"]


def phase2_targets(pid: str) -> list[Path]:
    """Phase 2 build artifacts: memory JSON (x3), FAISS index + mapping, LoRA weights."""
    md = REPO_ROOT / "memory_data"
    mw = REPO_ROOT / "model_weights"
    return [
        md / "in_context" / f"{pid}_memory.json",
        md / "retrieval" / f"{pid}_memory.json",
        md / "retrieval" / "index" / f"{pid}_faiss.index",
        md / "retrieval" / "mapping" / f"{pid}_mapping.json",
        md / "parametric" / f"{pid}_memory.json",
        md / "parametric" / f"{pid}_memory_QA.json",
        mw / f"{pid}_lora_raw",
        mw / f"{pid}_lora_persona.gguf",
    ]


def phase3_targets(pid: str, condition: str | None) -> list[Path]:
    """Phase 3 per-condition logs under participants/<pid>/phase3/<cond>_log.json."""
    p3 = participant_dir(pid) / "phase3"
    if condition:
        return [p3 / f"{condition}_log.json"]
    return [p3 / f"{c}_log.json" for c in CONDITIONS]


def all_targets(pid: str) -> list[Path]:
    """Full wipe: participant dir plus every external artifact."""
    targets: list[Path] = [participant_dir(pid)]
    targets.extend(phase2_targets(pid))
    return targets


# ───────────────────────── status updates ─────────────────────────

def load_status(pid: str) -> dict | None:
    sp = status_path(pid)
    if not sp.exists():
        return None
    try:
        with sp.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [warn] could not read status.json: {e}")
        return None


def save_status(pid: str, status: dict) -> None:
    sp = status_path(pid)
    sp.parent.mkdir(parents=True, exist_ok=True)
    with sp.open("w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def reset_status_phase1(pid: str) -> None:
    s = load_status(pid)
    if s is None:
        return
    s["phase1_done"] = False
    s.pop("phase1_ended_at", None)
    s.pop("phase1_usage_total", None)
    save_status(pid, s)
    print(f"  [status] phase1_done -> False, phase1_usage_total cleared")


def reset_status_phase2(pid: str) -> None:
    s = load_status(pid)
    if s is None:
        return
    s["phase2_status"] = "not_started"
    s.pop("phase2_error", None)
    save_status(pid, s)
    print(f"  [status] phase2_status -> not_started")


def reset_status_phase3(pid: str, condition: str | None) -> None:
    s = load_status(pid)
    if s is None:
        return
    completed = s.get("conditions_completed", [])
    usage_totals = s.get("phase3_usage_total") or {}
    if condition:
        if condition in completed:
            completed.remove(condition)
            print(f"  [status] removed {condition!r} from conditions_completed")
        if condition in usage_totals:
            usage_totals.pop(condition, None)
            print(f"  [status] removed phase3_usage_total[{condition!r}]")
    else:
        if completed:
            print(f"  [status] cleared conditions_completed ({completed})")
        completed = []
        if usage_totals:
            print(f"  [status] cleared phase3_usage_total ({list(usage_totals.keys())})")
        usage_totals = {}
    s["conditions_completed"] = completed
    s["phase3_usage_total"] = usage_totals
    save_status(pid, s)


# ───────────────────────── deletion ─────────────────────────

def delete_path(p: Path) -> None:
    """Delete a file or directory; no-op if missing."""
    if not p.exists() and not p.is_symlink():
        print(f"  [skip] {p} (not found)")
        return
    try:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"  [del]  {p}")
    except OSError as e:
        print(f"  [err]  {p}: {e}")


def existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists() or p.is_symlink()]


# ───────────────────────── confirmation ─────────────────────────

def confirm(pid: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        typed = input(f"Type the participant id to confirm ({pid}): ").strip()
    except EOFError:
        return False
    if typed != pid:
        print("Confirmation mismatch — aborting.")
        return False
    return True


# ───────────────────────── main ─────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wipe a single participant's data for a specific phase."
    )
    parser.add_argument("-p", "--participant", required=True, help="Participant id, e.g. P01")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase", type=int, choices=(1, 2, 3))
    group.add_argument("--all", action="store_true", help="Wipe everything for this participant")
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        help="Only for --phase 3: restrict wipe to one condition",
    )
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    args = parser.parse_args()

    pid = args.participant

    if args.condition and args.phase != 3:
        parser.error("--condition is only valid with --phase 3")

    # Plan targets + status mutation.
    if args.all:
        label = f"ALL data for participant {pid}"
        targets = all_targets(pid)
        status_action = None  # participant dir will be removed entirely
    elif args.phase == 1:
        label = f"Phase 1 data for {pid}"
        targets = phase1_targets(pid)
        status_action = reset_status_phase1
    elif args.phase == 2:
        label = f"Phase 2 build artifacts for {pid}"
        targets = phase2_targets(pid)
        status_action = reset_status_phase2
    else:  # phase 3
        cond_label = args.condition if args.condition else "all 3 conditions"
        label = f"Phase 3 data ({cond_label}) for {pid}"
        targets = phase3_targets(pid, args.condition)
        status_action = lambda p: reset_status_phase3(p, args.condition)  # noqa: E731

    present = existing(targets)
    missing = [p for p in targets if p not in present]

    print(f"\nAbout to reset: {label}")
    print("Repo root:", REPO_ROOT)
    print("\nTargets that WILL be deleted:")
    if present:
        for p in present:
            kind = "dir " if p.is_dir() else "file"
            print(f"  [{kind}] {p}")
    else:
        print("  (none found)")
    if missing:
        print("\nTargets that do not exist (will be skipped):")
        for p in missing:
            print(f"  [miss] {p}")

    if not present:
        print("\nNothing to delete. Exiting.")
        return 0

    print()
    if not confirm(pid, args.yes):
        return 1

    print("\nDeleting...")
    for p in present:
        delete_path(p)

    if status_action is not None:
        print("\nUpdating status.json...")
        status_action(pid)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
