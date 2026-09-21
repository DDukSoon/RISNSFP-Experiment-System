# -*- coding: utf-8 -*-
"""HCI voice-experiment server entry point.

Usage:
    python server.py -p P001 --phase 1
    python server.py -p P001 --phase 3 -c {0|1|2}

Output is redirected to ``participants/<pid>/server_phase<N>_<ts>.log``
unless ``--debug`` is passed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


class _TeeStream:
    """Mirror writes to a log file (and optionally still echo to original)."""

    def __init__(self, log_file, mirror=None):
        self._log = log_file
        self._mirror = mirror

    def write(self, data):
        try:
            self._log.write(data)
            self._log.flush()
        except Exception:
            pass
        if self._mirror is not None:
            try:
                self._mirror.write(data)
            except Exception:
                pass

    def flush(self):
        try:
            self._log.flush()
        except Exception:
            pass
        if self._mirror is not None:
            try:
                self._mirror.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        return self._log.fileno()

    def writable(self):
        return True

    def readable(self):
        return False

    @property
    def encoding(self):
        return getattr(self._log, "encoding", "utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HCI experiment voice server (single-endpoint).",
    )
    parser.add_argument("-p", "--participant", required=True,
                        help="Participant ID, e.g. P01")
    parser.add_argument("--phase", required=True, choices=["1", "3"],
                        help="Experiment phase (1 or 3)")
    parser.add_argument("-c", "--c", dest="condition_index",
                        required=False, type=int, choices=[0, 1, 2],
                        help="Phase 3 condition index into status.json "
                             "condition_order (0/1/2). Required when --phase 3.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Echo server stdout/stderr to the console "
                             "(by default redirected to "
                             "participants/<pid>/server_<phase>_<ts>.log)")
    args = parser.parse_args(argv)

    if args.phase == "3" and args.condition_index is None:
        parser.error("-c/--c (condition index 0/1/2) is required when --phase 3")
    if args.phase == "1" and args.condition_index is not None:
        parser.error("-c/--c must not be provided when --phase 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from app.config import Settings, require_elevenlabs_key
    from app.store import SessionStore

    require_elevenlabs_key()

    settings = Settings.load()
    host = args.host or settings.host
    port = args.port or settings.port

    log_path: Path | None = None
    if not args.debug:
        log_dir = Path(settings.base_dir) / "participants" / args.participant
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"server_phase{args.phase}_{ts}.log"
        log_file = open(log_path, "a", encoding="utf-8")
        sys.stdout = _TeeStream(log_file)
        sys.stderr = _TeeStream(log_file)
        print(f"[server] All output is written to {log_path} "
              f"(pass --debug to echo to console).",
              file=sys.__stdout__, flush=True)

    # Resolve condition index -> name only for Phase 3.
    condition_name: str | None = None
    if args.phase == "3":
        store = SessionStore(settings.base_dir)
        status = store.get_status(args.participant)  # auto-generated from a Latin square if absent
        order = status.get("condition_order") or []
        if args.condition_index >= len(order):
            print(f"[server] ERROR: condition_index={args.condition_index} is "
                  f"out of range for condition_order (len={len(order)})")
            return 2
        condition_name = order[args.condition_index]
        print(f"[server] participant={args.participant} condition_order={order}")
        print(f"[server] -c {args.condition_index} → condition='{condition_name}'")

    import uvicorn

    from app.server import create_app

    app = create_app(
        participant=args.participant,
        phase=args.phase,
        condition=condition_name,
        settings=settings,
    )
    uvicorn_kwargs = {"host": host, "port": port}
    if not args.debug:
        uvicorn_kwargs["log_level"] = "warning"
        uvicorn_kwargs["access_log"] = False
    uvicorn.run(app, **uvicorn_kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
