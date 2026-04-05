"""
noteweave-bfts entry point.

Usage:
    python -m noteweave_bfts --config bfts_config.yaml [--mode noteweave|standalone]

Modes:
    noteweave   — Control protocol on stdin/stdout, LLM calls routed to NoteWeave server.
                  Used when launched by the CLI runner or VS Code extension.
    standalone  — Direct LLM calls (OpenAI/Anthropic/Ollama). For testing without NoteWeave.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger("noteweave-bfts")


def main():
    parser = argparse.ArgumentParser(description="NoteWeave BFTS — Best-First Tree Search")
    parser.add_argument("--config", required=True, help="Path to bfts_config.yaml")
    parser.add_argument(
        "--mode", choices=["noteweave", "standalone"], default="noteweave",
        help="noteweave: control protocol + LLM proxy. standalone: direct LLM calls.",
    )
    args = parser.parse_args()

    # Set backend mode BEFORE importing anything that reads it
    if args.mode == "noteweave":
        os.environ["NOTEWEAVE_BFTS_BACKEND"] = "noteweave"

    # Now safe to import (backend/__init__.py reads the env var at import time)
    from .control import get_channel
    from .perform_experiments_bfts_with_agentmanager import perform_experiments_bfts
    from .utils.config import load_cfg

    if args.mode == "noteweave":
        channel = get_channel()
        _setup_logging_to_channel(channel)
        _register_command_handlers(channel, args.config)
        channel.send_log("BFTS starting in NoteWeave mode")
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        perform_experiments_bfts(args.config)

        if args.mode == "noteweave":
            # Collect summary from logs
            cfg = load_cfg(args.config)
            summary = _collect_summary(cfg)
            channel.send_done(success=True, summary=summary)
    except KeyboardInterrupt:
        if args.mode == "noteweave":
            channel.send_done(success=False, summary={"reason": "interrupted"})
    except Exception as e:
        if args.mode == "noteweave":
            channel.send_error(str(e))
            channel.send_done(success=False, summary={"reason": str(e)})
        else:
            raise


def _setup_logging_to_channel(channel):
    """Route Python logging to the control channel."""

    class ChannelHandler(logging.Handler):
        def emit(self, record):
            try:
                channel.send_log(
                    message=self.format(record),
                    level=record.levelname.lower(),
                )
            except Exception:
                pass

    handler = ChannelHandler()
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _register_command_handlers(channel, config_path: str):
    """Register handlers for runner commands (query, stop)."""
    from .utils.config import load_cfg

    def handle_query(msg):
        target = msg.get("target", "")
        try:
            cfg = load_cfg(config_path)
            if target == "journal":
                data = _get_journal_snapshot(cfg)
            elif target == "logs":
                stage = msg.get("stage", "")
                data = _get_stage_logs(cfg, stage)
            elif target == "config":
                data = {"config_path": config_path}
            else:
                data = {"error": f"Unknown query target: {target}"}
            channel.send_query_result(target, data)
        except Exception as e:
            channel.send_query_result(target, {"error": str(e)})

    channel.on_command("query", handle_query)
    # "stop" is handled automatically by the channel (sets stop_event)


def _get_journal_snapshot(cfg) -> dict:
    """Read the latest journal state from disk."""
    import glob

    log_dir = str(cfg.log_dir)
    journals = {}

    for stage_dir in sorted(glob.glob(os.path.join(log_dir, "stage_*"))):
        stage_name = os.path.basename(stage_dir)
        journal_path = os.path.join(stage_dir, "journal.json")
        if os.path.exists(journal_path):
            with open(journal_path) as f:
                journals[stage_name] = json.load(f)

    return {"stages": journals}


def _get_stage_logs(cfg, stage: str) -> dict:
    """Read logs for a specific stage."""
    log_dir = os.path.join(str(cfg.log_dir), stage)
    if not os.path.isdir(log_dir):
        return {"error": f"Stage dir not found: {log_dir}"}

    files = {}
    for fname in os.listdir(log_dir):
        fpath = os.path.join(log_dir, fname)
        if fname.endswith(".json") and os.path.isfile(fpath):
            try:
                with open(fpath) as f:
                    files[fname] = json.load(f)
            except Exception:
                files[fname] = "(unreadable)"
    return files


def _collect_summary(cfg) -> dict:
    """Collect final summary after BFTS completes."""
    summary: dict = {"log_dir": str(cfg.log_dir)}

    # Check for stage summaries
    for name in ["draft_summary", "baseline_summary", "research_summary", "ablation_summary"]:
        path = os.path.join(str(cfg.log_dir), f"{name}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    summary[name] = json.load(f)
            except Exception:
                pass

    return summary


if __name__ == "__main__":
    main()
