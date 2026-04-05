"""
control.py — JSON-lines control protocol for runner ↔ BFTS communication.

The BFTS process communicates with its runner (CLI bfts_runner.py or
extension BftsRunner.ts) via JSON lines on stdout (BFTS → runner) and
stdin (runner → BFTS).

## BFTS → Runner (stdout)

    {"type": "status", "stage": "stage_1", "node_id": 3, "metric": 0.83, "total_nodes": 5}
    {"type": "log", "level": "info", "message": "Generating code..."}
    {"type": "llm_request", "req_id": "abc123", "system": "...", "user": "...", "model": "...", "temperature": 0.7}
    {"type": "done", "success": true, "summary": {...}}
    {"type": "error", "message": "..."}

## Runner → BFTS (stdin)

    {"type": "llm_response", "req_id": "abc123", "content": "..."}
    {"type": "command", "action": "stop"}
    {"type": "command", "action": "query", "target": "journal"}
    {"type": "command", "action": "query", "target": "logs", "stage": "stage_1"}

## Query responses (stdout)

    {"type": "query_result", "target": "journal", "data": {...}}
    {"type": "query_result", "target": "logs", "data": {...}}
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import queue
from typing import Any, Callable

logger = logging.getLogger("noteweave-bfts")


class ControlChannel:
    """
    Bidirectional JSON-lines channel over stdin/stdout.

    - send(): write JSON line to stdout (BFTS → runner)
    - recv(): read JSON line from stdin (runner → BFTS), blocking
    - Incoming commands are dispatched to registered handlers.
    - LLM responses are matched to pending requests by req_id.
    """

    def __init__(self) -> None:
        self._llm_responses: dict[str, queue.Queue] = {}
        self._command_handlers: dict[str, Callable] = {}
        self._stop_event = threading.Event()
        self._stdin_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Separate stdout for control messages (original stdout)
        # Redirect Python's stdout so print() doesn't corrupt the protocol
        self._control_out = sys.stdout
        sys.stdout = sys.stderr  # all print() goes to stderr

    def start(self) -> None:
        """Start the stdin listener thread."""
        self._stdin_thread = threading.Thread(target=self._listen_stdin, daemon=True)
        self._stdin_thread.start()

    def stop(self) -> None:
        """Signal stop."""
        self._stop_event.set()

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def send(self, msg: dict) -> None:
        """Send a JSON message to the runner."""
        line = json.dumps(msg, default=str)
        with self._lock:
            self._control_out.write(line + "\n")
            self._control_out.flush()

    def send_status(self, stage: str, node_id: int, metric: float | None,
                    total_nodes: int, buggy_nodes: int = 0) -> None:
        self.send({
            "type": "status",
            "stage": stage,
            "node_id": node_id,
            "metric": metric,
            "total_nodes": total_nodes,
            "buggy_nodes": buggy_nodes,
        })

    def send_log(self, message: str, level: str = "info") -> None:
        self.send({"type": "log", "level": level, "message": message})

    def send_done(self, success: bool, summary: dict | None = None) -> None:
        self.send({"type": "done", "success": success, "summary": summary or {}})

    def send_error(self, message: str) -> None:
        self.send({"type": "error", "message": message})

    def send_query_result(self, target: str, data: Any) -> None:
        self.send({"type": "query_result", "target": target, "data": data})

    def request_llm(self, req_id: str, system: str | None, user: str | None,
                    model: str, temperature: float = 0.7,
                    func_spec: dict | None = None,
                    max_tokens: int | None = None) -> str:
        """
        Send an LLM request to the runner and block until the response arrives.
        The runner forwards this to the NoteWeave backend /bfts/llm endpoint.
        Returns the LLM response content string.
        """
        response_queue: queue.Queue = queue.Queue()
        self._llm_responses[req_id] = response_queue

        self.send({
            "type": "llm_request",
            "req_id": req_id,
            "system": system,
            "user": user,
            "model": model,
            "temperature": temperature,
            "func_spec": func_spec,
            "max_tokens": max_tokens,
        })

        # Block until response arrives (or stop signaled)
        try:
            result = response_queue.get(timeout=600)  # 10 min timeout
            return result
        except queue.Empty:
            raise TimeoutError(f"LLM request {req_id} timed out (600s)")
        finally:
            self._llm_responses.pop(req_id, None)

    def on_command(self, action: str, handler: Callable) -> None:
        """Register a handler for a command action (e.g. 'stop', 'query')."""
        self._command_handlers[action] = handler

    def _listen_stdin(self) -> None:
        """Background thread: read JSON lines from stdin."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "llm_response":
                req_id = msg.get("req_id")
                content = msg.get("content", "")
                q = self._llm_responses.get(req_id)
                if q:
                    q.put(content)

            elif msg_type == "command":
                action = msg.get("action")
                if action == "stop":
                    self._stop_event.set()
                handler = self._command_handlers.get(action)
                if handler:
                    try:
                        handler(msg)
                    except Exception as e:
                        self.send_error(f"Command handler error: {e}")


# Global singleton
_channel: ControlChannel | None = None


def get_channel() -> ControlChannel:
    global _channel
    if _channel is None:
        _channel = ControlChannel()
        _channel.start()
    return _channel
