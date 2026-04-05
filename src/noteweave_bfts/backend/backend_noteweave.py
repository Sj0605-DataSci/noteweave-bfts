"""
backend_noteweave.py — LLM backend that routes calls through the NoteWeave
control channel. The runner (CLI/extension) receives the request and forwards
it to the NoteWeave backend server's /bfts/llm endpoint.

This replaces direct OpenAI/Anthropic/Ollama calls when BFTS runs as a tool
inside the NoteWeave ecosystem.
"""

from __future__ import annotations

import logging
import uuid

from .utils import FunctionSpec, OutputType, PromptType, opt_messages_to_list

logger = logging.getLogger("noteweave-bfts")


def query(
    system_message: str | None,
    user_message: str | None,
    model: str = "noteweave",
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    Send an LLM query through the control channel to the NoteWeave backend.

    Returns: (output, req_time, in_tokens, out_tokens, info)
    Same signature as other backends for drop-in compatibility.
    """
    from ..control import get_channel

    channel = get_channel()
    req_id = str(uuid.uuid4())[:8]

    content = channel.request_llm(
        req_id=req_id,
        system=system_message,
        user=user_message,
        model=model,
        temperature=temperature or 0.7,
        func_spec=func_spec.to_dict() if func_spec else None,
        max_tokens=max_tokens,
    )

    # If func_spec was requested, try to parse as JSON
    if func_spec is not None:
        import json
        try:
            return json.loads(content), 0, 0, 0, {}
        except json.JSONDecodeError:
            pass

    return content, 0, 0, 0, {}


def get_ai_client(model: str = "noteweave", **kwargs):
    """No client needed — calls go through the control channel."""
    return None
