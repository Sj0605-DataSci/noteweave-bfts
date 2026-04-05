"""
llm_compat.py — Slim reimplementation of noteweave.llm functions used by
log_summarization.py. Routes through the BFTS backend (which in NoteWeave
mode goes through the control channel).
"""

from __future__ import annotations

import json
import re
from typing import Any

from .backend import query, compile_prompt_to_md


def get_response_from_llm(
    prompt: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict] | None = None,
    temperature: float = 0.7,
) -> tuple[str, list[dict[str, Any]]]:
    """Call LLM via the BFTS backend and return (content, msg_history)."""
    if msg_history is None:
        msg_history = []

    # Build user message from history + prompt
    user_text = prompt
    if msg_history:
        # Append prior history context
        parts = []
        for m in msg_history:
            parts.append(f"{m['role']}: {m['content']}")
        parts.append(f"user: {prompt}")
        user_text = "\n\n".join(parts)

    output = query(
        system_message=system_message,
        user_message=user_text,
        model=model,
        temperature=temperature,
    )

    # query() returns either str or tuple depending on backend
    if isinstance(output, tuple):
        content = output[0]
    else:
        content = output

    new_history = msg_history + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": content},
    ]
    return content, new_history


def extract_json_between_markers(llm_output: str) -> dict | None:
    """Extract JSON from LLM output (between ```json...``` or <JSON>...</JSON> markers)."""
    llm_output = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.DOTALL).strip()

    matches = re.findall(r"```json\s*(.*?)```", llm_output, re.DOTALL)
    if not matches:
        matches = re.findall(r"<JSON>\s*(.*?)\s*</JSON>", llm_output, re.DOTALL)
    if not matches:
        matches = re.findall(r"```\s*(.*?)```", llm_output, re.DOTALL)
    if not matches:
        matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                return json.loads(json_string_clean)
            except json.JSONDecodeError:
                continue

    return None
