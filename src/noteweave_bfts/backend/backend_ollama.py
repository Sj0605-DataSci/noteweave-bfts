"""
Ollama backend for treesearch — pure curl, streaming, no timeout.

Replaces backend_openai.py for all ollama/ prefixed models.
Uses Ollama's native /api/chat endpoint via curl subprocess (no openai dependency).
Streams responses token-by-token and assembles the full output.
"""
import json
import logging
import os
import subprocess
import sys
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list

logger = logging.getLogger("noteweave-bfts")

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _gpu_env_for_model(model: str) -> dict[str, str]:
    """Return env with CUDA_VISIBLE_DEVICES set if OLLAMA_GPU_MAP maps this model."""
    raw = os.environ.get("OLLAMA_GPU_MAP", "").strip()
    if not raw:
        return {}
    bare = model.replace("ollama/", "")
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.rsplit(":", 1)
        if len(parts) != 2:
            continue
        name, gpu_str = parts[0].strip(), parts[1].strip()
        if name == bare or name == bare.split(":")[0]:
            try:
                gpu_ids = [int(g.strip()) for g in gpu_str.split("+")]
                return {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids)}
            except ValueError:
                continue
    return {}


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query Ollama via curl streaming. No timeout — waits as long as needed.
    Streams tokens to stderr so progress is visible during long generations.
    Returns (output, req_time, in_tokens, out_tokens, info).
    """
    messages = opt_messages_to_list(system_message, user_message)

    model = model_kwargs.get("model", "")
    bare_model = model.replace("ollama/", "") if model.startswith("ollama/") else model

    payload = {
        "model": bare_model,
        "messages": messages,
        "stream": True,
        "options": {},
    }

    if model_kwargs.get("temperature") is not None:
        payload["options"]["temperature"] = model_kwargs["temperature"]
    if model_kwargs.get("max_tokens") is not None:
        payload["options"]["num_predict"] = model_kwargs["max_tokens"]

    # For function calling, use Ollama's format parameter with JSON schema
    if func_spec is not None:
        payload["format"] = func_spec.json_schema
        # Add instruction to respond as the function
        fn_instruction = (
            f"\n\nYou MUST respond with a JSON object matching this schema for the function '{func_spec.name}': "
            f"{json.dumps(func_spec.json_schema)}\n"
            f"Description: {func_spec.description}\n"
            f"Respond ONLY with the JSON object, no other text."
        )
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += fn_instruction
        elif messages and messages[-1]["role"] == "system":
            messages[-1]["content"] += fn_instruction
        else:
            messages.append({"role": "user", "content": fn_instruction})

    cmd = [
        "curl", "-s", "--no-buffer",
        "-X", "POST", f"{OLLAMA_BASE}/api/chat",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
    ]

    t0 = time.time()
    env_override = _gpu_env_for_model(model)
    proc_env = {**os.environ, **env_override} if env_override else None
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=proc_env
    )

    full_content = ""
    in_tokens = 0
    out_tokens = 0
    model_name = bare_model
    created = 0

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            chunk = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        delta = chunk.get("message", {}).get("content", "")
        full_content += delta

        # Stream to stderr so user sees progress
        if delta:
            sys.stderr.write(delta)
            sys.stderr.flush()

        if chunk.get("done"):
            in_tokens = chunk.get("prompt_eval_count", 0)
            out_tokens = chunk.get("eval_count", 0)
            model_name = chunk.get("model", bare_model)
            created = chunk.get("created_at", "")
            break

    proc.wait()
    sys.stderr.write("\n")
    sys.stderr.flush()

    req_time = time.time() - t0

    if func_spec is not None:
        try:
            output = json.loads(full_content)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding function call JSON from Ollama: {full_content[:500]}")
            raise e
    else:
        output = full_content

    info = {
        "system_fingerprint": None,
        "model": model_name,
        "created": created,
    }

    return output, req_time, in_tokens, out_tokens, info
