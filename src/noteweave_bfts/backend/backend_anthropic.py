import json
import time
import os

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import anthropic


ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.APIStatusError,
)

def get_ai_client(model: str, max_retries=2) -> anthropic.Anthropic:
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_retries=max_retries,
    )
    return client

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    # Strip provider prefix (e.g. "anthropic/claude-sonnet-4-6" -> "claude-sonnet-4-6")
    raw_model = model_kwargs.get("model", "")
    if "/" in raw_model:
        model_kwargs["model"] = raw_model.split("/", 1)[1]

    client = get_ai_client(model_kwargs.get("model"), max_retries=0)

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192  # default for Claude models

    # Anthropic doesn't allow not having user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system message as a separate top-level argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    # ── Tool use (function calling) ──────────────────────────────────────────
    if func_spec is not None:
        tool_def = {
            "name": func_spec.name,
            "description": func_spec.description,
            "input_schema": func_spec.json_schema,
        }
        filtered_kwargs["tools"] = [tool_def]
        filtered_kwargs["tool_choice"] = {"type": "tool", "name": func_spec.name}

        t0 = time.time()
        message = backoff_create(
            client.messages.create,
            ANTHROPIC_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
        req_time = time.time() - t0

        # Extract the tool_use block
        tool_block = next(
            (b for b in message.content if b.type == "tool_use"), None
        )
        if tool_block is None:
            raise RuntimeError(f"Anthropic returned no tool_use block. Content: {message.content}")
        output = tool_block.input  # already a dict

        in_tokens = message.usage.input_tokens
        out_tokens = message.usage.output_tokens
        info = {"stop_reason": message.stop_reason}
        return output, req_time, in_tokens, out_tokens, info

    # ── Plain text completion ─────────────────────────────────────────────────
    t0 = time.time()
    message = backoff_create(
        client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0
    print(filtered_kwargs)

    if "thinking" in filtered_kwargs:
        assert (
            len(message.content) == 2
            and message.content[0].type == "thinking"
            and message.content[1].type == "text"
        )
        output: str = message.content[1].text
    else:
        text_block = next((b for b in message.content if b.type == "text"), None)
        if text_block is None:
            raise RuntimeError(f"Anthropic returned no text block. Content: {message.content}")
        output: str = text_block.text

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
