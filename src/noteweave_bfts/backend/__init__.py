"""
BFTS LLM backend router.

When running under NoteWeave (NOTEWEAVE_BFTS_BACKEND=noteweave), all LLM
calls route through the control channel to the NoteWeave server.
Otherwise falls back to direct OpenAI/Anthropic/Ollama calls.
"""

import os

from . import backend_anthropic, backend_ollama, backend_openai, backend_noteweave
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

# When launched by the NoteWeave runner, this env var is set
_USE_NOTEWEAVE = os.environ.get("NOTEWEAVE_BFTS_BACKEND", "").lower() == "noteweave"


def get_ai_client(model: str, **model_kwargs):
    if _USE_NOTEWEAVE:
        return None
    if "claude-" in model:
        return backend_anthropic.get_ai_client(model=model, **model_kwargs)
    elif model.startswith("ollama/"):
        return None
    else:
        return backend_openai.get_ai_client(model=model, **model_kwargs)


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query — routes to the appropriate backend.
    Under NoteWeave, all calls go through the control channel.
    """
    # NoteWeave mode: route everything through the control channel
    if _USE_NOTEWEAVE:
        return backend_noteweave.query(
            system_message=compile_prompt_to_md(system_message) if system_message else None,
            user_message=compile_prompt_to_md(user_message) if user_message else None,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            func_spec=func_spec,
        )

    # Direct mode: route by model name
    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    if model.startswith("o1"):
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs["max_completion_tokens"] = 100000
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens

    if "claude-" in model:
        query_func = backend_anthropic.query
    elif model.startswith("ollama/"):
        query_func = backend_ollama.query
    else:
        query_func = backend_openai.query

    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
