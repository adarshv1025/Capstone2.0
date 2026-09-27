"""
llm_client.py

Thin, provider-agnostic wrapper around LiteLLM (CLAUDE.md section 3.3).
Every agent imports `chat` / `chat_json` from here instead of calling
litellm.completion() directly, so switching providers is a .env change,
never a code change.

Model selection:
    LLM_MODEL env var, e.g. "anthropic/claude-sonnet-5" or "openai/gpt-4o".
    LiteLLM picks the matching API key (ANTHROPIC_API_KEY, OPENAI_API_KEY,
    ...) off the model string's prefix automatically.
"""

import json
import os
import random
import time

from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import RateLimitError

load_dotenv()

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-5")


def chat(messages: list[dict], model: str = None, max_retries: int = 5, retry_backoff_s: float = 3.0, **kwargs) -> str:
    """
    Plain text completion. Returns the assistant's message content.

    Retries on RateLimitError with exponential backoff + jitter (3s, 6s,
    12s, 24s, 48s baseline by default, each randomized +-30%) before giving
    up -- a real pipeline chains several agents' worth of calls back to
    back (Route Agents, then the Coordinator's negotiation rounds, then the
    Carbon Optimizer's propose/reflect calls), which can add up to more
    tokens/minute than a free-tier provider allows even though no single
    call is oversized.

    The jitter matters specifically for a burst of concurrent calls (e.g.
    pipeline.run_batch's ThreadPoolExecutor firing several Route Agent
    calls at once): found via testing that a batch of 12 requests
    consistently failed 4-5 of them even on a freshly-reset daily quota --
    not a cumulative-budget problem, a thundering-herd one. Several threads
    hit the per-minute cap in the same instant, all retry after the exact
    same fixed backoff, collide on the limit again together, and some
    exhaust max_retries while still colliding in lockstep. Randomizing each
    thread's wait spreads retries out so they stop re-colliding as a group.
    Only RateLimitError is retried; other errors (bad request, auth, etc.)
    propagate immediately since retrying won't fix them.
    """
    for attempt in range(max_retries + 1):
        try:
            response = completion(model=model or DEFAULT_MODEL, messages=messages, **kwargs)
            return response.choices[0].message.content
        except RateLimitError:
            if attempt == max_retries:
                raise
            backoff = retry_backoff_s * (2 ** attempt)
            time.sleep(backoff * random.uniform(0.7, 1.3))


def chat_json(messages: list[dict], model: str = None, **kwargs) -> dict:
    """
    Completion where the model is expected to return a single JSON object.
    Strips markdown code fences if the model wraps its output in them.
    Raises ValueError (with the raw content attached) on parse failure so
    the caller can decide whether to retry or fall back to a default.
    """
    content = chat(messages, model=model, **kwargs)
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {content!r}") from e
