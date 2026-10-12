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

Repeatability (added 2026-10-11, CLAUDE.md 9):
    LLM_TEMPERATURE env var, default 0. Every call is sent at this
    temperature unless the caller passes its own. Measured before this:
    one baseline's on_time_rate moved 0.07 -> 0.27 between two runs with
    identical routes and seed, purely from the Execution agent's LLM judging
    the same deviations differently. Set LLM_TEMPERATURE to an empty string
    for a model that rejects the parameter.

Call log:
    LLM_LOG_PATH env var, default logs/llm_calls.jsonl at the repo root
    (git-ignored). One JSON line per successful call: messages, reply,
    model, temperature, latency, token counts. Set it to an empty string to
    turn logging off.
"""

import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import RateLimitError

load_dotenv()

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-5")

_temperature_env = os.environ.get("LLM_TEMPERATURE", "0").strip()
DEFAULT_TEMPERATURE = float(_temperature_env) if _temperature_env else None

LOG_PATH = os.environ.get("LLM_LOG_PATH", str(Path(__file__).resolve().parent.parent / "logs" / "llm_calls.jsonl"))
_log_lock = threading.Lock()  # Route Agents call chat() from several threads at once


def _log_call(record: dict) -> None:
    if not LOG_PATH:
        return
    try:
        Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # a log that can't be written must never fail a pipeline run


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
    if DEFAULT_TEMPERATURE is not None:
        kwargs.setdefault("temperature", DEFAULT_TEMPERATURE)

    for attempt in range(max_retries + 1):
        try:
            started = time.perf_counter()
            response = completion(model=model or DEFAULT_MODEL, messages=messages, **kwargs)
            content = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            _log_call({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": model or DEFAULT_MODEL,
                "temperature": kwargs.get("temperature"),
                "latency_s": round(time.perf_counter() - started, 3),
                "rate_limit_retries": attempt,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "messages": messages,
                "reply": content,
            })
            return content
        except RateLimitError:
            if attempt == max_retries:
                raise
            backoff = retry_backoff_s * (2 ** attempt)
            time.sleep(backoff * random.uniform(0.7, 1.3))


def align_ids(by_id, expected_ids) -> dict:
    """
    Best-effort repair for an LLM reply keyed by request_id when the model
    used the wrong key. Found while preparing demo runs: gpt-oss-120b
    sometimes copied the prompt's example id (e.g. "req-0") instead of the
    real one ("live-0", "mp-3", ...) -- ~2 in 6 replies for a single-request
    Carbon Optimizer reflection -- so the lookup missed and the agent
    silently fell back to its default (reject / no re-plan).

    Only remaps when it's unambiguous: exactly one requested id is missing
    AND exactly one returned key is unknown. Otherwise returns the dict
    unchanged (callers keep their existing conservative fallbacks).
    """
    if not isinstance(by_id, dict):
        return {}
    expected = list(expected_ids)
    unknown = [k for k in by_id if k not in expected]
    missing = [e for e in expected if e not in by_id]
    if len(unknown) == 1 and len(missing) == 1:
        return {**by_id, missing[0]: by_id[unknown[0]]}
    return by_id


def chat_json(messages: list[dict], model: str = None, max_parse_retries: int = 2, **kwargs) -> dict:
    """
    Completion where the model is expected to return a single JSON object.
    Strips markdown code fences if the model wraps its output in them.
    Raises ValueError (with the raw content attached) on parse failure so
    the caller can decide whether to retry or fall back to a default.

    Re-asks up to `max_parse_retries` times on an empty/unparseable reply
    before raising. Found while preparing demo runs: groq/openai/gpt-oss-120b
    intermittently returns an empty or non-JSON reply (~3 of 10 single-request
    pipeline runs), and callers' conservative fallbacks then silently turned
    that into e.g. "no reflection response -- reject" on a Carbon Optimizer
    candidate that saved 0.5kg for +1s. The same prompt re-asked parsed fine
    4/4 times, so this is transient, not a prompt problem.
    """
    for attempt in range(max_parse_retries + 1):
        call_kwargs = dict(kwargs)
        if attempt and "temperature" not in kwargs and DEFAULT_TEMPERATURE is not None:
            # Re-asking at temperature 0 can return the same unusable reply;
            # nudge it up for the retries only.
            call_kwargs["temperature"] = min(DEFAULT_TEMPERATURE + 0.3 * attempt, 1.0)
        content = chat(messages, model=model, **call_kwargs) or ""
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            if attempt == max_parse_retries:
                raise ValueError(f"Model did not return valid JSON: {content!r}") from e
