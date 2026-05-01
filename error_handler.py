"""
Structured error parsing for StataAgent.

Classifies exceptions by source (Stata, HuggingFace, LiteLLM, PydanticAI,
Config) and formats them into a single readable message so the REPL never
dumps a raw traceback at the user.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ErrorSource(Enum):
    STATA = "Stata"
    HUGGINGFACE = "HuggingFace"
    LITELLM = "LiteLLM"
    PYDANTIC_AI = "PydanticAI"
    CONFIG = "Configuration"
    UNKNOWN = "Unknown"


@dataclass
class ParsedError:
    source: ErrorSource
    message: str
    suggestion: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_hf_json_message(text: str) -> str | None:
    """Pull the human-readable message out of a HuggingFace JSON error body."""
    match = re.search(r'\{.*"message"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    try:
        outer = re.search(r'(\{.*\})', text, re.DOTALL)
        if outer:
            data = json.loads(outer.group(1))
            return (
                data.get("error", {}).get("message")
                or data.get("message")
            )
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def _module(exc: BaseException) -> str:
    return type(exc).__module__ or ""


def _qualname(exc: BaseException) -> str:
    return f"{_module(exc)}.{type(exc).__name__}"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(exc: BaseException) -> ParsedError:
    """Return a ParsedError with source attribution and a clean message."""
    qualname = _qualname(exc)
    text = str(exc)

    # --- HuggingFace / LiteLLM layer ----------------------------------------
    if "HuggingFace" in qualname or "huggingface" in qualname.lower():
        hf_msg = _extract_hf_json_message(text) or text
        suggestion = _hf_suggestion(hf_msg)
        return ParsedError(ErrorSource.HUGGINGFACE, hf_msg, suggestion)

    if "litellm" in qualname.lower() or "BadRequestError" in qualname:
        hf_msg = _extract_hf_json_message(text)
        if hf_msg:
            return ParsedError(ErrorSource.HUGGINGFACE, hf_msg, _hf_suggestion(hf_msg))
        return ParsedError(ErrorSource.LITELLM, text, _litellm_suggestion(text))

    # --- PydanticAI / model layer -------------------------------------------
    if "pydantic_ai" in qualname or "ModelHTTPError" in qualname:
        # ModelHTTPError body often contains a nested LiteLLM/HF message
        hf_msg = _extract_hf_json_message(text)
        if hf_msg:
            return ParsedError(ErrorSource.HUGGINGFACE, hf_msg, _hf_suggestion(hf_msg))
        return ParsedError(ErrorSource.PYDANTIC_AI, text, None)

    # --- Stata / pystata layer -----------------------------------------------
    if "pystata" in qualname or "stata" in qualname.lower():
        return ParsedError(ErrorSource.STATA, text, None)

    # Stata errors surfaced as plain strings from stata_tools.py
    if any(token in text.lower() for token in ("r(", "invalid syntax", "not found", "stata error")):
        return ParsedError(ErrorSource.STATA, text, None)

    # --- Config / startup layer ----------------------------------------------
    if isinstance(exc, (FileNotFoundError, EnvironmentError, RuntimeError)):
        return ParsedError(ErrorSource.CONFIG, text, None)

    return ParsedError(ErrorSource.UNKNOWN, text, None)


def _hf_suggestion(msg: str) -> str | None:
    msg_lower = msg.lower()
    if "single tool" in msg_lower or "tool-call" in msg_lower:
        return (
            "This model only accepts one tool call per turn. "
            "Try a model with parallel tool-call support "
            "(e.g. mistralai/Mistral-7B-Instruct-v0.3) or switch provider."
        )
    if "401" in msg or "unauthorized" in msg_lower or "token" in msg_lower:
        return "Check that your API key is valid and the model is accessible with your account."
    if "403" in msg or "forbidden" in msg_lower or "gated" in msg_lower:
        return "This model is gated — accept the license at huggingface.co before using it."
    if "404" in msg or "not found" in msg_lower:
        return "Model ID not found. Verify the model name in config.yaml."
    if "429" in msg or "rate limit" in msg_lower:
        return "Rate limit hit. Wait a moment and try again, or upgrade your HF plan."
    return None


def _litellm_suggestion(msg: str) -> str | None:
    if "api_base" in msg.lower() or "base_url" in msg.lower():
        return "Check that base_url in config.yaml points to a valid /v1 endpoint."
    return None


# ---------------------------------------------------------------------------
# Public formatting API
# ---------------------------------------------------------------------------

def format_error(exc: BaseException) -> str:
    """Return a single multi-line string describing the error for the REPL."""
    parsed = classify(exc)
    lines = [f"[{parsed.source.value} error] {parsed.message}"]
    if parsed.suggestion:
        lines.append(f"Hint: {parsed.suggestion}")
    return "\n".join(lines)


def print_error(exc: BaseException, *, file=None) -> None:
    """Print a formatted error to *file* (default: stderr)."""
    print(format_error(exc), file=file or sys.stderr)
