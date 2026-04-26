"""
HuggingFace-backed StataAgent.

Reads provider / model / auth settings from config.yaml (or a path you pass)
and builds a PydanticAI agent backed by any OpenAI-compatible inference
provider that hosts HuggingFace models.

Supported providers (set in config.yaml):
  huggingface       — HF Serverless Inference API
  together          — Together AI
  fireworks         — Fireworks AI
  groq              — Groq
  ollama            — local Ollama server
  openai_compatible — any other OpenAI-compatible endpoint

Usage (REPL):
  python hf_agent.py                     # uses config.yaml in the same dir
  python hf_agent.py --config my.yaml    # use a different config file
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from stata_tools import (
    StataResult,
    describe_data,
    load_dataset,
    regress,
    run_raw_stata,
    summarize,
    tabulate,
)

# ---------------------------------------------------------------------------
# Known provider endpoints
# ---------------------------------------------------------------------------
_PROVIDER_BASE_URLS: dict[str, str] = {
    "huggingface": "https://api-inference.huggingface.co/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
}

_PROVIDERS_WITHOUT_AUTH = {"ollama"}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    provider: str
    model: str
    api_key_env: str | None
    base_url: str | None
    temperature: float
    max_tokens: int

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "AgentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                "Copy config.yaml and edit it, or pass --config <path>."
            )
        with path.open() as f:
            raw = yaml.safe_load(f)

        provider = str(raw.get("provider", "huggingface")).lower()
        supported = set(_PROVIDER_BASE_URLS) | {"openai_compatible"}
        if provider not in supported:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose one of: {', '.join(sorted(supported))}"
            )

        return cls(
            provider=provider,
            model=str(raw["model"]),
            api_key_env=raw.get("api_key_env") or None,
            base_url=raw.get("base_url") or None,
            temperature=float(raw.get("temperature", 0.3)),
            max_tokens=int(raw.get("max_tokens", 4096)),
        )

    def apply_overrides(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Apply CLI overrides on top of the YAML values for this session only.

        Only non-None arguments are applied, so callers can pass argparse
        values directly without pre-filtering unset flags.
        Raises ValueError for an unrecognised provider.
        """
        if provider is not None:
            supported = set(_PROVIDER_BASE_URLS) | {"openai_compatible"}
            if provider not in supported:
                raise ValueError(
                    f"Unknown provider '{provider}'. "
                    f"Choose one of: {', '.join(sorted(supported))}"
                )
            self.provider = provider
        if model is not None:
            self.model = model
        if api_key_env is not None:
            self.api_key_env = api_key_env
        if base_url is not None:
            self.base_url = base_url
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(cfg: AgentConfig) -> OpenAIModel:
    """Construct a PydanticAI OpenAIModel pointed at the right provider."""
    # Resolve base URL
    if cfg.base_url:
        base_url = cfg.base_url
    elif cfg.provider == "openai_compatible":
        raise ValueError(
            "provider: openai_compatible requires a base_url in config.yaml"
        )
    else:
        base_url = _PROVIDER_BASE_URLS[cfg.provider]

    # Resolve API key
    if cfg.provider in _PROVIDERS_WITHOUT_AUTH:
        api_key = "ollama"  # placeholder; Ollama ignores it
    elif cfg.api_key_env:
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{cfg.api_key_env}' is not set.\n"
                f"Export it before running: export {cfg.api_key_env}=<your-key>"
            )
    else:
        raise ValueError(
            "api_key_env is required for this provider. "
            "Set it in config.yaml (e.g. api_key_env: HF_TOKEN)."
        )

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return OpenAIModel(cfg.model, openai_client=client)


# ---------------------------------------------------------------------------
# Shared agent state
# ---------------------------------------------------------------------------
@dataclass
class StataContext:
    """Tracks dataset state across tool calls."""
    dataset_loaded: bool = False
    dataset_path: str | None = None


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def build_agent(cfg: AgentConfig) -> Agent[StataContext, str]:
    """Build and return a configured PydanticAI agent."""
    model = build_model(cfg)
    model_settings = ModelSettings(
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    hf_agent: Agent[StataContext, str] = Agent(
        model,
        deps_type=StataContext,
        model_settings=model_settings,
        system_prompt=(
            "You are StataAgent, an AI assistant that helps users analyze "
            "data in Stata. When a user asks a question:\n"
            "1. If no dataset is loaded, ask the user for a file path.\n"
            "2. Call `describe_dataset` first to understand the variables.\n"
            "3. Use the structured tools (summarize_vars, tabulate_vars, "
            "   run_regression) when possible — they are safer and produce "
            "   cleaner output.\n"
            "4. Fall back to `run_stata` for commands not covered by the "
            "   structured tools (e.g. xtreg, margins, xtset).\n"
            "5. Interpret the output for the user in plain language. "
            "   Don't just dump the Stata log — explain what it means.\n"
            "6. If a command fails, read the error, fix it, and retry once. "
            "   If it fails again, tell the user clearly what went wrong."
        ),
    )

    # --- helpers -----------------------------------------------------------
    def _require_dataset(ctx: RunContext[StataContext]) -> str | None:
        """Return an error string if no dataset is loaded, else None."""
        if not ctx.deps.dataset_loaded:
            return (
                "No dataset is loaded. "
                "Call load_data first with a path to a .dta file."
            )
        return None

    # --- tool registrations ------------------------------------------------

    @hf_agent.tool
    def load_data(ctx: RunContext[StataContext], path: str) -> str:
        """Load a Stata .dta file into memory. Must be called before any analysis."""
        result: StataResult = load_dataset(path)
        if result.success:
            ctx.deps.dataset_loaded = True
            ctx.deps.dataset_path = path
            return f"Dataset loaded from {path}.\n{result.output}"
        return f"Failed to load: {result.error}"

    @hf_agent.tool
    def describe_dataset(ctx: RunContext[StataContext]) -> str:
        """List all variables, their types, and labels in the loaded dataset.
        Call this early to understand what's available."""
        if err := _require_dataset(ctx):
            return err
        result = describe_data()
        return result.output if result.success else f"Error: {result.error}"

    @hf_agent.tool
    def summarize_vars(
        ctx: RunContext[StataContext],
        varlist: str = "",
        detail: bool = False,
    ) -> str:
        """Compute descriptive statistics (mean, SD, min, max) for variables.
        Leave varlist empty to summarize all numeric variables. Set detail=True
        for percentiles and skewness."""
        if err := _require_dataset(ctx):
            return err
        result = summarize(varlist, detail)
        return result.output if result.success else f"Error: {result.error}"

    @hf_agent.tool
    def tabulate_vars(
        ctx: RunContext[StataContext],
        var1: str,
        var2: str | None = None,
    ) -> str:
        """Frequency table for one variable, or cross-tab for two variables."""
        if err := _require_dataset(ctx):
            return err
        result = tabulate(var1, var2)
        return result.output if result.success else f"Error: {result.error}"

    @hf_agent.tool
    def run_regression(
        ctx: RunContext[StataContext],
        dependent: str,
        independents: list[str],
        robust: bool = False,
        condition: str | None = None,
    ) -> str:
        """Run an OLS regression. 'condition' is an optional 'if' clause
        (e.g. 'age > 25'). Set robust=True for heteroskedasticity-robust SEs."""
        if err := _require_dataset(ctx):
            return err
        result = regress(dependent, independents, robust, condition)
        return result.output if result.success else f"Error: {result.error}"

    @hf_agent.tool
    def run_stata(ctx: RunContext[StataContext], code: str) -> str:
        """Execute arbitrary Stata code. Use this ONLY when the structured
        tools above don't cover your need (e.g. xtreg, margins, xtset,
        user-contributed commands). Prefer structured tools whenever possible."""
        if err := _require_dataset(ctx):
            return err
        result = run_raw_stata(code)
        if result.success:
            return result.output
        return f"Stata error: {result.error}\nOutput before error:\n{result.output}"

    return hf_agent


# ---------------------------------------------------------------------------
# REPL entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="StataAgent — HuggingFace edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "CLI flags override the corresponding config.yaml values for this\n"
            "session only — the file is never modified.\n\n"
            "examples:\n"
            "  python hf_agent.py --reset\n"
            "  python hf_agent.py --provider groq --model llama3-8b-8192\n"
            "  python hf_agent.py --temperature 0.1 --max-tokens 2048\n"
            "  python hf_agent.py --config my_project.yaml --provider together"
        ),
    )

    # Config file
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="YAML config file to load (default: config.yaml)",
    )

    # Stata reset
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the saved Stata installation config and re-run discovery",
    )

    # Model / provider overrides
    parser.add_argument(
        "--provider",
        metavar="NAME",
        help="Override provider (huggingface | together | fireworks | groq | ollama | openai_compatible)",
    )
    parser.add_argument(
        "--model",
        metavar="ID",
        help="Override model identifier as the provider expects it",
    )
    parser.add_argument(
        "--api-key-env",
        metavar="VAR",
        dest="api_key_env",
        help="Override the env-var name that holds the API key",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        dest="base_url",
        help="Override the provider base URL (required for openai_compatible)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        metavar="FLOAT",
        help="Override sampling temperature (0.0 – 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        metavar="INT",
        dest="max_tokens",
        help="Override maximum tokens in the model response",
    )

    args = parser.parse_args()

    # Handle Stata config reset before anything else
    if args.reset:
        from config import reset_saved_config
        reset_saved_config()

    # Load base config then apply any CLI overrides
    try:
        cfg = AgentConfig.from_yaml(args.config)
        cfg.apply_overrides(
            provider=args.provider,
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"StataAgent ready  |  provider: {cfg.provider}  |  model: {cfg.model}\n"
        "Type your question, or 'quit' to exit.\n"
    )

    agent = build_agent(cfg)
    ctx = StataContext()
    history: list[ModelMessage] = []

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        result = agent.run_sync(query, deps=ctx, message_history=history)
        history = result.all_messages()
        print(f"\n{result.data}\n")


if __name__ == "__main__":
    main()
