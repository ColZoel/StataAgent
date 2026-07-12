"""
HuggingFace-backed StataAgent — powered by LiteLLM.

Reads provider / model / auth settings from config.yaml (or a path you pass)
and builds a PydanticAI agent via pydantic-ai-litellm, which supports 100+
providers through a unified prefix-based model string.

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

from __init__ import __version__

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pyfiglet
import yaml
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pydantic_ai.messages import ModelMessage
from pydantic_ai.settings import ModelSettings
from pydantic_ai_litellm import LiteLLMModel

from error_handler import print_error
from stata_tools import (
    StataResult,
    _STATA_INIT_EXC,
    check_stata,
    describe_data,
    get_varnames,
    load_dataset,
    regress,
    run_raw_stata,
    summarize,
    tabulate,
)

# ---------------------------------------------------------------------------
# LiteLLM provider prefixes
# LiteLLM model strings take the form "<prefix>/<model-id>".
# ---------------------------------------------------------------------------
_LITELLM_PREFIXES: dict[str, str] = {
    "huggingface": "huggingface",
    "together": "together_ai",
    "fireworks": "fireworks_ai",
    "groq": "groq",
    "ollama": "ollama_chat",      # ollama_chat is preferred over ollama/
    "openai_compatible": "openai",
}

_PROVIDERS_WITHOUT_AUTH = {"ollama"}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
_VALID_RIGOR_LEVELS = {"silent", "colleague", "reviewer", "journal_editor"}


@dataclass
class AgentConfig:
    provider: str
    model: str
    api_key_env: str | None
    base_url: str | None
    temperature: float
    max_tokens: int
    commentary: bool = True
    rigor: str = "colleague"
    ui: bool = False

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
        supported = set(_LITELLM_PREFIXES)
        if provider not in supported:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose one of: {', '.join(sorted(supported))}"
            )

        rigor = str(raw.get("rigor", "colleague")).lower()
        if rigor not in _VALID_RIGOR_LEVELS:
            raise ValueError(
                f"Unknown rigor level '{rigor}' in config. "
                f"Choose one of: {', '.join(sorted(_VALID_RIGOR_LEVELS))}"
            )

        return cls(
            provider=provider,
            model=str(raw["model"]),
            api_key_env=raw.get("api_key_env") or None,
            base_url=raw.get("base_url") or None,
            temperature=float(raw.get("temperature", 0.3)),
            max_tokens=int(raw.get("max_tokens", 4096)),
            commentary=bool(raw.get("commentary", True)),
            rigor=rigor,
            ui=bool(raw.get("ui", False)),
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
        commentary: bool | None = None,
        rigor: str | None = None,
        ui: bool | None = None,
    ) -> None:
        """Apply CLI overrides on top of the YAML values for this session only.

        Only non-None arguments are applied, so callers can pass argparse
        values directly without pre-filtering unset flags.
        Raises ValueError for an unrecognised provider or rigor level.
        """
        if provider is not None:
            supported = set(_LITELLM_PREFIXES)
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
        if commentary is not None:
            self.commentary = commentary
        if rigor is not None:
            if rigor not in _VALID_RIGOR_LEVELS:
                raise ValueError(
                    f"Unknown rigor level '{rigor}'. "
                    f"Choose one of: {', '.join(sorted(_VALID_RIGOR_LEVELS))}"
                )
            self.rigor = rigor
        if ui is not None:
            self.ui = ui


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(cfg: AgentConfig) -> LiteLLMModel:
    """Construct a LiteLLMModel for the configured provider.

    LiteLLM model strings take the form "<prefix>/<model-id>", e.g.
    "groq/llama3-8b-8192" or "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1".
    api_key and api_base are passed as kwargs straight through to litellm.completion().
    """
    if cfg.provider == "openai_compatible" and not cfg.base_url:
        raise ValueError(
            "provider: openai_compatible requires a base_url in config.yaml"
        )

    prefix = _LITELLM_PREFIXES[cfg.provider]
    litellm_model = f"{prefix}/{cfg.model}"

    kwargs: dict[str, str] = {}

    if cfg.base_url:
        kwargs["api_base"] = cfg.base_url

    if cfg.provider not in _PROVIDERS_WITHOUT_AUTH:
        if not cfg.api_key_env:
            raise ValueError(
                "api_key_env is required for this provider. "
                "Set it in config.yaml (e.g. api_key_env: HF_TOKEN)."
            )
        load_dotenv()
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{cfg.api_key_env}' is not set.\n"
                f"Set it in your shell or add it to a .env file: "
                f"{cfg.api_key_env}=<your-key>"
            )
        kwargs["api_key"] = api_key

    return LiteLLMModel(litellm_model, **kwargs)


# ---------------------------------------------------------------------------
# Shared agent state
# ---------------------------------------------------------------------------
@dataclass
class StataContext:
    """Tracks dataset state across tool calls."""
    dataset_loaded: bool = False
    dataset_path: str | None = None
    varnames: set[str] = field(default_factory=set)
    rigor: str = "colleague"  # silent | colleague | reviewer | journal_editor


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_COMMENTARY_ON = (
    "5. Always show the raw Stata output (tables, logs) first, then add a "
    "   concise plain-language interpretation — explain what the results mean "
    "   in context (e.g. number of observations, key variables, notable "
    "   patterns). Keep commentary brief and factual."
)
_SYSTEM_PROMPT_COMMENTARY_OFF = (
    "5. Return the raw Stata output only. Do NOT add unsolicited explanation "
    "   or commentary after tool output. However, if the user explicitly asks "
    "   you to interpret, explain, or summarize results (e.g. 'what does this "
    "   mean?', 'explain this coefficient'), answer that question directly."
)


def build_agent(cfg: AgentConfig) -> Agent[StataContext, str]:
    """Build and return a configured PydanticAI agent."""
    model = build_model(cfg)
    model_settings = ModelSettings(
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    commentary_instruction = (
        _SYSTEM_PROMPT_COMMENTARY_ON if cfg.commentary else _SYSTEM_PROMPT_COMMENTARY_OFF
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
            f"{commentary_instruction}\n"
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

    def _check_vars(ctx: RunContext[StataContext], *varnames: str | None) -> str | None:
        """Return an error string if any variable name is not in the dataset.

        Catches hallucinated variable names before Stata runs, so the agent
        gets a clear correction rather than a cryptic Stata error.
        Unknown names are reported with the full variable list so the model
        can self-correct on its next attempt.
        """
        known = ctx.deps.varnames
        if not known:
            return None  # varnames not yet populated; let Stata catch it
        bad = [v for v in varnames if v and v not in known]
        if not bad:
            return None
        bad_str = ", ".join(f"'{v}'" for v in bad)
        known_str = ", ".join(sorted(known))
        return (
            f"Variable(s) {bad_str} not found in the loaded dataset. "
            f"Available variables: {known_str}"
        )

    # --- tool registrations ------------------------------------------------

    @hf_agent.tool
    def load_data(ctx: RunContext[StataContext], path: str) -> str:
        """Load a Stata .dta file into memory. Must be called before any analysis."""
        result: StataResult = load_dataset(path)
        if result.success:
            ctx.deps.dataset_loaded = True
            ctx.deps.dataset_path = path
            ctx.deps.varnames = set(get_varnames())
            return f"Dataset loaded from {path}.\n{result.output}"
        return f"Failed to load: {result.error}"

    @hf_agent.tool
    def describe_dataset(ctx: RunContext[StataContext]) -> str:
        """List all variables, their types, and labels in the loaded dataset.
        Call this early to understand what's available."""
        if err := _require_dataset(ctx):
            return err
        # Refresh varnames in case the dataset changed since load.
        ctx.deps.varnames = set(get_varnames())
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
        if varlist:
            if err := _check_vars(ctx, *varlist.split()):
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
        if err := _check_vars(ctx, var1, var2):
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
        if err := _check_vars(ctx, dependent, *independents):
            return err
        result = regress(dependent, independents, robust, condition)
        return result.output if result.success else f"Error: {result.error}"

    @hf_agent.tool
    def run_stata(ctx: RunContext[StataContext], code: str) -> str:
        """Execute arbitrary Stata code. Use this ONLY when the structured
        tools above don't cover your need (e.g. xtreg, margins, xtset,
        correlate, user-contributed commands).
        DO NOT use for: regress/regression → use run_regression instead;
        summarize/describe → use summarize_vars/describe_dataset;
        tabulate → use tabulate_vars."""
        if err := _require_dataset(ctx):
            return err
        result = run_raw_stata(code)
        if result.success:
            return result.output
        return f"Stata error: {result.error}\nOutput before error:\n{result.output}"

    return hf_agent


# ---------------------------------------------------------------------------
# Rich console (shared)
# ---------------------------------------------------------------------------
console = Console()


# ---------------------------------------------------------------------------
# LLM health check
# ---------------------------------------------------------------------------
# Update check
# ---------------------------------------------------------------------------

_GITHUB_RELEASES_URL = (
    "https://api.github.com/repos/ColZoel/StataAgent/releases/latest"
)


def _version_gt(a: str, b: str) -> bool:
    """Return True if semver string *a* is strictly greater than *b*."""
    try:
        return (
            tuple(int(x) for x in a.split("."))
            > tuple(int(x) for x in b.split("."))
        )
    except ValueError:
        return False


def check_for_update() -> str | None:
    """Fetch the latest GitHub release tag and compare with __version__.

    Returns the new version string (e.g. 'v0.1.1') if an update is available,
    or None if already up-to-date or the check could not be completed.
    """
    try:
        req = urllib.request.Request(
            _GITHUB_RELEASES_URL,
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": f"StataAgent/{__version__}"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        latest = data.get("tag_name", "").lstrip("v")
        if latest and _version_gt(latest, __version__):
            return f"v{latest}"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------

def _brief_llm_error(exc: BaseException) -> str:
    """Map a LiteLLM / PydanticAI exception to a 1-3 word splash label."""
    text = str(exc)
    tl = text.lower()
    for code, phrase in (
        ("401", "Bad API key"),
        ("403", "Access forbidden"),
        ("404", "Model not found"),
        ("429", "Rate limited"),
        ("500", "Server error"),
        ("502", "Server error"),
        ("503", "Service unavailable"),
    ):
        if code in text:
            return f"Error {code}: {phrase}"
    if "400" in text:
        if any(w in tl for w in ("api key", "api_key", "auth", "token")):
            return "Error 400: Bad API key"
        return "Error 400: Bad request"
    if "timeout" in tl or "timed out" in tl:
        return "Request timeout"
    if any(w in tl for w in ("network", "unreachable", "connection refused", "no route")):
        return "No network"
    return "Unknown error"


def check_llm(cfg: AgentConfig) -> tuple[bool, str | None, BaseException | None]:
    """Send a silent one-word ping to the configured LLM.

    Returns:
        ok          – True if the model responded
        brief_error – short label for the splash ✗, or None on success
        exc         – the raw exception for print_error(), or None on success
    """
    try:
        model = build_model(cfg)
        ping = Agent(model, system_prompt="You are a connectivity test.")
        ping.run_sync("Reply with exactly one word: ready")
        return True, None, None
    except Exception as exc:
        return False, _brief_llm_error(exc), exc


# ---------------------------------------------------------------------------
# Splash screen
# ---------------------------------------------------------------------------

def _check_cell(ok: bool, brief: str | None) -> str:
    """Render the ✓ / ✗ indicator for a status-table row."""
    if ok:
        return "[green]✓[/green]"
    return f"[red]✗  ({brief})[/red]"


def display_splash(
    cfg: AgentConfig,
    stata_ok: bool,
    stata_edition: str,
    stata_version: str,
    stata_brief: str | None,
    llm_ok: bool,
    llm_brief: str | None,
    update_version: str | None = None,
) -> None:
    """Print the StataAgent banner and status table."""
    ascii_art = pyfiglet.figlet_format("StataAgent", font="slant")

    console.print(Panel(
        ascii_art.rstrip(),
        subtitle=f"AI-Powered Stata  •  v{__version__}",
        style="bold #0071bc",
        expand=False,
    ))

    commentary_label = "on" if cfg.commentary else "off"
    rigor_labels = {
        "silent":         "silent",
        "colleague":      "colleague",
        "reviewer":       "reviewer",
        "journal_editor": "journal editor",
    }
    rigor_label = rigor_labels.get(cfg.rigor, cfg.rigor)

    t = Table(show_header=False, box=None, padding=(0, 1))
    t.add_row("[bold]Stata Edition:[/bold]", stata_edition, "")
    t.add_row("[bold]Stata Version:[/bold]", stata_version, _check_cell(stata_ok, stata_brief))
    t.add_row("[bold]Model:[/bold]",         cfg.model,     _check_cell(llm_ok, llm_brief))
    t.add_row("[bold]Commentary:[/bold]",    commentary_label, "")
    t.add_row("[bold]Rigor:[/bold]",         rigor_label,   "")
    t.add_row(
        "[bold]GitHub:[/bold]",
        "[link=https://github.com/ColZoel/StataAgent]StataAgent[/link]",
        "",
    )
    if update_version:
        t.add_row(
            "[bold yellow]Update Available:[/bold yellow]",
            f"[yellow]{update_version}[/yellow]",
            "",
        )

    console.print(t)
    console.rule(style="#0071bc")


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

    commentary_group = parser.add_mutually_exclusive_group()
    commentary_group.add_argument(
        "--commentary",
        dest="commentary",
        action="store_true",
        default=None,
        help="Enable plain-language commentary after Stata output (default: on)",
    )
    commentary_group.add_argument(
        "--no-commentary",
        dest="commentary",
        action="store_false",
        help="Return raw Stata output only, no interpretation",
    )

    ui_group = parser.add_mutually_exclusive_group()
    ui_group.add_argument(
        "--ui",
        dest="ui",
        action="store_true",
        default=None,
        help="Launch the web UI instead of the CLI (overrides config.yaml ui setting)",
    )
    ui_group.add_argument(
        "--no-ui",
        dest="ui",
        action="store_false",
        help="Force CLI mode even if config.yaml has ui: true",
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
            commentary=args.commentary,
            ui=args.ui,
        )
    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print_error(e)
        sys.exit(1)

    # --- UI mode ----------------------------------------------------------
    if cfg.ui:
        from webui.server import launch
        console.print(
            "[bold]Launching StataAgent UI[/bold] at "
            "[link=http://127.0.0.1:8765]http://127.0.0.1:8765[/link]  (Ctrl+C to stop)"
        )
        try:
            launch(cfg)
        except KeyboardInterrupt:
            pass
        return

    # --- startup health checks -------------------------------------------
    stata_ok, stata_edition, stata_version, stata_brief = check_stata()
    llm_ok, llm_brief, llm_exc = check_llm(cfg)
    update_version = check_for_update()

    display_splash(cfg, stata_ok, stata_edition, stata_version, stata_brief,
                   llm_ok, llm_brief, update_version)

    if not stata_ok:
        console.print()
        exc = _STATA_INIT_EXC or RuntimeError(f"Stata check failed: {stata_brief}")
        print_error(exc)
    if not llm_ok and llm_exc is not None:
        console.print()
        print_error(llm_exc)
    if not stata_ok or not llm_ok:
        sys.exit(1)

    _quit_key = "Ctrl+D" if sys.platform == "darwin" else "Ctrl+C"
    console.print(
        f"Type your question. "
        f"Type [bold]quit[/bold] or press [bold]{_quit_key}[/bold] to exit.\n"
    )

    agent = build_agent(cfg)
    ctx = StataContext(rigor=cfg.rigor)
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

        try:
            result = agent.run_sync(query, deps=ctx, message_history=history)
            history = result.all_messages()
            print(f"\n{result.output}\n")
        except Exception as exc:
            print_error(exc)
            print()


if __name__ == "__main__":
    main()
