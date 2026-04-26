"""
The PydanticAI agent. This is where the LLM meets the tools.

Key concepts you'll see:
- Agent: the orchestrator. Holds the model, system prompt, and tools.
- @agent.tool: decorator that registers a function as a tool the
  LLM can call. PydanticAI reads type hints to build the schema.
- RunContext: gives tools access to shared state (like the loaded dataset).
"""
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

from stata_tools import (
    StataResult,
    load_dataset,
    describe_data,
    summarize,
    tabulate,
    regress,
    run_raw_stata,
)


@dataclass
class StataContext:
    """Shared state across tool calls. Right now just tracks
    whether a dataset has been loaded — useful for better error
    messages and for the system prompt."""
    dataset_loaded: bool = False
    dataset_path: str | None = None


# The agent itself. 'claude-sonnet-4-5' is a reasonable default;
# swap for any model PydanticAI supports.
agent = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=StataContext,
    system_prompt=(
        "You are StataAgent, an AI assistant that helps users analyze "
        "data in Stata. When a user asks a question:\n"
        "1. If no dataset is loaded, ask the user for a file path.\n"
        "2. Call `describe_dataset` first to understand the variables.\n"
        "3. Use the structured tools (summarize, tabulate, regress) "
        "   when possible. They are safer and produce cleaner output.\n"
        "4. Fall back to `run_stata` for commands not covered by the "
        "   structured tools.\n"
        "5. Interpret the output for the user in plain language. "
        "   Don't just dump the Stata log — explain what it means.\n"
        "6. If a command fails, read the error, fix it, and retry once. "
        "   If it fails again, tell the user what went wrong."
    ),
)


# --- Tool registrations ---
# Each @agent.tool function becomes a tool the LLM can call.
# PydanticAI uses the type hints and docstring to build the schema
# the model sees. Write good docstrings — the model reads them.

@agent.tool
def load_data(ctx: RunContext[StataContext], path: str) -> str:
    """Load a Stata .dta file into memory. Must be called before any analysis."""
    result = load_dataset(path)
    if result.success:
        ctx.deps.dataset_loaded = True
        ctx.deps.dataset_path = path
        return f"Dataset loaded from {path}.\n{result.output}"
    return f"Failed to load: {result.error}"


def _require_dataset(ctx: RunContext[StataContext]) -> str | None:
    """Return an error string if no dataset is loaded, else None."""
    if not ctx.deps.dataset_loaded:
        return "No dataset is loaded. Call load_data first with a path to a .dta file."
    return None


@agent.tool
def describe_dataset(ctx: RunContext[StataContext]) -> str:
    """List all variables, their types, and labels in the loaded dataset.
    Call this early to understand what's available."""
    if err := _require_dataset(ctx):
        return err
    result = describe_data()
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
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


@agent.tool
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


@agent.tool
def run_regression(
    ctx: RunContext[StataContext],
    dependent: str,
    independents: list[str],
    robust: bool = False,
    condition: str | None = None,
) -> str:
    """Run an OLS regression. 'condition' is an optional 'if' clause
    (e.g., 'age > 25'). Set robust=True for heteroskedasticity-robust SEs."""
    if err := _require_dataset(ctx):
        return err
    result = regress(dependent, independents, robust, condition)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def run_stata(ctx: RunContext[StataContext], code: str) -> str:
    """Execute arbitrary Stata code. Use this ONLY when the structured
    tools above don't cover your need (e.g., xtreg, margins, user-contributed
    commands). Prefer structured tools whenever possible."""
    if err := _require_dataset(ctx):
        return err
    result = run_raw_stata(code)
    if result.success:
        return result.output
    return f"Stata error: {result.error}\nOutput before error:\n{result.output}"