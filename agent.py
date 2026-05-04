"""
The PydanticAI agent. This is where the LLM meets the tools.

Key concepts you'll see:
- Agent: the orchestrator. Holds the model, system prompt, and tools.
- @agent.tool: decorator that registers a function as a tool the
  LLM can call. PydanticAI reads type hints to build the schema.
- RunContext: gives tools access to shared state (like the loaded dataset).
"""
import urllib.request
import urllib.error
from html.parser import HTMLParser
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

from stata_tools import (
    StataResult,
    load_dataset,
    describe_data,
    summarize,
    tabulate,
    regress,
    collapse,
    reshape,
    egen,
    gstats_tab,
    gstats_sum,
    gcollapse,
    greshape,
    gegen,
    gquantiles,
    gunique,
    check_package_installed,
    ssc_install,
    reghdfe,
    ppmlhdfe,
    run_raw_stata,
    set_stata_locale,
)


@dataclass
class StataContext:
    """Shared state across tool calls."""
    dataset_loaded: bool = False
    dataset_path: str | None = None
    language: str = "en"       # ISO 639-1 code detected from first query
    language_set: bool = False  # True after the first detection; never re-detect


# The agent itself. 'claude-sonnet-4-5' is a reasonable default;
# swap for any model PydanticAI supports.
def _build_system_prompt(commentary: bool = True) -> str:
    commentary_instruction = (
        "5. Always show the raw Stata output first, then add a concise "
        "   plain-language interpretation — explain what the results mean "
        "   in context. Keep commentary brief and factual."
        if commentary else
        "5. Return the raw Stata output only. Do NOT add unsolicited explanation "
        "   or commentary after tool output. However, if the user explicitly asks "
        "   you to interpret, explain, or summarize results (e.g. 'what does this "
        "   mean?', 'explain this coefficient'), answer that question directly."
    )
    return (
        "You are StataAgent, an AI assistant that helps users analyze "
        "data in Stata. When a user asks a question:\n"
        "1. If no dataset is loaded, ask the user for a file path.\n"
        "2. Call `describe_dataset` first to understand the variables.\n"
        "3. Use the structured tools (summarize, tabulate, regress) "
        "   when possible. They are safer and produce cleaner output.\n"
        "4. Fall back to `run_stata` for commands not covered by the "
        "   structured tools.\n"
        f"{commentary_instruction}\n"
        "6. If a command fails, read the error, fix it, and retry once. "
        "   If it fails again, tell the user what went wrong.\n"
        "7. Use collapse_dataset, reshape_dataset, and generate_group_var (native Stata) "
        "   by default. Switch to gcollapse_dataset, greshape_dataset, or gegen_var "
        "   ONLY when the dataset has more than 50 million observations (check the "
        "   observation count shown by describe_dataset).\n"
        "8. Use gstats_tabstat or gstats_summarize instead of native tabstat/summarize, "
        "   and use gquantiles instead of native pctile/xtile/_pctile, whenever the "
        "   operation must be performed WITHIN by() groups. For ungrouped summary stats "
        "   or quantiles, use the native commands via run_stata.\n"
        "9. For fixed-effects regression, use run_hdfe_regression. Choose the estimator "
        "   based on the dependent variable: call summarize_vars on it first, then:\n"
        "   - Use 'continuous' (reghdfe) if the variable has non-integer or negative values.\n"
        "   - Use 'count' (ppmlhdfe) if all values are non-negative integers with a "
        "     distribution typical of counts (mean << variance or heavy right skew).\n"
        "   - If genuinely unclear, ask the user: 'Is [varname] a count variable or "
        "     can it be treated as continuous?'\n"
        "10. Before running any user-contributed command (reghdfe, ppmlhdfe, gtools, etc.), "
        "    use install_stata_package to check if it is installed. If it is not, tell the "
        "    user which package is missing and ask for confirmation before installing.\n"
        "    Also call lookup_stata_docs for any command whose exact option syntax you are "
        "    not certain of — treat the returned text as ground truth and adjust your "
        "    arguments accordingly before running the command.\n"
        "11. Language detection (do this ONCE on the very first user message, never again):\n"
        "    a. Detect the user's language and call set_session_language with the ISO 639-1\n"
        "       code (e.g. 'es') and, if supported, the matching Stata locale\n"
        "       (es→es_ES, fr→fr_FR, de→de_DE, it→it_IT, pt→pt_PT, zh→zh_CN,\n"
        "        ja→ja_JP, ko→ko_KR, ru→ru_RU, ar→ar_SA). Omit stata_locale for English\n"
        "       or for languages without a known Stata locale.\n"
        "    b. After calling set_session_language the tool will attempt\n"
        "       `set locale_functions <locale>` and `label language <code>` in Stata.\n"
        "       Errors from those commands are non-fatal — proceed regardless.\n"
        "12. Variable label translation: when you show variable names and labels\n"
        "    (e.g. from describe_dataset output), translate each label into the user's\n"
        "    language and append it in parentheses — e.g. 'ingreso (income)'. If you\n"
        "    cannot confidently translate a label, show it as-is.\n"
        "13. Response language: Stata commands are ALWAYS written in English. The raw\n"
        "    Stata log output is ALWAYS shown as-is (it is always in English). Your own\n"
        "    interpretation, explanations, and conversational text must be in the user's\n"
        "    detected language."
    )


agent = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=StataContext,
    system_prompt=_build_system_prompt(commentary=True),
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


@agent.tool
def set_session_language(
    ctx: RunContext[StataContext],
    language_code: str,
    stata_locale: str | None = None,
) -> str:
    """Record the language detected from the user's first message. Call this
    exactly once per session, before any other interaction.

    language_code: ISO 639-1 code, e.g. "es", "fr", "de", "zh".
    stata_locale: POSIX locale for `set locale_functions`, e.g. "es_ES".
        Omit for English or unsupported languages.

    Attempts `set locale_functions <stata_locale>` and
    `label language <language_code>` in Stata. Both failures are
    non-fatal — the session language is stored regardless.
    """
    ctx.deps.language = language_code
    ctx.deps.language_set = True

    messages: list[str] = [f"Session language set to '{language_code}'."]

    if stata_locale:
        loc_result = set_stata_locale(stata_locale)
        if loc_result.success:
            messages.append(f"Stata locale_functions set to '{stata_locale}'.")
        else:
            messages.append(
                f"set locale_functions '{stata_locale}' failed (non-fatal): {loc_result.error}"
            )
        lbl_result = run_raw_stata(f"label language {language_code}")
        if lbl_result.success:
            messages.append(f"Stata label language set to '{language_code}'.")
        else:
            messages.append(
                f"label language '{language_code}' not available in this dataset (non-fatal)."
            )

    return " ".join(messages)


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
def gstats_tabstat(
    ctx: RunContext[StataContext],
    varlist: str,
    by: str | None = None,
    statistics: str | None = None,
    columns: str | None = None,
    condition: str | None = None,
) -> str:
    """Summary statistics table via gstats tabstat. Use this instead of native
    tabstat when the operation is performed within by() groups.

    varlist: space-separated variable names.
    by: grouping variable(s) — supports multiple variables, unlike native tabstat.
    statistics: space-separated stat names, e.g. "mean sd min max p50".
        Supported: mean, geomean, count, nmissing, nunique, median, sum, sd,
        variance, cv, range, min, max, first, last, p1–p99, iqr, skewness,
        kurtosis, gini, percent, and more.
    columns: "stat" (default, columns = statistics) or "var" (columns = variables).
    condition: optional if-expression, e.g. "year > 2000".
    """
    if err := _require_dataset(ctx):
        return err
    result = gstats_tab(varlist, by, statistics, columns, condition)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def gstats_summarize(
    ctx: RunContext[StataContext],
    varlist: str,
    by: str | None = None,
    detail: bool = False,
    meanonly: bool = False,
    condition: str | None = None,
) -> str:
    """Descriptive statistics via gstats summarize. Use this instead of native
    summarize when the operation is performed within by() groups.

    varlist: space-separated variable names.
    by: grouping variable(s).
    detail: include percentiles, skewness, and kurtosis.
    meanonly: compute only count, sum, mean, min, max (faster).
    condition: optional if-expression.
    """
    if err := _require_dataset(ctx):
        return err
    result = gstats_sum(varlist, by, detail, meanonly, condition)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def collapse_dataset(
    ctx: RunContext[StataContext],
    clist: str,
    by: str | None = None,
    condition: str | None = None,
    cw: bool = False,
    fast: bool = False,
) -> str:
    """Collapse dataset to group-level statistics (native Stata collapse).
    Default collapse tool — use for any dataset under 50 million observations.
    For N > 50M use gcollapse_dataset instead.

    clist: stat/variable specification, e.g.:
        "(mean) income (sum) sales"
        "(p50) wage_med=wage (sd) wage_sd=wage"
    by: grouping variable(s).
    cw: casewise deletion — drop observations missing any variable in clist.
    fast: skip preserve/restore (safe in scripts).
    """
    if err := _require_dataset(ctx):
        return err
    result = collapse(clist, by, condition, cw, fast)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def gcollapse_dataset(
    ctx: RunContext[StataContext],
    clist: str,
    by: str | None = None,
    condition: str | None = None,
    fast: bool = False,
    merge: bool = False,
    replace: bool = False,
    freq: str | None = None,
) -> str:
    """Collapse dataset to group-level statistics via gcollapse (gtools).
    ONLY use when N > 50 million observations; otherwise use collapse_dataset.

    clist: Stata-style stat/variable specification, e.g.:
        "(mean) income (sum) sales"
        "(p50) wage_med=wage (sd) wage_sd=wage"
    Supports all native collapse stats plus: geomean, nunique, cv, gini,
    skewness, kurtosis, select#, and more.
    by: grouping variable(s).
    fast: skip preserve/restore (safe in scripts).
    merge: merge collapsed results back into the original dataset in memory.
    replace: with merge=True, overwrite existing variables.
    freq: variable name to store per-group observation count.
    """
    if err := _require_dataset(ctx):
        return err
    result = gcollapse(clist, by, condition, fast, merge, replace, freq)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def reshape_dataset(
    ctx: RunContext[StataContext],
    direction: str,
    stubnames: str,
    i: str,
    j: str | None = None,
    string: bool = False,
    condition: str | None = None,
) -> str:
    """Reshape dataset between wide and long (native Stata reshape).
    Default reshape tool — use for any dataset under 50 million observations.
    For N > 50M use greshape_dataset instead.

    direction: "long" or "wide".
    stubnames: stub variable names, e.g. "inc wage" or "x".
    i: ID variable(s) that uniquely identify observations.
    j: variable whose values label the wide columns (required for wide;
       optional for long — defaults to "j").
    string: allow string values in j (long only).
    """
    if err := _require_dataset(ctx):
        return err
    result = reshape(direction, stubnames, i, j, string, condition)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def greshape_dataset(
    ctx: RunContext[StataContext],
    direction: str,
    varlist: str,
    by: str | None = None,
    keys: str | None = None,
    values: str | None = None,
    string: bool = False,
    dropmiss: bool = False,
    nochecks: bool = False,
    condition: str | None = None,
) -> str:
    """Reshape dataset via greshape (gtools).
    ONLY use when N > 50 million observations; otherwise use reshape_dataset.

    direction: "long", "wide", "gather", or "spread".
        long/wide: mirror native reshape syntax.
        gather: pivot multiple columns into key-value rows (tidyr-style).
        spread: pivot key-value rows into columns (tidyr-style).
    varlist: stub names (long/wide) or variable list (gather/spread).
    by: ID variable(s) — required for long/wide.
    keys: variable for stub suffixes (long/wide) or key column (gather/spread).
    values: values variable — required for gather.
    string: allow string stub matching in long.
    dropmiss: drop rows where all reshaped variables are missing.
    nochecks: skip duplicate/missing checks and sorting (fastest).
    """
    if err := _require_dataset(ctx):
        return err
    result = greshape(direction, varlist, by, keys, values, string, dropmiss, nochecks, condition)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def generate_group_var(
    ctx: RunContext[StataContext],
    newvar: str,
    function: str,
    expression: str,
    by: str | None = None,
    replace: bool = False,
    condition: str | None = None,
    extra_opts: str | None = None,
) -> str:
    """Generate a new variable using native Stata egen.
    Default tool for group variable generation — use for any dataset under
    50 million observations. For N > 50M use gegen_var instead.

    newvar: name for the new variable.
    function: e.g. mean, sum, sd, count, min, max, median, iqr, pctile,
        first, last, tag, group, rank, rowmean, rowmax, rowmin, rowtotal.
    expression: argument to the function, e.g. "income" or "industry occupation"
        (for group/tag, pass the grouping variables as a space-separated string).
    by: group variable(s) for within-group computation.
    extra_opts: function-specific options, e.g. "p(75)" for pctile.
    replace: overwrite newvar if it already exists.
    condition: optional if-expression.
    """
    if err := _require_dataset(ctx):
        return err
    result = egen(newvar, function, expression, by, replace, condition, extra_opts)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def gegen_var(
    ctx: RunContext[StataContext],
    newvar: str,
    function: str,
    expression: str,
    by: str | None = None,
    replace: bool = False,
    condition: str | None = None,
    extra_opts: str | None = None,
) -> str:
    """Generate a new variable using gegen (gtools).
    ONLY use when N > 50 million observations; otherwise use generate_group_var.

    newvar: name for the new variable.
    function: all native egen functions are supported, plus gtools extras:
        nunique, geomean, variance, cv, skewness, kurtosis, gini, percent,
        cumsum, demean, demedian, xtile, rank, winsor, moving_stat, range_stat.
    expression: argument to the function.
    by: group variable(s) for within-group computation.
    extra_opts: function-specific options, e.g. "p(75)" for pctile,
        "missing" for group or tag, "n(3)" for select.
    replace: overwrite newvar if it already exists.
    condition: optional if-expression.
    """
    if err := _require_dataset(ctx):
        return err
    result = gegen(newvar, function, expression, by, replace, condition, extra_opts)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def compute_quantiles(
    ctx: RunContext[StataContext],
    expression: str,
    mode: str,
    newvar: str | None = None,
    nquantiles: int | None = None,
    quantiles: str | None = None,
    cutpoints: str | None = None,
    by: str | None = None,
    altdef: bool = False,
    genp: str | None = None,
    replace: bool = False,
    strict: bool = False,
    condition: str | None = None,
) -> str:
    """Compute quantiles, bin assignments, or percentile scalars via gquantiles.
    Use this instead of native pctile/xtile/_pctile when the operation is
    performed within by() groups.

    expression: variable or Stata expression to quantile.
    mode: "pctile"  — store quantile breakpoints in newvar.
          "xtile"   — store bin assignments (1..nquantiles) in newvar.
          "_pctile" — store results as r(r1), r(r2), … scalars (no newvar needed).
    newvar: output variable name (required for pctile and xtile).
    nquantiles: number of equal-sized groups (default 2 = median split).
    quantiles: specific percentiles as a space-separated list, e.g. "10 25 50 75 90".
    cutpoints: existing variable whose values define the cut boundaries.
    by: group variable(s) — computes separate quantiles per group.
    altdef: use alternative interpolation formula (cannot combine with weights).
    genp: new variable to store the percentage associated with each pctile value.
    replace: overwrite newvar if it already exists.
    strict: with by(), skip groups with too few obs rather than erroring.
    condition: optional if-expression.
    """
    if err := _require_dataset(ctx):
        return err
    result = gquantiles(expression, mode, newvar, nquantiles, quantiles, cutpoints,
                        by, altdef, genp, replace, strict, condition)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def count_unique(
    ctx: RunContext[StataContext],
    varlist: str,
    by: str | None = None,
    generate: str | None = None,
    replace: bool = False,
    detail: bool = False,
    missing: bool = False,
    condition: str | None = None,
) -> str:
    """Count unique values of one or more variables using gunique (gtools).

    Variables in varlist are evaluated jointly — e.g. 'a b' counts unique
    combinations of a and b, not unique values of each separately.

    by: group varlist; creates a per-observation variable tracking unique
        counts within each group (default name _Unique).
    generate: alternative variable name for the by() result.
    replace: overwrite the generate variable if it already exists.
    detail: report summary statistics on group sizes.
    missing: treat missing values as a distinct unique value.
    condition: optional 'if' expression, e.g. 'age > 25'.
    """
    if err := _require_dataset(ctx):
        return err
    result = gunique(varlist, by, generate, replace, detail, missing, condition)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def install_stata_package(ctx: RunContext[StataContext], pkg: str) -> str:
    """Check whether a user-contributed Stata package is installed and, if not,
    install it from SSC.

    Always call this before running any user-contributed command (reghdfe,
    ppmlhdfe, gtools, etc.). If the package is missing, inform the user and
    ask for confirmation before calling this tool to install.

    Returns a status message indicating whether the package was already
    present or was freshly installed.
    """
    cmd = pkg.lower().strip()
    if check_package_installed(cmd):
        return f"Package '{pkg}' is already installed."
    result = ssc_install(pkg)
    if result.success:
        return f"Package '{pkg}' installed successfully.\n{result.output}"
    return f"Installation failed: {result.error}\n{result.output}"


@agent.tool
def run_hdfe_regression(
    ctx: RunContext[StataContext],
    dependent: str,
    independents: list[str],
    absorb: str,
    outcome_type: str,
    vce: str | None = None,
    condition: str | None = None,
    residuals: str | None = None,
    exposure: str | None = None,
) -> str:
    """Run a high-dimensional fixed-effects regression.

    outcome_type must be one of:
        "continuous" — OLS via reghdfe. Use when the dependent variable takes
            non-integer or negative values, or is otherwise a continuous measure.
        "count"      — Poisson PPML via ppmlhdfe. Use when the dependent variable
            contains non-negative integers (trade flows, patent counts, events, etc.).

    Before calling this tool:
      1. Run summarize_vars on the dependent variable to inspect its distribution.
      2. Determine outcome_type from that output (see rules in system prompt).
      3. Confirm the required package is installed via install_stata_package
         (reghdfe for continuous, ppmlhdfe for count).

    absorb: space-separated fixed-effect variables to absorb, e.g. "firm year".
    vce: standard error type, e.g. "robust" or "cluster firmid".
    residuals: (continuous only) variable name to store absorbed residuals.
    exposure: (count only) exposure variable for the Poisson offset.
    condition: optional if-expression.
    """
    if err := _require_dataset(ctx):
        return err
    if outcome_type == "continuous":
        result = reghdfe(dependent, independents, absorb, vce, condition, residuals)
    elif outcome_type == "count":
        result = ppmlhdfe(dependent, independents, absorb, vce, condition, exposure)
    else:
        return (
            f"Unknown outcome_type '{outcome_type}'. "
            "Must be 'continuous' (reghdfe) or 'count' (ppmlhdfe)."
        )
    return result.output if result.success else f"Error: {result.error}"


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and collapse whitespace into readable plain text."""

    _SKIP_TAGS = {"script", "style", "head", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)


@agent.tool
def lookup_stata_docs(ctx: RunContext[StataContext], cmd: str) -> str:
    """Look up the official syntax and options for a Stata command.

    Fetches the help page from stata.com for built-in commands, then
    falls back to running `help <cmd>` inside Stata for user-contributed
    commands that are already installed (e.g. reghdfe, ppmlhdfe, gtools).

    Use this tool before running any command you are uncertain about —
    especially user-contributed commands or commands with complex option
    syntax. The returned text is the canonical reference; use it to
    verify that arguments and option names are correct before calling
    run_stata or a structured tool.
    """
    STATA_HELP_URL = f"https://www.stata.com/help.cgi?{urllib.request.quote(cmd)}"
    web_text: str | None = None
    web_error: str | None = None

    try:
        req = urllib.request.Request(
            STATA_HELP_URL,
            headers={"User-Agent": "StataAgent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        raw = extractor.get_text()
        # Stata's help.cgi returns "not found" pages for unknown commands.
        if "not found" in raw.lower() or len(raw) < 100:
            web_text = None
        else:
            # Cap at ~4 000 chars so the context doesn't balloon.
            web_text = raw[:4000].strip()
    except urllib.error.URLError as exc:
        web_error = str(exc)
    except Exception as exc:
        web_error = str(exc)

    # For user-contributed or unlisted commands, also try `help` inside Stata.
    stata_help_text: str | None = None
    if web_text is None:
        stata_result = run_raw_stata(f"help {cmd}")
        if stata_result.success and stata_result.output.strip():
            stata_help_text = stata_result.output.strip()[:4000]

    if web_text and stata_help_text:
        return f"=== stata.com ===\n{web_text}\n\n=== Stata help ===\n{stata_help_text}"
    if web_text:
        return f"=== stata.com ===\n{web_text}"
    if stata_help_text:
        return f"=== Stata help (local) ===\n{stata_help_text}"
    return (
        f"Could not retrieve documentation for '{cmd}'.\n"
        f"Web error: {web_error or 'none'}\n"
        "Check that the command name is spelled correctly and the package is installed."
    )


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