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
from dataclasses import dataclass, field
from pydantic import BaseModel
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
    eststo,
    esttab,
    merge_datasets,
    merge_report,
    append_datasets,
    ivregress,
    ivreghdfe,
    margins_effects,
    marginsplot_cmd,
)


class AnalysisStep(BaseModel):
    """One step in a proposed analysis plan."""
    number: int
    description: str
    stata_intent: str       # the Stata command this step maps to (loosely)
    status: str = "pending" # pending | done | skipped


@dataclass
class AnalysisPlan:
    """Structured plan produced during MODE 1 (CLARIFY)."""
    question: str
    steps: list[AnalysisStep] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    clarifications: list[str] = field(default_factory=list)
    proposed_spec: str = ""        # core regression command as currently planned
    confirmed: bool = False
    amendments: list[str] = field(default_factory=list)  # history of user changes


RIGOR_PROMPTS: dict[str, str] = {
    "silent": (
        "Rigor level: SILENT. Run exactly what the user asks and report results. "
        "Flag only hard errors (missing variable, wrong type, Stata error). "
        "No unsolicited methodological commentary, no suggestions, no caveats."
    ),
    "colleague": (
        "Rigor level: COLLEAGUE. Flag obvious issues once, non-intrusively — "
        "e.g. 'FYI: SEs aren't clustered — want me to add that?' "
        "Suggest one improvement after results. Do not repeat the same point."
    ),
    "reviewer": (
        "Rigor level: REVIEWER. Before reporting results, proactively check "
        "standard assumptions: parallel trends for DiD, VIF for multicollinearity, "
        "cluster count adequacy (warn if < 30), model fit. Ask explicitly about "
        "robustness checks. Note key caveats before interpreting any result."
    ),
    "journal_editor": (
        "Rigor level: JOURNAL EDITOR. Treat every specification as a submission "
        "under scrutiny. Require justification for modeling choices. Suggest "
        "alternative specifications, heterogeneity checks, placebo tests, and "
        "multiple testing corrections. Do not summarize results without a full "
        "assumption audit."
    ),
}


@dataclass
class StataContext:
    """Shared state across tool calls."""
    dataset_loaded: bool = False
    dataset_path: str | None = None
    language: str = "en"           # ISO 639-1 code detected from first query
    language_set: bool = False     # True after the first detection; never re-detect
    rigor: str = "colleague"       # silent | colleague | reviewer | journal_editor
    analysis_phase: str = "idle"   # idle | clarify | executing | reflecting
    current_plan: AnalysisPlan | None = None
    plan_version: int = 0          # increments with each amendment
    stored_estimates: list[str] = field(default_factory=list)  # names passed to store_estimation


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
        "    detected language.\n"
        "14. Three-mode analysis protocol — apply ONLY for complex analytical requests\n"
        "    (regressions, causal inference, DiD, IV, panel models, matching, event\n"
        "    studies). Skip this protocol for simple requests (describe, summarize,\n"
        "    tabulate, load, reshape).\n"
        "\n"
        "    MODE 1 — CLARIFY (call propose_analysis_plan):\n"
        "      Before touching the data, identify every ambiguity that could change the\n"
        "      model specification: treatment indicator, time variable, clustering level,\n"
        "      sample restriction, functional form. For each step include description\n"
        "      and the Stata command it maps to. State your proposed_spec (the core\n"
        "      regression command as currently planned, e.g.\n"
        "      'regress outcome i.treat##i.post i.year, vce(cluster schoolid)').\n"
        "      After presenting, ask: 'Ready to proceed, or would you like to adjust\n"
        "      anything?'\n"
        "\n"
        "    AMENDMENT LOOP (call amend_plan for any user changes):\n"
        "      When the user suggests changes — however small — call amend_plan with:\n"
        "        - amendment_description: a one-line summary of what changed\n"
        "        - updated_steps: the FULL revised step list (not just the changed steps)\n"
        "        - updated_spec: the revised core Stata command\n"
        "        - updated_assumptions: the revised assumption list\n"
        "      amend_plan re-states the full plan and asks for confirmation again.\n"
        "      Repeat until the user gives clear confirmation.\n"
        "      NEVER begin execution mid-amendment or infer confirmation from partial\n"
        "      approval (e.g. 'sure, and also change X' is an amendment, not a confirm).\n"
        "\n"
        "    MODE 2 — EXECUTE (call begin_execution ONLY on clear confirmation):\n"
        "      Clear confirmation = 'yes', 'looks good', 'go ahead', 'proceed',\n"
        "      or equivalent with no further requested changes.\n"
        "      Call begin_execution to record final confirmed assumptions and\n"
        "      transition to executing. Then run each step in order, reporting\n"
        "      findings before proceeding. At decision forks where data contradicts\n"
        "      an assumption, stop and ask rather than silently substituting.\n"
        "\n"
        "    MODE 3 — REFLECT (call reflect_on_results after the final step):\n"
        "      Interpret the key coefficient(s) in substantive terms.\n"
        "      Then assess these concerns in order — raise any that apply:\n"
        "        a. Cluster count: if N_clusters < 30, warn that clustered SEs may be\n"
        "           unreliable and suggest wild bootstrap (boottest).\n"
        "        b. Sample size: if total N or treated N is small (< ~100), flag\n"
        "           underpowered estimates.\n"
        "        c. Parallel trends (DiD only): suggest a pre-trend test or event-study\n"
        "           plot if one was not run.\n"
        "        d. Model fit: note R² or pseudo-R² and flag if suspiciously high or low.\n"
        "        e. Confounders: name one specific variable that is plausibly omitted\n"
        "           given the research question — don't give generic advice.\n"
        "      End with exactly ONE concrete next step as a specific Stata command or\n"
        "      test (e.g. 'Run boottest treat#post, reps(999) to get wild-bootstrap\n"
        "      p-values'), not a vague offer to 'improve the model'.\n"
        "      The depth of the reflection is governed by the active rigor level.\n"
        "      Call reflect_on_results to store the reflection in context.\n"
        "15. Rigor level: the active level is injected dynamically. Adjust the depth\n"
        "    of unsolicited commentary, assumption checking, and MODE 3 reflection\n"
        "    accordingly. Use set_rigor if the user asks to change level mid-session.\n"
        "16. Estimation tables: when the user wants to compare specifications side by\n"
        "    side, call store_estimation immediately after each regression to include\n"
        "    (run_regression, run_hdfe_regression, run_iv_regression, etc.), then call\n"
        "    build_estimation_table once all specs are stored. Requires the 'estout'\n"
        "    package — check install_stata_package(\"estout\") first if unsure.\n"
        "17. Merging and appending: before running merge_dataset, ask the user for the\n"
        "    match relationship (1:1, 1:m, m:1, m:m) if it isn't obvious, and always\n"
        "    call check_merge_quality immediately after — report the match breakdown\n"
        "    (matched / master-only / using-only) before proceeding with analysis.\n"
        "    Use append_dataset instead when stacking observations from datasets with\n"
        "    the same variables rather than matching on keys.\n"
        "18. Instrumental variables: use run_iv_regression for endogenous regressors.\n"
        "    Before calling, sanity-check the exclusion restriction against the research\n"
        "    question, and check install_stata_package(\"ivreghdfe\") if absorb is set.\n"
        "    At REVIEWER rigor and above, request or run first-stage/weak-instrument\n"
        "    diagnostics (e.g. estat firststage) before treating estimates as reliable.\n"
        "    Use compute_margins for marginal effects/adjusted predictions after any\n"
        "    regression (it uses the last estimation in memory) and plot_margins to\n"
        "    export a marginsplot — pystata is headless, so always give plot_margins an\n"
        "    export_path ending in .png.\n"
        "19. Weighted data: run_regression and summarize_vars accept weight_type\n"
        "    (\"aweight\", \"pweight\", \"fweight\", or \"iweight\") and weight_var. Ask the\n"
        "    user which weight variable and type applies for survey/complex-sample data\n"
        "    rather than guessing — pweight is typical for survey sampling weights."
    )


agent = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=StataContext,
    system_prompt=_build_system_prompt(commentary=True),
)


@agent.system_prompt
def inject_rigor(ctx: RunContext[StataContext]) -> str:
    """Appended to every request — tells the model which rigor level is active."""
    return RIGOR_PROMPTS.get(ctx.deps.rigor, RIGOR_PROMPTS["colleague"])


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
    condition: str | None = None,
    weight_type: str | None = None,
    weight_var: str | None = None,
) -> str:
    """Compute descriptive statistics (mean, SD, min, max) for variables.
    Leave varlist empty to summarize all numeric variables. Set detail=True
    for percentiles and skewness.

    weight_type: "aweight", "pweight", "fweight", or "iweight" — set together
        with weight_var for survey/complex-sample data.
    """
    if err := _require_dataset(ctx):
        return err
    result = summarize(varlist, detail, condition, weight_type, weight_var)
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
    weight_type: str | None = None,
    weight_var: str | None = None,
) -> str:
    """Run an OLS regression. 'condition' is an optional 'if' clause
    (e.g., 'age > 25'). Set robust=True for heteroskedasticity-robust SEs.

    weight_type: "aweight", "pweight", "fweight", or "iweight" — set together
        with weight_var. Ask the user rather than guessing for survey data;
        pweight is typical for sampling weights.
    """
    if err := _require_dataset(ctx):
        return err
    result = regress(dependent, independents, robust, condition, weight_type, weight_var)
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


@agent.tool
def store_estimation(ctx: RunContext[StataContext], name: str) -> str:
    """Store the most recently run estimation (regress, reghdfe, ivregress, etc.)
    under a name, for later inclusion in a side-by-side table via
    build_estimation_table. Call this right after each regression you want
    to compare.

    name: short identifier, e.g. "m1", "baseline", "with_fe".
    """
    if err := _require_dataset(ctx):
        return err
    result = eststo(name)
    if not result.success:
        return f"Error: {result.error}"
    if name not in ctx.deps.stored_estimates:
        ctx.deps.stored_estimates.append(name)
    return f"Stored estimation as '{name}'. Stored so far: {', '.join(ctx.deps.stored_estimates)}"


@agent.tool
def build_estimation_table(
    ctx: RunContext[StataContext],
    models: list[str] | None = None,
    export_path: str | None = None,
    stats: str | None = None,
    title: str | None = None,
) -> str:
    """Render a side-by-side regression table from estimations previously
    saved with store_estimation, via esttab. Requires the 'estout' package —
    call install_stata_package("estout") first if unsure it's installed.

    models: names previously passed to store_estimation, in display order.
        Omit to include every estimation stored so far.
    export_path: file path ending in .tex, .rtf, .csv, or .html to export
        the table (.rtf opens in Word). Omit to print to Results only.
    stats: space-separated summary stat rows, e.g. "N r2 r2_a". Defaults to
        observation count and R-squared.
    title: optional table title.
    """
    if err := _require_dataset(ctx):
        return err
    use_models = models if models else ctx.deps.stored_estimates
    if not use_models:
        return "No stored estimations. Call store_estimation after each regression to include."
    result = esttab(use_models, export_path, stats=stats or "N r2 r2_a", title=title)
    if not result.success:
        return f"Error: {result.error}"
    if export_path:
        return f"{result.output}\n\nTable exported to {export_path}."
    return result.output


@agent.tool
def merge_dataset(
    ctx: RunContext[StataContext],
    merge_type: str,
    keyvars: str,
    using: str,
    generate: str = "_merge",
    keep: str | None = None,
    update: bool = False,
    replace: bool = False,
    force: bool = False,
) -> str:
    """Merge another dataset into memory on shared key variables (Stata's
    merge command). ALWAYS call check_merge_quality immediately afterward
    and report the match breakdown before proceeding — don't assume a clean
    merge.

    merge_type: "1:1", "1:m", "m:1", or "m:m" — the match relationship
        between the in-memory (master) data and the using dataset.
    keyvars: space-separated key variable(s) present in both datasets.
    using: path to the using .dta file.
    generate: name for the match-status variable. Errors if this variable
        already exists from a prior merge — drop it first or pick a new name.
    keep: restrict to result codes, e.g. "3" (matched only) or
        "match master using" — decide using check_merge_quality's output.
    update/replace: fill in or overwrite missing values from using.
    force: allow the merge when variable storage types conflict.
    """
    if err := _require_dataset(ctx):
        return err
    result = merge_datasets(merge_type, keyvars, using, generate, keep, update, replace, force)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def check_merge_quality(ctx: RunContext[StataContext], generate: str = "_merge") -> str:
    """Tabulate the merge match-status variable to show how many observations
    matched, were master-only, or using-only. Call this immediately after
    merge_dataset."""
    if err := _require_dataset(ctx):
        return err
    result = merge_report(generate)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def append_dataset(
    ctx: RunContext[StataContext],
    using: list[str],
    generate: str | None = None,
    keep: str | None = None,
    force: bool = False,
) -> str:
    """Append (stack) one or more datasets below the data in memory (Stata's
    append command). Use this instead of merge_dataset when stacking
    observations from datasets with the same variables, rather than
    matching on key variables.

    using: path(s) to .dta files to append.
    generate: variable to mark source (1 = master, 2+ = each using file,
        in the order given).
    keep: restrict to specific variables.
    force: allow appending across string/numeric type mismatches.
    """
    if err := _require_dataset(ctx):
        return err
    result = append_datasets(using, generate, keep, force)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def run_iv_regression(
    ctx: RunContext[StataContext],
    estimator: str,
    dependent: str,
    exogenous: list[str],
    endogenous: list[str],
    instruments: list[str],
    absorb: str | None = None,
    vce: str | None = None,
    condition: str | None = None,
) -> str:
    """Instrumental-variables regression (2SLS/LIML/GMM), with optional
    high-dimensional fixed effects.

    estimator: "2sls", "liml", or "gmm" (ivregress). Ignored if absorb is
        set — ivreghdfe always uses 2SLS.
    dependent: outcome variable.
    exogenous: included exogenous regressors.
    endogenous: endogenous regressor(s) to be instrumented.
    instruments: excluded instrument(s) — need at least as many as
        endogenous variables (order condition).
    absorb: space-separated fixed-effect variables — if set, uses
        ivreghdfe instead of ivregress.
    vce: variance estimator, e.g. "robust" or "cluster firmid".

    Before calling: sanity-check the exclusion restriction against the
    research question, and call install_stata_package("ivreghdfe") if
    absorb is set (or the plain command if otherwise uncertain it's
    installed). After calling: at REVIEWER rigor and above, run or request
    first-stage/weak-instrument diagnostics (e.g. estat firststage after
    ivregress) before treating point estimates as reliable.
    """
    if err := _require_dataset(ctx):
        return err
    if absorb:
        result = ivreghdfe(dependent, exogenous, endogenous, instruments, absorb, vce, condition)
    else:
        result = ivregress(estimator, dependent, exogenous, endogenous, instruments, vce, condition)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def compute_margins(
    ctx: RunContext[StataContext],
    varlist: str | None = None,
    dydx: str | None = None,
    at: str | None = None,
    condition: str | None = None,
) -> str:
    """Compute marginal effects / adjusted predictions after a regression
    (Stata's margins command). Call this immediately after run_regression,
    run_hdfe_regression, or run_iv_regression — margins operates on the last
    estimation results in memory.

    varlist: factor-variable(s) to margin over, e.g. "i.treat" (omit for
        average marginal effects at means).
    dydx: variable(s) to compute marginal effects/derivatives for, e.g.
        "x1" or "*" for all.
    at: fix covariates at specific values, e.g. "x1=(10 20 30) x2=0".
    condition: optional if-expression restricting the estimation sample.
    """
    if err := _require_dataset(ctx):
        return err
    result = margins_effects(varlist, dydx, at, condition)
    return result.output if result.success else f"Error: {result.error}"


@agent.tool
def plot_margins(ctx: RunContext[StataContext], export_path: str) -> str:
    """Plot the results of the most recent compute_margins call
    (marginsplot) and export it as an image, since pystata runs headless
    and the plot window is not visible to the user.

    export_path: file path ending in .png to save the plot to.
    """
    if err := _require_dataset(ctx):
        return err
    result = marginsplot_cmd()
    if not result.success:
        return f"Error: {result.error}"
    export_result = run_raw_stata(f'graph export "{export_path}", replace')
    if export_result.success:
        return f"{result.output}\n\nPlot exported to {export_path}."
    return f"{result.output}\n\nWarning: failed to export plot: {export_result.error}"


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
def propose_analysis_plan(
    ctx: RunContext[StataContext],
    question: str,
    steps: list[AnalysisStep],
    proposed_spec: str,
    assumptions: list[str],
    clarifications: list[str],
) -> str:
    """MODE 1 — CLARIFY. Call this before touching the data for any complex
    analytical request (regression, DiD, IV, panel, matching, event study).

    question: the user's original analytical question, verbatim or paraphrased.
    steps: numbered execution plan. Each step has number, description, and
        stata_intent (the Stata command it maps to), e.g.:
        [{"number": 1, "description": "Load and describe data",
          "stata_intent": "use / describe"},
         {"number": 4, "description": "Run DiD regression",
          "stata_intent": "regress outcome i.treat##i.post i.year, vce(cluster schoolid)"}]
    proposed_spec: the core regression command as currently planned, e.g.
        "regress outcome i.treat##i.post i.year, vce(cluster schoolid)".
    assumptions: methodological choices you will make by default, e.g.
        ["treat is already binary in the data", "year >= 2019 is the post-period"].
    clarifications: open questions that could change the spec, e.g.
        ["Is there a pre-existing treatment indicator or should I construct one?"].

    Stores the plan in context and transitions analysis_phase to 'clarify'.
    Returns a formatted message to show the user — do NOT paraphrase it.
    """
    ctx.deps.current_plan = AnalysisPlan(
        question=question,
        steps=steps,
        assumptions=assumptions,
        clarifications=clarifications,
        proposed_spec=proposed_spec,
    )
    ctx.deps.analysis_phase = "clarify"
    ctx.deps.plan_version = 1

    return _format_plan(
        steps=steps,
        proposed_spec=proposed_spec,
        assumptions=assumptions,
        clarifications=clarifications,
        preamble="Here's my plan:" if not clarifications else
                 "Before I load the data, a few clarifying questions:",
        changes=None,
    )


def _format_plan(
    steps: list[AnalysisStep],
    proposed_spec: str,
    assumptions: list[str],
    clarifications: list[str],
    preamble: str,
    changes: str | None,
) -> str:
    lines: list[str] = [preamble]

    if clarifications:
        for i, q in enumerate(clarifications, 1):
            lines.append(f"  {i}. {q}")
        lines.append("")

    lines.append("Plan:")
    for s in steps:
        lines.append(f"  [{s.number}] {s.description}  → {s.stata_intent}")

    if proposed_spec:
        lines.append(f"\nCore specification: {proposed_spec}")

    if assumptions:
        lines.append("\nAssumptions:")
        for a in assumptions:
            lines.append(f"  - {a}")

    if changes:
        lines.append(f"\nChanges: {changes}")

    lines.append("\nReady to proceed, or would you like to adjust anything?")
    return "\n".join(lines)


@agent.tool
def amend_plan(
    ctx: RunContext[StataContext],
    amendment_description: str,
    updated_steps: list[AnalysisStep],
    updated_spec: str,
    updated_assumptions: list[str],
) -> str:
    """AMENDMENT LOOP. Call whenever the user requests any change to the plan,
    however small. Re-states the full updated plan and asks for confirmation.

    amendment_description: one-line summary of what changed, e.g.
        "Added year fixed effects (i.year) and clustering by schoolid".
    updated_steps: the COMPLETE revised step list (replace all, not append).
    updated_spec: the revised core Stata command.
    updated_assumptions: the revised assumption list.

    Records the amendment in history, increments plan_version, and asks for
    confirmation again. Do NOT call begin_execution until the user explicitly
    confirms the amended plan.
    """
    if ctx.deps.current_plan is None:
        return "No active plan to amend. Call propose_analysis_plan first."

    ctx.deps.current_plan.steps = updated_steps
    ctx.deps.current_plan.proposed_spec = updated_spec
    ctx.deps.current_plan.assumptions = updated_assumptions
    ctx.deps.current_plan.amendments.append(amendment_description)
    ctx.deps.plan_version += 1

    return _format_plan(
        steps=updated_steps,
        proposed_spec=updated_spec,
        assumptions=updated_assumptions,
        clarifications=[],
        preamble="Got it. Updated plan:",
        changes=amendment_description,
    )


@agent.tool
def begin_execution(
    ctx: RunContext[StataContext],
    confirmed_assumptions: list[str],
) -> str:
    """MODE 2 — EXECUTE. Call ONLY when the user gives clear confirmation
    (yes / looks good / go ahead / proceed) with no further requested changes.

    confirmed_assumptions: the final list of assumptions, incorporating all
        amendments the user approved.

    Marks the plan as confirmed, transitions analysis_phase to 'executing',
    and returns a go-ahead message. After calling this, run each step in
    order and report findings before proceeding to the next step.
    """
    if ctx.deps.current_plan is not None:
        ctx.deps.current_plan.assumptions = confirmed_assumptions
        ctx.deps.current_plan.confirmed = True
    ctx.deps.analysis_phase = "executing"
    return "Plan confirmed. Proceeding with execution."


@agent.tool
def reflect_on_results(
    ctx: RunContext[StataContext],
    results_summary: str,
    concerns: list[str],
    next_step: str,
) -> str:
    """MODE 3 — REFLECT. Call this after all execution steps are complete.

    results_summary: one or two sentences interpreting the key coefficient(s)
        in substantive terms (e.g. "The treat×post coefficient is 0.23 (p=0.04),
        suggesting the policy increased test scores by 0.23 SD among treated
        grade-10 students after 2019.").
    concerns: specific methodological issues identified — raise only those that
        apply given the active rigor level. E.g.:
        ["Only 12 clusters: wild-bootstrap SEs recommended (boottest)",
         "No parallel trends test run — consider an event-study plot"].
        Leave empty if none apply (appropriate for SILENT rigor).
    next_step: ONE concrete actionable step as a specific command or test.
        E.g. "boottest treat#post, reps(999) weighttype(webb)"
        Omit for SILENT rigor unless the user asked for suggestions.

    Transitions analysis_phase to 'reflecting'.
    Returns a formatted reflection block — do NOT paraphrase it.
    """
    ctx.deps.analysis_phase = "reflecting"

    lines: list[str] = [results_summary]

    if concerns:
        lines.append("\nMethodological notes:")
        for c in concerns:
            lines.append(f"  - {c}")

    if next_step:
        lines.append(f"\nSuggested next step: {next_step}")

    return "\n".join(lines)


@agent.tool
def set_rigor(ctx: RunContext[StataContext], level: str) -> str:
    """Change the analysis rigor level for the rest of the session.

    level must be one of:
        "silent"         — runs what you ask, flags only hard errors, no commentary.
        "colleague"      — flags obvious issues once, suggests one improvement.
        "reviewer"       — checks standard assumptions, asks about robustness checks.
        "journal_editor" — full assumption audit, requires justification, suggests
                           alternative specs and placebo tests.

    Call this when the user says something like 'be more critical', 'just run it',
    'act like a reviewer', or 'set rigor to X'.
    """
    normalized = level.lower().strip()
    valid = set(RIGOR_PROMPTS)
    if normalized not in valid:
        return (
            f"Unknown rigor level '{level}'. "
            f"Choose one of: {', '.join(sorted(valid))}."
        )
    ctx.deps.rigor = normalized
    labels = {
        "silent": "SILENT — results only, no unsolicited commentary",
        "colleague": "COLLEAGUE — flags obvious issues once",
        "reviewer": "REVIEWER — proactive assumption checks and robustness suggestions",
        "journal_editor": "JOURNAL EDITOR — full assumption audit on every result",
    }
    return f"Rigor level set to {labels[normalized]}."


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