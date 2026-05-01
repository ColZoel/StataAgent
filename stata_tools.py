"""
Thin wrappers around pystata. These are NOT agent tools yet —
they're plain Python functions. Keeping this layer separate means
you can test Stata integration without any LLM involved.
"""
from dataclasses import dataclass
from config import initialize_pystata

# Locate Stata, extend sys.path, and call pystata's config.init() —
# all in one call. Edition and path are resolved via the strategy chain
# in config.py (env vars → saved config → autodetect → interactive prompt).
# Must run before 'from pystata import stata'.
initialize_pystata()

from pystata import stata  # noqa: E402


@dataclass
class StataResult:
    """What every Stata operation returns to the agent.

    success: did the command run without error?
    output:  the captured log text (what the user would see in Stata's Results window)
    error:   error message if success=False, else None
    """
    success: bool
    output: str
    error: str | None = None


def _run(code: str) -> StataResult:
    """Private helper: run arbitrary Stata code and capture output.

    pystata writes through C-level Stata output, which bypasses
    redirect_stdout. We use echo=True so Stata prints to the console
    directly, and collect the log via stata.get_return() afterward
    for anything the agent needs to inspect programmatically.
    """
    import io
    import contextlib

    # Capture any Python-level prints (e.g. pystata warnings).
    py_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(py_buffer):
            stata.run(code, echo=True)
        py_output = py_buffer.getvalue()
        # Retrieve the full Stata log for this command if available.
        try:
            log = stata.get_return().get("r(log)", "") or ""
        except Exception:
            log = ""
        output = log or py_output
        return StataResult(success=True, output=output)
    except Exception as e:
        py_output = py_buffer.getvalue()
        return StataResult(success=False, output=py_output, error=str(e))


_SYSUSE_DATASETS = {
    "auto", "auto2", "bplong", "bpwide", "cancer", "census", "citytemp",
    "citytemp4", "educ99gdp", "exam", "gnp96", "lifeexp", "nlsw88",
    "nlswide1", "pop2000", "sandstone", "sp500", "surface", "tsline1",
    "tsline2", "urates", "uslifeexp", "uslifeexp2", "voter", "xtline1",
}


def load_dataset(path: str) -> StataResult:
    """Load a dataset into Stata's memory.

    Recognizes Stata's built-in example datasets (e.g. 'auto') and uses
    sysuse; otherwise treats path as a .dta file path and uses use.
    """
    name = path.strip().lower().removesuffix(".dta")
    if name in _SYSUSE_DATASETS:
        return _run(f"sysuse {name}, clear")
    return _run(f'use "{path}", clear')


def get_varnames() -> list[str]:
    """Return the list of variable names currently in memory.

    Uses Stata's `ds` command, which stores the full variable list in
    r(varlist). Returns an empty list if no dataset is loaded.
    """
    try:
        stata.run("ds", echo=False)
        raw = stata.get_return().get("r(varlist)", "") or ""
        return raw.split()
    except Exception:
        return []


def describe_data() -> StataResult:
    """Return variable names, types, and labels. Critical for the
    agent to understand what's in the dataset."""
    return _run("describe")


def summarize(varlist: str = "", detail: bool = False) -> StataResult:
    """Descriptive statistics."""
    code = f"summarize {varlist}"
    if detail:
        code += ", detail"
    return _run(code)


def tabulate(var1: str, var2: str | None = None) -> StataResult:
    """One-way or two-way frequency table."""
    code = f"tabulate {var1}"
    if var2:
        code += f" {var2}"
    return _run(code)


def regress(
    dependent: str,
    independents: list[str],
    robust: bool = False,
    condition: str | None = None,
) -> StataResult:
    """OLS regression."""
    xvars = " ".join(independents)
    code = f"regress {dependent} {xvars}"
    if condition:
        code += f" if {condition}"
    if robust:
        code += ", robust"
    return _run(code)


def gstats_tab(
    varlist: str,
    by: str | None = None,
    statistics: str | None = None,
    columns: str | None = None,
    condition: str | None = None,
) -> StataResult:
    """Summary statistics table via gstats tabstat.

    statistics: space-separated list of stats, e.g. "mean sd min max".
    columns: "stat" (default) or "var" — what the columns represent.
    by: group variable(s); supports multiple unlike native tabstat.
    """
    code = f"gstats tabstat {varlist}"
    if condition:
        code += f" if {condition}"
    opts = []
    if by:
        opts.append(f"by({by})")
    if statistics:
        opts.append(f"statistics({statistics})")
    if columns:
        opts.append(f"columns({columns})")
    if opts:
        code += ", " + " ".join(opts)
    return _run(code)


def gstats_sum(
    varlist: str,
    by: str | None = None,
    detail: bool = False,
    meanonly: bool = False,
    condition: str | None = None,
) -> StataResult:
    """Descriptive statistics via gstats summarize.

    detail: include percentiles, skewness, kurtosis.
    meanonly: compute only count, sum, mean, min, max (faster).
    by: group variable(s); output rendered in tabstat-style layout.
    """
    code = f"gstats summarize {varlist}"
    if condition:
        code += f" if {condition}"
    opts = []
    if by:
        opts.append(f"by({by})")
    if detail:
        opts.append("detail")
    if meanonly:
        opts.append("meanonly")
    if opts:
        code += ", " + " ".join(opts)
    return _run(code)


def gcollapse(
    clist: str,
    by: str | None = None,
    condition: str | None = None,
    fast: bool = False,
    merge: bool = False,
    replace: bool = False,
    freq: str | None = None,
) -> StataResult:
    """Collapse dataset to group-level statistics via gcollapse.

    clist: Stata collapse-style stat/variable list, e.g.
        "(mean) income (sum) sales" or "mean_inc=income".
    by: grouping variable(s).
    fast: skip preserve/restore (safe in scripts, risky interactively).
    merge: merge collapsed results back into the original dataset.
    replace: with merge, replace existing variables.
    freq: variable name to store group observation counts.
    """
    code = f"gcollapse {clist}"
    if condition:
        code += f" if {condition}"
    opts = []
    if by:
        opts.append(f"by({by})")
    if fast:
        opts.append("fast")
    if merge:
        opts.append("merge")
    if replace:
        opts.append("replace")
    if freq:
        opts.append(f"freq({freq})")
    if opts:
        code += ", " + " ".join(opts)
    return _run(code)


def greshape(
    direction: str,
    varlist: str,
    by: str | None = None,
    keys: str | None = None,
    values: str | None = None,
    string: bool = False,
    dropmiss: bool = False,
    nochecks: bool = False,
    condition: str | None = None,
) -> StataResult:
    """Reshape dataset via greshape.

    direction: "long", "wide", "gather", or "spread".
    varlist: stub names (long/wide) or variable list (gather/spread).
    by: ID variable(s) — required for long/wide, optional for gather/spread.
    keys: new/existing variable for stub suffixes or key column name.
    values: values variable name — required for gather.
    string: allow string stub matching (long only).
    dropmiss: drop observations where all reshaped variables are missing.
    nochecks: skip duplicate/missing checks and sorting (fastest).
    """
    code = f"greshape {direction} {varlist}"
    if condition:
        code += f" if {condition}"
    opts = []
    if by:
        opts.append(f"by({by})")
    if keys:
        opts.append(f"keys({keys})")
    if values:
        opts.append(f"values({values})")
    if string:
        opts.append("string")
    if dropmiss:
        opts.append("dropmiss")
    if nochecks:
        opts.append("nochecks")
    if opts:
        code += ", " + " ".join(opts)
    return _run(code)


def gegen(
    newvar: str,
    function: str,
    expression: str,
    by: str | None = None,
    replace: bool = False,
    condition: str | None = None,
    extra_opts: str | None = None,
) -> StataResult:
    """Generate a variable using a gtools group function via gegen.

    function: the function name, e.g. "mean", "sum", "tag", "group",
        "rank", "pctile", "sd", "count", "nunique", "first", "last".
    expression: argument to the function, e.g. "income" or "industry occupation".
    by: group variable(s) for within-group computation.
    extra_opts: function-specific options, e.g. "p(75)" for pctile
        or "missing" for group/tag.
    """
    code = f"gegen {newvar} = {function}({expression})"
    if condition:
        code += f" if {condition}"
    opts = []
    if by:
        opts.append(f"by({by})")
    if replace:
        opts.append("replace")
    if extra_opts:
        opts.append(extra_opts)
    if opts:
        code += ", " + " ".join(opts)
    return _run(code)


def gquantiles(
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
) -> StataResult:
    """Compute quantiles, bin assignments, or percentile values via gquantiles.

    mode: "pctile" (store quantile values in newvar),
          "xtile"  (store bin assignments in newvar),
          "_pctile" (store results in r(r1), r(r2), … — no newvar needed).
    expression: variable or expression to quantile.
    newvar: output variable name (required for pctile/xtile).
    nquantiles: number of quantiles (default 2).
    quantiles: space-separated percentile list, e.g. "10 25 50 75 90".
    cutpoints: existing variable whose values define cut boundaries.
    by: group variable(s) — enables within-group quantiles.
    altdef: use alternative interpolation formula.
    genp: new variable to store the percentage for each pctile result.
    strict: with by(), skip groups with too few obs instead of erroring.
    """
    lhs = f"{newvar} = {expression}" if newvar else expression
    code = f"gquantiles {lhs}, {mode}"
    if condition:
        # insert if/in before the comma+mode — restructure
        code = f"gquantiles {lhs} if {condition}, {mode}"
    opts = []
    if nquantiles:
        opts.append(f"nquantiles({nquantiles})")
    if quantiles:
        opts.append(f"quantiles({quantiles})")
    if cutpoints:
        opts.append(f"cutpoints({cutpoints})")
    if by:
        opts.append(f"by({by})")
    if altdef:
        opts.append("altdef")
    if genp:
        opts.append(f"genp({genp})")
    if replace:
        opts.append("replace")
    if strict:
        opts.append("strict")
    if opts:
        code += " " + " ".join(opts)
    return _run(code)


def gunique(
    varlist: str,
    by: str | None = None,
    generate: str | None = None,
    replace: bool = False,
    detail: bool = False,
    missing: bool = False,
    condition: str | None = None,
) -> StataResult:
    """Count unique values of varlist using gtools' gunique.

    Variables are evaluated jointly (like unique, not distinct).
    Stores r(unique), r(N), r(J), r(minJ), r(maxJ) after the call.
    When by= is set, creates a new variable with per-group unique counts.
    """
    code = f"gunique {varlist}"
    if condition:
        code += f" if {condition}"
    opts = []
    if by:
        opts.append(f"by({by})")
    if generate:
        opts.append(f"generate({generate})")
    if replace:
        opts.append("replace")
    if detail:
        opts.append("detail")
    if missing:
        opts.append("missing")
    if opts:
        code += ", " + " ".join(opts)
    return _run(code)


def run_raw_stata(code: str) -> StataResult:
    """Escape hatch: run arbitrary Stata code.

    The agent should prefer the structured tools above, but fall
    back to this for commands we haven't wrapped (xtreg, margins,
    user-contributed packages, etc.).
    """
    return _run(code)