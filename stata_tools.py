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
#
# Wrapped in try/except so the module always imports successfully even when
# Stata is not installed.  check_stata() exposes the status to the splash.

_STATA_OK: bool = False
_STATA_INIT_EXC: BaseException | None = None
stata = None  # type: ignore[assignment]  -- replaced below on success

try:
    initialize_pystata()
    from pystata import stata  # type: ignore[assignment]  # noqa: E402
    _STATA_OK = True
except Exception as _e:
    _STATA_INIT_EXC = _e


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


# ---------------------------------------------------------------------------
# Run observers — callbacks fired after every _run() so UIs can stream the
# Stata log live. Observer exceptions are swallowed: they must never break
# an agent run.
# ---------------------------------------------------------------------------
from typing import Callable

_RUN_OBSERVERS: list[Callable[[str, "StataResult"], None]] = []


def add_run_observer(fn: Callable[[str, "StataResult"], None]) -> None:
    """Register a callback invoked as fn(code, result) after each Stata run."""
    if fn not in _RUN_OBSERVERS:
        _RUN_OBSERVERS.append(fn)


def remove_run_observer(fn: Callable[[str, "StataResult"], None]) -> None:
    """Unregister a previously added run observer."""
    try:
        _RUN_OBSERVERS.remove(fn)
    except ValueError:
        pass


def _notify_observers(code: str, result: "StataResult") -> None:
    for fn in list(_RUN_OBSERVERS):
        try:
            fn(code, result)
        except Exception:
            pass


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
        result = StataResult(success=True, output=output)
        _notify_observers(code, result)
        return result
    except Exception as e:
        py_output = py_buffer.getvalue()
        result = StataResult(success=False, output=py_output, error=str(e))
        _notify_observers(code, result)
        return result


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
        stata.run("quietly ds", echo=False)
        raw = stata.get_return().get("r(varlist)", "") or ""
        return raw.split()
    except Exception:
        return []


def get_variable_meta() -> list[dict]:
    """Return [{name, type, label, format}] for every variable in memory.

    Uses the sfi Data API (available once pystata is initialized). Returns
    an empty list if Stata is unavailable or no dataset is loaded.
    """
    if not _STATA_OK:
        return []
    try:
        from sfi import Data
        meta = []
        for i in range(Data.getVarCount()):
            meta.append({
                "name": Data.getVarName(i),
                "type": Data.getVarType(i),
                "label": Data.getVarLabel(i) or "",
                "format": Data.getVarFormat(i) or "",
            })
        return meta
    except Exception:
        return []


def get_obs_count() -> int:
    """Return the number of observations in memory (0 if unavailable)."""
    if not _STATA_OK:
        return 0
    try:
        from sfi import Data
        return int(Data.getObsTotal())
    except Exception:
        return 0


def get_stata_version() -> str:
    """Return the Stata version string, e.g. 'StataNow 19' or '18.0'.

    Queries c(stata_version) first (available in Stata 18+, returns the
    branded string like 'StataNow 19'); falls back to c(version) for
    older installations.
    """
    try:
        stata.run("scalar __sv = c(stata_version)", echo=False)
        v = stata.get_scalar("__sv")
        stata.run("scalar drop __sv", echo=False)
        if v:
            return str(v).strip()
    except Exception:
        pass
    try:
        stata.run("scalar __sv = c(version)", echo=False)
        v = stata.get_scalar("__sv")
        stata.run("scalar drop __sv", echo=False)
        if v:
            # c(version) is numeric; drop trailing .0 for clean display
            return str(int(v)) if float(v) == int(float(v)) else str(v)
    except Exception:
        pass
    return "unknown"


def _brief_stata_error(exc: BaseException | None) -> str:
    """Map a Stata/pystata exception to a 1-3 word splash label."""
    if exc is None:
        return "Unknown error"
    text = str(exc).lower()
    if isinstance(exc, (ImportError, ModuleNotFoundError)) or "no module" in text:
        return "Stata not installed"
    if isinstance(exc, FileNotFoundError) or "not found" in text:
        return "Stata not found"
    if "license" in text:
        return "License error"
    if "permission" in text or "access denied" in text:
        return "Permission denied"
    if "timeout" in text or "timed out" in text:
        return "Connection timeout"
    return "PyStata not responsive"


def check_stata() -> tuple[bool, str, str, str | None]:
    """Verify Stata is reachable and return display info for the splash.

    Returns:
        ok           – True if Stata responded correctly
        edition      – e.g. "MP", "SE", or "unknown"
        version      – e.g. "StataNow 19" or "unknown"
        brief_error  – short label shown next to the ✗, or None on success
    """
    if not _STATA_OK or stata is None:
        return False, "unknown", "unknown", _brief_stata_error(_STATA_INIT_EXC)
    # Fire a silent no-op to confirm the engine is responsive.
    try:
        stata.run("local __chk = 1", echo=False)
    except Exception as e:
        return False, "unknown", "unknown", _brief_stata_error(e)
    # Engine responded — read edition and version.
    edition = "unknown"
    try:
        from config import find_stata  # noqa: PLC0415
        install = find_stata()
        edition = install.edition.upper()
    except Exception:
        pass
    version = get_stata_version()
    return True, edition, version, None


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


def collapse(
    clist: str,
    by: str | None = None,
    condition: str | None = None,
    cw: bool = False,
    fast: bool = False,
) -> StataResult:
    """Collapse dataset to group-level statistics (native Stata collapse).

    clist: stat/variable specification, e.g. "(mean) income (sum) sales".
    by: grouping variable(s).
    cw: casewise deletion — drop observations missing any variable.
    fast: skip preserve/restore (safe in scripts).
    """
    code = f"collapse {clist}"
    if condition:
        code += f" if {condition}"
    opts = []
    if by:
        opts.append(f"by({by})")
    if cw:
        opts.append("cw")
    if fast:
        opts.append("fast")
    if opts:
        code += ", " + " ".join(opts)
    return _run(code)


def reshape(
    direction: str,
    stubnames: str,
    i: str,
    j: str | None = None,
    string: bool = False,
    condition: str | None = None,
) -> StataResult:
    """Reshape dataset between wide and long (native Stata reshape).

    direction: "long" or "wide".
    stubnames: stub variable names, e.g. "inc wage" or "x".
    i: ID variable(s) that uniquely identify observations.
    j: variable whose values label the wide columns (required for wide;
       optional for long — defaults to "j").
    string: allow string values in j (long only).
    """
    code = f"reshape {direction} {stubnames}, i({i})"
    if j:
        code += f" j({j})"
        if string:
            code += " string"
    if condition:
        code = f"reshape {direction} {stubnames} if {condition}, i({i})"
        if j:
            code += f" j({j})"
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


def egen(
    newvar: str,
    function: str,
    expression: str,
    by: str | None = None,
    replace: bool = False,
    condition: str | None = None,
    extra_opts: str | None = None,
) -> StataResult:
    """Generate a variable using a native Stata egen function.

    function: e.g. "mean", "sum", "sd", "count", "min", "max", "tag",
        "group", "rank", "pctile", "median", "iqr", "first", "last".
    expression: argument to the function.
    by: group variable(s).
    extra_opts: function-specific options, e.g. "p(75)" for pctile.
    """
    code = f"egen {newvar} = {function}({expression})"
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


def check_package_installed(cmd: str) -> bool:
    """Return True if a user-contributed command is available in Stata."""
    try:
        stata.run(f"which {cmd}", echo=False)
        return True
    except Exception:
        return False


def ssc_install(pkg: str) -> StataResult:
    """Install a package from SSC (Stata Statistical Software Components)."""
    return _run(f"ssc install {pkg}, replace")


def reghdfe(
    dependent: str,
    independents: list[str],
    absorb: str,
    vce: str | None = None,
    condition: str | None = None,
    residuals: str | None = None,
) -> StataResult:
    """OLS with high-dimensional fixed effects via reghdfe.

    absorb: space-separated fixed-effect variables, e.g. "firm year".
    vce: variance estimator, e.g. "robust" or "cluster firmid".
    residuals: variable name to store residuals after absorption.
    """
    xvars = " ".join(independents)
    code = f"reghdfe {dependent} {xvars}"
    if condition:
        code += f" if {condition}"
    opts = [f"absorb({absorb})"]
    if vce:
        opts.append(f"vce({vce})")
    if residuals:
        opts.append(f"residuals({residuals})")
    code += ", " + " ".join(opts)
    return _run(code)


def ppmlhdfe(
    dependent: str,
    independents: list[str],
    absorb: str,
    vce: str | None = None,
    condition: str | None = None,
    exposure: str | None = None,
    offset: str | None = None,
) -> StataResult:
    """Poisson PPML with high-dimensional fixed effects via ppmlhdfe.

    Use for count or non-negative outcomes (trade flows, patent counts, etc.).
    absorb: space-separated fixed-effect variables, e.g. "exporter importer year".
    vce: variance estimator, e.g. "robust" or "cluster id".
    exposure: variable for Poisson exposure (logged offset with coef forced to 1).
    offset: variable for a free-coefficient log offset.
    """
    xvars = " ".join(independents)
    code = f"ppmlhdfe {dependent} {xvars}"
    if condition:
        code += f" if {condition}"
    opts = [f"absorb({absorb})"]
    if vce:
        opts.append(f"vce({vce})")
    if exposure:
        opts.append(f"exposure({exposure})")
    if offset:
        opts.append(f"offset({offset})")
    code += ", " + " ".join(opts)
    return _run(code)


def run_raw_stata(code: str) -> StataResult:
    """Escape hatch: run arbitrary Stata code.

    The agent should prefer the structured tools above, but fall
    back to this for commands we haven't wrapped (xtreg, margins,
    user-contributed packages, etc.).
    """
    return _run(code)


def set_stata_locale(locale: str) -> StataResult:
    """Set Stata's locale for string functions (e.g. strupper/strlower).

    locale: POSIX-style locale string, e.g. "es_ES", "fr_FR", "de_DE".
    Silently succeeds even if Stata does not support the locale on this OS.
    """
    return _run(f"set locale_functions {locale}")