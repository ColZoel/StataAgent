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


def run_raw_stata(code: str) -> StataResult:
    """Escape hatch: run arbitrary Stata code.

    The agent should prefer the structured tools above, but fall
    back to this for commands we haven't wrapped (xtreg, margins,
    user-contributed packages, etc.).
    """
    return _run(code)