"""Locate the Stata installation and edition. Tries multiple strategies in order."""
import os
import sys
import platform
from dataclasses import dataclass
from pathlib import Path
import yaml

CONFIG_PATH = Path.home() / ".stataagent" / "config.yaml"

VALID_EDITIONS = ("be", "se", "mp")

# Common install locations by OS.
# Paths that include an edition name (StataSE, StataMP, StataBE) let us
# infer the edition automatically; generic names (Stata, stata19) require
# a prompt if the edition isn't already saved.
COMMON_PATHS = {
    "Darwin": [
        "/Applications/Stata/utilities",
        "/Applications/StataSE/utilities",
        "/Applications/StataMP/utilities",
        "/Applications/StataBE/utilities",
        "/Applications/StataNow/utilities",
        "/Applications/StataNowSE/utilities",
        "/Applications/StataNowMP/utilities",
        "/Applications/StataNowBE/utilities",
    ],
    "Windows": [
        r"C:\Program Files\Stata19\utilities",
        r"C:\Program Files\Stata18\utilities",
        r"C:\Program Files\Stata17\utilities",
        r"C:\Program Files\StataSE19\utilities",
        r"C:\Program Files\StataSE18\utilities",
        r"C:\Program Files\StataMP19\utilities",
        r"C:\Program Files\StataMP18\utilities",
        r"C:\Program Files\StataBE19\utilities",
        r"C:\Program Files\StataBE18\utilities",
        r"C:\Program Files\StataNow19\utilities",
        r"C:\Program Files\StataNowSE19\utilities",
        r"C:\Program Files\StataNowMP19\utilities",
        r"C:\Program Files\StataNowBE19\utilities",
    ],
    "Linux": [
        "/usr/local/stata19/utilities",
        "/usr/local/stata18/utilities",
        "/usr/local/stata17/utilities",
        "/usr/local/statanow19/utilities",
    ],
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StataInstall:
    """A fully resolved Stata installation: where it lives and which edition."""
    path: Path     # path to the utilities/ directory
    edition: str   # 'be', 'se', or 'mp'


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _normalize_user_path(raw: str) -> Path:
    """Normalize a raw path string before any filesystem operations.

    Handles, in order:
      1. Surrounding whitespace from copy-paste or YAML values
      2. Surrounding quotes (single or double) from copy-paste
      3. Shell environment variables ($HOME, %APPDATA%, etc.)
      4. Home-directory shorthand (~)
      5. Resolution of relative segments (..) to an absolute path
      6. Platform path-separator normalization (/ ↔ \\) via pathlib
    """
    cleaned = raw.strip().strip('"').strip("'").strip()
    cleaned = os.path.expandvars(cleaned)
    return Path(cleaned).expanduser().resolve()


def _is_valid_stata_path(path: Path) -> bool:
    """Return True if *path* (already a resolved Path) contains pystata."""
    return (path / "pystata").is_dir()


def _infer_edition(path: Path) -> str | None:
    """Guess the Stata edition from the path name, then from executables inside.

    Two-pass strategy:
      1. Check each path component for edition tokens in the folder name
         (e.g. 'StataMP', 'StataNowSE').  StataNow variants are tested
         first so 'StataNowMP' binds to 'mp' before the generic 'Stata' test.
      2. If the folder name gives no signal (e.g. 'StataNow', 'Stata19'),
         scan the install directory (parent of utilities/) for an executable
         whose name contains an edition token — e.g. 'StataMP-64.app' → 'mp'.

    Returns None only when both passes fail.
    """
    for part in path.parts:
        upper = part.upper()
        if "STATANOWMP" in upper:
            return "mp"
        if "STATANOWSE" in upper:
            return "se"
        if "STATANOWBE" in upper:
            return "be"
        if "STATAMP" in upper:
            return "mp"
        if "STATASE" in upper:
            return "se"
        if "STATABE" in upper:
            return "be"

    # Folder name gave no edition — scan the install directory for executables.
    install_dir = path.parent  # e.g. /Applications/StataNow
    try:
        for entry in install_dir.iterdir():
            upper = entry.name.upper()
            if "STATAMP" in upper:
                return "mp"
            if "STATASE" in upper:
                return "se"
            if "STATABE" in upper:
                return "be"
    except OSError:
        pass

    return None


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def _prompt_edition(path: Path) -> str:
    """Ask the user which edition is installed at *path*.

    Pre-fills the default with whatever can be inferred from the path name,
    falling back to 'se' if the path is ambiguous.
    """
    default = _infer_edition(path) or "se"
    print(f"\nWhich Stata edition is installed at '{path.parent}'?")
    print("  be  —  Stata/BE  (Basic Edition)")
    print("  se  —  Stata/SE  (Standard Edition)")
    print("  mp  —  Stata/MP  (Multi-Processor Edition)")
    while True:
        try:
            raw = input(f"Edition [be/se/mp] (default: {default}): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return default
        if not raw:
            return default
        if raw in VALID_EDITIONS:
            return raw
        print(f"  '{raw}' is not a valid edition. Enter be, se, or mp.")


def _prompt_choice(installs: list[Path]) -> StataInstall:
    """Ask the user to choose among multiple detected Stata installations."""
    print("\nFound multiple Stata installations:")
    for i, path in enumerate(installs, 1):
        inferred = _infer_edition(path)
        hint = f"  →  edition: {inferred}" if inferred else ""
        print(f"  {i}.  {path.parent}{hint}")

    while True:
        try:
            raw = input(f"\nEnter number [1–{len(installs)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "1"
        if raw.isdigit() and 1 <= int(raw) <= len(installs):
            chosen = installs[int(raw) - 1]
            edition = _infer_edition(chosen) or _prompt_edition(chosen)
            return StataInstall(path=chosen, edition=edition)
        print(f"  Please enter a number between 1 and {len(installs)}.")


# ---------------------------------------------------------------------------
# Discovery strategies
# ---------------------------------------------------------------------------

def _from_env() -> StataInstall | None:
    """Strategy 1: STATA_PATH (and optionally STATA_EDITION) env vars."""
    val = os.environ.get("STATA_PATH")
    if not val:
        return None
    path = _normalize_user_path(val)
    if not _is_valid_stata_path(path):
        return None
    edition = os.environ.get("STATA_EDITION", "").strip().lower()
    if edition not in VALID_EDITIONS:
        edition = _infer_edition(path) or _prompt_edition(path)
    return StataInstall(path=path, edition=edition)


def _from_config() -> StataInstall | None:
    """Strategy 2: saved ~/.stataagent/config.yaml."""
    if not CONFIG_PATH.exists():
        return None
    try:
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        raw_path = data.get("stata_path", "")
        if not raw_path:
            return None
        path = _normalize_user_path(str(raw_path))
        if not _is_valid_stata_path(path):
            return None
        edition = str(data.get("stata_edition", "")).strip().lower()
        if edition not in VALID_EDITIONS:
            # Path was saved before edition tracking was added — ask once,
            # then fall through to _save_config so it's persisted.
            edition = _infer_edition(path) or _prompt_edition(path)
        return StataInstall(path=path, edition=edition)
    except (yaml.YAMLError, OSError):
        return None


def _autodetect() -> list[Path]:
    """Strategy 3: scan common install locations, returning ALL valid paths."""
    candidates = COMMON_PATHS.get(platform.system(), [])
    return [
        path
        for candidate in candidates
        if _is_valid_stata_path(path := _normalize_user_path(candidate))
    ]


def _interactive_prompt() -> StataInstall | None:
    """Strategy 4: ask the user for the utilities path, then the edition."""
    print("\nStataAgent could not automatically find your Stata installation.")
    print("Please enter the path to your Stata 'utilities' folder.")
    print("Examples:")
    print("  macOS:   /Applications/Stata/utilities")
    print("  Windows: C:\\Program Files\\Stata19\\utilities")
    print("  Linux:   /usr/local/stata19/utilities")
    print("(Press Ctrl+C to quit.)\n")

    while True:
        try:
            raw = input("Stata utilities path: ")
        except (EOFError, KeyboardInterrupt):
            return None

        path = _normalize_user_path(raw)
        if _is_valid_stata_path(path):
            edition = _infer_edition(path) or _prompt_edition(path)
            return StataInstall(path=path, edition=edition)
        print(f"  Couldn't find pystata at '{path}'. Try again, or Ctrl+C to quit.")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_config(install: StataInstall) -> None:
    """Persist the chosen installation to ~/.stataagent/config.yaml."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(
            {"stata_path": str(install.path), "stata_edition": install.edition},
            f,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_stata() -> StataInstall:
    """Resolve the Stata installation using the strategy chain.

    Order of preference:
      1. STATA_PATH / STATA_EDITION env vars (non-interactive)
      2. ~/.stataagent/config.yaml          (saved from a previous run)
      3. Scan common install locations       (auto-detect)
         - Exactly one found, edition clear  → use silently
         - Exactly one found, edition unclear → prompt for edition only
         - More than one found               → prompt user to choose
      4. Interactive path prompt             (last resort)

    The result is saved to ~/.stataagent/config.yaml so the user is only
    asked once (or again only if their installation moves).
    """
    # Strategies 1 & 2 — fully non-interactive when info is complete
    for strategy in (_from_env, _from_config):
        result = strategy()
        if result:
            return result

    # Strategy 3 — autodetect; may need a prompt for edition or choice
    found = _autodetect()
    if len(found) == 1:
        edition = _infer_edition(found[0])
        if edition:
            result = StataInstall(path=found[0], edition=edition)
            print(f"Found Stata/{edition.upper()} at '{found[0].parent}'.")
        else:
            result = StataInstall(path=found[0], edition=_prompt_edition(found[0]))
    elif len(found) > 1:
        result = _prompt_choice(found)
    else:
        # Strategy 4 — nothing found at all
        result = _interactive_prompt()

    if result is None:
        raise RuntimeError("Stata path not configured. Cannot continue.")

    _save_config(result)
    print(f"Saved Stata configuration to {CONFIG_PATH}. You won't be asked again.\n")
    return result


def reset_saved_config() -> None:
    """Delete the saved ~/.stataagent/config.yaml.

    The next call to find_stata() will re-run the full discovery chain
    (autodetect → interactive prompt) and save a fresh result.
    Has no effect if no config has been saved yet.
    """
    if CONFIG_PATH.exists():
        CONFIG_PATH.unlink()
        print(f"Cleared saved Stata configuration ({CONFIG_PATH}).")
    else:
        print("No saved Stata configuration found — nothing to reset.")


def initialize_pystata(*, reset: bool = False) -> None:
    """Resolve the Stata installation, extend sys.path, and call config.init().

    Steps executed in strict order:
      1. find_stata() → resolves path and edition (prompting if needed)
      2. sys.path.insert → makes 'from pystata import ...' importable
      3. pystata config.init(edition) → starts the Stata engine; pystata
         locates the executable automatically from its own position on
         sys.path (the utilities/ directory added in step 2)
    """
    if reset:
        reset_saved_config()

    install = find_stata()
    sys.path.insert(0, str(install.path))

    from pystata import config as _pystata_config  # noqa: PLC0415
    _pystata_config.init(install.edition)
