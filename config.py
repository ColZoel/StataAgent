"""Locate the Stata installation. Tries multiple strategies in order."""
import os
import sys
import platform
from pathlib import Path
import yaml

CONFIG_PATH = Path.home() / ".stataagent" / "config.yaml"

# Common install locations by OS. Add more as you discover them.
COMMON_PATHS = {
    "Darwin": [  # macOS
        "/Applications/Stata/utilities",
        "/Applications/StataSE/utilities",
        "/Applications/StataMP/utilities",
        "/Applications/StataBE/utilities",
    ],
    "Windows": [
        r"C:\Program Files\Stata19\utilities",
        r"C:\Program Files\Stata18\utilities",
        r"C:\Program Files\Stata17\utilities",
    ],
    "Linux": [
        "/usr/local/stata19/utilities",
        "/usr/local/stata18/utilities",
        "/usr/local/stata17/utilities",
    ],
}


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
    cleaned = os.path.expandvars(cleaned)   # $VAR / %VAR%
    return Path(cleaned).expanduser().resolve()


def _is_valid_stata_path(path: Path) -> bool:
    """Return True if *path* (already a resolved Path) contains pystata."""
    return (path / "pystata").is_dir()


def _from_env() -> Path | None:
    """Strategy 1: environment variable."""
    val = os.environ.get("STATA_PATH")
    if not val:
        return None
    path = _normalize_user_path(val)
    return path if _is_valid_stata_path(path) else None


def _from_config() -> Path | None:
    """Strategy 2: saved config file."""
    if not CONFIG_PATH.exists():
        return None
    try:
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("stata_path", "")
        if not raw:
            return None
        path = _normalize_user_path(str(raw))
        return path if _is_valid_stata_path(path) else None
    except (yaml.YAMLError, OSError):
        return None


def _autodetect() -> Path | None:
    """Strategy 3: scan common install locations."""
    candidates = COMMON_PATHS.get(platform.system(), [])
    for candidate in candidates:
        path = _normalize_user_path(candidate)
        if _is_valid_stata_path(path):
            return path
    return None


def _interactive_prompt() -> Path | None:
    """Strategy 4: ask the user."""
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
            return path
        print(f"Couldn't find pystata at '{path}'. Try again, or Ctrl+C to quit.")


def _save_config(path: Path) -> None:
    """Persist the discovered path so we don't ask again."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump({"stata_path": str(path)}, f)


def find_stata() -> Path:
    """Resolve the Stata path using the strategy chain. Raises if nothing works."""
    # Try each strategy in order of decreasing automation.
    for strategy in (_from_env, _from_config, _autodetect):
        result = strategy()
        if result:
            return result

    # Last resort: ask the user.
    result = _interactive_prompt()
    if result is None:
        raise RuntimeError("Stata path not configured. Cannot continue.")

    # Save for next time so this only happens once.
    _save_config(result)
    print(f"Saved Stata path to {CONFIG_PATH}. You won't be asked again.\n")
    return result


def initialize_pystata() -> None:
    """Resolve the path and add pystata to sys.path. Call once at startup."""
    stata_path = find_stata()
    sys.path.insert(0, str(stata_path))
    # Now the import works:
    # from pystata import config, stata