"""
StataAgent web UI — FastAPI backend.

Serves a zero-build single-page app (webui/static/) modeled on the Stata
desktop layout: Results window, Command window, Variables pane, and a
Review (history) pane. The agent itself is the same one the CLI uses
(hf_agent.build_agent); this module only adds an HTTP session around it.

Run directly:
    python web.py [--port 8765] [--config config.yaml]

Endpoints:
    GET  /                  the app
    GET  /api/state         session snapshot (dataset, variables, rigor, ...)
    POST /api/query         run one agent turn; streams SSE events
    POST /api/rigor         change rigor level
    POST /api/reset         clear conversation + context
    GET  /api/graph/{name}  serve a graph file produced during the session
"""
from __future__ import annotations

import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from __init__ import __version__
from hf_agent import _LITELLM_PREFIXES, AgentConfig, StataContext, build_agent
from stata_tools import (
    add_run_observer,
    check_stata,
    get_obs_count,
    get_stata_cwd,
    get_variable_meta,
    get_varnames,
    load_dataset,
    remove_run_observer,
)
import config as stata_config
import yaml
from dotenv import dotenv_values

_GRAPH_EXTS = (".png", ".svg", ".pdf")
_VALID_RIGOR = {"silent", "colleague", "reviewer", "journal_editor"}

# Default generation parameters — mirror AgentConfig.from_yaml's fallbacks.
# The UI's "Reset to defaults" targets these values.
_DEFAULT_TEMPERATURE = 0.3
_DEFAULT_MAX_TOKENS = 4096


# Request bodies. These must live at module level: with
# `from __future__ import annotations`, FastAPI resolves the string
# annotations against module globals — a class defined inside create_app()
# would not be found and the parameter would be misread as a query param.
class QueryIn(BaseModel):
    message: str


class RigorIn(BaseModel):
    level: str


class SettingsIn(BaseModel):
    """Partial settings update — only the fields present are applied.

    api_key is write-only: it is stored in .env and never returned to the
    client. base_url accepts "" to clear the override.
    """
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    commentary: bool | None = None
    stata_path: str | None = None
    stata_edition: str | None = None


# ---------------------------------------------------------------------------
# Settings persistence helpers
# ---------------------------------------------------------------------------
_ENV_PATH = _ROOT / ".env"
_CONFIG_YAML = _ROOT / "config.yaml"


def _yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(str(v))  # quoted string, YAML-safe


def _update_config_yaml(updates: dict) -> None:
    """Rewrite only the changed top-level keys in config.yaml, preserving
    every comment and all other lines."""
    text = _CONFIG_YAML.read_text() if _CONFIG_YAML.exists() else ""
    lines = text.splitlines()
    remaining = dict(updates)
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", line)
        if m and m.group(1) in remaining:
            key = m.group(1)
            lines[i] = f"{key}: {_yaml_scalar(remaining.pop(key))}"
    for key, value in remaining.items():
        lines.append(f"{key}: {_yaml_scalar(value)}")
    _CONFIG_YAML.write_text("\n".join(lines) + "\n")


def _write_env_var(name: str, value: str) -> None:
    """Set NAME=value in the project .env (replace or append) and in the
    running process environment so it takes effect without a restart."""
    lines = _ENV_PATH.read_text().splitlines() if _ENV_PATH.exists() else []
    pattern = re.compile(rf"^\s*(export\s+)?{re.escape(name)}\s*=")
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"{name}={value}"
            break
    else:
        lines.append(f"{name}={value}")
    _ENV_PATH.write_text("\n".join(lines) + "\n")
    os.environ[name] = value


def _api_key_is_set(env_name: str | None) -> bool:
    if not env_name:
        return False
    if os.environ.get(env_name):
        return True
    try:
        return bool(dotenv_values(_ENV_PATH).get(env_name))
    except Exception:
        return False


def _saved_stata_config() -> dict:
    """Read ~/.stataagent/config.yaml (path + edition), or {}."""
    try:
        if stata_config.CONFIG_PATH.exists():
            with open(stata_config.CONFIG_PATH) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


class Session:
    """One in-memory conversation. Stata is a single instance, so there is
    exactly one session and queries are serialized behind a lock."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.agent = None            # built lazily on first query
        self.agent_error: str | None = None
        self.ctx = StataContext(rigor=cfg.rigor)
        self.history: list = []
        self.busy = False
        self.stata_status = check_stata()  # (ok, edition, version, brief)
        # Drag-and-drop support: uploaded files land here, and notes about
        # them are prepended to the next query so the model has the context.
        self.upload_dir = Path(tempfile.mkdtemp(prefix="stataagent-uploads-"))
        self.context_notes: list[str] = []
        # Conversation size (for the UI meter): completed turns, and the raw
        # model-message count — the true proxy for context growth, since it
        # includes tool calls and their Stata logs.
        self.turn_count = 0
        # Graph tracking: watch the project root AND Stata's own cwd (which
        # the agent can change with `cd`). Files are served by registry name,
        # never by path lookup.
        self.session_started = time.time()
        self.stata_cwd: str | None = get_stata_cwd()
        self.graph_registry: dict[str, Path] = {}
        self.known_graphs: set[Path] = self._scan_graphs()

    def refresh_stata_cwd(self, force: bool = False) -> str | None:
        """Re-read c(pwd). Skipped while busy (Stata is single-threaded)
        unless force=True — the query worker forces it since it holds the lock."""
        if force or not self.busy:
            cwd = get_stata_cwd()
            if cwd:
                self.stata_cwd = cwd
        return self.stata_cwd

    def _graph_dirs(self) -> list[Path]:
        dirs = [_ROOT]
        if self.stata_cwd:
            p = Path(self.stata_cwd)
            if p.is_dir() and p != _ROOT:
                dirs.append(p)
        return dirs

    def _scan_graphs(self) -> set[Path]:
        found: set[Path] = set()
        for d in self._graph_dirs():
            for ext in _GRAPH_EXTS:
                found |= set(d.glob(f"*{ext}"))
        return found

    def new_graphs(self) -> list[str]:
        """Register graphs that appeared since the last scan; return their
        registry names. Files predating the session are ignored so that
        watching a newly cd'd directory doesn't flood the UI with old plots."""
        names: list[str] = []
        for p in sorted(self._scan_graphs()):
            if p in self.known_graphs:
                continue
            self.known_graphs.add(p)
            try:
                if p.stat().st_mtime < self.session_started:
                    continue
            except OSError:
                continue
            name, counter = p.name, 1
            while name in self.graph_registry and self.graph_registry[name] != p:
                name = f"{p.stem}_{counter}{p.suffix}"
                counter += 1
            self.graph_registry[name] = p
            names.append(name)
        return names

    def ensure_agent(self):
        if self.agent is None and self.agent_error is None:
            try:
                self.agent = build_agent(self.cfg)
            except Exception as exc:
                self.agent_error = f"{type(exc).__name__}: {exc}"
        return self.agent

    def take_context_notes(self) -> list[str]:
        """Return and clear pending context notes (from dropped files)."""
        notes, self.context_notes = self.context_notes, []
        return notes

    def reset(self):
        self.ctx = StataContext(rigor=self.cfg.rigor)
        self.history = []
        self.known_graphs = self._scan_graphs()
        self.context_notes = []
        self.turn_count = 0


SESSION: Session | None = None


def create_app(cfg: AgentConfig | None = None) -> FastAPI:
    global SESSION
    if cfg is None:
        cfg = AgentConfig.from_yaml(_ROOT / "config.yaml")
    SESSION = Session(cfg)

    app = FastAPI(title="StataAgent", version=__version__)

    # ------------------------------------------------------------------ state
    @app.get("/api/state")
    def state():
        s = SESSION
        stata_ok, edition, version, brief = s.stata_status
        return {
            "version": __version__,
            "stata": {
                "ok": stata_ok,
                "edition": edition,
                "version": version,
                "error": brief,
            },
            "provider": s.cfg.provider,
            "model": s.cfg.model,
            "rigor": s.ctx.rigor,
            "commentary": s.cfg.commentary,
            "busy": s.busy,
            "agent_error": s.agent_error,
            "dataset": {
                "loaded": s.ctx.dataset_loaded,
                "path": s.ctx.dataset_path,
                "obs": get_obs_count() if s.ctx.dataset_loaded else 0,
            },
            "variables": get_variable_meta() if s.ctx.dataset_loaded else [],
            "cwd": s.refresh_stata_cwd() or str(_ROOT),
            "conversation": {
                "turns": s.turn_count,
                "messages": len(s.history),
            },
        }

    # ------------------------------------------------------------------ query
    @app.post("/api/query")
    def run_query(q: QueryIn):
        s = SESSION
        prompt = q.message.strip()
        if not prompt:
            return JSONResponse({"error": "Empty query."}, status_code=400)
        if not s.lock.acquire(blocking=False):
            return JSONResponse(
                {"error": "A query is already running."}, status_code=409
            )

        events: queue.Queue = queue.Queue()

        def observer(code, result):
            events.put({
                "type": "stata",
                "code": code,
                "output": result.output,
                "success": result.success,
                "error": result.error,
            })

        def worker():
            s.busy = True
            add_run_observer(observer)
            notes: list[str] = []
            try:
                agent = s.ensure_agent()
                if agent is None:
                    events.put({
                        "type": "error",
                        "message": f"Agent could not be initialised: {s.agent_error}",
                    })
                    return
                # Prepend any pending drag-and-drop context (loaded datasets,
                # attached do-files) so the model knows about them.
                notes = s.take_context_notes()
                full_prompt = (
                    "\n\n".join(notes) + f"\n\n{prompt}" if notes else prompt
                )
                result = agent.run_sync(
                    full_prompt, deps=s.ctx, message_history=s.history
                )
                s.history = result.all_messages()
                s.turn_count += 1
                # The agent may have cd'd Stata elsewhere; follow it so graph
                # exports in the new directory are picked up.
                s.refresh_stata_cwd(force=True)
                text = getattr(result, "output", None)
                if text is None:
                    text = getattr(result, "data", "")
                events.put({
                    "type": "response",
                    "text": str(text),
                    "graphs": s.new_graphs(),
                })
            except Exception as exc:
                # Restore unconsumed context so a transient failure doesn't
                # silently drop the user's attached files.
                s.context_notes = notes + s.context_notes
                events.put({
                    "type": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                })
            finally:
                remove_run_observer(observer)
                s.busy = False
                events.put(None)  # sentinel: stream done
                s.lock.release()

        threading.Thread(target=worker, daemon=True).start()

        def sse():
            while True:
                ev = events.get()
                if ev is None:
                    yield "data: {\"type\": \"done\"}\n\n"
                    break
                yield f"data: {json.dumps(ev)}\n\n"

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ----------------------------------------------------------------- upload
    # Drag-and-drop files, Stata-GUI style: a .dta is loaded into memory
    # immediately (like dropping it on Stata); a .do file is attached as
    # context for the model's next turn. The file arrives as a raw request
    # body (?filename=...) to avoid a multipart dependency.
    _UPLOAD_LIMITS = {
        ".dta": 512 * 1024 * 1024,   # 512 MB
        ".do": 1 * 1024 * 1024,      # 1 MB — do-files are code, keep them sane
    }
    _DO_CONTEXT_CHARS = 100_000      # cap of do-file text passed to the model

    @app.post("/api/upload")
    async def upload(request: Request, filename: str):
        s = SESSION
        name = Path(filename).name.strip()
        ext = Path(name).suffix.lower()
        if not name or ext not in _UPLOAD_LIMITS:
            return JSONResponse(
                {"error": "Only .dta and .do files are supported."},
                status_code=400,
            )

        body = await request.body()
        if not body:
            return JSONResponse({"error": "Empty file."}, status_code=400)
        if len(body) > _UPLOAD_LIMITS[ext]:
            limit_mb = _UPLOAD_LIMITS[ext] // (1024 * 1024)
            return JSONResponse(
                {"error": f"{name} is too large (limit {limit_mb} MB for {ext})."},
                status_code=413,
            )

        # Save under a collision-free name in the session upload dir.
        target = s.upload_dir / name
        counter = 1
        while target.exists():
            target = s.upload_dir / f"{Path(name).stem}_{counter}{ext}"
            counter += 1
        target.write_bytes(body)

        # ---- .do: attach contents as context for the next turn -----------
        if ext == ".do":
            text = body.decode("utf-8", errors="replace")
            truncated = len(text) > _DO_CONTEXT_CHARS
            context_text = text[:_DO_CONTEXT_CHARS]
            s.context_notes.append(
                f'[Attached file] The user attached the Stata do-file '
                f'"{target.name}" (saved at {target}) as context:\n'
                f"```stata\n{context_text}\n```"
                + ("\n(Contents truncated for length.)" if truncated else "")
            )
            lines = text.splitlines()
            return {
                "kind": "do",
                "name": target.name,
                "success": True,
                "lines": len(lines),
                "preview": "\n".join(lines[:12]),
                "truncated": truncated,
                "message": f"{target.name} attached — its contents will be "
                           "shared with the agent on your next question.",
            }

        # ---- .dta: load into Stata now (serialized with queries) ---------
        if not s.stata_status[0]:
            return {
                "kind": "dta",
                "name": target.name,
                "success": False,
                "code": f'use "{target}", clear',
                "log": "",
                "error": "Stata is not available on this machine — check the "
                         "Stata path in Settings and restart StataAgent.",
            }
        if not s.lock.acquire(blocking=False):
            return JSONResponse(
                {"error": "A query is running — try again when it finishes."},
                status_code=409,
            )
        try:
            s.busy = True
            code = f'use "{target}", clear'
            result = load_dataset(str(target))
            if not result.success:
                return {
                    "kind": "dta",
                    "name": target.name,
                    "success": False,
                    "code": code,
                    "log": result.output,
                    "error": result.error,
                }
            s.ctx.dataset_loaded = True
            s.ctx.dataset_path = str(target)
            try:
                s.ctx.varnames = set(get_varnames())
            except Exception:
                pass
            s.context_notes.append(
                f'[Loaded dataset] The user loaded "{target.name}" by '
                f"drag-and-drop (path: {target}). It is now in Stata memory; "
                "call describe_dataset to inspect it before analysis."
            )
            return {
                "kind": "dta",
                "name": target.name,
                "success": True,
                "code": code,
                "log": result.output,
            }
        finally:
            s.busy = False
            s.lock.release()

    # ------------------------------------------------------------------ rigor
    @app.post("/api/rigor")
    def set_rigor(r: RigorIn):
        level = r.level.lower().strip()
        if level not in _VALID_RIGOR:
            return JSONResponse(
                {"error": f"Unknown rigor level '{r.level}'."}, status_code=400
            )
        SESSION.ctx.rigor = level
        return {"rigor": level}

    # --------------------------------------------------------------- settings
    @app.get("/api/settings")
    def get_settings():
        s = SESSION
        saved = _saved_stata_config()
        stata_ok = s.stata_status[0]
        return {
            "version": __version__,
            "provider": s.cfg.provider,
            "providers": sorted(_LITELLM_PREFIXES),
            "model": s.cfg.model,
            "base_url": s.cfg.base_url,
            "api_key_env": s.cfg.api_key_env,
            "api_key_set": _api_key_is_set(s.cfg.api_key_env),
            "temperature": s.cfg.temperature,
            "max_tokens": s.cfg.max_tokens,
            "commentary": s.cfg.commentary,
            "defaults": {
                "temperature": _DEFAULT_TEMPERATURE,
                "max_tokens": _DEFAULT_MAX_TOKENS,
            },
            "stata": {
                "path": saved.get("stata_path") or "",
                "edition": saved.get("stata_edition") or "",
                "env_override": bool(os.environ.get("STATA_PATH")),
                "running": stata_ok,
            },
            "config_yaml": str(_CONFIG_YAML),
            "env_file": str(_ENV_PATH),
            "stata_config_yaml": str(stata_config.CONFIG_PATH),
        }

    @app.get("/api/settings/stata-detect")
    def stata_detect():
        found = stata_config._autodetect()
        return {
            "candidates": [
                {
                    "path": str(p),
                    "edition": stata_config._infer_edition(p) or "",
                }
                for p in found
            ]
        }

    @app.get("/api/update-check")
    def update_check():
        """Query GitHub for the latest release and compare with this version.

        status is one of: 'update' (newer available), 'current' (up to date),
        'error' (could not reach GitHub / parse the response).
        """
        import urllib.error
        import urllib.request
        from hf_agent import _GITHUB_RELEASES_URL, _version_gt

        try:
            req = urllib.request.Request(
                _GITHUB_RELEASES_URL,
                headers={"Accept": "application/vnd.github+json",
                         "User-Agent": f"StataAgent/{__version__}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            latest = str(data.get("tag_name", "")).lstrip("v")
            if not latest:
                return {"current": __version__, "status": "none"}
            if _version_gt(latest, __version__):
                return {"current": __version__, "latest": latest, "status": "update"}
            return {"current": __version__, "latest": latest, "status": "current"}
        except urllib.error.HTTPError as exc:
            # 404 from releases/latest means the repo has no published releases.
            if exc.code == 404:
                return {"current": __version__, "status": "none"}
            return {"current": __version__, "status": "error", "error": str(exc)}
        except Exception as exc:
            return {"current": __version__, "status": "error", "error": str(exc)}

    @app.post("/api/settings")
    def save_settings(body: SettingsIn):
        s = SESSION
        if s.busy:
            return JSONResponse(
                {"error": "Cannot change settings while a query is running."},
                status_code=409,
            )

        # ---- validate ---------------------------------------------------
        if body.provider is not None and body.provider not in _LITELLM_PREFIXES:
            return JSONResponse(
                {"error": f"Unknown provider '{body.provider}'."}, status_code=400
            )
        if body.temperature is not None and not (0.0 <= body.temperature <= 2.0):
            return JSONResponse(
                {"error": "Temperature must be between 0 and 2."}, status_code=400
            )
        if body.max_tokens is not None and body.max_tokens < 1:
            return JSONResponse(
                {"error": "max_tokens must be positive."}, status_code=400
            )
        if body.api_key is not None:
            key = body.api_key.strip()
            if "\n" in key or "\r" in key:
                return JSONResponse(
                    {"error": "API key must be a single line."}, status_code=400
                )
            body.api_key = key

        provider = body.provider if body.provider is not None else s.cfg.provider
        base_url = (
            (body.base_url.strip() or None)
            if body.base_url is not None
            else s.cfg.base_url
        )
        if provider == "openai_compatible" and not base_url:
            return JSONResponse(
                {"error": "Provider 'openai_compatible' requires a base URL."},
                status_code=400,
            )

        messages: list[str] = []
        restart_required = False

        # ---- model / provider config → config.yaml ----------------------
        yaml_updates: dict = {}
        for attr in ("provider", "model", "api_key_env", "temperature",
                     "max_tokens", "commentary"):
            val = getattr(body, attr)
            if val is not None and val != getattr(s.cfg, attr):
                setattr(s.cfg, attr, val)
                yaml_updates[attr] = val
        if body.base_url is not None and base_url != s.cfg.base_url:
            s.cfg.base_url = base_url
            yaml_updates["base_url"] = base_url
        if yaml_updates:
            _update_config_yaml(yaml_updates)
            messages.append("Model settings saved to config.yaml.")

        # ---- API key → .env ---------------------------------------------
        if body.api_key:
            env_name = s.cfg.api_key_env
            if not env_name:
                return JSONResponse(
                    {"error": "Set the API key variable name before saving a key."},
                    status_code=400,
                )
            _write_env_var(env_name, body.api_key)
            messages.append(f"API key saved to .env as {env_name}.")

        # Any of the above invalidates the cached agent — rebuild on next query.
        if yaml_updates or body.api_key:
            s.agent = None
            s.agent_error = None

        # ---- Stata install → ~/.stataagent/config.yaml -------------------
        if body.stata_path is not None and body.stata_path.strip():
            path = stata_config._normalize_user_path(body.stata_path)
            if not stata_config._is_valid_stata_path(path):
                return JSONResponse(
                    {"error": f"No pystata found at '{path}'. The path should "
                              "point to Stata's 'utilities' folder."},
                    status_code=400,
                )
            edition = (body.stata_edition or "").strip().lower()
            if edition not in stata_config.VALID_EDITIONS:
                edition = stata_config._infer_edition(path) or "se"
            saved = _saved_stata_config()
            changed = (
                str(path) != saved.get("stata_path")
                or edition != saved.get("stata_edition")
            )
            stata_config._save_config(
                stata_config.StataInstall(path=path, edition=edition)
            )
            if changed:
                messages.append(
                    f"Stata path saved ({edition.upper()}). Restart StataAgent "
                    "to load the new installation."
                )
                restart_required = True
            else:
                messages.append("Stata path unchanged.")
            if os.environ.get("STATA_PATH"):
                messages.append(
                    "Note: the STATA_PATH environment variable is set and "
                    "takes precedence over the saved path."
                )

        if not messages:
            messages.append("Nothing to save.")
        return {"ok": True, "restart_required": restart_required,
                "messages": messages}

    # ------------------------------------------------------------------ reset
    @app.post("/api/reset")
    def reset():
        if SESSION.busy:
            return JSONResponse(
                {"error": "Cannot reset while a query is running."},
                status_code=409,
            )
        SESSION.reset()
        return {"ok": True}

    # ----------------------------------------------------------------- graphs
    @app.get("/api/graph/{name}")
    def graph(name: str):
        # Served strictly from the session registry — no filesystem lookup,
        # so there is nothing to traverse.
        p = SESSION.graph_registry.get(name)
        if p is None or not p.exists():
            return JSONResponse({"error": "Not found."}, status_code=404)
        return FileResponse(p)

    # ------------------------------------------------------------------- app
    _NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate"}

    @app.get("/")
    def index():
        return FileResponse(_HERE / "static" / "index.html", headers=_NO_CACHE)

    # StaticFiles with caching disabled — this is a locally served dev tool,
    # so revalidate every asset. Prevents stale CSS/JS after a UI update.
    class NoCacheStatic(StaticFiles):
        def is_not_modified(self, response_headers, request_headers) -> bool:
            return False  # never serve a 304; always send fresh bytes

        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            resp.headers.update(_NO_CACHE)
            return resp

    app.mount("/static", NoCacheStatic(directory=_HERE / "static"), name="static")
    return app


def launch(cfg: AgentConfig | None = None, host: str = "127.0.0.1",
           port: int = 8765, open_browser: bool = True) -> None:
    """Build the app and serve it with uvicorn (blocking)."""
    import uvicorn

    app = create_app(cfg)
    if open_browser:
        import webbrowser

        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://{host}:{port}")
        ).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
