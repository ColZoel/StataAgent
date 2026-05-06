# ui.py — StataAgent Streamlit interface
import sys
import datetime
import traceback
from pathlib import Path

# ── File-based debug log (always written, regardless of Streamlit's stderr) ─
_HERE = Path(__file__).resolve().parent
_DEBUG_LOG = _HERE / ".ui_debug.log"


def _dlog(msg: str) -> None:
    """Append a timestamped line to .ui_debug.log — survives any stderr capture."""
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass


# Truncate the log on each fresh script start so old runs don't clutter it
try:
    _DEBUG_LOG.write_text("", encoding="utf-8")
except Exception:
    pass

_dlog("ui.py started — about to import streamlit")

import streamlit as st

_dlog("streamlit imported")

# ── Page config must be first st.* call ───────────────────────────────────
st.set_page_config(
    page_title="StataAgent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
_dlog("set_page_config done")

# Ensure the project directory is on sys.path regardless of how Streamlit
# was launched (subprocess, streamlit run from another cwd, etc.)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Local imports — wrapped so failures show in the browser rather than producing
# a blank page. BaseException covers SystemExit raised by sys.exit() in the
# import chain (e.g. from interactive prompts in config.find_stata).
_dlog("about to import hf_agent / agent / pydantic_ai")
try:
    from hf_agent import AgentConfig, build_agent
    _dlog("hf_agent imported")
    from agent import StataContext
    _dlog("agent imported")
    from pydantic_ai.messages import ModelMessage
    _dlog("pydantic_ai.messages imported")
    _IMPORT_ERROR: BaseException | None = None
except BaseException as exc:
    _IMPORT_ERROR = exc
    _dlog(f"IMPORT FAILED: {type(exc).__name__}: {exc}")
    _dlog(traceback.format_exc())

if _IMPORT_ERROR is not None:
    st.error(f"Failed to import StataAgent modules: {type(_IMPORT_ERROR).__name__}")
    st.exception(_IMPORT_ERROR)
    st.info(
        "Common causes on Windows:\n"
        "- Conda env not activated before launching\n"
        "- `pydantic-ai` or `pydantic-ai-litellm` not installed\n"
        "- Stata path not configured: run `python main.py --no-ui` once from the terminal "
        "to complete interactive Stata discovery, then retry the UI\n"
        f"\nSee {_DEBUG_LOG} for the full trace."
    )
    st.stop()

_dlog("all imports successful — entering main UI body")

# ── Stata blue theme ───────────────────────────────────────────────────────
st.markdown("""
<style>
    :root { --stata-blue: #0071bc; }

    .stChatMessage [data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] th {
        background: #0071bc;
        color: white;
        padding: 6px 12px;
        text-align: left;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] td {
        padding: 4px 12px;
        border-bottom: 1px solid #eee;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] blockquote {
        border-left: 4px solid #0071bc;
        background: #f0f7ff;
        padding: 8px 16px;
        margin: 8px 0;
        border-radius: 0 4px 4px 0;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] code {
        background: #1e1e2e;
        color: #cdd6f4;
        padding: 2px 6px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────
def _extract_tool_logs(messages: list) -> list[tuple[str, str]]:
    """Pull (label, output) pairs from pydantic_ai tool return parts."""
    logs = []
    call_map: dict[str, tuple[str, str]] = {}

    for msg in messages:
        for part in getattr(msg, "parts", []):
            ptype = type(part).__name__
            if "ToolCall" in ptype and "Return" not in ptype:
                tool_name = getattr(part, "tool_name", "")
                args = getattr(part, "args", {})
                if hasattr(args, "args_dict"):
                    args = args.args_dict()
                if isinstance(args, dict):
                    cmd = args.get("code", args.get("path", args.get("dependent", str(args))))
                else:
                    cmd = str(args)
                call_map[getattr(part, "tool_call_id", "")] = (tool_name, str(cmd)[:80])
            elif "ToolReturn" in ptype or "ReturnPart" in ptype:
                cid = getattr(part, "tool_call_id", "")
                label, cmd = call_map.pop(cid, ("tool", ""))
                content = str(getattr(part, "content", ""))
                logs.append((f"{label}({cmd})", content))

    return logs


def _detect_new_graphs(known: set[Path], workdir: Path) -> list[Path]:
    """Return PNG/SVG files in workdir that aren't already in known."""
    found = []
    for ext in ("*.png", "*.svg", "*.pdf"):
        for p in workdir.glob(ext):
            if p not in known:
                found.append(p)
    return found


def _get_agent():
    """Build the agent on first use and cache it in session state."""
    if "agent" not in st.session_state:
        with st.spinner("Initialising agent..."):
            try:
                st.session_state.agent = build_agent(load_config())
            except Exception as exc:
                st.error("Failed to build agent.")
                st.exception(exc)
                st.stop()
    return st.session_state.agent


# ── Config (cached across reruns) ──────────────────────────────────────────
@st.cache_resource
def load_config() -> "AgentConfig":
    try:
        return AgentConfig.from_yaml(_HERE / "config.yaml")
    except Exception as exc:
        st.error(f"Could not load config.yaml: {exc}")
        st.stop()


# ── Session state init ─────────────────────────────────────────────────────
cfg = load_config()

if "ctx" not in st.session_state:
    st.session_state.ctx = StataContext(rigor=cfg.rigor)
if "history" not in st.session_state:
    st.session_state.history: list[ModelMessage] = []
if "messages" not in st.session_state:
    st.session_state.messages: list[tuple[str, str]] = []
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer: list[tuple[str, str]] = []
if "graph_paths" not in st.session_state:
    st.session_state.graph_paths: list[Path] = []
if "known_graphs" not in st.session_state:
    st.session_state.known_graphs: set[Path] = (
        set(_HERE.glob("*.png")) | set(_HERE.glob("*.svg"))
    )


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 StataAgent")
    st.caption("An AI that speaks Stata")
    st.divider()

    st.markdown("**Dataset**")
    if st.session_state.ctx.dataset_loaded:
        dpath = st.session_state.ctx.dataset_path or ""
        st.success(f"`{Path(dpath).name}`")
    else:
        st.info("No dataset loaded")

    st.divider()

    st.markdown("**Rigor Level**")
    _RIGOR_OPTIONS = ["silent", "colleague", "reviewer", "journal_editor"]
    _RIGOR_DESCRIPTIONS = {
        "silent":         "Runs commands as requested. No unsolicited commentary.",
        "colleague":      "Flags obvious issues once, as a friendly suggestion.",
        "reviewer":       "Checks standard assumptions. States caveats explicitly.",
        "journal_editor": "Full assumption audit. Requires justification for specs.",
    }
    _current_rigor = st.session_state.ctx.rigor
    if _current_rigor not in _RIGOR_OPTIONS:
        _current_rigor = "colleague"

    selected_rigor = st.radio(
        label="rigor",
        options=_RIGOR_OPTIONS,
        index=_RIGOR_OPTIONS.index(_current_rigor),
        label_visibility="collapsed",
    )
    st.caption(_RIGOR_DESCRIPTIONS[selected_rigor])

    if selected_rigor != st.session_state.ctx.rigor:
        st.session_state.ctx.rigor = selected_rigor

    st.divider()

    st.markdown("**Analysis Plan**")
    plan = getattr(st.session_state.ctx, "current_plan", None)
    if plan is None:
        st.caption("*No analysis in progress*")
    else:
        for step in plan.steps:
            icon = {"done": "✅", "skipped": "⏭️", "pending": "⬜"}.get(step.status, "⬜")
            st.markdown(f"{icon} {step.description}")
        proposed = getattr(plan, "proposed_spec", "")
        if proposed:
            st.caption(f"Spec: `{proposed}`")

    st.divider()

    if st.button("🔄 New Session", use_container_width=True):
        for key in ["ctx", "history", "messages", "log_buffer", "graph_paths", "known_graphs", "agent"]:
            st.session_state.pop(key, None)
        st.rerun()

    if st.button("⚙️ Reset Stata Config", use_container_width=True):
        from config import reset_saved_config
        reset_saved_config()
        st.success("Stata config cleared. Restart to re-run discovery.")


# ── Main area ──────────────────────────────────────────────────────────────
chat_tab, log_tab, graph_tab = st.tabs(["💬 Analysis", "📋 Stata Log", "📈 Graphs"])

with chat_tab:
    for role, content in st.session_state.messages:
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "📊"):
            st.markdown(content)

    if prompt := st.chat_input("Ask a question about your data..."):
        agent = _get_agent()

        st.session_state.messages.append(("user", prompt))
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="📊"):
            with st.spinner("Thinking..."):
                try:
                    result = agent.run_sync(
                        prompt,
                        deps=st.session_state.ctx,
                        message_history=st.session_state.history,
                    )
                except Exception as exc:
                    st.error("Agent error — see details below.")
                    st.exception(exc)
                    st.stop()

            new_messages = result.all_messages()
            st.session_state.history = new_messages
            response = result.data

            st.markdown(response)

            new_logs = _extract_tool_logs(new_messages)
            st.session_state.log_buffer.extend(new_logs)

            new_graphs = _detect_new_graphs(st.session_state.known_graphs, _HERE)
            for g in new_graphs:
                st.session_state.known_graphs.add(g)
                st.session_state.graph_paths.append(g)

            if new_graphs:
                latest = new_graphs[-1]
                if latest.exists():
                    st.image(str(latest), caption=latest.name, width=500)

        st.session_state.messages.append(("assistant", response))


with log_tab:
    st.markdown("### Stata Log")
    st.caption(
        "Raw output from each tool call this session. "
        "Commands run in English regardless of UI language."
    )

    if not st.session_state.log_buffer:
        st.info("No commands have been run yet.")
    else:
        for i, (label, output) in enumerate(st.session_state.log_buffer, 1):
            header = label if len(label) <= 60 else label[:57] + "..."
            with st.expander(
                f"Call {i}: `{header}`",
                expanded=(i == len(st.session_state.log_buffer)),
            ):
                st.code(output, language=None)

        if st.button("Show full log"):
            full_log = "\n\n".join(
                f". {lbl}\n{out}" for lbl, out in st.session_state.log_buffer
            )
            st.code(full_log, language=None)


with graph_tab:
    st.markdown("### Graphs")
    st.caption("Graphs produced during this session.")

    if not st.session_state.graph_paths:
        st.info("No graphs produced yet. Ask the agent to plot something.")
    else:
        pairs = [
            st.session_state.graph_paths[i : i + 2]
            for i in range(0, len(st.session_state.graph_paths), 2)
        ]
        for pair in pairs:
            cols = st.columns(2)
            for col, path in zip(cols, pair):
                if path.exists():
                    with col:
                        st.image(str(path), use_column_width=True)
                        st.caption(path.name)
                        with open(path, "rb") as f:
                            mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
                            st.download_button(
                                "Download",
                                f,
                                file_name=path.name,
                                mime=mime,
                                key=str(path),
                            )
