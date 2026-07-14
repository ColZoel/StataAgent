/* StataAgent web UI — client logic (no build step, no dependencies) */
"use strict";

const $ = (sel) => document.querySelector(sel);
const results = $("#results");
const input = $("#command-input");
const runBtn = $("#run-btn");

let busy = false;
let cmdHistory = [];      // strings, oldest first
let histCursor = -1;      // -1 = not navigating
let graphCount = 0;

/* ── tiny markdown renderer (headers, code, tables, lists, emphasis) ── */
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inlineMd(s) {
  return s
    .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|\W)\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

function renderMarkdown(text) {
  const lines = esc(text).split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    if (line.startsWith("```")) {                      // fenced code
      const buf = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) buf.push(lines[i++]);
      i++;
      out.push(`<pre><code>${buf.join("\n")}</code></pre>`);
      continue;
    }
    if (/^\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\|[\s\-:|]+\|\s*$/.test(lines[i + 1])) {
      const cells = (l) => l.replace(/^\||\|\s*$/g, "").split("|").map((c) => inlineMd(c.trim()));
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push(
        "<table><thead><tr>" + head.map((c) => `<th>${c}</th>`).join("") +
        "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("") +
        "</tbody></table>"
      );
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)/);
    if (h) { out.push(`<h${h[1].length + 1}>${inlineMd(h[2])}</h${h[1].length + 1}>`); i++; continue; }

    if (/^\s*[-*]\s+/.test(line)) {                    // unordered list
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i]))
        items.push(inlineMd(lines[i++].replace(/^\s*[-*]\s+/, "")));
      out.push("<ul>" + items.map((x) => `<li>${x}</li>`).join("") + "</ul>");
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {                   // ordered list
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i]))
        items.push(inlineMd(lines[i++].replace(/^\s*\d+\.\s+/, "")));
      out.push("<ol>" + items.map((x) => `<li>${x}</li>`).join("") + "</ol>");
      continue;
    }
    if (/^\s*&gt;\s?/.test(line)) {                    // blockquote
      const buf = [];
      while (i < lines.length && /^\s*&gt;\s?/.test(lines[i]))
        buf.push(inlineMd(lines[i++].replace(/^\s*&gt;\s?/, "")));
      out.push(`<blockquote>${buf.join("<br>")}</blockquote>`);
      continue;
    }
    if (/^(\s*[-*_]){3,}\s*$/.test(line)) { out.push("<hr>"); i++; continue; }
    if (line.trim() === "") { i++; continue; }

    const buf = [inlineMd(line)];                      // paragraph
    i++;
    while (i < lines.length && lines[i].trim() !== "" &&
           !/^(#{1,4}\s|```|\||\s*[-*]\s|\s*\d+\.\s|\s*&gt;)/.test(lines[i]))
      buf.push(inlineMd(lines[i++]));
    out.push(`<p>${buf.join(" ")}</p>`);
  }
  return out.join("\n");
}

/* ── results-stream DOM helpers ─────────────────────────────────────── */
function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}

function scrollToBottom() {
  results.scrollTop = results.scrollHeight;
}

function startEntry(prompt) {
  $("#welcome")?.remove();
  const entry = el("div", "entry");
  const echo = el("div", "user-echo");
  echo.textContent = prompt;
  entry.appendChild(echo);
  const spin = el("div", "running",
    '<span class="spinner"></span><span>running…</span>');
  entry.appendChild(spin);
  results.appendChild(entry);
  scrollToBottom();
  return entry;
}

function addStataBlock(entry, ev) {
  const block = el("div", "stata-block");
  const cmd = el("div", "stata-cmd" + (ev.success ? "" : " err"));
  cmd.textContent = ev.code;
  block.appendChild(cmd);
  const body = (ev.output || "").trimEnd() +
    (ev.success ? "" : (ev.error ? `\n${ev.error}` : ""));
  if (body.trim()) {
    const out = el("pre", "stata-out" + (ev.success ? "" : " err"));
    out.textContent = body;
    block.appendChild(out);
  }
  entry.insertBefore(block, entry.querySelector(".running"));
  addHistoryItem(ev.code, !ev.success);
  scrollToBottom();
}

function addCommentary(entry, text) {
  entry.insertBefore(el("div", "commentary", renderMarkdown(text)),
    entry.querySelector(".running"));
  scrollToBottom();
}

function addError(entry, msg) {
  const card = el("div", "error-card");
  card.textContent = msg;
  entry.insertBefore(card, entry.querySelector(".running"));
  scrollToBottom();
}

function addGraphs(entry, names) {
  for (const name of names) {
    const url = `/api/graph/${encodeURIComponent(name)}`;
    if (!name.endsWith(".pdf")) {
      const fig = el("figure", "inline-graph",
        `<img src="${url}" alt="${esc(name)}"><figcaption>${esc(name)}</figcaption>`);
      entry.insertBefore(fig, entry.querySelector(".running"));
    }
    addGraphToPane(name, url);
  }
  scrollToBottom();
}

/* ── Review (history) pane ──────────────────────────────────────────── */
function addHistoryItem(code, isErr) {
  $("#history-empty").hidden = true;
  const li = el("li", isErr ? "err" : "");
  li.textContent = code.split("\n")[0];
  li.title = code;
  li.onclick = () => { input.value = code; input.focus(); autosize(); };
  $("#history-list").appendChild(li);
  li.scrollIntoView({ block: "nearest" });
}

/* ── Graphs pane ────────────────────────────────────────────────────── */
function addGraphToPane(name, url) {
  $("#graphs-empty").hidden = true;
  graphCount++;
  const badge = $("#graph-count");
  badge.hidden = false;
  badge.textContent = graphCount;
  const inner = name.endsWith(".pdf")
    ? `<figcaption><span>${esc(name)}</span><a href="${url}" target="_blank">open</a></figcaption>`
    : `<img src="${url}" alt="${esc(name)}" onclick="window.open('${url}')">` +
      `<figcaption><span>${esc(name)}</span><a href="${url}" download>download</a></figcaption>`;
  $("#graph-grid").prepend(el("figure", "", inner));
}

/* ── state sync ─────────────────────────────────────────────────────── */
async function refreshState() {
  let s;
  try {
    s = await (await fetch("/api/state")).json();
  } catch { return; }

  $("#version-label").textContent = "v" + s.version;

  const dotS = $("#dot-stata");
  dotS.className = "dot " + (s.stata.ok ? "ok" : "bad");
  $("#stata-label").textContent = s.stata.ok
    ? `Stata ${s.stata.version || ""}`.trim()
    : `Stata unavailable`;
  $("#chip-stata").title = s.stata.ok
    ? `${s.stata.edition || "Stata"} ${s.stata.version || ""}`
    : (s.stata.error || "Stata not found");

  $("#dot-model").className = "dot " + (s.agent_error ? "bad" : "ok");
  $("#model-label").textContent = s.model.split("/").pop();
  $("#chip-model").title = `${s.provider} / ${s.model}` +
    (s.agent_error ? `\n${s.agent_error}` : "");

  $("#dataset-label").textContent = s.dataset.loaded
    ? (s.dataset.path || "dataset in memory").split("/").pop()
    : "no dataset";

  $("#rigor-select").value = s.rigor;

  // variables pane + properties
  const tbody = $("#var-tbody");
  tbody.innerHTML = "";
  const vars = s.variables || [];
  $("#vars-empty").hidden = vars.length > 0;
  for (const v of vars) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td class="vname">${esc(v.name)}</td>` +
      `<td class="vtype">${esc(v.type)}</td>` +
      `<td title="${esc(v.label)}">${esc(v.label)}</td>`;
    tr.querySelector(".vname").onclick = () => {
      input.value = (input.value + " " + v.name).trimStart();
      input.focus();
    };
    tbody.appendChild(tr);
  }
  filterVars();

  $("#prop-obs").textContent = s.dataset.loaded ? s.dataset.obs.toLocaleString() : "—";
  $("#prop-vars").textContent = s.dataset.loaded ? vars.length : "—";
  const f = s.dataset.path ? s.dataset.path.split(/[\\/]/).pop() : "—";
  $("#prop-file").textContent = f;
  $("#prop-file").title = s.dataset.path || "";

  // working directory: show the tail, keep the full path in the tooltip
  const cwd = s.cwd || "";
  const parts = cwd.split(/[\\/]/).filter(Boolean);
  $("#prop-cwd").textContent = cwd
    ? (parts.length > 2 ? "…/" + parts.slice(-2).join("/") : cwd)
    : "—";
  $("#prop-cwd").title = cwd;

  // conversation-size meter
  const conv = s.conversation || { turns: 0, messages: 0 };
  const pill = $("#conv-pill");
  pill.textContent = conv.messages
    ? `${conv.turns} turn${conv.turns === 1 ? "" : "s"} · ${conv.messages} messages in context`
    : "new conversation";
  pill.classList.toggle("warn", conv.messages >= 60);
}

/* ── query execution (SSE over fetch) ───────────────────────────────── */
async function runQuery(prompt) {
  if (busy || !prompt.trim()) return;
  busy = true;
  runBtn.disabled = true;
  cmdHistory.push(prompt);
  histCursor = -1;

  const entry = startEntry(prompt);
  try {
    const resp = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: prompt }),
    });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({}));
      addError(entry, e.error || `Request failed (${resp.status})`);
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const line = chunk.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "stata") addStataBlock(entry, ev);
        else if (ev.type === "response") {
          if (ev.text) addCommentary(entry, ev.text);
          if (ev.graphs?.length) addGraphs(entry, ev.graphs);
        } else if (ev.type === "error") addError(entry, ev.message);
      }
    }
  } catch (err) {
    addError(entry, "Connection lost: " + err.message);
  } finally {
    entry.querySelector(".running")?.remove();
    busy = false;
    runBtn.disabled = false;
    input.focus();
    refreshState();
  }
}

/* ── drag-and-drop (.dta loads into Stata, .do attaches as context) ──── */
let dragDepth = 0;

function dragHasFiles(e) {
  return [...(e.dataTransfer?.types || [])].includes("Files");
}

document.addEventListener("dragenter", (e) => {
  if (!dragHasFiles(e)) return;
  e.preventDefault();
  dragDepth++;
  $("#drop-overlay").hidden = false;
});
document.addEventListener("dragover", (e) => {
  if (dragHasFiles(e)) e.preventDefault();
});
document.addEventListener("dragleave", (e) => {
  if (!dragHasFiles(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) $("#drop-overlay").hidden = true;
});
document.addEventListener("drop", (e) => {
  if (!dragHasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  $("#drop-overlay").hidden = true;
  handleDroppedFiles([...e.dataTransfer.files]);
});

function fileEntry(name) {
  $("#welcome")?.remove();
  const entry = el("div", "entry");
  const echo = el("div", "file-echo");
  echo.textContent = `📎 ${name}`;
  entry.appendChild(echo);
  const spin = el("div", "running",
    '<span class="spinner"></span><span>uploading…</span>');
  entry.appendChild(spin);
  results.appendChild(entry);
  scrollToBottom();
  return entry;
}

async function handleDroppedFiles(files) {
  for (const f of files) {
    const ext = f.name.toLowerCase().slice(f.name.lastIndexOf("."));
    const entry = fileEntry(f.name);
    if (ext !== ".dta" && ext !== ".do") {
      entry.querySelector(".running")?.remove();
      addError(entry, "Only .dta and .do files are supported.");
      continue;
    }
    if (busy) {
      entry.querySelector(".running")?.remove();
      addError(entry, "A query is running — drop the file again when it finishes.");
      continue;
    }
    await uploadFile(f, entry);   // sequential: Stata is a single instance
  }
}

async function uploadFile(f, entry) {
  try {
    const resp = await fetch(
      `/api/upload?filename=${encodeURIComponent(f.name)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: f,
      },
    );
    const d = await resp.json();
    if (!resp.ok) {
      addError(entry, d.error || `Upload failed (${resp.status})`);
      return;
    }
    if (d.kind === "dta") {
      addStataBlock(entry, {
        code: d.code,
        output: d.log,
        success: d.success,
        error: d.error,
      });
      if (d.success) {
        const note = el("div", "attach-card",
          `<div class="attach-note">Dataset loaded — the Variables pane is ` +
          `updated and the agent will be told with your next question.</div>`);
        entry.insertBefore(note, entry.querySelector(".running"));
      }
    } else if (d.kind === "do") {
      const card = el("div", "attach-card");
      card.innerHTML =
        `<div>📄 <strong>${esc(d.name)}</strong> attached ` +
        `(${d.lines} line${d.lines === 1 ? "" : "s"}` +
        `${d.truncated ? ", truncated for context" : ""})</div>` +
        `<div class="attach-note">Its contents will be shared with the agent ` +
        `on your next question — try “explain this do-file” or “run this”.</div>` +
        (d.preview ? `<pre>${esc(d.preview)}</pre>` : "");
      entry.insertBefore(card, entry.querySelector(".running"));
    }
    scrollToBottom();
  } catch (err) {
    addError(entry, "Upload failed: " + err.message);
  } finally {
    entry.querySelector(".running")?.remove();
    refreshState();
  }
}

/* ── command window behavior ────────────────────────────────────────── */
function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}

function submit() {
  const v = input.value.trim();
  if (!v) return;
  input.value = "";
  autosize();
  runQuery(v);
}

input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  else if (e.key === "ArrowUp" && !input.value.includes("\n") &&
           input.selectionStart === input.value.length) {
    if (!cmdHistory.length) return;
    e.preventDefault();
    histCursor = histCursor < 0 ? cmdHistory.length - 1 : Math.max(0, histCursor - 1);
    input.value = cmdHistory[histCursor];
    autosize();
  } else if (e.key === "ArrowDown" && histCursor >= 0) {
    e.preventDefault();
    histCursor++;
    if (histCursor >= cmdHistory.length) { histCursor = -1; input.value = ""; }
    else input.value = cmdHistory[histCursor];
    autosize();
  }
});
runBtn.addEventListener("click", submit);

/* ── top-bar controls ───────────────────────────────────────────────── */
$("#rigor-select").addEventListener("change", async (e) => {
  await fetch("/api/rigor", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ level: e.target.value }),
  });
});

$("#reset-btn").addEventListener("click", async () => {
  if (busy) return;
  if (!confirm("Start a new session? The conversation and dataset context will be cleared.")) return;
  await fetch("/api/reset", { method: "POST" });
  location.reload();
});

$("#theme-btn").addEventListener("click", () => {
  const root = document.documentElement;
  const next = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  localStorage.setItem("stataagent-theme", next);
});

/* ── settings modal ─────────────────────────────────────────────────── */
const overlay = $("#settings-overlay");
let settingsSnapshot = null;   // values as loaded, to send only what changed

async function openSettings() {
  let s;
  try {
    s = await (await fetch("/api/settings")).json();
  } catch {
    return;
  }
  settingsSnapshot = s;

  const prov = $("#set-provider");
  prov.innerHTML = s.providers
    .map((p) => `<option value="${p}">${p}</option>`).join("");
  prov.value = s.provider;

  $("#set-model").value = s.model || "";
  $("#set-base-url").value = s.base_url || "";
  syncBaseUrlVisibility();
  $("#set-temperature").value = s.temperature;
  $("#set-max-tokens").value = s.max_tokens;
  $("#set-commentary").checked = !!s.commentary;
  collapseAdvanced();
  syncResetButton();

  $("#set-key-env").value = s.api_key_env || "";
  $("#set-api-key").value = "";
  const pill = $("#key-status");
  pill.textContent = s.api_key_set ? "set" : "not set";
  pill.className = "pill " + (s.api_key_set ? "ok" : "bad");

  $("#set-stata-path").value = s.stata.path || "";
  $("#set-stata-edition").value = s.stata.edition || "";
  $("#detect-results").hidden = true;
  $("#detect-results").innerHTML = "";
  $("#stata-hint").textContent =
    (s.stata.env_override
      ? "STATA_PATH is set in your environment and takes precedence over the saved path. "
      : "") +
    "Changing the Stata path takes effect after restarting StataAgent.";

  $("#about-version").textContent = "v" + s.version;
  $("#about-year").textContent = new Date().getFullYear();
  collapseAbout();

  const msg = $("#settings-msg");
  msg.textContent = "";
  msg.className = "settings-msg";

  overlay.hidden = false;
  $("#set-model").focus();
}

function closeSettings() { overlay.hidden = true; }

/* Base URL is only meaningful for openai_compatible; hide it otherwise. */
function syncBaseUrlVisibility() {
  const show = $("#set-provider").value === "openai_compatible";
  $("#set-base-url-label").hidden = !show;
  $("#set-base-url").hidden = !show;
}

/* ── Advanced Model Settings disclosure ─────────────────────────────── */
function collapseAdvanced() {
  $("#advanced-body").hidden = true;
  $("#advanced-toggle").setAttribute("aria-expanded", "false");
}

function toggleAdvanced() {
  const body = $("#advanced-body");
  const open = body.hidden;
  body.hidden = !open;
  $("#advanced-toggle").setAttribute("aria-expanded", String(open));
}

/* Enable "Reset to defaults" only when a parameter differs from its default. */
function syncResetButton() {
  const d = settingsSnapshot?.defaults || { temperature: 0.3, max_tokens: 4096 };
  const temp = parseFloat($("#set-temperature").value);
  const maxTok = parseInt($("#set-max-tokens").value, 10);
  const adjusted =
    (!Number.isNaN(temp) && temp !== d.temperature) ||
    (!Number.isNaN(maxTok) && maxTok !== d.max_tokens);
  $("#reset-params-btn").disabled = !adjusted;
}

function resetParams() {
  const d = settingsSnapshot?.defaults || { temperature: 0.3, max_tokens: 4096 };
  $("#set-temperature").value = d.temperature;
  $("#set-max-tokens").value = d.max_tokens;
  syncResetButton();
}

/* ── About StataAgent ───────────────────────────────────────────────── */
function collapseAbout() {
  $("#about-body").hidden = true;
  $("#about-toggle").setAttribute("aria-expanded", "false");
  $("#issue-choice").hidden = true;
  $("#update-status").hidden = true;
}

function toggleAbout() {
  const body = $("#about-body");
  const open = body.hidden;
  body.hidden = !open;
  $("#about-toggle").setAttribute("aria-expanded", String(open));
}

async function checkForUpdates() {
  const out = $("#update-status");
  const btn = $("#check-update-btn");
  out.hidden = false;
  out.className = "about-note dim";
  out.textContent = "Checking…";
  btn.disabled = true;
  try {
    const d = await (await fetch("/api/update-check")).json();
    if (d.status === "update") {
      out.className = "about-note warn";
      out.innerHTML =
        `Version <strong>v${esc(d.latest)}</strong> is available ` +
        `(you have v${esc(d.current)}). ` +
        `<a href="https://github.com/ColZoel/StataAgent/releases/latest" ` +
        `target="_blank" rel="noopener">Release notes</a>`;
    } else if (d.status === "current") {
      out.className = "about-note ok";
      out.textContent = `You’re on the latest version (v${d.current}).`;
    } else if (d.status === "none") {
      out.className = "about-note dim";
      out.textContent = `No published releases to compare against yet (you have v${d.current}).`;
    } else {
      out.className = "about-note dim";
      out.textContent = "Couldn’t reach GitHub to check for updates.";
    }
  } catch {
    out.className = "about-note dim";
    out.textContent = "Couldn’t reach GitHub to check for updates.";
  } finally {
    btn.disabled = false;
  }
}

async function saveSettings() {
  const snap = settingsSnapshot || {};
  const body = {};
  const prov = $("#set-provider").value;
  const model = $("#set-model").value.trim();
  const baseUrl = $("#set-base-url").value.trim();
  const keyEnv = $("#set-key-env").value.trim();
  const key = $("#set-api-key").value;
  const temp = parseFloat($("#set-temperature").value);
  const maxTok = parseInt($("#set-max-tokens").value, 10);
  const comm = $("#set-commentary").checked;
  const stataPath = $("#set-stata-path").value.trim();
  const stataEd = $("#set-stata-edition").value;

  if (prov !== snap.provider) body.provider = prov;
  if (model && model !== snap.model) body.model = model;
  if (baseUrl !== (snap.base_url || "")) body.base_url = baseUrl;
  if (keyEnv && keyEnv !== (snap.api_key_env || "")) body.api_key_env = keyEnv;
  if (key) body.api_key = key;
  if (!Number.isNaN(temp) && temp !== snap.temperature) body.temperature = temp;
  if (!Number.isNaN(maxTok) && maxTok !== snap.max_tokens) body.max_tokens = maxTok;
  if (comm !== !!snap.commentary) body.commentary = comm;
  if (stataPath && (stataPath !== (snap.stata?.path || "") ||
      stataEd !== (snap.stata?.edition || ""))) {
    body.stata_path = stataPath;
    if (stataEd) body.stata_edition = stataEd;
  }

  const msg = $("#settings-msg");
  if (!Object.keys(body).length) {
    msg.textContent = "No changes to save.";
    msg.className = "settings-msg";
    return;
  }

  msg.textContent = "Saving…";
  msg.className = "settings-msg";
  try {
    const resp = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) {
      msg.textContent = data.error || `Save failed (${resp.status})`;
      msg.className = "settings-msg err";
      return;
    }
    msg.textContent = data.messages.join(" ");
    msg.className = "settings-msg ok";
    $("#set-api-key").value = "";
    settingsSnapshot = null;   // force fresh load next open
    refreshState();
    // refresh the key-status pill without wiping the message
    const s = await (await fetch("/api/settings")).json();
    settingsSnapshot = s;
    const pill = $("#key-status");
    pill.textContent = s.api_key_set ? "set" : "not set";
    pill.className = "pill " + (s.api_key_set ? "ok" : "bad");
  } catch (err) {
    msg.textContent = "Save failed: " + err.message;
    msg.className = "settings-msg err";
  }
}

async function detectStata() {
  const box = $("#detect-results");
  box.hidden = false;
  box.innerHTML = '<div class="none">Scanning common install locations…</div>';
  try {
    const data = await (await fetch("/api/settings/stata-detect")).json();
    if (!data.candidates.length) {
      box.innerHTML = '<div class="none">No Stata installation found in the usual locations. Enter the path manually.</div>';
      return;
    }
    box.innerHTML = "";
    for (const c of data.candidates) {
      const b = el("button", "candidate",
        esc(c.path) + (c.edition ? `  →  ${c.edition.toUpperCase()}` : ""));
      b.onclick = () => {
        $("#set-stata-path").value = c.path;
        $("#set-stata-edition").value = c.edition || "";
      };
      box.appendChild(b);
    }
  } catch {
    box.innerHTML = '<div class="none">Detection failed.</div>';
  }
}

$("#settings-btn").addEventListener("click", openSettings);
$("#set-provider").addEventListener("change", syncBaseUrlVisibility);
$("#settings-close").addEventListener("click", closeSettings);
$("#settings-save").addEventListener("click", saveSettings);
$("#detect-btn").addEventListener("click", detectStata);
$("#advanced-toggle").addEventListener("click", toggleAdvanced);
$("#reset-params-btn").addEventListener("click", resetParams);
$("#set-temperature").addEventListener("input", syncResetButton);
$("#set-max-tokens").addEventListener("input", syncResetButton);
$("#about-toggle").addEventListener("click", toggleAbout);
$("#check-update-btn").addEventListener("click", checkForUpdates);
$("#submit-issue-btn").addEventListener("click", () => {
  const box = $("#issue-choice");
  box.hidden = !box.hidden;
});
overlay.addEventListener("click", (e) => { if (e.target === overlay) closeSettings(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !overlay.hidden) closeSettings();
});

/* ── right-pane tabs + variable filter ──────────────────────────────── */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-body").forEach((b) => b.classList.remove("active"));
    tab.classList.add("active");
    $("#tab-" + tab.dataset.tab).classList.add("active");
  });
});

function filterVars() {
  const q = $("#var-filter").value.toLowerCase();
  document.querySelectorAll("#var-tbody tr").forEach((tr) => {
    tr.hidden = q && !tr.textContent.toLowerCase().includes(q);
  });
}
$("#var-filter").addEventListener("input", filterVars);

/* ── examples on the welcome card ───────────────────────────────────── */
document.querySelectorAll(".example").forEach((b) => {
  b.addEventListener("click", () => { input.value = b.textContent; input.focus(); autosize(); });
});

/* ── init ───────────────────────────────────────────────────────────── */
const savedTheme = localStorage.getItem("stataagent-theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
else if (matchMedia("(prefers-color-scheme: dark)").matches)
  document.documentElement.dataset.theme = "dark";

refreshState();
input.focus();
