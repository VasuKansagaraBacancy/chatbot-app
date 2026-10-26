const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const inputEl = $("input");
const sendBtn = $("send");
const sessionListEl = $("sessionList");
const chatTitleEl = $("chatTitle");

let sessionId = null;
let busy = false;
let mode = "chat";          // "chat" | "document"
let docReady = false;
let docInfo = null;         // /api/document payload, once it has loaded
let corpora = [];           // every indexed document the server offers
let corpus = null;          // the one being searched

/* ---------- minimal markdown rendering ---------- */
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function renderMarkdown(text) {
  const blocks = [];
  // Pull fenced code out first so its contents are never treated as markdown.
  let src = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push(
      `<pre><code class="lang-${escapeHtml(lang)}">${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`
    );
    return `@@BLOCK${blocks.length - 1}@@`;
  });

  src = escapeHtml(src)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");

  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };

  for (const rawLine of src.split("\n")) {
    const line = rawLine.trimEnd();
    const ph = line.match(/^@@BLOCK(\d+)@@$/);
    if (ph) { closeList(); out.push(blocks[+ph[1]]); continue; }

    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const ol = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    const head = line.match(/^(#{1,4})\s+(.*)$/);

    if (ul || ol) {
      const tag = ul ? "ul" : "ol";
      // Carry the source ordinal onto the <li>. A bullet sub-list between two
      // numbered items closes the <ol>, so without this the next item reopens
      // a fresh list and the numbering restarts at 1 (1. 1. 1. instead of 1. 2. 3.).
      if (list !== tag) {
        closeList();
        out.push(tag === "ol" ? `<ol start="${ol[1]}">` : "<ul>");
        list = tag;
      }
      out.push(ol ? `<li value="${ol[1]}">${ol[2]}</li>` : `<li>${ul[1]}</li>`);
    } else if (head) {
      closeList();
      const level = Math.min(head[1].length + 2, 6);
      out.push(`<h${level}>${head[2]}</h${level}>`);
    } else if (line === "") {
      closeList();
    } else {
      closeList();
      out.push(`<p>${line}</p>`);
    }
  }
  closeList();
  return out.join("");
}

/* ---------- message rendering ---------- */
const CHAT_SUGGESTIONS = [
  "What is retrieval-augmented generation?",
  "Explain async/await in Python",
  "Write a haiku about winter",
];

// Document-mode prompts are written against the indexed report, and show off
// the three things the hierarchy buys: a figure lookup, a named table and a
// named box.
const DOC_SUGGESTIONS = [
  "What is global growth projected to be in 2026?",
  "What does Table 1.1 show for China in 2026?",
  "What does Box 1.1 say about AI and growth?",
];

// A flat report has no tables or boxes to name, so it gets plainer prompts.
const PLAIN_DOC_SUGGESTIONS = [
  "What does this report cover?",
  "Summarise the key figures.",
  "Who prepared it, and when?",
];

function emptyState() {
  const doc = mode === "document";
  const info = doc ? currentCorpus() : null;
  const name = info ? (info.documents || []).join(", ") : "";
  const docChips = info && info.structured ? DOC_SUGGESTIONS : PLAIN_DOC_SUGGESTIONS;
  const chips = (doc ? docChips : CHAT_SUGGESTIONS)
    .map((s) => `<button class="suggestion">${escapeHtml(s)}</button>`)
    .join("");
  return `
  <div class="empty" id="empty">
    <h1>${doc ? "Ask the document" : "What can I help with?"}</h1>
    <p>${doc
        ? `Answers come only from ${escapeHtml(name || "the indexed report")}, with the chapter and page for each one.`
        : "Ask anything. I remember this conversation, so follow-up questions work."}</p>
    <div class="suggestions">${chips}</div>
  </div>`;
}

function addMessage(role, content) {
  const empty = $("empty");
  if (empty) empty.remove();

  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "AI";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderMarkdown(content);
  wrap.append(avatar, bubble);
  messagesEl.appendChild(wrap);
  scrollToBottom();
  return bubble;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setBusy(state) {
  busy = state;
  sendBtn.disabled = state;
  inputEl.disabled = state;
  if (!state) inputEl.focus();
}

/* ---------- session handling ---------- */
async function refreshSessions() {
  const { sessions } = await fetch("/api/sessions").then((r) => r.json());
  sessionListEl.innerHTML = "";
  for (const s of sessions) {
    const item = document.createElement("div");
    item.className = "session-item" + (s.session_id === sessionId ? " active" : "");
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = s.title;
    const del = document.createElement("button");
    del.className = "del";
    del.textContent = "×";
    del.title = "Delete conversation";
    del.onclick = async (e) => {
      e.stopPropagation();
      await fetch(`/api/sessions/${s.session_id}`, { method: "DELETE" });
      if (s.session_id === sessionId) await newSession();
      else await refreshSessions();
    };
    item.onclick = () => loadSession(s.session_id);
    item.append(label, del);
    sessionListEl.appendChild(item);
  }
}

function paintConversation(data) {
  sessionId = data.session_id;
  chatTitleEl.textContent = data.title || "New chat";
  messagesEl.innerHTML = "";
  if (!data.messages.length) {
    messagesEl.innerHTML = emptyState();
    wireSuggestions();
  } else {
    for (const m of data.messages) addMessage(m.role, m.content);
  }
}

async function loadSession(id) {
  if (busy) return;
  const res = await fetch(`/api/sessions/${id}`);
  if (!res.ok) return;
  paintConversation(await res.json());
  await refreshSessions();
}

async function newSession() {
  if (busy) return;
  const data = await fetch("/api/session", { method: "POST" }).then((r) => r.json());
  paintConversation(data);
  await refreshSessions();
  inputEl.focus();
}

function wireSuggestions() {
  document.querySelectorAll(".suggestion").forEach((btn) => {
    btn.onclick = () => { inputEl.value = btn.textContent; sendMessage(); };
  });
}

/* ---------- sending ---------- */
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || busy) return;

  addMessage("user", text);
  inputEl.dataset.lastSent = text;
  inputEl.value = "";
  autoGrow();
  setBusy(true);

  const bubble = addMessage("assistant", "");
  bubble.innerHTML = '<span class="cursor"></span>';
  let answer = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId, mode, corpus }),
    });
    if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let split;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);

        let event = "message";
        let data = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        const payload = JSON.parse(data);

        if (event === "start") {
          sessionId = payload.session_id;
          chatTitleEl.textContent = payload.title;
        } else if (event === "sources") {
          renderSources(bubble, payload);
        } else if (event === "token") {
          answer += payload.text;
          bubble.innerHTML = renderMarkdown(answer) + '<span class="cursor"></span>';
          scrollToBottom();
        } else if (event === "error") {
          throw new Error(payload.detail);
        }
      }
    }
    bubble.innerHTML = renderMarkdown(answer.trim());
  } catch (err) {
    bubble.classList.add("error");
    bubble.textContent = `Something went wrong: ${err.message}`;
  } finally {
    setBusy(false);
    scrollToBottom();
    refreshSessions();
  }
}

/* ---------- document mode ---------- */
// Citations live as a sibling of the bubble: the bubble's innerHTML is
// rewritten on every streamed token, which would otherwise wipe them.
function renderSources(bubble, payload) {
  const wrap = bubble.parentElement;
  let box = wrap.querySelector(".sources");
  if (!box) {
    box = document.createElement("div");
    box.className = "sources";
    wrap.appendChild(box);
  }
  const cites = payload.citations || [];
  if (!cites.length) {
    box.innerHTML = '<div class="src-head">no matching passages</div>';
    return;
  }
  const rewritten = payload.standalone && payload.standalone !== inputEl.dataset.lastSent
    ? `<div class="src-rewrite">interpreted as: ${escapeHtml(payload.standalone)}</div>`
    : "";
  box.innerHTML =
    rewritten +
    `<div class="src-head">${cites.length} passages</div>` +
    cites.map((c) => {
      // Show the page number the report itself prints. The PDF index is
      // offset from it by the front matter, so citing the PDF page would
      // send a reader to the wrong place in the printed document.
      const page = c.label && c.label !== String(c.page)
        ? `p.${escapeHtml(c.label)} <span class="src-pdf">(pdf ${c.page})</span>`
        : `p.${c.page}`;
      const trail = c.path || c.heading || "";
      const kind = c.kind && c.kind !== "body"
        ? `<span class="src-kind">${escapeHtml(c.kind)}</span>` : "";
      return `
      <details class="src">
        <summary><span class="src-n">[${c.n}]</span> ${page} ${kind}
          <span class="src-score">${c.score}</span>
          ${trail ? `<div class="src-path">${escapeHtml(trail)}</div>` : ""}
        </summary>
        <div class="src-body">${escapeHtml(c.snippet)}…</div>
      </details>`;
    }).join("");
}

function setMode(next) {
  if (busy) return;
  if (next === "document" && !docReady) return;
  mode = next;
  document.querySelectorAll(".mode-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode));
  inputEl.placeholder = mode === "document"
    ? "Ask about the document…"
    : "Message the assistant…";
  document.body.classList.toggle("doc-mode", mode === "document");
  // The suggestions belong to the mode, so refresh them while the thread is
  // still empty; mid-conversation the transcript stays untouched.
  if ($("empty")) messagesEl.innerHTML = emptyState();
  wireSuggestions();
}

function currentCorpus() {
  return corpora.find((c) => c.id === corpus) || null;
}

// The picker is only worth showing when there is a choice to make.
function buildCorpusPicker() {
  const choice = $("sourceChoice");
  const select = $("corpusSelect");
  if (!choice || !select) return;
  choice.hidden = corpora.length < 2;
  select.innerHTML = corpora
    .map((c) => `<option value="${escapeHtml(c.id)}"${c.id === corpus ? " selected" : ""}>`
      + `${escapeHtml(c.label)}</option>`)
    .join("");
}

function describeCorpus() {
  const note = $("modeNote");
  const info = currentCorpus();
  if (!note) return;
  if (!info) {
    note.textContent = "no document indexed";
    return;
  }
  const parts = (info.parts || []).length;
  note.textContent = `${info.chunks} chunks`
    + (parts ? ` · ${parts} sections` : "")
    + (info.pages ? ` · ${info.pages} pages` : "");
}

function setCorpus(next) {
  if (!next || next === corpus) return;
  corpus = next;
  describeCorpus();
  // Switching document invalidates the prompts offered for the old one.
  if ($("empty")) messagesEl.innerHTML = emptyState();
  wireSuggestions();
}

async function initDocumentMode() {
  const note = $("modeNote");
  const btn = $("docModeBtn");
  try {
    const d = await fetch("/api/document").then((r) => r.json());
    docReady = !!d.available;
    if (docReady) {
      docInfo = d;
      corpora = (d.corpora || []).filter((c) => !c.error);
      corpus = d.default || (corpora[0] && corpora[0].id) || null;
      buildCorpusPicker();
      describeCorpus();
      btn.disabled = false;
      btn.title = "Answer only from the indexed document";
    } else {
      btn.disabled = true;
      btn.title = d.detail || "No index";
      note.textContent = "no document indexed";
    }
  } catch {
    docReady = false;
    btn.disabled = true;
    note.textContent = "document mode unavailable";
  }
}

document.querySelectorAll(".mode-btn").forEach((b) =>
  b.addEventListener("click", () => setMode(b.dataset.mode)));

const corpusSelect = $("corpusSelect");
if (corpusSelect) corpusSelect.addEventListener("change", (e) => setCorpus(e.target.value));

/* ---------- input behaviour ---------- */
function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + "px";
}

inputEl.addEventListener("input", autoGrow);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
$("composer").addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });
$("newChat").addEventListener("click", newSession);
$("toggleSidebar").addEventListener("click", () => $("sidebar").classList.toggle("hidden"));

/* ---------- boot ---------- */
(async function init() {
  wireSuggestions();
  try {
    const health = await fetch("/api/health").then((r) => r.json());
    $("statusDot").classList.add("ready");
    $("statusText").textContent = health.model.split(/[\\/]/).pop().replace(".gguf", "");
  } catch {
    $("statusDot").classList.add("error");
    $("statusText").textContent = "model unavailable";
  }
  await initDocumentMode();
  paintConversation(await fetch("/api/session").then((r) => r.json()));
  await refreshSessions();
  inputEl.focus();
})();
