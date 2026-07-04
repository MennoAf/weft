"""Weft board — the localhost triage dashboard (weft-board-epic Goal #4).

A single self-contained static page: no external CSS/JS/fonts, no build step,
works offline. Served by ``board_server`` at ``GET /``. All it does is speak the
``weft_board`` contract — ``GET /board`` to render, ``POST /act`` to triage —
holding no business logic beyond rendering and dispatch, so the same contract an
agent uses backs the human UI. Kept as a Python string constant (not a static
file) so it ships with the package and needs no asset-path/packaging wiring.
"""

from __future__ import annotations

BOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weft Board</title>
<style>
  :root {
    --bg: #f6f7f9; --card: #fff; --ink: #1c2430; --muted: #6b7785;
    --line: #e4e8ee; --overdue: #c0392b; --due_soon: #d97a1a;
    --pending: #2571b0; --no_date: #7a8592; --accent: #2571b0;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { position: sticky; top: 0; z-index: 5; background: var(--card);
    border-bottom: 1px solid var(--line); padding: 12px 20px;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; font-weight: 650; letter-spacing: .2px; }
  header .spacer { flex: 1; }
  header .meta { color: var(--muted); font-size: 12px; }
  button, select { font: inherit; }
  .btn { border: 1px solid var(--line); background: var(--card); color: var(--ink);
    border-radius: 7px; padding: 5px 10px; cursor: pointer; }
  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .btn:active { transform: translateY(1px); }
  #warnings { padding: 0 20px; }
  .warn { background: #fff6e6; border: 1px solid #f0d9a8; color: #8a5a00;
    border-radius: 8px; padding: 8px 12px; margin: 10px 0 0; font-size: 13px; }
  main { padding: 8px 20px 60px; max-width: 980px; margin: 0 auto; }
  .bucket { margin-top: 22px; }
  .bucket > h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .6px;
    margin: 0 0 8px; display: flex; align-items: center; gap: 8px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .count { color: var(--muted); font-weight: 500; }
  .src-group > summary { cursor: pointer; color: var(--muted); font-size: 12px;
    padding: 6px 2px; list-style: none; }
  .src-group > summary::-webkit-details-marker { display: none; }
  .src-group > summary::before { content: "\25B8 "; }
  .src-group[open] > summary::before { content: "\25BE "; }
  .item { background: var(--card); border: 1px solid var(--line); border-left: 3px solid;
    border-radius: 9px; padding: 10px 12px; margin: 7px 0;
    display: flex; align-items: center; gap: 12px; }
  .item .body { flex: 1; min-width: 0; }
  .item .title { font-weight: 500; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; }
  .item .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .badge { display: inline-block; background: #eef1f5; color: #4a5563;
    border-radius: 5px; padding: 1px 6px; font-size: 11px; margin-right: 6px; }
  .actions { display: flex; gap: 6px; flex-shrink: 0; }
  .actions .btn { padding: 4px 9px; font-size: 13px; }
  .btn.danger:hover { border-color: var(--overdue); color: var(--overdue); }
  .ro { color: var(--muted); font-size: 12px; font-style: italic; }
  .empty { color: var(--muted); padding: 40px 0; text-align: center; }
  .snooze { display: inline-flex; align-items: center; gap: 3px; }
  .snooze .lbl { color: var(--muted); font-size: 12px; }
  #toast { position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%);
    background: #1c2430; color: #fff; padding: 9px 16px; border-radius: 8px;
    font-size: 13px; opacity: 0; transition: opacity .2s; pointer-events: none; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <h1>Weft Board</h1>
  <label class="meta">horizon
    <select id="days">
      <option>3</option><option selected>7</option><option>14</option><option>30</option>
    </select> days
  </label>
  <button class="btn" id="refresh">Refresh</button>
  <div class="spacer"></div>
  <span class="meta" id="gen"></span>
</header>
<div id="warnings"></div>
<main id="board"><div class="empty">Loading…</div></main>
<div id="toast"></div>

<script>
const BUCKETS = ["overdue", "due_soon", "pending", "no_date"];
const LABEL = { overdue: "Overdue", due_soon: "Due soon", pending: "Pending", no_date: "No date" };
const DAY = 86400000;

function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2200);
}
function fmtDue(iso) {
  if (!iso) return "no due date";
  const d = new Date(iso);
  return "due " + d.toLocaleDateString(undefined, { month: "short", day: "numeric" })
    + " " + d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}
function el(tag, cls, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
}

async function loadBoard() {
  const days = document.getElementById("days").value;
  const board = document.getElementById("board");
  try {
    const res = await fetch("/board?days=" + days);
    if (!res.ok) throw new Error("HTTP " + res.status);
    render(await res.json());
  } catch (e) {
    board.innerHTML = "";
    board.appendChild(el("div", "empty", "Could not reach the board server: " + e.message));
  }
}

function render(data) {
  document.getElementById("gen").textContent =
    "updated " + new Date(data.generated_at).toLocaleTimeString();
  const warns = document.getElementById("warnings");
  warns.innerHTML = "";
  (data.warnings || []).forEach(w => {
    const msg = w.source === "identity" ? w.message
      : w.truncated ? (w.source + " truncated at " + w.cap + " items")
      : (w.source + ": " + (w.error || "unavailable"));
    warns.appendChild(el("div", "warn", "⚠ " + msg));
  });

  const board = document.getElementById("board");
  board.innerHTML = "";
  if ((data.counts && data.counts.total) === 0) {
    board.appendChild(el("div", "empty", "Nothing open. 🎉"));
    return;
  }
  for (const b of BUCKETS) {
    const items = (data.buckets && data.buckets[b]) || [];
    if (!items.length) continue;
    const sec = el("div", "bucket");
    const h = el("h2");
    const dot = el("span", "dot"); dot.style.background = "var(--" + b + ")";
    h.appendChild(dot);
    h.appendChild(document.createTextNode(LABEL[b] + " "));
    h.appendChild(el("span", "count", "(" + items.length + ")"));
    sec.appendChild(h);

    // Group by source; read-only sources (no actions) collapse by default.
    const bySource = {};
    for (const it of items) (bySource[it.source] ||= []).push(it);
    for (const src of Object.keys(bySource)) {
      const group = bySource[src];
      const readOnly = group.every(it => !(it.actions && it.actions.length));
      const wrap = el("details", "src-group");
      wrap.open = !readOnly;
      const sum = el("summary", null, src + " (" + group.length + ")" + (readOnly ? " · read-only" : ""));
      wrap.appendChild(sum);
      for (const it of group) wrap.appendChild(renderItem(it, b));
      sec.appendChild(wrap);
    }
    board.appendChild(sec);
  }
}

function renderItem(it, bucket) {
  const row = el("div", "item");
  row.style.borderLeftColor = "var(--" + bucket + ")";
  const body = el("div", "body");
  body.appendChild(el("div", "title", it.title || "(untitled)"));
  const sub = el("div", "sub");
  sub.appendChild(el("span", "badge", it.source + "/" + it.kind));
  sub.appendChild(document.createTextNode(fmtDue(it.due_at)));
  if (it.state) sub.appendChild(document.createTextNode(" · " + it.state));
  body.appendChild(sub);
  row.appendChild(body);

  const acts = el("div", "actions");
  if (!it.actions || !it.actions.length) {
    acts.appendChild(el("span", "ro", "read-only"));
  } else {
    for (const a of it.actions) {
      if (a.verb === "snooze") acts.appendChild(snoozeControl(it, a));
      else acts.appendChild(actionButton(it, a));
    }
  }
  row.appendChild(acts);
  return row;
}

const DESTRUCTIVE = new Set(["close", "dismiss", "delete"]);

function actionButton(it, a) {
  const b = el("button", "btn" + (DESTRUCTIVE.has(a.verb) ? " danger" : ""), a.verb);
  b.onclick = () => {
    if (DESTRUCTIVE.has(a.verb) &&
        !confirm(a.verb + " this " + it.source + "?\n\n" + (it.title || ""))) return;
    fireAct(it, a, a.args);
  };
  return b;
}

function snoozeControl(it, a) {
  const wrap = el("span", "snooze");
  wrap.appendChild(el("span", "lbl", "snooze"));
  for (const [lbl, days] of [["1d", 1], ["3d", 3], ["1w", 7]]) {
    const b = el("button", "btn", lbl);
    b.onclick = () => {
      const until = new Date(Date.now() + days * DAY).toISOString();
      fireAct(it, a, Object.assign({}, a.args, { until }), days);
    };
    wrap.appendChild(b);
  }
  return wrap;
}

async function fireAct(it, action, args, snoozeDays) {
  const payload = {
    tool: action.tool, args,
    item_id: it.id, source: it.source, kind: it.kind,
    urgency_at_surface: it.urgency, age_days_at_surface: it.age_days,
    verb: action.verb,
  };
  if (snoozeDays != null) payload.snooze_duration_days = snoozeDays;
  try {
    const res = await fetch("/act", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || ("HTTP " + res.status));
    }
    toast(action.verb + " ✓");
    await loadBoard();  // refetch — the board reflects the write
  } catch (e) {
    toast("Failed: " + e.message);
  }
}

document.getElementById("refresh").onclick = loadBoard;
document.getElementById("days").onchange = loadBoard;
loadBoard();
</script>
</body>
</html>
"""
