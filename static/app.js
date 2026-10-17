"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  phase: "idle", // idle | countdown | racing | done
  text: "",
  typed: "",
  spans: [],
  startTime: 0,
  youDone: null,
  modelPos: 0,
  modelDone: null,
  modelRejects: 0,
  youErrors: 0,
};

// --- settings -------------------------------------------------------------

const sliders = {
  temperature: { out: "temp-out", fmt: (v) => Number(v).toFixed(1) },
  top_k: { out: "topk-out", fmt: (v) => v },
  max_tps: { out: "tps-out", fmt: (v) => (Number(v) === 0 ? "∞" : v) },
};
for (const [id, { out, fmt }] of Object.entries(sliders)) {
  const el = $(id);
  try { el.value = localStorage.getItem("bt." + id) ?? el.value; } catch {}
  const sync = () => {
    $(out).textContent = fmt(el.value);
    try { localStorage.setItem("bt." + id, el.value); } catch {}
  };
  el.addEventListener("input", sync);
  sync();
}
const settings = () => ({
  temperature: Number($("temperature").value),
  top_k: Number($("top_k").value),
  max_tps: Number($("max_tps").value),
});

// --- socket ---------------------------------------------------------------

let ws;
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => { setStatus("disconnected – retrying…", "error"); setTimeout(connect, 1000); };
}
connect();

function setStatus(text, cls = "") {
  $("status").textContent = text;
  $("status").className = "status " + cls;
}

function handle(msg) {
  switch (msg.type) {
    case "status":
      if (msg.error) setStatus(msg.error, "error");
      else setStatus(msg.ready ? msg.runtime : `loading ${msg.runtime}…`, msg.ready ? "ready" : "");
      $("model-name").textContent = (msg.runtime || "Model").split(" · ")[1] || "Model";
      break;
    case "race": return beginCountdown(msg.text, msg.countdown);
    case "go": return go();
    case "model": return modelEvent(msg);
  }
}

// --- race lifecycle -------------------------------------------------------

function start() {
  if (state.phase === "countdown" || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "start", settings: settings() }));
  overlay("…", "");
}

function beginCountdown(text, seconds) {
  Object.assign(state, {
    phase: "countdown", text, typed: "", youDone: null, modelPos: 0,
    modelDone: null, modelRejects: 0, youErrors: 0,
  });
  $("log").innerHTML = "";
  $("text").classList.remove("finished");
  $("text").innerHTML = "";
  state.spans = [...text].map((ch) => {
    const s = document.createElement("span");
    s.className = "c";
    s.textContent = ch;
    $("text").appendChild(s);
    return s;
  });
  // Trailing span so both cursors have somewhere to sit at the end.
  const end = document.createElement("span");
  end.className = "c";
  $("text").appendChild(end);
  state.spans.push(end);
  render();
  let n = seconds;
  overlay(n, "");
  const tick = setInterval(() => {
    n -= 1;
    if (n <= 0 || state.phase !== "countdown") return clearInterval(tick);
    overlay(n, "");
  }, 1000);
}

function go() {
  state.phase = "racing";
  state.startTime = performance.now();
  overlay(null);
  render();
}

function elapsed() { return (performance.now() - state.startTime) / 1000; }
function wpm(chars, secs) { return secs > 0 ? Math.round(chars / 5 / (secs / 60)) : 0; }

function checkFinish() {
  const you = state.youDone, model = state.modelDone;
  if (state.phase !== "racing" || (you === null && model === null)) return;
  // First finisher decides the result; keep racing visuals until both are done.
  if (you !== null && model !== null) {
    state.phase = "done";
    $("text").classList.add("finished");
  }
  if (!$("overlay").dataset.result) {
    const win = you !== null && (model === null || you <= model);
    const t = (win ? you : model).toFixed(2);
    $("overlay").dataset.result = "1";
    overlay(win ? "You win" : "Model wins", win ? "win" : "lose",
      `${t}s · press Enter for a rematch`);
  }
}

// --- model ----------------------------------------------------------------

function modelEvent(msg) {
  if (state.phase !== "racing") return;
  if (msg.ev === "accept") {
    state.modelPos = msg.pos;
  } else if (msg.ev === "reject") {
    state.modelRejects += 1;
    showTypo(msg.pos, msg.tok);
  } else if (msg.ev === "finish") {
    state.modelPos = state.text.length;
    state.modelDone = elapsed();
  }
  render();
  checkFinish();
}

function showTypo(pos, tok) {
  const anchor = state.spans[Math.min(pos, state.spans.length - 1)];
  const chip = document.createElement("span");
  chip.className = "typo";
  chip.textContent = tok;
  chip.style.left = anchor.offsetLeft + "px";
  chip.style.top = anchor.offsetTop + "px";
  $("board").appendChild(chip);
  setTimeout(() => chip.remove(), 900);

  const entry = document.createElement("span");
  entry.className = "t";
  entry.innerHTML = `<b>@${pos}</b> `;
  entry.append(JSON.stringify(tok));
  $("log").appendChild(entry);
}

// --- you ------------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (state.phase !== "racing" || state.youDone !== null)) { e.preventDefault(); return start(); }
  if (state.phase !== "racing" || state.youDone !== null) return;
  if (e.metaKey || e.ctrlKey || e.altKey || e.key.length !== 1) return;
  e.preventDefault();

  // Same rule as the model: a wrong keystroke is rejected on the spot and you
  // retry that character, so `typed` is always a correct prefix of the text.
  const i = state.typed.length;
  if (e.key !== state.text[i]) {
    state.youErrors += 1;
    flashMiss(state.spans[i]);
  } else {
    state.typed += e.key;
    if (state.typed === state.text) state.youDone = elapsed();
  }
  render();
  checkFinish();
});

function flashMiss(span) {
  span.classList.remove("miss");
  void span.offsetWidth; // restart the animation on repeated misses
  span.classList.add("miss");
}

// --- rendering ------------------------------------------------------------

function render() {
  const { text, typed, spans, modelPos } = state;
  for (let i = 0; i < spans.length; i++) {
    const s = spans[i];
    let cls = "c";
    if (i < typed.length) cls += " ok";
    if (s.classList.contains("miss")) cls += " miss";
    if (i < modelPos) cls += " m";
    if (i === typed.length) cls += " cursor";
    if (i === modelPos && state.modelDone === null) cls += " mcursor";
    s.className = cls;
  }

  const secs = state.phase === "racing" || state.phase === "done" ? elapsed() : 0;
  const good = typed.length;
  const youT = state.youDone ?? secs;
  const modelT = state.modelDone ?? secs;
  const pct = (n) => (text.length ? (100 * n) / text.length : 0) + "%";
  $("you-bar").style.width = pct(good);
  $("model-bar").style.width = pct(modelPos);
  $("you-stats").textContent = text ? `${wpm(good, youT)} wpm · ${state.youErrors} typos` : "–";
  $("model-stats").textContent = text ? `${wpm(modelPos, modelT)} wpm · ${state.modelRejects} retries` : "–";
}

// Keep WPM ticking while racing even when nobody is typing.
setInterval(() => { if (state.phase === "racing") render(); }, 250);

function overlay(big, cls, small = "") {
  const el = $("overlay");
  if (big === null) { el.className = "overlay hidden"; delete el.dataset.result; return; }
  el.className = "overlay " + cls;
  el.textContent = big;
  if (small) {
    const s = document.createElement("small");
    s.textContent = small;
    el.appendChild(s);
  }
}

$("start").addEventListener("click", (e) => { e.currentTarget.blur(); start(); });
