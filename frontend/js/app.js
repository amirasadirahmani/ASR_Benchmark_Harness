"use strict";

const SAMPLE_RATE = 16000;

const state = {
  mode: "assistant",
  models: [],
  availableIds: new Set(),
  preflightOk: false,
  selected: { assistant: new Set(), benchmark: new Set() },
  ws: null,
  audioCtx: null,
  workletNode: null,
  micStream: null,
  results: new Map(),
  cardEls: new Map(),
  sortKey: "wer",
  sortDir: 1,
};

const el = (id) => document.getElementById(id);
const statusPill = el("statusPill");
const statusText = el("statusText");
const tabAssistant = el("tabAssistant");
const tabBenchmark = el("tabBenchmark");
const panelAssistant = el("panelAssistant");
const panelBenchmark = el("panelBenchmark");
const modelPickerAssistant = el("modelPickerAssistant");
const modelPickerBenchmark = el("modelPickerBenchmark");
const refTextAssistant = el("refTextAssistant");
const refTextBenchmark = el("refTextBenchmark");
const refWarningAssistant = el("refWarningAssistant");
const refWarningBenchmark = el("refWarningBenchmark");
const btnStartSession = el("btnStartSession");
const btnStopSession = el("btnStopSession");
const btnTriggerWake = el("btnTriggerWake");
const btnRunBenchmark = el("btnRunBenchmark");
const audioFileInput = el("audioFile");
const levelStrip = el("levelStrip");
const levelLabel = el("levelLabel");
const levelBars = el("levelBars");
const cardRow = el("cardRow");
const emptyState = el("emptyState");
const resultsHint = el("resultsHint");
const tableWrap = el("tableWrap");
const compareBody = el("compareBody");
const compareTable = el("compareTable");
const errorBanner = el("errorBanner");
const summaryBanner = el("summaryBanner");
const footerMeta = el("footerMeta");

init();

async function init() {
  buildLevelBars(28);
  wireTabs();
  wireRefWarnings();
  wireButtons();
  wireTableSort();
  await Promise.all([loadModels(), loadPreflight()]);
  applyDefaultSelection();
  renderModelPickers();
}

function applyDefaultSelection() {
  const selectable = state.preflightOk
    ? state.models.filter((m) => state.availableIds.has(m.id))
    : state.models;
  state.selected.assistant = new Set(selectable.map((m) => m.id));
  state.selected.benchmark = new Set(selectable.map((m) => m.id));
}

function buildLevelBars(n) {
  levelBars.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const s = document.createElement("span");
    levelBars.appendChild(s);
  }
}

async function loadModels() {
  try {
    const r = await fetch("/api/models");
    state.models = await r.json();
  } catch (err) {
    showError("فهرست مدل‌ها بارگذاری نشد: " + err.message);
  }
}

async function loadPreflight() {
  try {
    const r = await fetch("/api/preflight");
    const data = await r.json();
    const ok = (data.validation && data.validation.ok) || [];
    state.availableIds = new Set(ok);
    state.preflightOk = true;
  } catch {
    state.availableIds = new Set(state.models.map((m) => m.id));
    state.preflightOk = false;
  }
}

function renderModelPickers() {
  renderOnePicker(modelPickerAssistant, "assistant");
  renderOnePicker(modelPickerBenchmark, "benchmark");
}

function renderOnePicker(container, panelKey) {
  container.innerHTML = "";
  if (state.models.length === 0) {
    container.innerHTML = '<span class="hint">مدلی در models.yaml فعال نیست.</span>';
    return;
  }
  for (const m of state.models) {
    const known = !state.preflightOk || state.availableIds.has(m.id);
    const label = document.createElement("label");
    label.className = "model-chip"
      + (state.selected[panelKey].has(m.id) ? " checked" : "")
      + (known ? "" : " disabled");
    label.innerHTML = `
      <input type="checkbox" ${state.selected[panelKey].has(m.id) ? "checked" : ""} ${known ? "" : "disabled"} />
      <span>${escapeHtml(m.display_name)}</span>
      ${known ? "" : '<span class="unavailable-mark" title="فایل مدل روی دیسک پیدا نشد — ابتدا setup_models.py را اجرا کنید">⚠</span>'}
    `;
    const cb = label.querySelector("input");
    cb.addEventListener("change", () => {
      if (cb.checked) state.selected[panelKey].add(m.id);
      else state.selected[panelKey].delete(m.id);
      label.classList.toggle("checked", cb.checked);
    });
    container.appendChild(label);
  }
}

function wireTabs() {
  tabAssistant.addEventListener("click", () => switchMode("assistant"));
  tabBenchmark.addEventListener("click", () => switchMode("benchmark"));
}

function switchMode(mode) {
  if (mode === state.mode) return;
  if (state.mode === "assistant" && state.ws) stopAssistantSession();
  state.mode = mode;
  tabAssistant.classList.toggle("active", mode === "assistant");
  tabBenchmark.classList.toggle("active", mode === "benchmark");
  panelAssistant.classList.toggle("active", mode === "assistant");
  panelBenchmark.classList.toggle("active", mode === "benchmark");
  clearResults();
}

function wireRefWarnings() {
  const sync = (input, banner) => {
    const update = () => banner.classList.toggle("show", input.value.trim().length === 0);
    input.addEventListener("input", update);
    update();
  };
  sync(refTextAssistant, refWarningAssistant);
  sync(refTextBenchmark, refWarningBenchmark);
}

function wireButtons() {
  btnStartSession.addEventListener("click", startAssistantSession);
  btnStopSession.addEventListener("click", stopAssistantSession);
  btnTriggerWake.addEventListener("click", () => wsSend({ type: "trigger_wake" }));
  btnRunBenchmark.addEventListener("click", runBenchmarkMode);
}

function setStatus(stateKey, text) {
  statusPill.dataset.state = stateKey;
  statusText.textContent = text;
}

function setLevel(dbfs, active) {
  levelStrip.dataset.active = active ? "1" : "0";
  const pct = Math.max(0, Math.min(100, ((dbfs + 60) / 60) * 100));
  const bars = levelBars.children;
  const n = bars.length;
  const lit = Math.round((pct / 100) * n);
  for (let i = 0; i < n; i++) {
    const fromCenter = Math.abs(i - n / 2);
    const h = i < lit ? 15 + (1 - fromCenter / (n / 2)) * 70 : 8;
    bars[i].style.height = Math.max(8, Math.min(100, h)) + "%";
  }
}

async function startAssistantSession() {
  if (state.models.length > 0 && state.selected.assistant.size === 0) {
    showError("هیچ مدلی انتخاب نشده — حداقل یک مدل را از فهرست تیک بزنید.");
    return;
  }

  try {
    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (err) {
    showError("دسترسی به میکروفون رد شد: " + err.message);
    return;
  }

  clearResults();
  hideError();

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/session`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.onopen = async () => {
    const ref = refTextAssistant.value.trim();
    if (ref) wsSend({ type: "set_reference", text: ref });

    const all = state.models.map((m) => m.id);
    const chosen = [...state.selected.assistant];
    if (chosen.length > 0 && chosen.length < all.length) {
      wsSend({ type: "select_models", model_ids: chosen });
    }

    await startMicPump();
    btnStartSession.style.display = "none";
    btnStopSession.style.display = "";
    btnTriggerWake.style.display = "";
    setStatus("waiting", "در انتظار «آرینا» …");
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") return;
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    handleServerEvent(msg);
  };

  ws.onerror = () => showError("اتصال WebSocket با خطا مواجه شد.");

  ws.onclose = () => {
    state.ws = null;
    stopMicPump();
    btnStartSession.style.display = "";
    btnStopSession.style.display = "none";
    btnTriggerWake.style.display = "none";
    setStatus("idle", "جلسه پایان یافت");
    levelStrip.dataset.active = "0";
    levelLabel.textContent = "در انتظار شروع جلسه";
  };
}

function stopAssistantSession() {
  if (state.ws) {
    wsSend({ type: "stop_session" });
    state.ws.close();
  }
  stopMicPump();
}

function wsSend(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(obj));
  }
}

const WORKLET_SRC = `
class PcmFramer extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0][0];
    if (ch && ch.length) {
      const buf = new Int16Array(ch.length);
      for (let i = 0; i < ch.length; i++) {
        let s = Math.max(-1, Math.min(1, ch[i]));
        buf[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(buf.buffer, [buf.buffer]);
    }
    return true;
  }
}
registerProcessor("pcm-framer", PcmFramer);
`;

async function startMicPump() {
  state.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  const blobUrl = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
  await state.audioCtx.audioWorklet.addModule(blobUrl);
  URL.revokeObjectURL(blobUrl);

  const source = state.audioCtx.createMediaStreamSource(state.micStream);
  state.workletNode = new AudioWorkletNode(state.audioCtx, "pcm-framer");
  state.workletNode.port.onmessage = (ev) => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(ev.data);
  };
  source.connect(state.workletNode);
}

function stopMicPump() {
  if (state.micStream) {
    state.micStream.getTracks().forEach((t) => t.stop());
    state.micStream = null;
  }
  if (state.workletNode) {
    state.workletNode.disconnect();
    state.workletNode = null;
  }
  if (state.audioCtx) {
    state.audioCtx.close();
    state.audioCtx = null;
  }
}

async function runBenchmarkMode() {
  const file = audioFileInput.files[0];
  if (!file) {
    showError("ابتدا یک فایل صوتی WAV انتخاب کنید.");
    return;
  }
  if (state.models.length > 0 && state.selected.benchmark.size === 0) {
    showError("هیچ مدلی انتخاب نشده — حداقل یک مدل را از فهرست تیک بزنید.");
    return;
  }

  clearResults();
  hideError();
  setStatus("processing", "در حال اجرای محک …");
  btnRunBenchmark.disabled = true;

  const chosen = [...state.selected.benchmark];
  for (const id of chosen) showPendingCard(id, displayNameOf(id));

  const fd = new FormData();
  fd.append("audio", file);
  fd.append("reference_text", refTextBenchmark.value.trim());
  if (chosen.length > 0 && chosen.length < state.models.length) {
    fd.append("model_ids", chosen.join(","));
  }

  try {
    const r = await fetch("/api/benchmark/run", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) {
      showError(data.detail || "اجرای محک ناموفق بود.");
      setStatus("error", "خطا");
      return;
    }
    for (const raw of data.results || []) renderResultFromRest(raw);
    finalizeBenchmark(data);
    setStatus("idle", "پایان محک");
  } catch (err) {
    showError("درخواست به سرور ناموفق بود: " + err.message);
    setStatus("error", "خطا");
  } finally {
    btnRunBenchmark.disabled = false;
  }
}

function displayNameOf(id) {
  const m = state.models.find((x) => x.id === id);
  return m ? m.display_name : id;
}

function handleServerEvent(msg) {
  switch (msg.type) {
    case "session_started":
      setStatus("waiting", "در انتظار «آرینا» …");
      break;
    case "level":
      setLevel(msg.dbfs, false);
      break;
    case "wake_detected":
      setStatus("listening", "بیدارباش شنیده شد — در حال شنیدن دستور …");
      levelLabel.textContent = "در حال شنیدن دستور";
      break;
    case "speech_start":
      setStatus("listening", "در حال شنیدن دستور …");
      break;
    case "speech":
      setLevel(msg.level_dbfs, true);
      break;
    case "silence":
      levelLabel.textContent = "سکوت — در حال بررسی پایان دستور …";
      break;
    case "wake_timeout":
      setStatus("waiting", "زمان بیدارباش تمام شد — دوباره «آرینا» بگویید");
      break;
    case "utterance_too_short":
      setStatus("waiting", "دستور خیلی کوتاه بود — دوباره تلاش کنید");
      break;
    case "utterance_ready":
      setStatus("processing", "در حال اجرای محک روی همهٔ مدل‌ها …");
      levelLabel.textContent = "در حال پردازش";
      break;
    case "utterance_cancelled":
      setStatus("waiting", "لغو شد — در انتظار «آرینا» …");
      break;
    case "listening_resumed":
      setStatus("waiting", "در انتظار «آرینا» …");
      break;
    case "session_stopped":
    case "session_ended":
      setStatus("idle", "جلسه پایان یافت");
      break;
    case "benchmark_started":
      clearResults();
      resultsHint.textContent = `در حال اجرا روی ${msg.models_total} مدل …`;
      for (const id of msg.model_ids || []) showPendingCard(id, displayNameOf(id));
      break;
    case "model_started":
      showPendingCard(msg.model_id, msg.display_name, msg.index, msg.total);
      break;
    case "model_completed":
      renderResultFromEvent(msg);
      break;
    case "benchmark_completed":
      finalizeBenchmark(msg);
      setStatus("waiting", "در انتظار «آرینا» …");
      break;
    case "error":
      showError(msg.message || "خطای نامشخص");
      break;
    case "pong":
      break;
    default:
      break;
  }
}

function clearResults() {
  cardRow.innerHTML = "";
  state.results.clear();
  state.cardEls.clear();
  emptyState.style.display = "";
  tableWrap.style.display = "none";
  compareBody.innerHTML = "";
  summaryBanner.classList.remove("show");
  resultsHint.textContent = "";
}

function ensureCard(modelId, displayName) {
  emptyState.style.display = "none";
  let card = state.cardEls.get(modelId);
  if (card) return card;

  card = document.createElement("div");
  card.className = "model-card pending";
  card.dataset.modelId = modelId;
  card.innerHTML = `
    <div class="bar"></div>
    <div class="body">
      <div class="name">${escapeHtml(displayName || modelId)}</div>
      <div class="runtime">در حال آماده‌سازی …</div>
    </div>
  `;
  cardRow.appendChild(card);
  state.cardEls.set(modelId, card);
  return card;
}

function showPendingCard(modelId, displayName, index, total) {
  const card = ensureCard(modelId, displayName);
  card.className = "model-card pending";
  const rt = card.querySelector(".runtime");
  if (rt) rt.textContent = index && total ? `مدل ${index} از ${total} …` : "در حال اجرا …";
}

function renderCard(res) {
  const card = ensureCard(res.modelId, res.displayName);
  card.className = "model-card " + (res.success ? "ok" : "fail");
  const wer = fmtPct(res.wer);
  const cer = fmtPct(res.cer);

  card.innerHTML = `
    <div class="bar"></div>
    <div class="body">
      <div class="name">${escapeHtml(res.displayName)}</div>
      <div class="runtime">${escapeHtml(res.runtime || "")}${res.runs && res.runs > 1 ? ` · ${res.runs} اجرا` : ""}</div>
      <div class="text-out">${res.success ? escapeHtml(res.text || "(متنی برنگشت)") : escapeHtml(res.error || "خطای نامشخص")}</div>
      ${res.success ? `
      <div class="metric-grid">
        <div class="metric ${res.wer != null ? (res.wer <= 0.15 ? "good" : res.wer >= 0.40 ? "bad" : "") : ""}">
          <span class="v data-num">${wer}</span><span class="k">WER٪</span>
        </div>
        <div class="metric ${res.cer != null ? (res.cer <= 0.10 ? "good" : res.cer >= 0.30 ? "bad" : "") : ""}">
          <span class="v data-num">${cer}</span><span class="k">CER٪</span>
        </div>
        <div class="metric"><span class="v data-num">${fmtNum(res.rtf, 2)}</span><span class="k">RTF</span></div>
        <div class="metric"><span class="v data-num">${fmtNum(res.peakRamMb, 0)}</span><span class="k">RAM (MB)</span></div>
      </div>` : ""}
    </div>
  `;
}

function renderResultFromEvent(e) {
  const res = {
    modelId: e.model_id, displayName: e.display_name, runtime: "",
    success: e.success, error: e.error, errorType: e.error_type,
    text: e.text, wer: e.wer, cer: e.cer, rtf: e.rtf,
    loadTime: e.load_time, inferenceTime: e.inference_time,
    peakRamMb: e.peak_ram_mb, runs: null,
  };
  state.results.set(res.modelId, res);
  renderCard(res);
  addTableRow(res);
}

function renderResultFromRest(r) {
  const m = r.metrics || {};
  const res = {
    modelId: r.model_id, displayName: r.display_name, runtime: r.runtime,
    success: r.success, error: r.error, errorType: r.error_type,
    text: r.text, wer: m.wer, cer: m.cer, rtf: r.rtf,
    loadTime: r.load_time, inferenceTime: r.inference_time,
    peakRamMb: r.peak_ram_mb, runs: r.runs,
  };
  state.results.set(res.modelId, res);
  renderCard(res);
  addTableRow(res);
}

function addTableRow() {
  tableWrap.style.display = "";
  renderTableRows();
}

function renderTableRows() {
  const rows = [...state.results.values()];
  const key = state.sortKey;
  rows.sort((a, b) => {
    const av = a[key], bv = b[key];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    return (av - bv) * state.sortDir;
  });

  const bestFor = {};
  for (const k of ["wer", "cer", "rtf", "loadTime", "inferenceTime", "peakRamMb"]) {
    const vals = rows.filter((r) => r.success && r[k] != null).map((r) => r[k]);
    bestFor[k] = vals.length ? Math.min(...vals) : null;
  }

  compareBody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    if (!r.success) tr.className = "fail";
    const cell = (k, fmt) => {
      const v = r[k];
      const isBest = r.success && bestFor[k] != null && v === bestFor[k];
      return `<td class="data-num${isBest ? " best" : ""}">${fmt(v)}</td>`;
    };
    tr.innerHTML = `
      <td class="model-name">${escapeHtml(r.displayName)}</td>
      ${cell("wer", fmtPct)}
      ${cell("cer", fmtPct)}
      ${cell("rtf", (v) => fmtNum(v, 2))}
      ${cell("loadTime", (v) => fmtNum(v, 2))}
      ${cell("inferenceTime", (v) => fmtNum(v, 2))}
      ${cell("peakRamMb", (v) => fmtNum(v, 0))}
    `;
    compareBody.appendChild(tr);
  }
}

function wireTableSort() {
  compareTable.querySelectorAll("thead th[data-key]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (state.sortKey === key) state.sortDir *= -1;
      else {
        state.sortKey = key;
        state.sortDir = 1;
      }
      renderTableRows();
    });
  });
}

function finalizeBenchmark(data) {
  resultsHint.textContent = `${data.models_succeeded || 0} موفق، ${data.models_failed || 0} ناموفق`;
  if (!data.has_reference && data.no_reference_message) showError(data.no_reference_message);

  if (data.best_model || data.fastest_model || data.lightest_model) {
    const parts = [];
    if (data.best_model) parts.push(`دقیق‌ترین: <b>${escapeHtml(displayNameOf(data.best_model))}</b>`);
    if (data.fastest_model) parts.push(`سریع‌ترین: <b>${escapeHtml(displayNameOf(data.fastest_model))}</b>`);
    if (data.lightest_model) parts.push(`کم‌مصرف‌ترین (RAM): <b>${escapeHtml(displayNameOf(data.lightest_model))}</b>`);
    summaryBanner.innerHTML = parts.join(" &nbsp;·&nbsp; ");
    summaryBanner.classList.add("show");
  }
  footerMeta.textContent = data.benchmark_id ? `شناسهٔ اجرا: ${data.benchmark_id}` : "";
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.add("show");
}

function hideError() {
  errorBanner.classList.remove("show");
}

function fmtPct(v) {
  return v == null ? "—" : (v * 100).toFixed(1);
}

function fmtNum(v, digits) {
  return v == null ? "—" : Number(v).toFixed(digits);
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
