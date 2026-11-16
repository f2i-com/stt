import init, { STT, decodeWav } from "../pkg/stt.js";

const $ = (id) => document.getElementById(id);
const els = {
  brandDot: $("brandDot"),
  perfBadge: $("perfBadge"),
  themeBtn: $("themeBtn"),
  dropPanel: $("dropPanel"),
  modeDictationBtn: $("modeDictationBtn"),
  modeBothBtn: $("modeBothBtn"),
  modeCommandsBtn: $("modeCommandsBtn"),
  micBtn: $("micBtn"),
  sampleBtn: $("sampleBtn"),
  fileInput: $("fileInput"),
  copyBtn: $("copyBtn"),
  downloadTxtBtn: $("downloadTxtBtn"),
  clearBtn: $("clearBtn"),
  statusLine: $("statusLine"),
  statusText: $("statusText"),
  speedStat: $("speedStat"),
  meterCanvas: $("meterCanvas"),
  meterState: $("meterState"),
  meterDb: $("meterDb"),
  transcript: $("transcript"),
  wordChips: $("wordChips"),
  heardLabel: $("heardLabel"),
  heardMeta: $("heardMeta"),
  teachForm: $("teachForm"),
  teachInput: $("teachInput"),
  teachBtn: $("teachBtn"),
  cmdList: $("cmdList"),
  exportCmdsBtn: $("exportCmdsBtn"),
  importCmdsInput: $("importCmdsInput"),
  forgetCmdsBtn: $("forgetCmdsBtn"),
  hangSlider: $("hangSlider"),
  hangVal: $("hangVal"),
  onsetSlider: $("onsetSlider"),
  onsetVal: $("onsetVal"),
  strictSlider: $("strictSlider"),
  strictVal: $("strictVal"),
  resetCfgBtn: $("resetCfgBtn"),
  log: $("log"),
};

const STORAGE_KEY = "stt.computer/commands/v1";

let stt = null;
let session = null;
let activeMode = "dictation";
let teachingLabel = null;
let audioCtx = null;

// Rolling history for the live level + floor visualizer
const HISTORY = 120;
const levelHist = new Float32Array(HISTORY).fill(-90);
const floorHist = new Float32Array(HISTORY).fill(-85);
const speakHist = new Uint8Array(HISTORY);

function log(msg) {
  const t = new Date().toLocaleTimeString();
  els.log.textContent = `${t}  ${msg}\n` + els.log.textContent.slice(0, 6000);
}

// ---- Theme switcher -----------------------------------------------------------------------
const themes = ["auto", "light", "dark"];
let themeIdx = 0;
els.themeBtn.onclick = () => {
  themeIdx = (themeIdx + 1) % themes.length;
  const mode = themes[themeIdx];
  if (mode === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = mode;
  els.themeBtn.textContent = `Theme: ${mode[0].toUpperCase() + mode.slice(1)}`;
};

// ---- Visualizer canvas --------------------------------------------------------------------
function drawMeter() {
  const canvas = els.meterCanvas;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const dbToY = (db) => {
    const clamped = Math.max(-85, Math.min(0, db));
    return h - ((clamped + 85) / 85) * (h - 14) - 4;
  };

  const barW = w / HISTORY;
  for (let i = 0; i < HISTORY; i++) {
    const y = dbToY(levelHist[i]);
    const barH = Math.max(2, h - y);
    ctx.fillStyle = speakHist[i] ? "#34d399" : "rgba(120, 145, 165, 0.42)";
    ctx.fillRect(i * barW + 1, y, Math.max(1, barW - 1.5), barH);
  }

  // Adaptive floor line
  ctx.beginPath();
  ctx.strokeStyle = "#fbbf24";
  ctx.lineWidth = 1.6;
  for (let i = 0; i < HISTORY; i++) {
    const x = i * barW + barW * 0.5;
    const y = dbToY(floorHist[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function pushMeterSample(levelDb, floorDb, speaking) {
  levelHist.copyWithin(0, 1);
  floorHist.copyWithin(0, 1);
  speakHist.copyWithin(0, 1);
  levelHist[HISTORY - 1] = levelDb;
  floorHist[HISTORY - 1] = floorDb;
  speakHist[HISTORY - 1] = speaking ? 1 : 0;
  els.meterState.textContent = teachingLabel
    ? `TEACHING “${teachingLabel}”`
    : speaking
      ? "SPEAKING"
      : session
        ? "LISTENING"
        : "MIC IDLE";
  els.meterDb.textContent = `level ${levelDb.toFixed(0)} dB · floor ${floorDb.toFixed(0)} dB`;
  drawMeter();
}

// ---- Transcript & Word Timings ------------------------------------------------------------
function appendTranscript(text, words, duration, tookMs) {
  if (!text) return;
  els.transcript.textContent += (els.transcript.textContent ? " " : "") + text;
  els.transcript.scrollTop = els.transcript.scrollHeight;

  els.wordChips.textContent = "";
  for (const w of words || []) {
    const chip = document.createElement("span");
    chip.className = "word-chip";
    const wordSpan = document.createElement("strong");
    wordSpan.textContent = w.word;
    const metaSpan = document.createElement("span");
    metaSpan.className = "wt";
    metaSpan.textContent = `${w.start.toFixed(2)}s · ${Math.round(w.confidence * 100)}%`;
    chip.append(wordSpan, metaSpan);
    els.wordChips.append(chip);
  }

  if (duration && tookMs > 0) {
    const rtf = (duration * 1000) / tookMs;
    els.speedStat.textContent = `${duration.toFixed(1)}s in ${Math.round(tookMs)}ms (${rtf.toFixed(0)}× real time)`;
    els.perfBadge.textContent = `${rtf.toFixed(0)}× Real Time · Zero Network`;
  }
}

// ---- Persistence for Taught Commands ------------------------------------------------------
function saveCommands() {
  if (!stt) return;
  try {
    const bytes = stt.exportCommands();
    let bin = "";
    for (const b of bytes) bin += String.fromCharCode(b);
    localStorage.setItem(STORAGE_KEY, btoa(bin));
  } catch (err) {
    log(`could not save commands: ${err.message}`);
  }
}

function restoreCommands() {
  if (!stt) return;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const bytes = Uint8Array.from(atob(raw), (c) => c.charCodeAt(0));
    if (stt.importCommands(bytes)) {
      const n = stt.labels().length;
      if (n > 0) log(`restored ${n} saved command(s) from browser storage`);
    }
  } catch {}
}

function renderCommands() {
  els.cmdList.textContent = "";
  const cmds = stt ? stt.commands() : [];
  if (!cmds.length) {
    const li = document.createElement("li");
    li.style.color = "var(--muted)";
    li.textContent = "No commands taught yet. Enter a phrase above and click Teach.";
    els.cmdList.append(li);
    return;
  }
  for (const { label, count } of cmds) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "cmd-name";
    name.textContent = label;
    const cnt = document.createElement("span");
    cnt.className = "cmd-count";
    cnt.textContent = `${count} take${count === 1 ? "" : "s"}`;
    const addBtn = document.createElement("button");
    addBtn.className = "sm";
    addBtn.textContent = "+ Add take";
    addBtn.onclick = () => armTeach(label);
    const rmBtn = document.createElement("button");
    rmBtn.className = "sm";
    rmBtn.textContent = "Remove";
    rmBtn.onclick = async () => {
      if (session) await session.removeCommand(label);
      else stt.removeCommand(label);
      saveCommands();
      renderCommands();
    };
    li.append(name, cnt, addBtn, rmBtn);
    els.cmdList.append(li);
  }
}

function updateStatus() {
  if (teachingLabel) {
    els.statusLine.className = "status-line teaching";
    els.statusText.textContent = `Say “${teachingLabel}” clearly into your microphone now…`;
    return;
  }
  els.statusLine.className = "status-line";
  if (session) {
    const modeDesc =
      activeMode === "dictation"
        ? "Dictation active — pause briefly to transcribe."
        : activeMode === "both"
          ? "Hybrid active — short taught commands trigger actions; longer speech dictates."
          : "Commands mode active — say any taught command.";
    els.statusText.textContent = modeDesc;
  } else {
    els.statusText.textContent = "Microphone is off. Start listening, try the sample audio, or drop an audio file.";
  }
}

// ---- Mode & Controls ----------------------------------------------------------------------
async function selectMode(mode) {
  activeMode = mode;
  els.modeDictationBtn.classList.toggle("active", mode === "dictation");
  els.modeBothBtn.classList.toggle("active", mode === "both");
  els.modeCommandsBtn.classList.toggle("active", mode === "commands");
  if (!stt) return;
  const next = session ? await session.configure({ mode }) : stt.configure({ mode });
  els.hangSlider.value = Math.round(next.hangMs);
  els.hangVal.textContent = `${Math.round(next.hangMs)} ms`;
  updateStatus();
}

async function applySliders() {
  if (!stt) return;
  const hangMs = Number(els.hangSlider.value);
  const onsetDb = Number(els.onsetSlider.value);
  const t = Number(els.strictSlider.value) / 100;
  const maxDistance = +(3.8 + t * 1.8).toFixed(2);
  const maxRatio = +(0.85 + t * 0.13).toFixed(2);

  els.hangVal.textContent = `${hangMs} ms`;
  els.onsetVal.textContent = `+${onsetDb} dB`;
  els.strictVal.textContent = maxDistance.toFixed(2);

  const patch = { hangMs, onsetDb, maxDistance, maxRatio };
  if (session) await session.configure(patch);
  else stt.configure(patch);
}

// ---- Live Microphone Session (`stt.listen()`) ---------------------------------------------
async function startMic() {
  els.micBtn.disabled = true;
  try {
    session = await stt.listen({
      mode: activeMode,
      onPhrase: (p) => {
        appendTranscript(p.text, p.words, p.durationSeconds, p.tookMs || 0);
        log(`dictated ${p.durationSeconds.toFixed(1)}s in ${Math.round(p.tookMs || 0)}ms: "${p.text}"`);
      },
      onCommand: (e) => {
        if (e.label === null) {
          els.heardLabel.className = "heard-label muted";
          els.heardLabel.textContent = "Speech detected (no commands taught yet)";
          els.heardMeta.textContent = "Add a command phrase below.";
        } else if (e.accepted) {
          els.heardLabel.className = "heard-label";
          els.heardLabel.textContent = e.label;
          const ru = e.runnerUp ? ` · runner-up "${e.runnerUp}" (${e.runnerUpDistance.toFixed(2)})` : "";
          els.heardMeta.textContent = `accepted · distance ${e.distance.toFixed(2)}${ru}`;
          log(`command "${e.label}" accepted (d=${e.distance.toFixed(2)})`);
        } else {
          els.heardLabel.className = "heard-label muted";
          els.heardLabel.textContent = `Not sure (closest: ${e.label})`;
          els.heardMeta.textContent = `rejected · distance ${e.distance.toFixed(2)}`;
          log(`rejected weak match "${e.label}" (d=${e.distance.toFixed(2)})`);
        }
      },
      onTaught: (e) => {
        teachingLabel = null;
        saveCommands();
        renderCommands();
        updateStatus();
        els.heardLabel.className = "heard-label";
        els.heardLabel.textContent = `Taught “${e.label}”`;
        els.heardMeta.textContent = `${e.count} example${e.count === 1 ? "" : "s"} stored (${e.durationSeconds.toFixed(2)}s)`;
        log(`taught "${e.label}" (${e.count} total)`);
      },
      onIgnored: (e) => {
        log(`ignored short noise (${Math.round(e.durationSeconds * 1000)} ms)`);
      },
      onLevel: ({ levelDb, floorDb, speaking }) => {
        pushMeterSample(levelDb, floorDb, speaking);
      },
      onError: (err) => log(`error: ${err.message}`),
    });

    if (teachingLabel) {
      await session.teachNext(teachingLabel);
    }

    els.brandDot.classList.add("recording");
    els.micBtn.textContent = "■ Stop Microphone";
    els.micBtn.className = "danger";
    log(
      `microphone open at ${session.sampleRate} Hz (${session.usingWorker ? "Web Worker" : "main thread"})`
    );
    updateStatus();
  } catch (err) {
    els.statusText.textContent = `Could not access microphone: ${err.message}`;
    log(`mic error: ${err.message}`);
  } finally {
    els.micBtn.disabled = false;
  }
}

async function stopMic() {
  if (!session) return;
  els.micBtn.disabled = true;
  try {
    await session.stop();
  } finally {
    session = null;
    teachingLabel = null;
    els.brandDot.classList.remove("recording");
    els.micBtn.textContent = "● Start Microphone";
    els.micBtn.className = "primary";
    els.micBtn.disabled = false;
    pushMeterSample(-90, -85, false);
    updateStatus();
    log("microphone stopped");
  }
}

async function armTeach(label) {
  const clean = label.trim().toLowerCase();
  if (!clean || !stt) return;
  teachingLabel = clean;
  updateStatus();
  if (!session) {
    await startMic();
  } else {
    await session.teachNext(clean);
  }
}

// ---- Sample Audio & File Transcription ----------------------------------------------------
async function playPcmPreview(samples, sampleRate) {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    audioCtx ??= new Ctx();
    if (audioCtx.state === "suspended") await audioCtx.resume();
    const buf = audioCtx.createBuffer(1, samples.length, sampleRate);
    buf.copyToChannel(samples, 0);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    src.start();
  } catch {}
}

els.sampleBtn.onclick = async () => {
  els.sampleBtn.disabled = true;
  try {
    els.statusText.textContent = "Transcribing sample clip (henry.wav)…";
    const resp = await fetch("../tests/data/henry.wav");
    const wavBytes = new Uint8Array(await resp.arrayBuffer());
    const decoded = decodeWav(wavBytes);
    playPcmPreview(decoded.samples, decoded.sampleRate);

    const t0 = performance.now();
    const res = stt.transcribe(wavBytes);
    const tookMs = performance.now() - t0;
    appendTranscript(res.text, res.words, res.duration, tookMs);
    log(`sample clip (${res.duration.toFixed(2)}s) transcribed in ${Math.round(tookMs)}ms: "${res.text}"`);
    updateStatus();
  } catch (err) {
    log(`sample error: ${err.message}`);
  } finally {
    els.sampleBtn.disabled = false;
  }
};

async function handleAudioFile(file) {
  if (!file || !stt) return;
  els.statusText.textContent = `Transcribing ${file.name}…`;
  try {
    const t0 = performance.now();
    const res = await stt.transcribeFile(file);
    const tookMs = performance.now() - t0;
    appendTranscript(res.text, res.words, res.duration, tookMs);
    log(`file "${file.name}" (${res.duration.toFixed(2)}s) transcribed in ${Math.round(tookMs)}ms`);
  } catch (err) {
    els.statusText.textContent = `Could not transcribe ${file.name}: ${err.message}`;
    log(`file error: ${err.message}`);
  }
}

els.fileInput.onchange = () => {
  if (els.fileInput.files?.[0]) handleAudioFile(els.fileInput.files[0]);
  els.fileInput.value = "";
};

els.dropPanel.addEventListener("dragover", (e) => {
  e.preventDefault();
  els.dropPanel.classList.add("drag-over");
});
els.dropPanel.addEventListener("dragleave", () => els.dropPanel.classList.remove("drag-over"));
els.dropPanel.addEventListener("drop", (e) => {
  e.preventDefault();
  els.dropPanel.classList.remove("drag-over");
  const file = e.dataTransfer?.files?.[0];
  if (file) handleAudioFile(file);
});

// ---- Buttons & Events ---------------------------------------------------------------------
els.micBtn.onclick = () => (session ? stopMic() : startMic());
els.modeDictationBtn.onclick = () => selectMode("dictation");
els.modeBothBtn.onclick = () => selectMode("both");
els.modeCommandsBtn.onclick = () => selectMode("commands");

els.copyBtn.onclick = () => navigator.clipboard?.writeText(els.transcript.textContent);
els.clearBtn.onclick = () => {
  els.transcript.textContent = "";
  els.wordChips.textContent = "";
};
els.downloadTxtBtn.onclick = () => {
  const blob = new Blob([els.transcript.textContent], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "transcript.txt";
  a.click();
  URL.revokeObjectURL(a.href);
};

els.teachForm.onsubmit = (e) => {
  e.preventDefault();
  armTeach(els.teachInput.value);
  els.teachInput.value = "";
};

els.exportCmdsBtn.onclick = () => {
  if (!stt) return;
  const bytes = stt.exportCommands();
  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "commands.sttcmds";
  a.click();
  URL.revokeObjectURL(a.href);
};

els.importCmdsInput.onchange = async () => {
  const f = els.importCmdsInput.files?.[0];
  els.importCmdsInput.value = "";
  if (!f || !stt) return;
  const bytes = new Uint8Array(await f.arrayBuffer());
  const ok = session ? await session.importCommands(bytes) : stt.importCommands(bytes);
  if (ok) {
    saveCommands();
    renderCommands();
    log(`imported ${stt.labels().length} command(s) from ${f.name}`);
  } else {
    log(`failed to import ${f.name}: invalid or incompatible command file`);
  }
};

els.forgetCmdsBtn.onclick = async () => {
  if (!stt) return;
  if (session) await session.clearCommands();
  else stt.clearCommands();
  saveCommands();
  renderCommands();
  log("cleared all taught commands");
};

els.hangSlider.oninput = applySliders;
els.onsetSlider.oninput = applySliders;
els.strictSlider.oninput = applySliders;
els.resetCfgBtn.onclick = async () => {
  els.hangSlider.value = activeMode === "commands" ? 350 : 600;
  els.onsetSlider.value = 15;
  els.strictSlider.value = 45;
  await applySliders();
};

// ---- Boot `stt.wasm` ----------------------------------------------------------------------
drawMeter();
(async () => {
  try {
    const t0 = performance.now();
    await init();
    stt = new STT();
    stt.warmup();
    const loadMs = Math.round(performance.now() - t0);

    restoreCommands();
    renderCommands();
    await applySliders();

    els.micBtn.disabled = false;
    els.sampleBtn.disabled = false;
    els.teachBtn.disabled = false;
    els.perfBadge.textContent = `16 kHz Conformer + DTW · Ready (${loadMs} ms)`;
    updateStatus();
    log(`stt.wasm loaded & model warmed up in ${loadMs} ms`);
  } catch (err) {
    els.perfBadge.textContent = "Load Failed";
    els.statusText.textContent = `Failed to initialize stt.wasm: ${err.message}`;
    log(`fatal: ${err.message}`);
  }
})();
