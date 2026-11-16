// Smoke & integration test of the built self-contained WebAssembly package (`pkg/stt.wasm` + `pkg/stt.js`).
//
//   npm run build
//   node tests/smoke.mjs
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const wasmBytes = fs.readFileSync(path.join(root, "pkg", "stt.wasm"));
const henryWav = fs.readFileSync(path.join(root, "tests", "data", "henry.wav"));
const SENTENCE = "and henry might return to england at any moment";

// --- 1. Pure single-file `stt.wasm` test (ZERO `.js` files, ZERO external imports!) -------
const rawMod = new WebAssembly.Module(wasmBytes);
assert.equal(WebAssembly.Module.imports(rawMod).length, 0, "stt.wasm has zero external imports");
const rawInst = new WebAssembly.Instance(rawMod, {});
const w = rawInst.exports;
const internalGlue = new TextDecoder().decode(
  new Uint8Array(w.memory.buffer, w.stt_glue_ptr(), w.stt_glue_len())
);
const { STT: StandaloneSTT, decodeWav: standaloneDecodeWav } = Function("wasm", internalGlue)(w);
assert.equal(StandaloneSTT.hasEmbeddedModel(), true);

const standalone = new StandaloneSTT();
const standaloneTr = standalone.transcribe(henryWav);
assert.equal(standaloneTr.text, SENTENCE);
assert.equal(standaloneTr.words.length, 9);
assert.ok(standaloneTr.confidence > 0.7 && standaloneTr.confidence <= 1.0);
assert.ok(standaloneTr.duration > 3.0 && standaloneTr.duration < 5.0);
assert.ok(
  standaloneTr.words.every(
    (wd, i, a) =>
      wd.word.length > 0 &&
      wd.start < wd.end &&
      wd.confidence > 0.4 &&
      (i === 0 || a[i - 1].end <= wd.start + 1e-9)
  ),
  JSON.stringify(standaloneTr.words)
);
standalone.free();

// Also verify raw C-ABI directly on `stt.wasm` without even running the JS glue:
const rawHandle = w.stt_new(0, 0, 0, 0);
assert.ok(rawHandle >= 1);
const wavPtr = w.stt_alloc(henryWav.length);
new Uint8Array(w.memory.buffer, wavPtr, henryWav.length).set(henryWav);
assert.equal(w.stt_transcribe_wav(rawHandle, wavPtr, henryWav.length), 0);
const cAbiJson = JSON.parse(
  new TextDecoder().decode(new Uint8Array(w.memory.buffer, w.stt_json_ptr(), w.stt_json_len()))
);
assert.equal(cAbiJson.text, SENTENCE);
w.stt_drop(rawHandle);

// --- 2. ES module loader (`pkg/stt.js`) ---------------------------------------------------
const { initSync, STT, Dictation, Listener, decodeWav, encodeWav } = await import(
  pathToFileURL(path.join(root, "pkg", "stt.js")).href
);
initSync(wasmBytes);

assert.deepEqual(STT.modes(), ["dictation", "commands", "both"]);
assert.equal(STT.modelSampleRate(), 16000);

const clip = decodeWav(henryWav);
assert.equal(clip.sampleRate, 16000);
assert.ok(clip.samples instanceof Float32Array && clip.samples.length > 48000);
assert.equal(String.fromCharCode(...encodeWav(clip.samples.subarray(0, 64), 16000).slice(0, 4)), "RIFF");

// --- 3. One-shot transcription across WAV, Float32Array, and `{ samples, sampleRate }` ----
const stt = new STT();
const began = performance.now();
const fromPcm = stt.transcribe(clip.samples, 16000);
const tookSec = (performance.now() - began) / 1000;
const rtf = clip.duration / tookSec;
assert.equal(fromPcm.text, SENTENCE);
assert.ok(rtf > 2, `expected >2x real time in WASM, got ${rtf.toFixed(1)}x`);

// Accepts `{ samples, sampleRate }` (just like `TTS.speak()` output!) directly:
assert.equal(stt.transcribeText(clip), SENTENCE);
// Empty input returns empty string cleanly:
assert.equal(stt.transcribeText(new Float32Array(0), 16000), "");

// --- 4. Configuration & validation --------------------------------------------------------
assert.equal(stt.mode, "dictation");
assert.equal(stt.sampleRate, 16000);
const beforeSettings = JSON.stringify(stt.settings);
assert.throws(() => stt.configure({ unknownOption: 123 }), /unknown setting/);
assert.throws(() => stt.configure({ hangMs: NaN }), /hangMs must be a finite number/);
assert.throws(() => stt.configure({ mode: "magic" }), /unknown mode/);
assert.equal(JSON.stringify(stt.settings), beforeSettings, "failed configure leaves settings untouched");

// --- 5. Live 48 kHz streaming dictation + hybrid `"both"` mode ----------------------------
let seed = 1;
const rand = () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 2 ** 31 - 1;
};
const noise = (n, amp) => Float32Array.from({ length: n }, () => amp * rand());
function toneWord(sr, tones, stretch = 1) {
  const per = Math.round(sr * 0.08 * stretch);
  const out = new Float32Array(per * tones.length);
  let phase = 0;
  tones.forEach((hz, k) => {
    for (let i = 0; i < per; i++) {
      phase += (2 * Math.PI * hz) / sr;
      out[k * per + i] = 0.3 * Math.sin(phase) + 0.1 * Math.sin(2.3 * phase) + 0.0005 * rand();
    }
  });
  return out;
}
const concat = (...parts) => {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
};
const feed = (target, samples, chunk) => {
  const events = [];
  for (let i = 0; i < samples.length; i += chunk) {
    events.push(...target.push(samples.subarray(i, i + chunk)));
  }
  return events;
};

// Upsample henry.wav to 48 kHz and stream twice with pauses:
const up48 = new Float32Array(clip.samples.length * 3);
for (let i = 0; i < up48.length; i++) {
  const at = i / 3;
  const k = Math.floor(at);
  const next = Math.min(k + 1, clip.samples.length - 1);
  up48[i] = clip.samples[k] + (at - k) * (clip.samples[next] - clip.samples[k]);
}

const liveStt = new STT({ sampleRate: 48000, mode: "both" });
const CLEAR_TONES = [400, 1200, 700, 2000];
assert.equal(
  liveStt.teach("clear screen", concat(noise(9600, 0.0005), toneWord(48000, CLEAR_TONES), noise(9600, 0.0005)), 48000),
  true
);

const mixedStream = concat(
  noise(48000, 0.0005),
  toneWord(48000, CLEAR_TONES, 1.1), // taught command!
  noise(48000, 0.0005),
  up48, // free-form English sentence!
  noise(72000, 0.0005)
);
const mixedEvents = [...feed(liveStt, mixedStream, 1024), ...liveStt.flush()];
assert.equal(mixedEvents.length, 2, JSON.stringify(mixedEvents));
assert.equal(mixedEvents[0].type, "heard");
assert.equal(mixedEvents[0].label, "clear screen");
assert.equal(mixedEvents[0].accepted, true);
assert.equal(mixedEvents[1].type, "phrase");
assert.equal(mixedEvents[1].text, SENTENCE);
liveStt.free();

// Also verify standalone `Dictation` class (works both without and with explicit model bytes):
const dict = new Dictation(48000);
const dictStream = concat(noise(48000, 0.0005), up48, noise(72000, 0.0005), up48, noise(72000, 0.0005));
const phrases = [...feed(dict, dictStream, 1024), ...dict.flush()];
assert.deepEqual(
  phrases.map((p) => p.text),
  [SENTENCE, SENTENCE]
);
assert.equal(dict.speaking, false);
dict.free();

assert.throws(() => new Dictation(16000, new Uint8Array(100)), /weight file/);

// --- 6. Taught commands (`Listener` / `STT` in commands mode) -----------------------------
const SR = 48000;
const WORDS = {
  lights: [400, 1200, 700, 2000],
  music: [2000, 700, 1200, 400],
  stop: [900, 900, 3000, 500],
};
const gap = () => noise(SR * 0.8, 0.0005);

const listener = new Listener(SR);
assert.equal(listener.sampleRate, SR);
assert.deepEqual(listener.labels(), []);

let events = feed(listener, concat(gap(), toneWord(SR, WORDS.stop), gap()), 960);
assert.equal(events.length, 1);
assert.deepEqual([events[0].type, events[0].label, events[0].accepted], ["heard", null, false]);
assert.equal(events[0].distance, Infinity);

for (const [label, tones] of Object.entries(WORDS)) {
  listener.teachNext(label);
  assert.equal(listener.teaching, label);
  events = feed(listener, concat(toneWord(SR, tones), gap()), 960);
  assert.equal(events.length, 1, label);
  assert.deepEqual([events[0].type, events[0].label, events[0].count], ["taught", label, 1]);
  assert.equal(listener.teaching, undefined);
}
assert.deepEqual(listener.labels(), ["lights", "music", "stop"]);
assert.equal(listener.count("music"), 1);

for (const chunk of [128, 960, 4096]) {
  const s = concat(
    toneWord(SR, WORDS.music, 1.25),
    gap(),
    toneWord(SR, WORDS.lights, 0.85),
    gap(),
    toneWord(SR, WORDS.stop, 1.1),
    gap()
  );
  events = feed(listener, s, chunk);
  assert.deepEqual(
    events.map((e) => e.label),
    ["music", "lights", "stop"],
    `chunk ${chunk}`
  );
  assert.ok(events.every((e) => e.type === "heard" && e.accepted));
  assert.ok(events[0].startSeconds < events[1].startSeconds);
  assert.ok(events.every((e) => e.durationSeconds > 0.3 && e.durationSeconds < 1.5));
  assert.ok(events.every((e) => e.runnerUp && e.runnerUpDistance > e.distance));
}

// Unrecognised word rejected, click ignored:
events = feed(listener, concat(toneWord(SR, [3200, 250, 3200, 250]), gap()), 960);
assert.equal(events.length, 1);
assert.equal(events[0].accepted, false);
const click = noise(SR * 0.03, 0.5);
events = feed(listener, concat(click, gap()), 960);
assert.ok(events.every((e) => e.type === "ignored"));

// Saving, restoring, forgetting:
const saved = listener.exportCommands();
assert.ok(saved instanceof Uint8Array && saved.length > 1000);
const restored = new Listener(SR);
assert.equal(restored.importCommands(saved), true);
assert.deepEqual(restored.labels(), listener.labels());
const probe = concat(noise(SR * 0.2, 0.0005), toneWord(SR, WORDS.lights, 0.95), noise(SR * 0.2, 0.0005));
assert.equal(restored.recognizeClip(probe).label, "lights");
assert.equal(restored.importCommands(saved.subarray(0, saved.length - 5)), false);
assert.equal(new Listener(8000).importCommands(saved), false, "different analysis band");
restored.remove("lights");
assert.deepEqual(restored.labels(), ["music", "stop"]);
restored.clear();
assert.equal(restored.recognizeClip(probe), null);
assert.equal(listener.teachClip("silence", new Float32Array(SR)), false);

listener.maxDistance = 0.01;
assert.equal(listener.recognizeClip(probe).accepted, false);
listener.maxDistance = 4.6;
assert.equal(listener.recognizeClip(probe).accepted, true);

listener.free();
restored.free();
stt.free();

// --- 7. Real spoken digit recordings (optional, runs when data/fsdd is present) -----------
let fsddNote = "";
const fsdd = path.join(root, "data", "fsdd", "recordings");
if (fs.existsSync(fsdd)) {
  const names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
  let right = 0;
  let total = 0;
  for (const speaker of ["george", "jackson", "lucas", "nicolas", "theo", "yweweler"]) {
    const l = new Listener(8000);
    l.maxDistance = Infinity;
    l.maxRatio = Infinity;
    for (let d = 0; d < 10; d++) {
      for (let take = 0; take < 3; take++) {
        const wav = decodeWav(fs.readFileSync(path.join(fsdd, `${d}_${speaker}_${take}.wav`)));
        assert.ok(l.teachClip(names[d], wav.samples));
      }
    }
    for (let d = 0; d < 10; d++) {
      for (let take = 3; take < 50; take++) {
        const wav = decodeWav(fs.readFileSync(path.join(fsdd, `${d}_${speaker}_${take}.wav`)));
        const heard = l.recognizeClip(wav.samples);
        total += 1;
        right += heard && heard.label === names[d] ? 1 : 0;
      }
    }
    l.free();
  }
  const accuracy = (100 * right) / total;
  assert.ok(accuracy > 94, `FSDD accuracy ${accuracy}`);
  fsddNote = `, FSDD digits ${accuracy.toFixed(1)}% of ${total}`;
}

console.log(
  `ok: self-contained stt.wasm works (dictation ${rtf.toFixed(0)}x real time, hybrid commands+dictation ok${fsddNote})`
);
