# STT (`stt.computer`)

**STT** ([stt.computer](https://stt.computer)) is an English speech-to-text and taught voice-command engine written in pure Rust and compiled to a **single self-contained WebAssembly file (`stt.wasm`)**. Sister project to [**TTS** (`tts.computer`)](https://github.com/f2i-com/tts).

Microphone streams, `.wav` / `.mp3` / `.webm` files, or `Float32Array` PCM go in — English transcripts with word-level timestamps & confidences, or matched voice commands, come out. **Zero servers, zero external WASM imports, and zero runtime dependencies.**

- **Single-file self-contained `stt.wasm`** (~13.3 MiB raw, ~12.0 MiB gzipped) — **0 external WASM imports**, with the 13M-parameter quantized Conformer-CTC weights, DTW command matcher, and complete JavaScript `STT` SDK baked directly inside `stt.wasm` (grab it from [GitHub Releases](https://github.com/f2i-com/stt/releases))
- **3.71% Word Error Rate** on LibriSpeech `test-clean` (matching NVIDIA's published 3.7% figure), with **word-level start/end timestamps and CTC confidence scores**
- **~15–27× faster than real time** with 128-bit WASM SIMD; `stt.listen()` automatically runs inference inside an inline background Web Worker so the browser UI never drops a frame
- **Taught voice commands + Hybrid mode** (`stt.teach()`, `stt.teachNext()`, `mode: "both"`) — enroll custom spoken commands in 1–3 examples via Dynamic Time Warping (**96.4%** digit accuracy, **93.9%** consumer microphone command accuracy) with zero neural network, or combine live commands and dictation simultaneously
- **Seamless companion to [`TTS`](https://github.com/f2i-com/tts)** — `stt.transcribe()` accepts RIFF `.wav` bytes, `Float32Array`, Web Audio `AudioBuffer`, browser `File`/`Blob`, or `{ samples, sampleRate }` objects returned directly by `TTS`

---

## Quick Start: Embed `stt.wasm` Anywhere

### Option 1: Single `.wasm` File Only (Zero `.js` Files, Zero `.bin` Files!)

Download **`stt.wasm`** from [Releases](https://github.com/f2i-com/stt/releases) and drop it next to any HTML file. Because `stt.wasm` has **zero external imports**, embeds the speech model, and carries its own JavaScript SDK internally (`stt_glue_ptr` / `stt_glue_len`), 4 lines load the complete `STT` engine:

```html
<!doctype html>
<button id="mic">Start Dictation</button>
<p id="transcript"></p>

<script type="module">
  // Load ONLY stt.wasm — the model and complete JavaScript `STT` SDK live inside the WASM!
  async function loadSTT(url = "./stt.wasm") {
    const { module, instance: { exports: w } } = await WebAssembly.instantiateStreaming(fetch(url));
    return Function("wasm", "wasmModule", new TextDecoder().decode(
      new Uint8Array(w.memory.buffer, w.stt_glue_ptr(), w.stt_glue_len())
    ))(w, module);
  }

  const { STT } = await loadSTT("./stt.wasm");
  const stt = new STT();

  document.getElementById("mic").onclick = async () => {
    await stt.listen({
      onPhrase: (p) => { document.getElementById("transcript").textContent += p.text + " "; },
    });
  };
</script>
```

### Option 2: ES Module Loader (`stt.js` + `stt.wasm`)

If you use a bundler (Vite, Webpack, Rollup) or strict CSP (`script-src 'self'`), import `stt.js`:

```js
import init, { STT, decodeWav } from "./stt.js";

await init(); // fetches ./stt.wasm (or pass a custom URL / Uint8Array)

const stt = new STT();
const mic = await stt.listen({
  onPhrase: (phrase) => console.log(phrase.text, phrase.words),
});
```

### Option 3: Node.js / Bun / Deno

```js
import fs from "node:fs";
import { initSync, STT } from "./pkg/stt.js";

initSync(fs.readFileSync("./pkg/stt.wasm"));

const stt = new STT();
const result = stt.transcribe(fs.readFileSync("./speech.wav"));
console.log(result.text, result.words);
```

*(Or without `stt.js` in Node: `const { module, instance: { exports: w } } = await WebAssembly.instantiate(fs.readFileSync("./stt.wasm"));` and unpack `stt_glue_ptr` / `stt_glue_len` identically to Option 1.)*

---

## JavaScript & WebAssembly API

### 1. One-Shot Transcription & Word Timestamps

`stt.transcribe(input, sampleRate?)` accepts:
- `Uint8Array` or `ArrayBuffer` of any RIFF `.wav` file (8/16/24/32-bit PCM or 32-bit float, mono or stereo, any sample rate — automatically downmixed and resampled to 16 kHz)
- `Float32Array` mono PCM in `-1..1` at `sampleRate` (defaults to `stt.sampleRate`)
- Browser `AudioBuffer`
- `{ samples, sampleRate }` (including the `Speech` object returned directly by `TTS.speak()`)

```js
const stt = new STT();

// Transcribe WAV bytes, PCM Float32Array, AudioBuffer, or TTS output:
const res = stt.transcribe(wavBytes);
console.log(res.text);  // "and henry might return to england at any moment"
console.log(res.words); // [{ word: "and", start: 0.56, end: 0.64, confidence: 0.99 }, ...]

// Plain string helper:
const text = stt.transcribeText(pcmFloat32, 48000);

// Async browser/URL helper (handles File, Blob, URL, .wav, .mp3, .webm, .ogg, .m4a):
const fromFile = await stt.transcribeFile(fileInput.files[0]);
```

`stt.transcribe()` and `stt.transcribeFile()` return:

| Field | Type | Description |
|---|---|---|
| `text` | `string` | Recognized lower-case English text |
| `words` | `Array<{ word, start, end, confidence }>` | Word timestamps in seconds and CTC confidence (`0..1`) |
| `duration` | `number` | Clip duration in seconds |
| `confidence` | `number` | Mean word confidence (`0..1`) |

---

### 2. Live Browser Microphone (`stt.listen()`)

`await stt.listen(options)` opens the microphone via `AudioWorklet`, tracks the room's background noise floor in real time, splits speech cleanly at natural pauses, and automatically runs `STT` inside an **inline background Web Worker** (using the already-compiled `WebAssembly.Module` — zero extra network fetches) so transcription never blocks the main thread:

```js
const stt = new STT();

const session = await stt.listen({
  mode: "dictation", // "dictation" (default) | "commands" | "both"
  onPhrase: (p) => {
    console.log(p.text, p.words, `${p.durationSeconds.toFixed(1)}s in ${Math.round(p.tookMs)}ms`);
  },
  onSpeaking: (speaking) => console.log("speaking:", speaking),
  onLevel: ({ levelDb, floorDb, speaking }) => updateMeter(levelDb, floorDb, speaking),
});

// Control the live session at any time:
session.pause();
session.resume();
await session.flush(); // transcribe any phrase currently in progress
await session.stop();  // closes microphone tracks, AudioContext, and background Worker
```

---

### 3. Taught Voice Commands & Hybrid Mode (`"commands"` / `"both"`)

Teach custom spoken commands in 1–3 examples — matched by Dynamic Time Warping over normalized mel-cepstral + delta features with **zero neural network**. Use `mode: "commands"` for ultra-lightweight command control, or `mode: "both"` to trigger taught commands on short phrases while dictating everything else:

```js
const stt = new STT({ mode: "both" });

// Enroll from clips (WAV, Float32Array, AudioBuffer, or TTS output) or live via teachNext():
stt.teach("lights on", lightsOnWav);

const mic = await stt.listen({
  mode: "both",
  onCommand: (cmd) => {
    if (cmd.accepted) console.log("Command triggered:", cmd.label, cmd.distance);
  },
  onPhrase: (phrase) => {
    console.log("Dictated:", phrase.text);
  },
  onTaught: (ev) => {
    console.log(`Taught "${ev.label}" (${ev.count} examples)`);
  },
});

// Arm live microphone enrollment for the next spoken utterance:
await mic.teachNext("clear screen");

// Persist and restore taught commands (e.g. localStorage or IndexedDB):
const savedBytes = stt.exportCommands(); // Uint8Array
stt.importCommands(savedBytes);          // true
```

---

### 4. Low-Level Chunked Streaming (`push` / `flush`)

For custom audio pipelines (Node.js streams, WebRTC tracks, custom Workers, Electron):

```js
const stt = new STT({ sampleRate: 48000, mode: "dictation" });

for (const event of stt.push(float32Chunk)) {
  if (event.type === "phrase") {
    console.log(event.text, event.startSeconds, event.durationSeconds);
  }
}
for (const event of stt.flush()) {
  if (event.type === "phrase") console.log(event.text);
}
stt.free();
```

Dedicated single-purpose classes `Dictation` (`new Dictation(sampleRate, customModelBytes?)`) and `Listener` (`new Listener(sampleRate)`) are also exported alongside `STT`, plus `decodeWav(bytes)` and `encodeWav(samples, sampleRate)`.

---

### 5. Detection & Matching Controls

```js
new STT();                                        // 16 kHz, "dictation", embedded Conformer model
new STT(48000);                                   // 48 kHz input shorthand
new STT("commands");                              // zero-model DTW command mode
new STT({ sampleRate: 48000, mode: "both", hangMs: 500 });

stt.configure({ hangMs: 450, maxDistance: 4.2 }); // atomic validation & update
stt.settings;                                     // plain JSON-serializable settings object
```

| Option | Default | Effect |
|---|---|---|
| `sampleRate` | `16000` | Input PCM sample rate in Hz (`4000–384000`; auto-detected for `.wav` and `listen()`) |
| `mode` | `"dictation"` | `"dictation"`, `"commands"`, or `"both"` |
| `model` | *(embedded)* | Optional custom `STTW`/`VRCW` Conformer weight buffer (`Uint8Array`) |
| `hangMs` | `600` / `350` | Quiet duration in ms required to finish a phrase (`600` dictation, `350` commands) |
| `preRollMs` | `300` / `250` | Pre-onset audio kept in ms so soft initial consonants are preserved |
| `maxMs` | `20000` / `5000` | Maximum phrase duration in ms before automatic split |
| `softMaxMs` | `8000` | Phrase length in ms after which a brief pause (`softHangMs`) splits long speech |
| `softHangMs` | `150` | Brief pause in ms used once `softMaxMs` is reached |
| `onsetDb` | `15` | dB above the adaptive room noise floor required to start speech detection |
| `releaseDb` | `8` | dB above the room floor below which speech counts as paused |
| `dropDb` | `35` | dB drop below utterance peak that also counts as paused |
| `minLevelDb` | `-60` | Minimum block level in dBFS to count as speech |
| `maxDistance` | `4.6` | Maximum DTW distance accepted for taught commands (lower = stricter) |
| `maxRatio` | `0.92` | Maximum `distance / runnerUpDistance` ratio accepted for taught commands |

---

### 6. Raw C-ABI WebAssembly Exports (Non-JS Hosts)

Because `stt.wasm` has **zero host imports**, any WebAssembly runtime (`wasmtime`, `wazero`, `wasmer`, Rust, Go, Python, C#) can call its linear-memory exports directly without JavaScript:

- `stt_new(opts_ptr, opts_len, model_ptr, model_len) -> handle` (`(0, 0, 0, 0)` for default `STT` with embedded Conformer model)
- `stt_alloc(len) -> ptr` / `stt_alloc_f32(len) -> ptr`
- `stt_transcribe_wav(handle, wav_ptr, wav_len) -> 0` / `stt_transcribe_pcm(handle, f32_ptr, f32_len, sample_rate) -> 0`
- `stt_push(handle, f32_ptr, f32_len) -> event_count` / `stt_flush(handle) -> event_count`
- `stt_json_ptr() -> ptr`, `stt_json_len() -> len` (UTF-8 JSON transcript or events)
- `stt_drop(handle)`

---

## How It Works & Benchmarks

1. **Streaming Resampler & Adaptive Speech Segmenter** (`src/resample.rs`, `src/stream.rs`): Windowed-sinc anti-aliased resampling converts arbitrary microphone rates (8–384 kHz) to 16 kHz. Every 10 ms block updates a rolling 20th-percentile room noise floor, capturing 250–300 ms of pre-roll so soft plosives and fricatives are never clipped.
2. **Conformer-CTC Speech Recognition** (`src/conformer/`): Independent pure-Rust implementation of NVIDIA's 16-layer `stt_en_conformer_ctc_small` architecture (80-band log-mel frontend, 4× stride-2 2D conv subsampling, relative-positional multi-head self-attention, depthwise separable convolutions, and 1,024-piece CTC decoding with per-word timestamps & probabilities), verified to within `1e-4` log-probability against the reference ONNX export.
3. **Few-Shot Voice Commands** (`src/frontend.rs`, `src/features.rs`, `src/dtw.rs`, `src/commands.rs`): Utterance-normalized MFCC + regression delta extraction with Sakoe-Chiba banded Dynamic Time Warping and runner-up ratio rejection.

### Conformer Dictation Accuracy

| Benchmark | Word Error Rate (WER) |
|---|---|
| LibriSpeech `test-clean` (2,620 recordings, `f32` weights) | **3.67%** |
| LibriSpeech `test-clean` (2,620 recordings, shipped 13.7 MB `int8` weights) | **3.71%** |
| CMU ARCTIC (7 speakers × 200 sentences, `int8` weights) | **2.45–7.19%** |

Speed: **18–27× faster than real time** natively on a single desktop core, **15–26× faster than real time** in WebAssembly (Node.js / Chrome with 128-bit SIMD) — ~0.15–0.20 s for a 3.4-second sentence.

### Taught Voice Commands (Zero Neural Network)

| Benchmark | 1 example | 2 examples | 3 examples |
|---|---|---|---|
| Free Spoken Digit Dataset (10 digits, 6 speakers, 2,820 test clips) | 93.8% | — | **96.4%** |
| Google Speech Commands (~22 words, 400 speakers, consumer microphones) | 83.8% | 91.2% | **93.9%** |

---

## Build, Test & Run Demo Locally

Requires Rust (with `rustup target add wasm32-unknown-unknown`) and Node.js 18+:

```sh
npm run build      # builds self-contained pkg/stt.wasm + pkg/stt.js + pkg/stt.d.ts
npm test           # runs Rust release unit tests + Node WASM smoke tests
npm run demo       # starts local studio at http://localhost:8081/demo/

# Native CLI example:
cargo run --release --example transcribe -- --words tests/data/henry.wav
cargo run --release --example transcribe -- --json tests/data/henry.wav
```

*(Optional lightweight build without the embedded 13.7 MB model: `cargo build --release --no-default-features` compiles the ~80 KB core + DTW command engine where custom weights can still be passed at runtime.)*

---

## Third-Party Data & Attribution

- `models/conformer_small_int8.bin` (embedded in `stt.wasm` by default) is a quantized flat conversion of **NVIDIA's `stt_en_conformer_ctc_small`**, licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (see [`models/README.md`](./models/README.md)).
- `tests/data/henry.wav` is utterance `61-70968-0048` from LibriSpeech `test-clean` (© Vassil Panayotov, [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); see [`tests/data/README.md`](./tests/data/README.md)).
- All Rust DSP, Conformer inference, DTW command recognition, and WebAssembly code is original to this project with zero runtime dependencies.
