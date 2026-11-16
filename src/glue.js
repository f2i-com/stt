// Embedded JavaScript SDK for STT (stt.computer).
// Compiled directly inside `stt.wasm` (`stt_glue_ptr` / `stt_glue_len`) so a single `.wasm`
// file contains its own complete JS bindings, Conformer speech model, DTW command matcher,
// and inline Web Worker + AudioWorklet microphone pipeline with zero external dependencies.
//
// Evaluated via `Function("wasm", "wasmModule", code)(instance.exports, module)`
// -> `{ STT, Dictation, Listener, decodeWav, encodeWav }`.

const enc = new TextEncoder();
const dec = new TextDecoder();

function bindSTT(w, wasmMod = null) {
  if (!w || !w.memory || typeof w.stt_new !== "function") {
    throw new Error("Invalid STT WebAssembly exports object");
  }

  const writeUtf8 = (str) => {
    const bytes = enc.encode(String(str));
    if (bytes.length === 0) return [0, 0];
    const ptr = w.stt_alloc(bytes.length);
    new Uint8Array(w.memory.buffer, ptr, bytes.length).set(bytes);
    return [ptr, bytes.length];
  };

  const writeBytes = (u8) => {
    if (!u8 || u8.length === 0) return [0, 0];
    const ptr = w.stt_alloc(u8.length);
    new Uint8Array(w.memory.buffer, ptr, u8.length).set(u8);
    return [ptr, u8.length];
  };

  const writeAuxBytes = (u8, labelStr = null) => {
    let lPtr = 0;
    let lLen = 0;
    if (labelStr !== null) {
      [lPtr, lLen] = writeUtf8(labelStr);
    }
    if (!u8 || u8.length === 0) return [lPtr, lLen, 0, 0];
    const ptr = w.stt_alloc_aux(u8.length);
    if (labelStr !== null) {
      // Re-write label in case stt_alloc_aux grew WASM linear memory
      [lPtr, lLen] = writeUtf8(labelStr);
    }
    new Uint8Array(w.memory.buffer, ptr, u8.length).set(u8);
    return [lPtr, lLen, ptr, u8.length];
  };

  const writeF32 = (arr) => {
    const f32 = arr instanceof Float32Array ? arr : Float32Array.from(arr || []);
    if (f32.length === 0) return [0, 0];
    const byteOffset = w.stt_alloc_f32(f32.length);
    new Float32Array(w.memory.buffer, byteOffset, f32.length).set(f32);
    return [byteOffset, f32.length];
  };

  const readJsonText = () => {
    const len = w.stt_json_len();
    if (!len) return "";
    const ptr = w.stt_json_ptr();
    return dec.decode(new Uint8Array(w.memory.buffer, ptr, len));
  };

  const readJson = () =>
    JSON.parse(readJsonText(), (_k, v) =>
      v === "__STT_POS_INF__" ? Infinity : v === "__STT_NEG_INF__" ? -Infinity : v
    );

  const copySamples = () => {
    const len = w.stt_samples_len();
    if (!len) return new Float32Array(0);
    const ptr = w.stt_samples_ptr();
    return new Float32Array(w.memory.buffer, ptr, len).slice();
  };

  const copyBytes = () => {
    const len = w.stt_bytes_len();
    if (!len) return new Uint8Array(0);
    const ptr = w.stt_bytes_ptr();
    return new Uint8Array(w.memory.buffer, ptr, len).slice();
  };

  const serializePayload = (val) =>
    JSON.stringify(val, (_k, v) => {
      if (typeof v === "number") {
        if (Number.isNaN(v)) return "__STT_NAN__";
        if (v === Infinity) return "__STT_POS_INF__";
        if (v === -Infinity) return "__STT_NEG_INF__";
      }
      return v;
    });

  const toUint8Array = (val) => {
    if (val instanceof Uint8Array) return val;
    if (val instanceof ArrayBuffer) return new Uint8Array(val);
    if (ArrayBuffer.isView(val)) return new Uint8Array(val.buffer, val.byteOffset, val.byteLength);
    return null;
  };

  const isRiffWav = (u8) =>
    u8 &&
    u8.length >= 12 &&
    u8[0] === 0x52 && // R
    u8[1] === 0x49 && // I
    u8[2] === 0x46 && // F
    u8[3] === 0x46 && // F
    u8[8] === 0x57 && // W
    u8[9] === 0x41 && // A
    u8[10] === 0x56 && // V
    u8[11] === 0x45; // E

  const unpackAudioInput = (input, explicitRate, defaultRate) => {
    if (input === null || input === undefined) {
      return { kind: "pcm", samples: new Float32Array(0), sampleRate: explicitRate || defaultRate };
    }
    if (input instanceof Float32Array) {
      return { kind: "pcm", samples: input, sampleRate: explicitRate || defaultRate };
    }
    if (input instanceof Float64Array || Array.isArray(input)) {
      return { kind: "pcm", samples: Float32Array.from(input), sampleRate: explicitRate || defaultRate };
    }
    // Web Audio API AudioBuffer
    if (
      typeof input === "object" &&
      typeof input.getChannelData === "function" &&
      typeof input.numberOfChannels === "number" &&
      typeof input.sampleRate === "number"
    ) {
      const ch = Math.max(1, input.numberOfChannels);
      let mono;
      if (ch === 1) {
        mono = input.getChannelData(0);
      } else {
        mono = new Float32Array(input.length);
        for (let c = 0; c < ch; c++) {
          const d = input.getChannelData(c);
          for (let i = 0; i < mono.length; i++) mono[i] += d[i] / ch;
        }
      }
      return { kind: "pcm", samples: mono, sampleRate: explicitRate || Math.round(input.sampleRate) };
    }
    // `{ samples, sampleRate }` object (e.g. TTS `Speech` output or `decodeWav` output)
    if (typeof input === "object" && "samples" in input && input.samples != null) {
      const s =
        input.samples instanceof Float32Array
          ? input.samples
          : Float32Array.from(input.samples);
      return {
        kind: "pcm",
        samples: s,
        sampleRate: explicitRate || Math.round(Number(input.sampleRate) || defaultRate),
      };
    }
    // Uint8Array / ArrayBuffer (RIFF WAV file bytes)
    const u8 = toUint8Array(input);
    if (u8) {
      return { kind: "wav", bytes: u8 };
    }
    throw new Error(
      "Audio input must be a Float32Array, WAV Uint8Array/ArrayBuffer, AudioBuffer, or { samples, sampleRate }"
    );
  };

  const fixEventNumbers = (ev) => {
    if (ev && ev.type === "heard") {
      if (ev.distance === null) ev.distance = Infinity;
      if (ev.runnerUpDistance === null) ev.runnerUpDistance = Infinity;
    }
    return ev;
  };

  w.stt_meta_json();
  const META = readJson();

  const instRegistry =
    typeof FinalizationRegistry !== "undefined"
      ? new FinalizationRegistry((h) => {
          if (h > 0) w.stt_drop(h);
        })
      : null;

  function decodeWav(wavBytes) {
    const u8 = toUint8Array(wavBytes);
    if (!u8) throw new Error("decodeWav expects a Uint8Array or ArrayBuffer");
    const [ptr, len] = writeBytes(u8);
    if (w.stt_decode_wav(ptr, len) < 0) throw new Error(readJsonText());
    const meta = readJson();
    return {
      samples: copySamples(),
      sampleRate: meta.sampleRate,
      duration: meta.duration,
    };
  }

  function encodeWav(samples, sampleRate = META.modelSampleRate) {
    const [ptr, len] = writeF32(samples);
    const sr = Math.max(1000, Math.floor(Number(sampleRate) || META.modelSampleRate));
    w.stt_encode_wav(ptr, len, sr);
    return copyBytes();
  }

  // Inline Web Worker source used by `stt.listen()` so Conformer inference runs off the
  // browser's main UI thread with zero external `.js` or `.wasm` network requests.
  const WORKER_SOURCE = `
    let stt = null;
    let lastSpeaking = false;
    self.onmessage = async (ev) => {
      const m = ev.data;
      try {
        if (m.type === "init") {
          const inst = await WebAssembly.instantiate(m.wasmModule, {});
          const w = inst.exports;
          const glue = new TextDecoder().decode(
            new Uint8Array(w.memory.buffer, w.stt_glue_ptr(), w.stt_glue_len())
          );
          const { STT } = Function("wasm", "wasmModule", glue)(w, m.wasmModule);
          const initOpts = m.model
            ? { ...m.settings, model: new Uint8Array(m.model) }
            : m.settings;
          stt = new STT(initOpts);
          if (m.commands && m.commands.byteLength) {
            stt.importCommands(new Uint8Array(m.commands));
          }
          if (stt.mode !== "commands") stt.warmup();
          self.postMessage({ type: "ready", id: m.id });
        } else if (m.type === "audio" && stt) {
          const t0 = performance.now();
          const events = stt.push(m.samples);
          const tookMs = performance.now() - t0;
          const speaking = stt.speaking;
          const levelDb = stt.levelDb;
          const floorDb = stt.floorDb;
          if (events.length) {
            for (const e of events) e.tookMs = tookMs;
            let syncedCommands = null;
            if (events.some((e) => e.type === "taught")) {
              syncedCommands = stt.exportCommands();
            }
            self.postMessage(
              { type: "events", events, syncedCommands },
              syncedCommands ? [syncedCommands.buffer] : []
            );
          }
          if (speaking !== lastSpeaking) {
            lastSpeaking = speaking;
            self.postMessage({ type: "speaking", speaking });
          }
          self.postMessage({ type: "level", levelDb, floorDb, speaking });
        } else if (m.type === "configure" && stt) {
          const s = stt.configure(m.options);
          self.postMessage({ type: "reply", id: m.id, value: s });
        } else if (m.type === "teachNext" && stt) {
          if (m.label) stt.teachNext(m.label);
          else stt.cancelTeach();
          self.postMessage({ type: "reply", id: m.id, value: stt.teaching });
        } else if (m.type === "removeCommand" && stt) {
          stt.removeCommand(m.label);
          const bytes = stt.exportCommands();
          self.postMessage({ type: "reply", id: m.id, value: bytes }, [bytes.buffer]);
        } else if (m.type === "clearCommands" && stt) {
          stt.clearCommands();
          self.postMessage({ type: "reply", id: m.id, value: true });
        } else if (m.type === "importCommands" && stt) {
          const ok = stt.importCommands(new Uint8Array(m.bytes));
          self.postMessage({ type: "reply", id: m.id, value: ok });
        } else if (m.type === "flush" && stt) {
          const t0 = performance.now();
          const events = stt.flush();
          const tookMs = performance.now() - t0;
          for (const e of events) e.tookMs = tookMs;
          if (events.length) self.postMessage({ type: "events", events });
          self.postMessage({ type: "reply", id: m.id, value: events });
        } else if (m.type === "stop") {
          let bytes = null;
          if (stt) {
            bytes = stt.exportCommands();
            stt.free();
            stt = null;
          }
          self.postMessage({ type: "reply", id: m.id, value: bytes }, bytes ? [bytes.buffer] : []);
        }
      } catch (err) {
        self.postMessage({ type: "error", id: m.id, message: String((err && err.message) || err) });
      }
    };
  `;

  const WORKLET_SOURCE = `
    class SttTapProcessor extends AudioWorkletProcessor {
      constructor() {
        super();
        this.buffer = new Float32Array(1024);
        this.fill = 0;
      }
      process(inputs) {
        const input = inputs[0];
        const ch = input && input[0];
        if (ch) {
          for (let i = 0; i < ch.length; i++) {
            this.buffer[this.fill++] = ch[i];
            if (this.fill === this.buffer.length) {
              const out = this.buffer.slice();
              this.port.postMessage(out, [out.buffer]);
              this.fill = 0;
            }
          }
        }
        return true;
      }
    }
    registerProcessor("stt-tap", SttTapProcessor);
  `;

  class STT {
    #handle;
    #customModelBytes = null;

    constructor(options) {
      let optsPayload = options;
      let modelU8 = null;

      if (options && typeof options === "object" && !Array.isArray(options) && !(options instanceof Uint8Array)) {
        if ("model" in options && options.model !== undefined && options.model !== null) {
          modelU8 = toUint8Array(options.model);
          if (!modelU8) {
            throw new Error("model option must be a Uint8Array or ArrayBuffer");
          }
          this.#customModelBytes = modelU8.slice();
        }
        const rest = { ...options };
        delete rest.model;
        optsPayload = rest;
      }

      let oPtr = 0;
      let oLen = 0;
      let mPtr = 0;
      let mLen = 0;

      if (modelU8) {
        const optsStr = optsPayload !== undefined && optsPayload !== null ? serializePayload(optsPayload) : null;
        [oPtr, oLen, mPtr, mLen] = writeAuxBytes(modelU8, optsStr);
        if (mLen === 0) mPtr = 1; // empty Uint8Array should still hit model validation
      } else if (optsPayload !== undefined && optsPayload !== null) {
        [oPtr, oLen] = writeUtf8(serializePayload(optsPayload));
      }

      const h = w.stt_new(oPtr, oLen, mPtr, mLen);
      if (h < 0) throw new Error(readJsonText());
      this.#handle = h;
      if (instRegistry) instRegistry.register(this, h, this);
    }

    #checkHandle() {
      if (!this.#handle) throw new Error("STT instance has been freed");
      return this.#handle;
    }

    static modes() {
      return META.modes.slice();
    }

    static defaults(mode = "dictation") {
      const key = String(mode || "dictation").toLowerCase();
      return structuredClone(META.defaults[key] || META.defaults.dictation);
    }

    static hasEmbeddedModel() {
      return Boolean(META.hasEmbeddedModel);
    }

    static modelSampleRate() {
      return META.modelSampleRate;
    }

    warmup() {
      const h = this.#checkHandle();
      if (w.stt_warmup(h) < 0) throw new Error(readJsonText());
    }

    configure(options) {
      const h = this.#checkHandle();
      if (options === undefined || options === null) return this.settings;
      const [ptr, len] = writeUtf8(serializePayload(options));
      if (w.stt_configure(h, ptr, len) < 0) throw new Error(readJsonText());
      return readJson();
    }

    reset() {
      const h = this.#checkHandle();
      if (w.stt_reset(h) < 0) throw new Error(readJsonText());
    }

    get settings() {
      const h = this.#checkHandle();
      if (w.stt_get_settings(h) < 0) throw new Error(readJsonText());
      return readJson();
    }

    get sampleRate() {
      return this.settings.sampleRate;
    }

    set sampleRate(sr) {
      this.configure({ sampleRate: sr });
    }

    setSampleRate(sr) {
      return this.configure({ sampleRate: sr });
    }

    get mode() {
      return this.settings.mode;
    }

    set mode(mode) {
      this.configure({ mode });
    }

    setMode(mode) {
      return this.configure({ mode });
    }

    get speaking() {
      const h = this.#checkHandle();
      return w.stt_speaking(h) !== 0;
    }

    get levelDb() {
      const h = this.#checkHandle();
      return w.stt_level_db(h);
    }

    get floorDb() {
      const h = this.#checkHandle();
      return w.stt_floor_db(h);
    }

    /**
     * Transcribes audio (`Float32Array`, RIFF `.wav` `Uint8Array`/`ArrayBuffer`, Web Audio
     * `AudioBuffer`, or `{ samples, sampleRate }`) into `{ text, words, duration, confidence }`.
     */
    transcribe(input, sampleRate) {
      const h = this.#checkHandle();
      const unpacked = unpackAudioInput(input, sampleRate, 0);
      if (unpacked.kind === "wav") {
        const [ptr, len] = writeBytes(unpacked.bytes);
        if (w.stt_transcribe_wav(h, ptr, len) < 0) throw new Error(readJsonText());
        return readJson();
      }
      const [ptr, len] = writeF32(unpacked.samples);
      const sr = unpacked.sampleRate ? Math.max(4000, Math.round(Number(unpacked.sampleRate))) : 0;
      if (w.stt_transcribe_pcm(h, ptr, len, sr) < 0) throw new Error(readJsonText());
      return readJson();
    }

    /** Convenience wrapper returning just the transcribed `string`. */
    transcribeText(input, sampleRate) {
      return this.transcribe(input, sampleRate).text;
    }

    /** Transcribes a RIFF `.wav` buffer (`Uint8Array` or `ArrayBuffer`). */
    transcribeWav(wavBytes) {
      const h = this.#checkHandle();
      const u8 = toUint8Array(wavBytes);
      if (!u8) throw new Error("transcribeWav expects a Uint8Array or ArrayBuffer");
      const [ptr, len] = writeBytes(u8);
      if (w.stt_transcribe_wav(h, ptr, len) < 0) throw new Error(readJsonText());
      return readJson();
    }

    /**
     * Async helper for browsers & runtimes: accepts a `File`, `Blob`, URL `string` / `URL`,
     * `Response`, `ArrayBuffer`, or `Uint8Array`. Decodes `.wav` natively in WASM and falls
     * back to Web Audio `decodeAudioData` in browsers for `.mp3`, `.webm`, `.ogg`, `.m4a`, `.flac`.
     */
    async transcribeFile(source) {
      this.#checkHandle();
      let ab;
      if (source instanceof ArrayBuffer) {
        ab = source;
      } else if (ArrayBuffer.isView(source)) {
        ab = source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
      } else if (typeof Blob !== "undefined" && source instanceof Blob) {
        ab = await source.arrayBuffer();
      } else if (typeof Response !== "undefined" && source instanceof Response) {
        ab = await source.arrayBuffer();
      } else if (typeof source === "string" || source instanceof URL) {
        const resp = await fetch(source);
        if (!resp.ok) throw new Error(`Failed to fetch audio (${resp.status})`);
        ab = await resp.arrayBuffer();
      } else {
        return this.transcribe(source);
      }

      const u8 = new Uint8Array(ab);
      if (isRiffWav(u8)) {
        return this.transcribeWav(u8);
      }
      const Ctx = globalThis.OfflineAudioContext || globalThis.AudioContext || globalThis.webkitAudioContext;
      if (Ctx) {
        const ctx =
          Ctx === globalThis.OfflineAudioContext
            ? new Ctx(1, 16000, 16000)
            : new Ctx();
        try {
          const audioBuf = await ctx.decodeAudioData(ab.slice(0));
          return this.transcribe(audioBuf);
        } finally {
          if (typeof ctx.close === "function" && !(ctx instanceof (globalThis.OfflineAudioContext || Function))) {
            try { await ctx.close(); } catch {}
          }
        }
      }
      throw new Error("Unsupported audio file format (expected RIFF WAV, or run in a browser with Web Audio API)");
    }

    /** Feeds mono PCM chunk (`Float32Array` in `-1..1` at `this.sampleRate`) and returns completed events. */
    push(samples) {
      const h = this.#checkHandle();
      const [ptr, len] = writeF32(samples);
      if (w.stt_push(h, ptr, len) < 0) throw new Error(readJsonText());
      return readJson().map(fixEventNumbers);
    }

    /** Finishes any utterance currently in progress at the end of a stream. */
    flush() {
      const h = this.#checkHandle();
      if (w.stt_flush(h) < 0) throw new Error(readJsonText());
      return readJson().map(fixEventNumbers);
    }

    // ---- Taught voice commands -----------------------------------------------------------

    teachNext(label) {
      const h = this.#checkHandle();
      const [ptr, len] = writeUtf8(String(label ?? ""));
      if (w.stt_teach_next(h, ptr, len) < 0) throw new Error(readJsonText());
    }

    cancelTeach() {
      const h = this.#checkHandle();
      if (w.stt_teach_next(h, 0, 0) < 0) throw new Error(readJsonText());
    }

    get teaching() {
      const h = this.#checkHandle();
      const r = w.stt_teaching(h);
      if (r < 0) throw new Error(readJsonText());
      return r > 0 ? readJsonText() : undefined;
    }

    teach(label, input, sampleRate) {
      const h = this.#checkHandle();
      const unpacked = unpackAudioInput(input, sampleRate, 0);
      if (unpacked.kind === "wav") {
        const [lPtr, lLen, wPtr, wLen] = writeAuxBytes(unpacked.bytes, String(label));
        const r = w.stt_teach_wav(h, lPtr, lLen, wPtr, wLen);
        if (r < 0) throw new Error(readJsonText());
        return r > 0;
      }
      const [fPtr, fLen] = writeF32(unpacked.samples);
      const [lPtr, lLen] = writeUtf8(String(label));
      const sr = unpacked.sampleRate ? Math.max(4000, Math.round(Number(unpacked.sampleRate))) : 0;
      const r = w.stt_teach_pcm(h, lPtr, lLen, fPtr, fLen, sr);
      if (r < 0) throw new Error(readJsonText());
      return r > 0;
    }

    teachClip(label, samples, sampleRate) {
      return this.teach(label, samples, sampleRate);
    }

    recognizeCommand(input, sampleRate) {
      const h = this.#checkHandle();
      const unpacked = unpackAudioInput(input, sampleRate, 0);
      let r;
      if (unpacked.kind === "wav") {
        const [ptr, len] = writeBytes(unpacked.bytes);
        r = w.stt_recognize_wav(h, ptr, len);
      } else {
        const [ptr, len] = writeF32(unpacked.samples);
        const sr = unpacked.sampleRate ? Math.max(4000, Math.round(Number(unpacked.sampleRate))) : 0;
        r = w.stt_recognize_pcm(h, ptr, len, sr);
      }
      if (r < 0) throw new Error(readJsonText());
      if (r === 0) return null;
      return fixEventNumbers(readJson());
    }

    recognizeClip(samples, sampleRate) {
      return this.recognizeCommand(samples, sampleRate);
    }

    commands() {
      const h = this.#checkHandle();
      if (w.stt_commands_json(h) < 0) throw new Error(readJsonText());
      return readJson();
    }

    labels() {
      return this.commands().map((c) => c.label);
    }

    count(label) {
      const entry = this.commands().find((c) => c.label === label);
      return entry ? entry.count : 0;
    }

    commandCount(label) {
      return this.count(label);
    }

    removeCommand(label) {
      const h = this.#checkHandle();
      const [ptr, len] = writeUtf8(String(label));
      if (w.stt_remove_command(h, ptr, len) < 0) throw new Error(readJsonText());
    }

    remove(label) {
      this.removeCommand(label);
    }

    clearCommands() {
      const h = this.#checkHandle();
      if (w.stt_clear_commands(h) < 0) throw new Error(readJsonText());
    }

    clear() {
      this.clearCommands();
    }

    exportCommands() {
      const h = this.#checkHandle();
      if (w.stt_export_commands(h) < 0) throw new Error(readJsonText());
      return copyBytes();
    }

    importCommands(bytes) {
      const h = this.#checkHandle();
      const u8 = toUint8Array(bytes);
      if (!u8) return false;
      const [ptr, len] = writeBytes(u8);
      const r = w.stt_import_commands(h, ptr, len);
      if (r < 0) throw new Error(readJsonText());
      return r > 0;
    }

    /**
     * Opens the browser microphone and streams live speech into `STT` (running inside an
     * automatic inline Web Worker when available so Conformer inference never blocks the UI).
     * Returns a `ListenSession` controller (`{ stop(), flush(), pause(), resume(), teachNext(), ... }`).
     */
    async listen(options = {}) {
      const self = this;
      this.#checkHandle();
      if (!globalThis.navigator?.mediaDevices?.getUserMedia && !options.stream) {
        throw new Error("Microphone capture (navigator.mediaDevices.getUserMedia) is not available in this environment");
      }

      const ownStream = !options.stream;
      const mediaStream =
        options.stream ||
        (await globalThis.navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            ...(options.audioConstraints || {}),
          },
        }));

      const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
      if (!Ctx && !options.audioContext) {
        if (ownStream) mediaStream.getTracks().forEach((t) => t.stop());
        throw new Error("Web Audio API (AudioContext) is not available in this environment");
      }
      const ownContext = !options.audioContext;
      const ctx = options.audioContext || new Ctx();
      if (ctx.state === "suspended") await ctx.resume();

      const sessionSettings = {
        ...this.settings,
        sampleRate: Math.round(ctx.sampleRate),
      };
      if (options.mode) sessionSettings.mode = options.mode;
      this.configure(sessionSettings);

      const dispatchEvents = (events) => {
        for (const ev of events) {
          fixEventNumbers(ev);
          if (typeof options.onEvent === "function") options.onEvent(ev);
          if (ev.type === "phrase" && typeof options.onPhrase === "function") options.onPhrase(ev);
          else if (ev.type === "heard" && typeof options.onCommand === "function") options.onCommand(ev);
          else if (ev.type === "taught" && typeof options.onTaught === "function") options.onTaught(ev);
          else if (ev.type === "ignored" && typeof options.onIgnored === "function") options.onIgnored(ev);
        }
      };

      // Try spawning an inline Web Worker when `wasmMod` is available so transcription stays 100% off-main-thread.
      let worker = null;
      let rpcSeq = 0;
      const pendingRpc = new Map();

      const callWorker = (payload, transfer = []) =>
        new Promise((resolve, reject) => {
          const id = ++rpcSeq;
          pendingRpc.set(id, { resolve, reject });
          worker.postMessage({ ...payload, id }, transfer);
        });

      const wantWorker = options.worker !== false && wasmMod instanceof WebAssembly.Module && typeof Worker === "function";
      if (wantWorker) {
        try {
          const blobUrl = URL.createObjectURL(new Blob([WORKER_SOURCE], { type: "text/javascript" }));
          worker = new Worker(blobUrl);
          URL.revokeObjectURL(blobUrl);

          await new Promise((resolve, reject) => {
            const initId = ++rpcSeq;
            pendingRpc.set(initId, { resolve, reject });

            worker.onmessage = (msg) => {
              const d = msg.data;
              if (d.type === "ready" || d.type === "reply") {
                const entry = pendingRpc.get(d.id);
                if (entry) {
                  pendingRpc.delete(d.id);
                  entry.resolve(d.value);
                }
              } else if (d.type === "events") {
                if (d.syncedCommands) {
                  this.importCommands(new Uint8Array(d.syncedCommands));
                }
                dispatchEvents(d.events);
              } else if (d.type === "speaking") {
                if (typeof options.onSpeaking === "function") options.onSpeaking(d.speaking);
              } else if (d.type === "level") {
                if (typeof options.onLevel === "function") {
                  options.onLevel({ levelDb: d.levelDb, floorDb: d.floorDb, speaking: d.speaking });
                }
              } else if (d.type === "error") {
                const err = new Error(d.message);
                if (d.id && pendingRpc.has(d.id)) {
                  const entry = pendingRpc.get(d.id);
                  pendingRpc.delete(d.id);
                  entry.reject(err);
                } else if (typeof options.onError === "function") {
                  options.onError(err);
                }
              }
            };

            worker.onerror = (ev) => {
              reject(new Error(ev.message || "STT Web Worker failed to start"));
            };

            const cmds = this.exportCommands();
            const customModel = this.#customModelBytes ? this.#customModelBytes.slice() : null;
            const transfers = [cmds.buffer];
            if (customModel) transfers.push(customModel.buffer);
            worker.postMessage(
              {
                type: "init",
                id: initId,
                wasmModule: wasmMod,
                settings: this.settings,
                commands: cmds.buffer,
                model: customModel ? customModel.buffer : null,
              },
              transfers
            );
          });
        } catch {
          if (worker) {
            try { worker.terminate(); } catch {}
            worker = null;
          }
          pendingRpc.clear();
        }
      }

      if (!worker && this.mode !== "commands") {
        this.warmup();
      }

      let paused = false;
      let stopped = false;
      let lastMainSpeaking = false;

      const onSamples = (samples) => {
        if (paused || stopped) return;
        if (worker) {
          const copy = samples.slice();
          worker.postMessage({ type: "audio", samples: copy }, [copy.buffer]);
        } else {
          try {
            const t0 = typeof performance !== "undefined" ? performance.now() : 0;
            const events = this.push(samples);
            const tookMs = (typeof performance !== "undefined" ? performance.now() : 0) - t0;
            for (const e of events) e.tookMs = tookMs;
            if (events.length) dispatchEvents(events);
            const sp = this.speaking;
            if (sp !== lastMainSpeaking) {
              lastMainSpeaking = sp;
              if (typeof options.onSpeaking === "function") options.onSpeaking(sp);
            }
            if (typeof options.onLevel === "function") {
              options.onLevel({ levelDb: this.levelDb, floorDb: this.floorDb, speaking: sp });
            }
          } catch (err) {
            if (typeof options.onError === "function") options.onError(err);
          }
        }
      };

      const sourceNode = ctx.createMediaStreamSource(mediaStream);
      let tapNode = null;
      if (ctx.audioWorklet && typeof AudioWorkletNode !== "undefined") {
        const tapUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "text/javascript" }));
        try {
          await ctx.audioWorklet.addModule(tapUrl);
        } finally {
          URL.revokeObjectURL(tapUrl);
        }
        tapNode = new AudioWorkletNode(ctx, "stt-tap", { numberOfInputs: 1, numberOfOutputs: 0 });
        tapNode.port.onmessage = (m) => onSamples(m.data);
        sourceNode.connect(tapNode);
      } else {
        tapNode = ctx.createScriptProcessor(1024, 1, 1);
        tapNode.onaudioprocess = (e) => onSamples(e.inputBuffer.getChannelData(0).slice());
        const mute = ctx.createGain();
        mute.gain.value = 0;
        sourceNode.connect(tapNode);
        tapNode.connect(mute);
        mute.connect(ctx.destination);
      }

      return {
        sampleRate: Math.round(ctx.sampleRate),
        usingWorker: Boolean(worker),
        get paused() { return paused; },
        get active() { return !stopped; },
        pause() { paused = true; },
        resume() { paused = false; },
        async configure(opts) {
          const next = self.configure(opts);
          if (worker) return await callWorker({ type: "configure", options: opts });
          return next;
        },
        async teachNext(label) {
          self.teachNext(label);
          if (worker) await callWorker({ type: "teachNext", label: label ? String(label) : "" });
        },
        async cancelTeach() {
          self.cancelTeach();
          if (worker) await callWorker({ type: "teachNext", label: "" });
        },
        async removeCommand(label) {
          self.removeCommand(label);
          if (worker) {
            const bytes = await callWorker({ type: "removeCommand", label: String(label) });
            if (bytes) self.importCommands(new Uint8Array(bytes));
          }
        },
        async clearCommands() {
          self.clearCommands();
          if (worker) await callWorker({ type: "clearCommands" });
        },
        async importCommands(bytes) {
          const ok = self.importCommands(bytes);
          if (worker && ok) {
            const u8 = toUint8Array(bytes).slice();
            await callWorker({ type: "importCommands", bytes: u8.buffer }, [u8.buffer]);
          }
          return ok;
        },
        async flush() {
          if (stopped) return [];
          if (worker) return await callWorker({ type: "flush" });
          const events = self.flush();
          if (events.length) dispatchEvents(events);
          return events;
        },
        async stop() {
          if (stopped) return;
          stopped = true;
          try { sourceNode.disconnect(); } catch {}
          try { if (tapNode) tapNode.disconnect(); } catch {}
          if (ownStream) {
            for (const track of mediaStream.getTracks()) track.stop();
          }
          if (ownContext && typeof ctx.close === "function") {
            try { await ctx.close(); } catch {}
          }
          if (worker) {
            try {
              await callWorker({ type: "flush" });
              const synced = await callWorker({ type: "stop" });
              if (synced) self.importCommands(new Uint8Array(synced));
            } catch {}
            worker.terminate();
            worker = null;
          } else {
            const rem = self.flush();
            if (rem.length) dispatchEvents(rem);
          }
        },
      };
    }

    free() {
      if (this.#handle) {
        if (instRegistry) instRegistry.unregister(this);
        w.stt_drop(this.#handle);
        this.#handle = 0;
      }
    }
  }

  class Dictation {
    #stt;
    constructor(sampleRate = 16000, model) {
      const opts = { sampleRate, mode: "dictation" };
      if (model !== undefined) opts.model = model;
      this.#stt = new STT(opts);
      this.#stt.warmup();
    }
    push(samples) {
      return this.#stt.push(samples).filter((e) => e.type === "phrase");
    }
    flush() {
      return this.#stt.flush().filter((e) => e.type === "phrase");
    }
    transcribeClip(samples) {
      return this.#stt.transcribeText(samples, this.#stt.sampleRate);
    }
    transcribe(input, sampleRate) {
      return this.#stt.transcribe(input, sampleRate);
    }
    get levelDb() { return this.#stt.levelDb; }
    get floorDb() { return this.#stt.floorDb; }
    get speaking() { return this.#stt.speaking; }
    get sampleRate() { return this.#stt.sampleRate; }
    free() { this.#stt.free(); }
  }

  class Listener {
    #stt;
    constructor(sampleRate = 48000) {
      this.#stt = new STT({ sampleRate, mode: "commands" });
    }
    push(samples) { return this.#stt.push(samples); }
    flush() { return this.#stt.flush(); }
    teachNext(label) { this.#stt.teachNext(label); }
    cancelTeach() { this.#stt.cancelTeach(); }
    get teaching() { return this.#stt.teaching; }
    teachClip(label, samples) { return this.#stt.teach(label, samples, this.#stt.sampleRate); }
    recognizeClip(samples) { return this.#stt.recognizeCommand(samples, this.#stt.sampleRate); }
    remove(label) { this.#stt.removeCommand(label); }
    clear() { this.#stt.clearCommands(); }
    labels() { return this.#stt.labels(); }
    count(label) { return this.#stt.count(label); }
    exportCommands() { return this.#stt.exportCommands(); }
    importCommands(bytes) { return this.#stt.importCommands(bytes); }
    get levelDb() { return this.#stt.levelDb; }
    get floorDb() { return this.#stt.floorDb; }
    get speaking() { return this.#stt.speaking; }
    get sampleRate() { return this.#stt.sampleRate; }
    get maxDistance() { return this.#stt.settings.maxDistance; }
    set maxDistance(value) { this.#stt.configure({ maxDistance: value }); }
    get maxRatio() { return this.#stt.settings.maxRatio; }
    set maxRatio(value) { this.#stt.configure({ maxRatio: value }); }
    free() { this.#stt.free(); }
  }

  return { STT, Dictation, Listener, decodeWav, encodeWav };
}

if (typeof wasm !== "undefined" && wasm) {
  return bindSTT(wasm, typeof wasmModule !== "undefined" ? wasmModule : null);
}
