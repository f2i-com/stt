// Builds the standalone, zero-import `pkg/stt.wasm` binary (with the 13M-parameter Conformer
// speech model, DTW command matcher, and JS `STT` SDK embedded directly inside `stt.wasm`)
// plus the optional ES module loader `pkg/stt.js` and TypeScript declarations `pkg/stt.d.ts`.
//
//   node tools/build_wasm.mjs
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = path.join(root, "pkg");
fs.mkdirSync(pkgDir, { recursive: true });

console.log("[stt] Compiling Rust -> wasm32-unknown-unknown (release, +simd128)...");
execFileSync("cargo", ["build", "--target", "wasm32-unknown-unknown", "--release", "--lib"], {
  cwd: root,
  stdio: "inherit",
});

const rawWasmPath = path.join(root, "target", "wasm32-unknown-unknown", "release", "stt.wasm");
const outWasmPath = path.join(pkgDir, "stt.wasm");
fs.copyFileSync(rawWasmPath, outWasmPath);

// Run wasm-opt if installed on PATH for fast SIMD execution & compact code size
const runWasmOpt = (args) =>
  process.platform === "win32"
    ? spawnSync("cmd.exe", ["/d", "/s", "/c", "wasm-opt", ...args], {
        stdio: args[0] === "--version" ? "ignore" : "inherit",
      })
    : spawnSync("wasm-opt", args, {
        stdio: args[0] === "--version" ? "ignore" : "inherit",
      });

const optProbe = runWasmOpt(["--version"]);
if (optProbe.status === 0) {
  console.log("[stt] Optimizing pkg/stt.wasm with wasm-opt -O3 --enable-simd...");
  const optRes = runWasmOpt([
    "-O3",
    "--enable-simd",
    "--enable-bulk-memory",
    "--enable-bulk-memory-opt",
    "--enable-nontrapping-float-to-int",
    "--enable-sign-ext",
    "--enable-mutable-globals",
    outWasmPath,
    "-o",
    outWasmPath,
  ]);
  if (optRes.status !== 0) {
    console.warn("[stt] wasm-opt returned non-zero; keeping unoptimized release wasm.");
    fs.copyFileSync(rawWasmPath, outWasmPath);
  }
}

// Verify zero external imports & embedded JS glue inside stt.wasm!
const wasmBytes = fs.readFileSync(outWasmPath);
const wasmMod = new WebAssembly.Module(wasmBytes);
const imports = WebAssembly.Module.imports(wasmMod);
if (imports.length !== 0) {
  throw new Error(
    `Expected 0 external imports in stt.wasm, found ${imports.length}: ${JSON.stringify(imports)}`
  );
}
const inst = new WebAssembly.Instance(wasmMod, {});
const glueLen = inst.exports.stt_glue_len();
if (glueLen < 500) {
  throw new Error(`Embedded JS glue inside stt.wasm looks truncated (${glueLen} bytes)`);
}

// Extract `bindSTT` source from `src/glue.js` (stripping the trailing `Function(...)` return)
const glueSrc = fs
  .readFileSync(path.join(root, "src", "glue.js"), "utf8")
  .replace(/if\s*\(typeof wasm !== "undefined"[\s\S]*$/, "")
  .trim();

const esmLoader = `// Auto-generated ES module loader for stt.wasm (stt.computer).
// Note: stt.wasm is also 100% self-contained (zero imports, speech model & JS glue embedded inside stt_glue_ptr/stt_glue_len).

${glueSrc}

let _bound = null;

function requireBound() {
  if (!_bound) {
    throw new Error("STT WebAssembly is not initialized yet: call await init() or initSync(wasmBytes) first.");
  }
  return _bound;
}

function resolveModuleInput(input) {
  if (input && typeof input === "object" && "module_or_path" in input) return input.module_or_path;
  if (input && typeof input === "object" && "module" in input) return input.module;
  return input;
}

/** Synchronously initializes STT from a \`WebAssembly.Module\` or \`BufferSource\` (\`Uint8Array\` / \`ArrayBuffer\`). */
export function initSync(moduleOrBytes) {
  const raw = resolveModuleInput(moduleOrBytes);
  const mod = raw instanceof WebAssembly.Module ? raw : new WebAssembly.Module(raw);
  const instance = new WebAssembly.Instance(mod, {});
  _bound = bindSTT(instance.exports, mod);
  return { ..._bound, module: mod, exports: instance.exports };
}

/**
 * Loads \`stt.wasm\` (defaults to \`new URL("./stt.wasm", import.meta.url)\`) and initializes \`STT\`.
 * Returns \`{ STT, Dictation, Listener, decodeWav, encodeWav, module, exports }\`.
 */
export default async function init(moduleOrPath) {
  let src = resolveModuleInput(moduleOrPath);
  if (src === undefined) {
    src = new URL("./stt.wasm", import.meta.url);
  }
  let module = null;
  let instance;
  if (src instanceof WebAssembly.Instance) {
    instance = src;
  } else if (src instanceof WebAssembly.Module) {
    module = src;
    instance = await WebAssembly.instantiate(src, {});
  } else if (ArrayBuffer.isView(src) || src instanceof ArrayBuffer) {
    ({ module, instance } = await WebAssembly.instantiate(src, {}));
  } else {
    const resp = src instanceof Response ? src : fetch(src);
    if (typeof WebAssembly.instantiateStreaming === "function") {
      try {
        ({ module, instance } = await WebAssembly.instantiateStreaming(resp, {}));
      } catch {
        const r = await (src instanceof Response ? src : fetch(src));
        ({ module, instance } = await WebAssembly.instantiate(await r.arrayBuffer(), {}));
      }
    } else {
      const r = await resp;
      ({ module, instance } = await WebAssembly.instantiate(await r.arrayBuffer(), {}));
    }
  }
  _bound = bindSTT(instance.exports, module);
  return { ..._bound, module, exports: instance.exports };
}

/** One-step helper: loads \`stt.wasm\` and returns \`{ STT, Dictation, Listener, decodeWav, encodeWav, module, exports }\`. */
export const loadSTT = init;

export class STT {
  #inner;
  constructor(options) {
    const { STT: BoundSTT } = requireBound();
    this.#inner = new BoundSTT(options);
  }
  static modes() { return requireBound().STT.modes(); }
  static defaults(mode) { return requireBound().STT.defaults(mode); }
  static hasEmbeddedModel() { return requireBound().STT.hasEmbeddedModel(); }
  static modelSampleRate() { return requireBound().STT.modelSampleRate(); }

  warmup() { return this.#inner.warmup(); }
  configure(options) { return this.#inner.configure(options); }
  reset() { return this.#inner.reset(); }
  get settings() { return this.#inner.settings; }
  get sampleRate() { return this.#inner.sampleRate; }
  set sampleRate(sr) { this.#inner.sampleRate = sr; }
  setSampleRate(sr) { return this.#inner.setSampleRate(sr); }
  get mode() { return this.#inner.mode; }
  set mode(m) { this.#inner.mode = m; }
  setMode(m) { return this.#inner.setMode(m); }
  get speaking() { return this.#inner.speaking; }
  get levelDb() { return this.#inner.levelDb; }
  get floorDb() { return this.#inner.floorDb; }

  transcribe(input, sampleRate) { return this.#inner.transcribe(input, sampleRate); }
  transcribeText(input, sampleRate) { return this.#inner.transcribeText(input, sampleRate); }
  transcribeWav(wavBytes) { return this.#inner.transcribeWav(wavBytes); }
  transcribeFile(source) { return this.#inner.transcribeFile(source); }

  push(samples) { return this.#inner.push(samples); }
  flush() { return this.#inner.flush(); }

  teachNext(label) { return this.#inner.teachNext(label); }
  cancelTeach() { return this.#inner.cancelTeach(); }
  get teaching() { return this.#inner.teaching; }
  teach(label, input, sampleRate) { return this.#inner.teach(label, input, sampleRate); }
  teachClip(label, samples, sampleRate) { return this.#inner.teachClip(label, samples, sampleRate); }
  recognizeCommand(input, sampleRate) { return this.#inner.recognizeCommand(input, sampleRate); }
  recognizeClip(samples, sampleRate) { return this.#inner.recognizeClip(samples, sampleRate); }
  commands() { return this.#inner.commands(); }
  labels() { return this.#inner.labels(); }
  count(label) { return this.#inner.count(label); }
  commandCount(label) { return this.#inner.commandCount(label); }
  removeCommand(label) { return this.#inner.removeCommand(label); }
  remove(label) { return this.#inner.remove(label); }
  clearCommands() { return this.#inner.clearCommands(); }
  clear() { return this.#inner.clear(); }
  exportCommands() { return this.#inner.exportCommands(); }
  importCommands(bytes) { return this.#inner.importCommands(bytes); }

  listen(options) { return this.#inner.listen(options); }
  free() { return this.#inner.free(); }
}

export class Dictation {
  #inner;
  constructor(sampleRate, model) {
    const { Dictation: BoundDictation } = requireBound();
    this.#inner = new BoundDictation(sampleRate, model);
  }
  push(samples) { return this.#inner.push(samples); }
  flush() { return this.#inner.flush(); }
  transcribeClip(samples) { return this.#inner.transcribeClip(samples); }
  transcribe(input, sampleRate) { return this.#inner.transcribe(input, sampleRate); }
  get levelDb() { return this.#inner.levelDb; }
  get floorDb() { return this.#inner.floorDb; }
  get speaking() { return this.#inner.speaking; }
  get sampleRate() { return this.#inner.sampleRate; }
  free() { return this.#inner.free(); }
}

export class Listener {
  #inner;
  constructor(sampleRate) {
    const { Listener: BoundListener } = requireBound();
    this.#inner = new BoundListener(sampleRate);
  }
  push(samples) { return this.#inner.push(samples); }
  flush() { return this.#inner.flush(); }
  teachNext(label) { return this.#inner.teachNext(label); }
  cancelTeach() { return this.#inner.cancelTeach(); }
  get teaching() { return this.#inner.teaching; }
  teachClip(label, samples) { return this.#inner.teachClip(label, samples); }
  recognizeClip(samples) { return this.#inner.recognizeClip(samples); }
  remove(label) { return this.#inner.remove(label); }
  clear() { return this.#inner.clear(); }
  labels() { return this.#inner.labels(); }
  count(label) { return this.#inner.count(label); }
  exportCommands() { return this.#inner.exportCommands(); }
  importCommands(bytes) { return this.#inner.importCommands(bytes); }
  get levelDb() { return this.#inner.levelDb; }
  get floorDb() { return this.#inner.floorDb; }
  get speaking() { return this.#inner.speaking; }
  get sampleRate() { return this.#inner.sampleRate; }
  get maxDistance() { return this.#inner.maxDistance; }
  set maxDistance(v) { this.#inner.maxDistance = v; }
  get maxRatio() { return this.#inner.maxRatio; }
  set maxRatio(v) { this.#inner.maxRatio = v; }
  free() { return this.#inner.free(); }
}

export function decodeWav(wavBytes) {
  return requireBound().decodeWav(wavBytes);
}

export function encodeWav(samples, sampleRate) {
  return requireBound().encodeWav(samples, sampleRate);
}
`;

const dts = `// TypeScript declarations for STT (stt.computer)
export type SttMode = "dictation" | "commands" | "both";

export type AudioInput =
  | Float32Array
  | number[]
  | Uint8Array
  | ArrayBuffer
  | AudioBuffer
  | { samples: Float32Array | number[]; sampleRate?: number };

export interface SttOptions {
  /** Input audio sample rate in Hz (default 16000; auto-updated by \`listen()\` to \`AudioContext.sampleRate\`). */
  sampleRate?: number;
  /** Streaming recognition mode: \`"dictation"\` (default), \`"commands"\`, or \`"both"\`. */
  mode?: SttMode;
  /** Optional custom Conformer weight file (\`STTW\` / \`VRCW\` binary); defaults to the embedded 13M Conformer model. */
  model?: Uint8Array | ArrayBuffer;
  /** Pause duration in ms required to end a phrase/utterance (default 600 for dictation, 350 for commands). */
  hangMs?: number;
  /** Pre-onset audio kept in ms so soft initial consonants are preserved (default 300). */
  preRollMs?: number;
  /** Maximum phrase length in ms before automatic split (default 20000). */
  maxMs?: number;
  /** Length in ms after which a brief pause (\`softHangMs\`) splits a long phrase (default 8000). */
  softMaxMs?: number;
  /** Brief pause duration in ms once \`softMaxMs\` is reached (default 150). */
  softHangMs?: number;
  /** dB above tracked room floor required to start speech detection (default 15). */
  onsetDb?: number;
  /** dB above tracked room floor below which speech counts as paused (default 8). */
  releaseDb?: number;
  /** dB drop below phrase peak that also counts as paused (default 35). */
  dropDb?: number;
  /** Minimum block level in dBFS to ever count as speech (default -60). */
  minLevelDb?: number;
  /** Maximum DTW distance accepted for taught voice commands (default 4.6; lower is stricter). */
  maxDistance?: number;
  /** Maximum \`distance / runnerUpDistance\` ratio accepted for taught voice commands (default 0.92). */
  maxRatio?: number;
}

export type SttSettings = Required<Omit<SttOptions, "model">>;

export interface WordTiming {
  word: string;
  start: number;
  end: number;
  confidence: number;
}

export interface Transcript {
  text: string;
  words: WordTiming[];
  duration: number;
  confidence: number;
}

export interface PhraseEvent {
  type: "phrase";
  text: string;
  words: WordTiming[];
  confidence: number;
  startSeconds: number;
  durationSeconds: number;
  tookMs?: number;
}

export interface HeardEvent {
  type: "heard";
  label: string | null;
  accepted: boolean;
  distance: number;
  runnerUp: string | null;
  runnerUpDistance: number;
  startSeconds: number;
  durationSeconds: number;
  tookMs?: number;
}

export interface TaughtEvent {
  type: "taught";
  label: string;
  count: number;
  startSeconds: number;
  durationSeconds: number;
  tookMs?: number;
}

export interface IgnoredEvent {
  type: "ignored";
  startSeconds: number;
  durationSeconds: number;
  tookMs?: number;
}

export type ListenerEvent = HeardEvent | TaughtEvent | IgnoredEvent;
export type SttEvent = PhraseEvent | HeardEvent | TaughtEvent | IgnoredEvent;

export interface CommandSummary {
  label: string;
  count: number;
}

export interface LevelTelemetry {
  levelDb: number;
  floorDb: number;
  speaking: boolean;
}

export interface ListenOptions {
  mode?: SttMode;
  /** Run STT inside an automatic inline Web Worker so transcription never blocks the main UI thread (default \`true\`). */
  worker?: boolean;
  stream?: MediaStream;
  audioConstraints?: MediaTrackConstraints;
  audioContext?: AudioContext;
  onPhrase?: (phrase: PhraseEvent) => void;
  onCommand?: (event: HeardEvent) => void;
  onTaught?: (event: TaughtEvent) => void;
  onIgnored?: (event: IgnoredEvent) => void;
  onEvent?: (event: SttEvent) => void;
  onSpeaking?: (speaking: boolean) => void;
  onLevel?: (telemetry: LevelTelemetry) => void;
  onError?: (err: Error) => void;
}

export interface ListenSession {
  readonly sampleRate: number;
  readonly usingWorker: boolean;
  readonly paused: boolean;
  readonly active: boolean;
  pause(): void;
  resume(): void;
  configure(options: Omit<SttOptions, "model"> | SttMode | number): Promise<SttSettings>;
  teachNext(label: string): Promise<void>;
  cancelTeach(): Promise<void>;
  removeCommand(label: string): Promise<void>;
  clearCommands(): Promise<void>;
  importCommands(bytes: Uint8Array | ArrayBuffer): Promise<boolean>;
  flush(): Promise<SttEvent[]>;
  stop(): Promise<void>;
}

export interface DecodedWav {
  samples: Float32Array;
  sampleRate: number;
  duration: number;
}

export class STT {
  constructor(options?: SttOptions | SttMode | number);
  static modes(): SttMode[];
  static defaults(mode?: SttMode): SttSettings;
  static hasEmbeddedModel(): boolean;
  static modelSampleRate(): number;

  warmup(): void;
  configure(options: Omit<SttOptions, "model"> | SttMode | number): SttSettings;
  reset(): void;
  readonly settings: SttSettings;
  sampleRate: number;
  setSampleRate(sampleRate: number): SttSettings;
  mode: SttMode;
  setMode(mode: SttMode): SttSettings;
  readonly speaking: boolean;
  readonly levelDb: number;
  readonly floorDb: number;

  transcribe(input: AudioInput, sampleRate?: number): Transcript;
  transcribeText(input: AudioInput, sampleRate?: number): string;
  transcribeWav(wavBytes: Uint8Array | ArrayBuffer): Transcript;
  transcribeFile(source: Blob | File | Response | URL | string | ArrayBuffer | Uint8Array): Promise<Transcript>;

  push(samples: Float32Array | number[]): SttEvent[];
  flush(): SttEvent[];

  teachNext(label: string): void;
  cancelTeach(): void;
  readonly teaching: string | undefined;
  teach(label: string, input: AudioInput, sampleRate?: number): boolean;
  teachClip(label: string, samples: AudioInput, sampleRate?: number): boolean;
  recognizeCommand(input: AudioInput, sampleRate?: number): HeardEvent | null;
  recognizeClip(samples: AudioInput, sampleRate?: number): HeardEvent | null;
  commands(): CommandSummary[];
  labels(): string[];
  count(label: string): number;
  commandCount(label: string): number;
  removeCommand(label: string): void;
  remove(label: string): void;
  clearCommands(): void;
  clear(): void;
  exportCommands(): Uint8Array;
  importCommands(bytes: Uint8Array | ArrayBuffer): boolean;

  listen(options?: ListenOptions): Promise<ListenSession>;
  free(): void;
}

export class Dictation {
  constructor(sampleRate?: number, model?: Uint8Array | ArrayBuffer);
  push(samples: Float32Array | number[]): PhraseEvent[];
  flush(): PhraseEvent[];
  transcribeClip(samples: Float32Array | number[]): string;
  transcribe(input: AudioInput, sampleRate?: number): Transcript;
  readonly levelDb: number;
  readonly floorDb: number;
  readonly speaking: boolean;
  readonly sampleRate: number;
  free(): void;
}

export class Listener {
  constructor(sampleRate?: number);
  push(samples: Float32Array | number[]): ListenerEvent[];
  flush(): ListenerEvent[];
  teachNext(label: string): void;
  cancelTeach(): void;
  readonly teaching: string | undefined;
  teachClip(label: string, samples: Float32Array | number[]): boolean;
  recognizeClip(samples: Float32Array | number[]): HeardEvent | null;
  remove(label: string): void;
  clear(): void;
  labels(): string[];
  count(label: string): number;
  exportCommands(): Uint8Array;
  importCommands(bytes: Uint8Array | ArrayBuffer): boolean;
  readonly levelDb: number;
  readonly floorDb: number;
  readonly speaking: boolean;
  readonly sampleRate: number;
  maxDistance: number;
  maxRatio: number;
  free(): void;
}

export function decodeWav(wavBytes: Uint8Array | ArrayBuffer): DecodedWav;
export function encodeWav(samples: Float32Array | number[], sampleRate?: number): Uint8Array;

export interface SttExports {
  STT: typeof STT;
  Dictation: typeof Dictation;
  Listener: typeof Listener;
  decodeWav: typeof decodeWav;
  encodeWav: typeof encodeWav;
  module: WebAssembly.Module | null;
  exports: WebAssembly.Exports;
}

export function initSync(
  moduleOrBytes: WebAssembly.Module | BufferSource | { module: WebAssembly.Module | BufferSource }
): SttExports;
export default function init(
  moduleOrPath?: RequestInfo | URL | Response | BufferSource | WebAssembly.Module
): Promise<SttExports>;
export const loadSTT: typeof init;
`;

fs.writeFileSync(path.join(pkgDir, "stt.js"), esmLoader);
fs.writeFileSync(path.join(pkgDir, "stt.d.ts"), dts);
fs.writeFileSync(
  path.join(pkgDir, "package.json"),
  JSON.stringify(
    {
      name: "stt",
      version: "1.0.0",
      type: "module",
      description:
        "STT (stt.computer): self-contained speech recognition & taught voice commands in WebAssembly.",
      homepage: "https://stt.computer",
      repository: { type: "git", url: "https://github.com/f2i-com/stt.git" },
      main: "stt.js",
      types: "stt.d.ts",
      files: ["stt.wasm", "stt.js", "stt.d.ts"],
    },
    null,
    2
  ) + "\n"
);

const gzSize = zlib.gzipSync(wasmBytes).length;
console.log(
  `[stt] Built self-contained pkg/stt.wasm (${(wasmBytes.length / (1024 * 1024)).toFixed(2)} MB raw, ${(gzSize / (1024 * 1024)).toFixed(2)} MB gzipped, 0 external imports, ${glueLen} bytes internal JS glue) + pkg/stt.js + pkg/stt.d.ts`
);
