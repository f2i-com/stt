//! Self-contained WebAssembly ABI + zero-dependency JSON marshaling for STT (`stt.computer`).
//!
//! When compiled to `wasm32-unknown-unknown`, this module exports a pure C-ABI (`stt_*`)
//! with **zero external imports** and embeds the complete JavaScript `STT` SDK (`src/glue.js`)
//! right inside `stt.wasm` (`stt_glue_ptr` / `stt_glue_len`).

#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

use crate::config::{Mode, SttConfig, NUMBER_SETTINGS};
use crate::conformer::{Transcript, WordTiming, SAMPLE_RATE};
use crate::{CommandMatch, Phrase, SttEvent, HAS_EMBEDDED_MODEL, STT};

/// Complete JavaScript `STT` SDK embedded directly inside `stt.wasm`.
pub const EMBEDDED_JS_GLUE: &str = include_str!("glue.js");

const SENTINEL_NAN: &str = "__STT_NAN__";
const SENTINEL_POS_INF: &str = "__STT_POS_INF__";
const SENTINEL_NEG_INF: &str = "__STT_NEG_INF__";

// ============================================================================
// Minimal RFC 8259 JSON parser & serializer (zero external crates)
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum JsonVal {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(Vec<(String, JsonVal)>),
}

impl JsonVal {
    fn as_str(&self) -> Option<&str> {
        match self {
            JsonVal::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.s.get(self.i).copied()
    }

    fn parse_value(&mut self) -> Result<JsonVal, String> {
        let b = self.peek().ok_or_else(|| "unexpected end of JSON".to_string())?;
        match b {
            b'n' => {
                if self.s.get(self.i..self.i + 4) == Some(b"null") {
                    self.i += 4;
                    Ok(JsonVal::Null)
                } else {
                    Err("invalid JSON literal".into())
                }
            }
            b't' => {
                if self.s.get(self.i..self.i + 4) == Some(b"true") {
                    self.i += 4;
                    Ok(JsonVal::Bool(true))
                } else {
                    Err("invalid JSON literal".into())
                }
            }
            b'f' => {
                if self.s.get(self.i..self.i + 5) == Some(b"false") {
                    self.i += 5;
                    Ok(JsonVal::Bool(false))
                } else {
                    Err("invalid JSON literal".into())
                }
            }
            b'"' => self.parse_string().map(JsonVal::Str),
            b'[' => self.parse_array(),
            b'{' => self.parse_object(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            _ => Err(format!("unexpected JSON character '{}'", b as char)),
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.skip_ws();
        if self.s.get(self.i) != Some(&b'"') {
            return Err("expected JSON string".into());
        }
        self.i += 1;
        let mut out = String::new();
        while self.i < self.s.len() {
            let b = self.s[self.i];
            self.i += 1;
            if b == b'"' {
                return Ok(out);
            }
            if b == b'\\' {
                let esc = *self.s.get(self.i).ok_or("unterminated JSON escape")?;
                self.i += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let hex = std::str::from_utf8(
                            self.s.get(self.i..self.i + 4).ok_or("short \\u escape")?,
                        )
                        .map_err(|_| "bad \\u escape")?;
                        self.i += 4;
                        let cp = u32::from_str_radix(hex, 16).map_err(|_| "bad \\u hex")?;
                        if let Some(ch) = char::from_u32(cp) {
                            out.push(ch);
                        }
                    }
                    _ => out.push(esc as char),
                }
            } else {
                let start = self.i - 1;
                let rest = std::str::from_utf8(&self.s[start..]).map_err(|_| "invalid UTF-8")?;
                let ch = rest.chars().next().ok_or("empty UTF-8")?;
                out.push(ch);
                self.i = start + ch.len_utf8();
            }
        }
        Err("unterminated JSON string".into())
    }

    fn parse_number(&mut self) -> Result<JsonVal, String> {
        self.skip_ws();
        let start = self.i;
        while self.i < self.s.len()
            && matches!(self.s[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.i += 1;
        }
        let num_str = std::str::from_utf8(&self.s[start..self.i]).map_err(|_| "bad number")?;
        let n: f64 = num_str.parse().map_err(|_| format!("bad number {num_str:?}"))?;
        Ok(JsonVal::Num(n))
    }

    fn parse_array(&mut self) -> Result<JsonVal, String> {
        self.i += 1;
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(JsonVal::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(JsonVal::Arr(items));
                }
                _ => return Err("expected ',' or ']' in JSON array".into()),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonVal, String> {
        self.i += 1;
        let mut entries = Vec::new();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(JsonVal::Obj(entries));
        }
        loop {
            let key = self.parse_string()?;
            if self.peek() != Some(b':') {
                return Err("expected ':' in JSON object".into());
            }
            self.i += 1;
            let val = self.parse_value()?;
            entries.push((key, val));
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(JsonVal::Obj(entries));
                }
                _ => return Err("expected ',' or '}' in JSON object".into()),
            }
        }
    }
}

pub(crate) fn parse_json(raw: &str) -> Result<JsonVal, String> {
    let mut p = JsonParser {
        s: raw.as_bytes(),
        i: 0,
    };
    let v = p.parse_value()?;
    p.skip_ws();
    if p.i != p.s.len() {
        return Err("trailing characters after JSON value".into());
    }
    Ok(v)
}

fn push_json_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn push_json_num(out: &mut String, v: f64) {
    if v.is_finite() {
        let clean = if v == 0.0 { 0.0 } else { v };
        out.push_str(&clean.to_string());
    } else {
        out.push_str("null");
    }
}

// ============================================================================
// Options validation & JSON builders
// ============================================================================

fn parse_options_payload(raw: &str) -> Result<JsonVal, String> {
    let t = raw.trim();
    if t.is_empty() {
        return Ok(JsonVal::Null);
    }
    if let Ok(v) = parse_json(t) {
        return Ok(v);
    }
    if !t.starts_with('{') && !t.starts_with('[') && !t.starts_with('"') {
        return Ok(JsonVal::Str(t.to_string()));
    }
    Err("invalid STT options JSON".into())
}

fn finite_val(value: &JsonVal, key: &str) -> Result<f64, String> {
    match value {
        JsonVal::Num(v) if v.is_finite() => Ok(*v),
        _ => Err(format!("{key} must be a finite number")),
    }
}

fn threshold_val(value: &JsonVal, key: &str) -> Result<f32, String> {
    match value {
        JsonVal::Num(v) if v.is_finite() => Ok(*v as f32),
        JsonVal::Str(s) if s == SENTINEL_POS_INF => Ok(1.0e9),
        JsonVal::Str(s) if s == SENTINEL_NEG_INF => Ok(0.0),
        _ => Err(format!("{key} must be a number")),
    }
}

pub(crate) fn apply_json_options(cfg: &mut SttConfig, options: &JsonVal) -> Result<(), String> {
    match options {
        JsonVal::Null => return Ok(()),
        JsonVal::Num(sr) if sr.is_finite() => {
            cfg.sample_rate = (sr.round() as i64).clamp(4_000, 384_000) as u32;
            cfg.sanitize();
            return Ok(());
        }
        JsonVal::Str(s) => {
            if s == SENTINEL_NAN || s == SENTINEL_POS_INF || s == SENTINEL_NEG_INF {
                return Err("sampleRate must be a finite number".into());
            }
            let mode = Mode::parse(s).ok_or_else(|| {
                format!("unknown mode {s:?}; expected \"dictation\", \"commands\", or \"both\"")
            })?;
            cfg.set_mode(mode);
            cfg.sanitize();
            return Ok(());
        }
        JsonVal::Obj(_) => {}
        _ => {
            return Err(
                "STT options must be an object, a sampleRate number, or \"dictation\" / \"commands\" / \"both\""
                    .into(),
            )
        }
    }

    let mut next = cfg.clone();
    let JsonVal::Obj(entries) = options else {
        unreachable!()
    };

    // Apply `mode` first so mode-specific timing defaults take effect before explicit overrides.
    for (key, value) in entries {
        if key == "mode" && !matches!(value, JsonVal::Null) {
            let s = value.as_str().ok_or("mode must be a string")?;
            let mode = Mode::parse(s).ok_or_else(|| {
                format!("unknown mode {s:?}; expected \"dictation\", \"commands\", or \"both\"")
            })?;
            next.set_mode(mode);
        }
    }

    for (key, value) in entries {
        if matches!(value, JsonVal::Null) || key == "mode" {
            continue;
        }
        match key.as_str() {
            "sampleRate" => {
                let sr = finite_val(value, "sampleRate")?;
                if sr <= 0.0 {
                    return Err("sampleRate must be a positive number".into());
                }
                next.sample_rate = (sr.round() as i64).clamp(4_000, 384_000) as u32;
            }
            "hangMs" => next.hang_ms = finite_val(value, "hangMs")? as f32,
            "preRollMs" => next.pre_roll_ms = finite_val(value, "preRollMs")? as f32,
            "maxMs" => next.max_ms = finite_val(value, "maxMs")? as f32,
            "softMaxMs" => next.soft_max_ms = finite_val(value, "softMaxMs")? as f32,
            "softHangMs" => next.soft_hang_ms = finite_val(value, "softHangMs")? as f32,
            "onsetDb" => next.onset_db = finite_val(value, "onsetDb")? as f32,
            "releaseDb" => next.release_db = finite_val(value, "releaseDb")? as f32,
            "dropDb" => next.drop_db = finite_val(value, "dropDb")? as f32,
            "minLevelDb" => next.min_level_db = finite_val(value, "minLevelDb")? as f32,
            "maxDistance" => next.max_distance = threshold_val(value, "maxDistance")?,
            "maxRatio" => next.max_ratio = threshold_val(value, "maxRatio")?,
            _ => {
                return Err(format!(
                    "unknown setting {key:?}; valid settings are: mode, {}",
                    NUMBER_SETTINGS.join(", ")
                ));
            }
        }
    }

    next.sanitize();
    *cfg = next;
    Ok(())
}

pub(crate) fn settings_to_json(cfg: &SttConfig) -> String {
    let mut out = String::with_capacity(320);
    out.push_str("{\"sampleRate\":");
    out.push_str(&cfg.sample_rate.to_string());
    out.push_str(",\"mode\":");
    push_json_str(&mut out, cfg.mode.as_str());
    out.push_str(",\"hangMs\":");
    push_json_num(&mut out, cfg.hang_ms as f64);
    out.push_str(",\"preRollMs\":");
    push_json_num(&mut out, cfg.pre_roll_ms as f64);
    out.push_str(",\"maxMs\":");
    push_json_num(&mut out, cfg.max_ms as f64);
    out.push_str(",\"softMaxMs\":");
    push_json_num(&mut out, cfg.soft_max_ms as f64);
    out.push_str(",\"softHangMs\":");
    push_json_num(&mut out, cfg.soft_hang_ms as f64);
    out.push_str(",\"onsetDb\":");
    push_json_num(&mut out, cfg.onset_db as f64);
    out.push_str(",\"releaseDb\":");
    push_json_num(&mut out, cfg.release_db as f64);
    out.push_str(",\"dropDb\":");
    push_json_num(&mut out, cfg.drop_db as f64);
    out.push_str(",\"minLevelDb\":");
    push_json_num(&mut out, cfg.min_level_db as f64);
    out.push_str(",\"maxDistance\":");
    if cfg.max_distance >= 5.0e8 {
        push_json_str(&mut out, SENTINEL_POS_INF);
    } else {
        push_json_num(&mut out, cfg.max_distance as f64);
    }
    out.push_str(",\"maxRatio\":");
    if cfg.max_ratio >= 5.0e8 {
        push_json_str(&mut out, SENTINEL_POS_INF);
    } else {
        push_json_num(&mut out, cfg.max_ratio as f64);
    }
    out.push('}');
    out
}

fn meta_json_string() -> String {
    let mut out = String::with_capacity(640);
    out.push_str("{\"modelSampleRate\":");
    out.push_str(&SAMPLE_RATE.to_string());
    out.push_str(",\"hasEmbeddedModel\":");
    out.push_str(if HAS_EMBEDDED_MODEL { "true" } else { "false" });
    out.push_str(",\"modes\":[\"dictation\",\"commands\",\"both\"]");
    out.push_str(",\"defaults\":{\"dictation\":");
    out.push_str(&settings_to_json(&SttConfig::for_mode(Mode::Dictation)));
    out.push_str(",\"commands\":");
    out.push_str(&settings_to_json(&SttConfig::for_mode(Mode::Commands)));
    out.push_str(",\"both\":");
    out.push_str(&settings_to_json(&SttConfig::for_mode(Mode::Both)));
    out.push_str("}}");
    out
}

fn push_words_json(out: &mut String, words: &[WordTiming]) {
    out.push('[');
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"word\":");
        push_json_str(out, &w.word);
        out.push_str(",\"start\":");
        push_json_num(out, w.start);
        out.push_str(",\"end\":");
        push_json_num(out, w.end);
        out.push_str(",\"confidence\":");
        push_json_num(out, w.confidence);
        out.push('}');
    }
    out.push(']');
}

fn transcript_to_json(tr: &Transcript) -> String {
    let mut out = String::with_capacity(128 + tr.words.len() * 64);
    out.push_str("{\"text\":");
    push_json_str(&mut out, &tr.text);
    out.push_str(",\"duration\":");
    push_json_num(&mut out, tr.duration);
    out.push_str(",\"confidence\":");
    push_json_num(&mut out, tr.confidence);
    out.push_str(",\"words\":");
    push_words_json(&mut out, &tr.words);
    out.push('}');
    out
}

fn push_phrase_json(out: &mut String, p: &Phrase) {
    out.push_str("{\"type\":\"phrase\",\"text\":");
    push_json_str(out, &p.text);
    out.push_str(",\"confidence\":");
    push_json_num(out, p.confidence);
    out.push_str(",\"startSeconds\":");
    push_json_num(out, p.start_seconds);
    out.push_str(",\"durationSeconds\":");
    push_json_num(out, p.duration_seconds);
    out.push_str(",\"words\":");
    push_words_json(out, &p.words);
    out.push('}');
}

fn push_heard_json(out: &mut String, m: &CommandMatch) {
    out.push_str("{\"type\":\"heard\",\"label\":");
    match &m.label {
        Some(lbl) => push_json_str(out, lbl),
        None => out.push_str("null"),
    }
    out.push_str(",\"accepted\":");
    out.push_str(if m.accepted { "true" } else { "false" });
    out.push_str(",\"distance\":");
    push_json_num(out, m.distance as f64);
    out.push_str(",\"runnerUp\":");
    match &m.runner_up {
        Some(lbl) => push_json_str(out, lbl),
        None => out.push_str("null"),
    }
    out.push_str(",\"runnerUpDistance\":");
    push_json_num(out, m.runner_up_distance as f64);
    out.push_str(",\"startSeconds\":");
    push_json_num(out, m.start_seconds);
    out.push_str(",\"durationSeconds\":");
    push_json_num(out, m.duration_seconds);
    out.push('}');
}

fn events_to_json(events: &[SttEvent]) -> String {
    let mut out = String::with_capacity(64 + events.len() * 128);
    out.push('[');
    for (i, ev) in events.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        match ev {
            SttEvent::Phrase(p) => push_phrase_json(&mut out, p),
            SttEvent::Heard(m) => push_heard_json(&mut out, m),
            SttEvent::Taught {
                label,
                count,
                start_seconds,
                duration_seconds,
            } => {
                out.push_str("{\"type\":\"taught\",\"label\":");
                push_json_str(&mut out, label);
                out.push_str(",\"count\":");
                out.push_str(&count.to_string());
                out.push_str(",\"startSeconds\":");
                push_json_num(&mut out, *start_seconds);
                out.push_str(",\"durationSeconds\":");
                push_json_num(&mut out, *duration_seconds);
                out.push('}');
            }
            SttEvent::Ignored {
                start_seconds,
                duration_seconds,
            } => {
                out.push_str("{\"type\":\"ignored\",\"startSeconds\":");
                push_json_num(&mut out, *start_seconds);
                out.push_str(",\"durationSeconds\":");
                push_json_num(&mut out, *duration_seconds);
                out.push('}');
            }
        }
    }
    out.push(']');
    out
}

// ============================================================================
// Pure C-ABI WebAssembly Host (`stt_*`, zero external imports)
// ============================================================================

mod abi {
    use super::*;
    use std::cell::RefCell;

    #[derive(Default)]
    struct WasmHost {
        instances: Vec<Option<STT>>,
        scratch_in: Vec<u8>,
        scratch_aux: Vec<u8>,
        scratch_f32: Vec<f32>,
        out_json: String,
        out_bytes: Vec<u8>,
        out_samples: Vec<f32>,
    }

    impl WasmHost {
        fn set_err(&mut self, msg: impl Into<String>) -> i32 {
            self.out_json = msg.into();
            -1
        }

        fn read_bytes(&self, ptr: usize, len: usize) -> Vec<u8> {
            if len == 0 || ptr == 0 {
                return Vec::new();
            }
            unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }.to_vec()
        }

        fn read_str(&self, ptr: usize, len: usize) -> Result<String, String> {
            if len == 0 || ptr == 0 {
                return Ok(String::new());
            }
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
            std::str::from_utf8(slice)
                .map(|s| s.to_string())
                .map_err(|_| "invalid UTF-8 input".to_string())
        }

        fn read_f32(&self, ptr: usize, len: usize) -> Vec<f32> {
            if len == 0 || ptr == 0 {
                return Vec::new();
            }
            unsafe { std::slice::from_raw_parts(ptr as *const f32, len) }.to_vec()
        }

        fn get_inst(&self, handle: i32) -> Result<&STT, String> {
            if handle <= 0 {
                return Err("invalid STT handle".into());
            }
            self.instances
                .get((handle - 1) as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| "STT instance has been freed".into())
        }

        fn get_inst_mut(&mut self, handle: i32) -> Result<&mut STT, String> {
            if handle <= 0 {
                return Err("invalid STT handle".into());
            }
            self.instances
                .get_mut((handle - 1) as usize)
                .and_then(Option::as_mut)
                .ok_or_else(|| "STT instance has been freed".into())
        }
    }

    thread_local! {
        static HOST: RefCell<WasmHost> = RefCell::new(WasmHost::default());
    }

    #[no_mangle]
    pub extern "C" fn stt_glue_ptr() -> *const u8 {
        EMBEDDED_JS_GLUE.as_ptr()
    }

    #[no_mangle]
    pub extern "C" fn stt_glue_len() -> usize {
        EMBEDDED_JS_GLUE.len()
    }

    #[no_mangle]
    pub extern "C" fn stt_has_embedded_model() -> i32 {
        i32::from(HAS_EMBEDDED_MODEL)
    }

    #[no_mangle]
    pub extern "C" fn stt_alloc(len: usize) -> *mut u8 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            h.scratch_in.resize(len, 0);
            h.scratch_in.as_mut_ptr()
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_alloc_aux(len: usize) -> *mut u8 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            h.scratch_aux.resize(len, 0);
            h.scratch_aux.as_mut_ptr()
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_alloc_f32(len: usize) -> *mut f32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            h.scratch_f32.resize(len, 0.0);
            h.scratch_f32.as_mut_ptr()
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_json_ptr() -> *const u8 {
        HOST.with(|h| h.borrow().out_json.as_ptr())
    }

    #[no_mangle]
    pub extern "C" fn stt_json_len() -> usize {
        HOST.with(|h| h.borrow().out_json.len())
    }

    #[no_mangle]
    pub extern "C" fn stt_bytes_ptr() -> *const u8 {
        HOST.with(|h| h.borrow().out_bytes.as_ptr())
    }

    #[no_mangle]
    pub extern "C" fn stt_bytes_len() -> usize {
        HOST.with(|h| h.borrow().out_bytes.len())
    }

    #[no_mangle]
    pub extern "C" fn stt_samples_ptr() -> *const f32 {
        HOST.with(|h| h.borrow().out_samples.as_ptr())
    }

    #[no_mangle]
    pub extern "C" fn stt_samples_len() -> usize {
        HOST.with(|h| h.borrow().out_samples.len())
    }

    #[no_mangle]
    pub extern "C" fn stt_meta_json() -> i32 {
        HOST.with(|h| {
            h.borrow_mut().out_json = meta_json_string();
            0
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_new(
        opts_ptr: usize,
        opts_len: usize,
        model_ptr: usize,
        model_len: usize,
    ) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let mut cfg = SttConfig::default();
            if opts_len > 0 {
                let raw = match h.read_str(opts_ptr, opts_len) {
                    Ok(s) => s,
                    Err(e) => return h.set_err(e),
                };
                let opts = match parse_options_payload(&raw) {
                    Ok(v) => v,
                    Err(e) => return h.set_err(e),
                };
                if let Err(e) = apply_json_options(&mut cfg, &opts) {
                    return h.set_err(e);
                }
            }
            let stt = if model_len > 0 || model_ptr != 0 {
                let model_bytes = h.read_bytes(model_ptr, model_len);
                // Reclaim auxiliary model scratch memory promptly after parsing
                h.scratch_aux = Vec::new();
                match STT::with_model(cfg.clone(), &model_bytes) {
                    Ok(inst) => inst,
                    Err(e) => return h.set_err(e),
                }
            } else {
                STT::new(cfg.clone())
            };
            let json = settings_to_json(stt.config());
            let slot = h.instances.iter().position(Option::is_none);
            let handle = if let Some(idx) = slot {
                h.instances[idx] = Some(stt);
                (idx + 1) as i32
            } else {
                h.instances.push(Some(stt));
                h.instances.len() as i32
            };
            h.out_json = json;
            handle
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_drop(handle: i32) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            if handle > 0 {
                if let Some(slot) = h.instances.get_mut((handle - 1) as usize) {
                    *slot = None;
                }
            }
            0
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_warmup(handle: i32) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let res = match h.get_inst(handle) {
                Ok(inst) => inst.warmup(),
                Err(e) => Err(e),
            };
            match res {
                Ok(()) => 0,
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_get_settings(handle: i32) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let json = match h.get_inst(handle) {
                Ok(inst) => settings_to_json(inst.config()),
                Err(e) => return h.set_err(e),
            };
            h.out_json = json;
            0
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_configure(handle: i32, ptr: usize, len: usize) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let raw = match h.read_str(ptr, len) {
                Ok(s) => s,
                Err(e) => return h.set_err(e),
            };
            let opts = match parse_options_payload(&raw) {
                Ok(v) => v,
                Err(e) => return h.set_err(e),
            };
            let inst = match h.get_inst_mut(handle) {
                Ok(i) => i,
                Err(e) => return h.set_err(e),
            };
            let mut next = inst.config().clone();
            if let Err(e) = apply_json_options(&mut next, &opts) {
                return h.set_err(e);
            }
            inst.set_config(next);
            let json = settings_to_json(inst.config());
            h.out_json = json;
            0
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_reset(handle: i32) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            match h.get_inst_mut(handle) {
                Ok(inst) => {
                    inst.reset();
                    0
                }
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_speaking(handle: i32) -> i32 {
        HOST.with(|h| {
            let h = h.borrow();
            match h.get_inst(handle) {
                Ok(inst) => i32::from(inst.speaking()),
                Err(_) => 0,
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_level_db(handle: i32) -> f32 {
        HOST.with(|h| {
            let h = h.borrow();
            match h.get_inst(handle) {
                Ok(inst) => inst.level_db(),
                Err(_) => -100.0,
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_floor_db(handle: i32) -> f32 {
        HOST.with(|h| {
            let h = h.borrow();
            match h.get_inst(handle) {
                Ok(inst) => inst.floor_db(),
                Err(_) => -100.0,
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_transcribe_pcm(
        handle: i32,
        f32_ptr: usize,
        f32_len: usize,
        sample_rate: u32,
    ) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let pcm = h.read_f32(f32_ptr, f32_len);
            let res = {
                let inst = match h.get_inst(handle) {
                    Ok(i) => i,
                    Err(e) => return h.set_err(e),
                };
                let sr = if sample_rate == 0 {
                    inst.sample_rate()
                } else {
                    sample_rate
                };
                inst.transcribe(&pcm, sr)
            };
            match res {
                Ok(tr) => {
                    h.out_json = transcript_to_json(&tr);
                    0
                }
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_transcribe_wav(handle: i32, wav_ptr: usize, wav_len: usize) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let wav = h.read_bytes(wav_ptr, wav_len);
            let res = match h.get_inst(handle) {
                Ok(inst) => inst.transcribe_wav(&wav),
                Err(e) => return h.set_err(e),
            };
            match res {
                Ok(tr) => {
                    h.out_json = transcript_to_json(&tr);
                    0
                }
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_push(handle: i32, f32_ptr: usize, f32_len: usize) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let pcm = h.read_f32(f32_ptr, f32_len);
            let res = match h.get_inst_mut(handle) {
                Ok(inst) => inst.push(&pcm),
                Err(e) => return h.set_err(e),
            };
            match res {
                Ok(events) => {
                    h.out_json = events_to_json(&events);
                    events.len() as i32
                }
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_flush(handle: i32) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let res = match h.get_inst_mut(handle) {
                Ok(inst) => inst.flush(),
                Err(e) => return h.set_err(e),
            };
            match res {
                Ok(events) => {
                    h.out_json = events_to_json(&events);
                    events.len() as i32
                }
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_teach_next(handle: i32, ptr: usize, len: usize) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let label = match h.read_str(ptr, len) {
                Ok(s) => s,
                Err(e) => return h.set_err(e),
            };
            let inst = match h.get_inst_mut(handle) {
                Ok(i) => i,
                Err(e) => return h.set_err(e),
            };
            if label.trim().is_empty() {
                inst.cancel_teach();
            } else {
                inst.teach_next(&label);
            }
            0
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_teaching(handle: i32) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let label = match h.get_inst(handle) {
                Ok(inst) => inst.teaching().map(str::to_string),
                Err(e) => return h.set_err(e),
            };
            match label {
                Some(s) => {
                    h.out_json = s;
                    1
                }
                None => {
                    h.out_json.clear();
                    0
                }
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_teach_pcm(
        handle: i32,
        label_ptr: usize,
        label_len: usize,
        f32_ptr: usize,
        f32_len: usize,
        sample_rate: u32,
    ) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let label = match h.read_str(label_ptr, label_len) {
                Ok(s) => s,
                Err(e) => return h.set_err(e),
            };
            let pcm = h.read_f32(f32_ptr, f32_len);
            let inst = match h.get_inst_mut(handle) {
                Ok(i) => i,
                Err(e) => return h.set_err(e),
            };
            let sr = if sample_rate == 0 {
                inst.sample_rate()
            } else {
                sample_rate
            };
            i32::from(inst.teach(&label, &pcm, sr))
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_teach_wav(
        handle: i32,
        label_ptr: usize,
        label_len: usize,
        wav_ptr: usize,
        wav_len: usize,
    ) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let label = match h.read_str(label_ptr, label_len) {
                Ok(s) => s,
                Err(e) => return h.set_err(e),
            };
            let wav = h.read_bytes(wav_ptr, wav_len);
            let res = match h.get_inst_mut(handle) {
                Ok(inst) => inst.teach_wav(&label, &wav),
                Err(e) => return h.set_err(e),
            };
            match res {
                Ok(ok) => i32::from(ok),
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_recognize_pcm(
        handle: i32,
        f32_ptr: usize,
        f32_len: usize,
        sample_rate: u32,
    ) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let pcm = h.read_f32(f32_ptr, f32_len);
            let matched = {
                let inst = match h.get_inst(handle) {
                    Ok(i) => i,
                    Err(e) => return h.set_err(e),
                };
                let sr = if sample_rate == 0 {
                    inst.sample_rate()
                } else {
                    sample_rate
                };
                inst.recognize_command(&pcm, sr)
            };
            match matched {
                Some(m) => {
                    let mut out = String::with_capacity(160);
                    push_heard_json(&mut out, &m);
                    h.out_json = out;
                    1
                }
                None => {
                    h.out_json = "null".into();
                    0
                }
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_recognize_wav(handle: i32, wav_ptr: usize, wav_len: usize) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let wav = h.read_bytes(wav_ptr, wav_len);
            let res = match h.get_inst(handle) {
                Ok(inst) => inst.recognize_command_wav(&wav),
                Err(e) => return h.set_err(e),
            };
            match res {
                Ok(Some(m)) => {
                    let mut out = String::with_capacity(160);
                    push_heard_json(&mut out, &m);
                    h.out_json = out;
                    1
                }
                Ok(None) => {
                    h.out_json = "null".into();
                    0
                }
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_commands_json(handle: i32) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let json = {
                let inst = match h.get_inst(handle) {
                    Ok(i) => i,
                    Err(e) => return h.set_err(e),
                };
                let mut out = String::from("[");
                for (i, label) in inst.command_labels().iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str("{\"label\":");
                    push_json_str(&mut out, label);
                    out.push_str(",\"count\":");
                    out.push_str(&inst.command_count(label).to_string());
                    out.push('}');
                }
                out.push(']');
                out
            };
            h.out_json = json;
            0
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_remove_command(handle: i32, ptr: usize, len: usize) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let label = match h.read_str(ptr, len) {
                Ok(s) => s,
                Err(e) => return h.set_err(e),
            };
            match h.get_inst_mut(handle) {
                Ok(inst) => {
                    inst.remove_command(&label);
                    0
                }
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_clear_commands(handle: i32) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            match h.get_inst_mut(handle) {
                Ok(inst) => {
                    inst.clear_commands();
                    0
                }
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_export_commands(handle: i32) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let bytes = match h.get_inst(handle) {
                Ok(inst) => inst.export_commands(),
                Err(e) => return h.set_err(e),
            };
            let len = bytes.len() as i32;
            h.out_bytes = bytes;
            len
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_import_commands(handle: i32, ptr: usize, len: usize) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let bytes = h.read_bytes(ptr, len);
            match h.get_inst_mut(handle) {
                Ok(inst) => i32::from(inst.import_commands(&bytes)),
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_decode_wav(wav_ptr: usize, wav_len: usize) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let wav = h.read_bytes(wav_ptr, wav_len);
            match crate::wav::decode_wav(&wav) {
                Ok(audio) => {
                    let mut meta = String::with_capacity(64);
                    meta.push_str("{\"sampleRate\":");
                    meta.push_str(&audio.sample_rate.to_string());
                    meta.push_str(",\"duration\":");
                    push_json_num(&mut meta, audio.duration());
                    meta.push('}');
                    h.out_json = meta;
                    h.out_samples = audio.samples;
                    0
                }
                Err(e) => h.set_err(e),
            }
        })
    }

    #[no_mangle]
    pub extern "C" fn stt_encode_wav(f32_ptr: usize, f32_len: usize, sample_rate: u32) -> i32 {
        HOST.with(|h| {
            let mut h = h.borrow_mut();
            let pcm = h.read_f32(f32_ptr, f32_len);
            h.out_bytes = crate::wav::encode_wav(&pcm, sample_rate.max(1000));
            0
        })
    }
}
