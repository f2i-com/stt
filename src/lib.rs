//! # STT (`stt.computer`)
//!
//! Speech recognition and taught voice commands in pure Rust and self-contained WebAssembly:
//! microphone streams, PCM buffers, or `.wav` files go in — English transcripts with word
//! timestamps & confidences, or matched voice commands, come out.
//!
//! Zero external crate dependencies natively and on `wasm32-unknown-unknown`. Compiles to a
//! single self-contained WebAssembly binary (`stt.wasm`) with **zero external imports**, the
//! 13M-parameter Conformer-CTC weights, DTW command matcher, and complete JavaScript `STT`
//! SDK embedded directly inside `stt.wasm`.
//!
//! ```no_run
//! use stt::{STT, SttConfig};
//!
//! let stt = STT::default();
//! let wav = std::fs::read("speech.wav").unwrap();
//! let transcript = stt.transcribe_wav(&wav).unwrap();
//! println!("{} ({:.2}s)", transcript.text, transcript.duration);
//! ```

use std::sync::Arc;
#[cfg(feature = "embedded-model")]
use std::sync::OnceLock;

pub mod commands;
pub mod config;
pub mod conformer;
pub mod decode;
pub mod dictation;
pub mod dtw;
pub mod features;
pub mod fft;
pub mod frontend;
pub mod model;
pub mod phones;
pub mod resample;
pub mod stream;
pub mod wasm;
pub mod wav;

pub use commands::{CommandSet, Match};
pub use config::{Mode, SttConfig};
pub use conformer::{Conformer, Piece, Transcript, WordTiming, FRAME_SECONDS, SAMPLE_RATE};
pub use dictation::{Dictation, Phrase};
pub use dtw::{dtw, DtwOpts, DtwScratch};
pub use features::{assemble, assemble_with, endpoints, Feat, FeatConfig};
pub use frontend::{Frames, FrontEnd, FrontEndConfig};
pub use model::{PhoneSpan, PhoneTable};
pub use resample::Resampler;
pub use stream::{Segmenter, SegmenterConfig, Utterance};
pub use wav::{decode_wav, encode_wav, WavAudio};

/// Whether the built-in 13.7 MB int8 Conformer speech model is compiled into this binary.
pub const HAS_EMBEDDED_MODEL: bool = cfg!(feature = "embedded-model");

#[cfg(feature = "embedded-model")]
const EMBEDDED_MODEL_BYTES: &[u8] = include_bytes!("../models/conformer_small_int8.bin");

#[cfg(feature = "embedded-model")]
static DEFAULT_CONFORMER: OnceLock<Result<Arc<Conformer>, String>> = OnceLock::new();

/// Returns a shared reference to the built-in Conformer speech model, dequantizing once on first use.
#[cfg(feature = "embedded-model")]
pub fn embedded_conformer() -> Result<Arc<Conformer>, String> {
    DEFAULT_CONFORMER
        .get_or_init(|| Conformer::from_bytes(EMBEDDED_MODEL_BYTES).map(Arc::new))
        .clone()
}

/// Result of comparing an utterance against enrolled voice commands.
#[derive(Clone, Debug, PartialEq)]
pub struct CommandMatch {
    /// Closest taught command label, or `None` when no commands have been taught yet.
    pub label: Option<String>,
    /// `false` when the match is too distant or too close to the runner-up to act on.
    pub accepted: bool,
    /// DTW distance to the closest template of `label` (lower is closer).
    pub distance: f32,
    /// Second-closest command label, if at least two commands are enrolled.
    pub runner_up: Option<String>,
    /// DTW distance to `runner_up` (`f32::INFINITY` if none).
    pub runner_up_distance: f32,
    pub start_seconds: f64,
    pub duration_seconds: f64,
}

/// Streaming event emitted by [`STT::push`] and [`STT::flush`].
#[derive(Clone, Debug, PartialEq)]
pub enum SttEvent {
    /// A dictated phrase transcribed by the Conformer-CTC model when the speaker pauses.
    Phrase(Phrase),
    /// An utterance compared against taught voice commands.
    Heard(CommandMatch),
    /// An utterance stored as a new example of `label` after [`STT::teach_next`].
    Taught {
        label: String,
        count: usize,
        start_seconds: f64,
        duration_seconds: f64,
    },
    /// Sound detected above the floor that held no usable speech (a click, cough, or bump).
    Ignored {
        start_seconds: f64,
        duration_seconds: f64,
    },
}

fn stream_rate_for(cfg: &SttConfig) -> u32 {
    match cfg.mode {
        Mode::Commands => cfg.sample_rate,
        Mode::Dictation | Mode::Both => SAMPLE_RATE,
    }
}

fn command_rate_for(cfg: &SttConfig) -> u32 {
    match cfg.mode {
        Mode::Commands => cfg.sample_rate,
        Mode::Dictation | Mode::Both => {
            if cfg.sample_rate < 14_118 {
                cfg.sample_rate
            } else {
                SAMPLE_RATE
            }
        }
    }
}

/// Unified Speech-to-Text (`stt.computer`) engine combining:
/// - **Free-form English dictation** (13M-parameter Conformer-CTC with word timestamps & confidences)
/// - **Taught voice commands** (few-shot enrolment via Dynamic Time Warping, zero neural network)
/// - **Live adaptive speech segmentation** (tracks room noise floor in real time)
pub struct STT {
    config: SttConfig,
    custom_model: Option<Arc<Conformer>>,
    resampler: Resampler,
    segmenter: Segmenter,
    commands: CommandSet,
    teaching: Option<String>,
    scratch: Vec<f32>,
}

impl Default for STT {
    fn default() -> Self {
        Self::new(SttConfig::default())
    }
}

impl STT {
    /// Creates a new `STT` engine with the given configuration (using the built-in embedded
    /// Conformer model when compiled with the default `embedded-model` feature).
    pub fn new(mut config: SttConfig) -> STT {
        config.sanitize();
        let s_rate = stream_rate_for(&config);
        let c_rate = command_rate_for(&config);
        let mut commands = CommandSet::new(c_rate);
        commands.max_distance = config.max_distance;
        commands.max_ratio = config.max_ratio;
        STT {
            resampler: Resampler::new(config.sample_rate, s_rate),
            segmenter: Segmenter::new(s_rate, config.to_segmenter_config()),
            commands,
            teaching: None,
            scratch: Vec::new(),
            config,
            custom_model: None,
        }
    }

    /// Creates a new `STT` engine using custom Conformer weight bytes (`STTW` / `VRCW`).
    pub fn with_model(config: SttConfig, model_bytes: &[u8]) -> Result<STT, String> {
        let model = Arc::new(Conformer::from_bytes(model_bytes)?);
        let mut stt = Self::new(config);
        stt.custom_model = Some(model);
        Ok(stt)
    }

    fn conformer(&self) -> Result<Arc<Conformer>, String> {
        if let Some(m) = &self.custom_model {
            return Ok(Arc::clone(m));
        }
        #[cfg(feature = "embedded-model")]
        {
            return embedded_conformer();
        }
        #[cfg(not(feature = "embedded-model"))]
        {
            Err("no speech model loaded: pass custom model weights or enable the `embedded-model` feature".into())
        }
    }

    /// Pre-loads and validates the Conformer speech model so the first transcription has zero cold-start delay.
    pub fn warmup(&self) -> Result<(), String> {
        self.conformer().map(|_| ())
    }

    pub fn config(&self) -> &SttConfig {
        &self.config
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    pub fn mode(&self) -> Mode {
        self.config.mode
    }

    pub fn speaking(&self) -> bool {
        self.segmenter.speaking()
    }

    pub fn level_db(&self) -> f32 {
        self.segmenter.level_db()
    }

    pub fn floor_db(&self) -> f32 {
        self.segmenter.floor_db()
    }

    /// Updates engine configuration, preserving enrolled commands whenever the analysis band is compatible.
    pub fn set_config(&mut self, mut next: SttConfig) {
        next.sanitize();
        let prev_s_rate = stream_rate_for(&self.config);
        let next_s_rate = stream_rate_for(&next);
        let prev_c_rate = command_rate_for(&self.config);
        let next_c_rate = command_rate_for(&next);

        let rebuild_stream = self.config.sample_rate != next.sample_rate
            || prev_s_rate != next_s_rate
            || self.config.to_segmenter_config_tuple() != next.to_segmenter_config_tuple();

        if rebuild_stream {
            self.resampler = Resampler::new(next.sample_rate, next_s_rate);
            self.segmenter = Segmenter::new(next_s_rate, next.to_segmenter_config());
        }

        if prev_c_rate != next_c_rate {
            let saved = self.commands.to_bytes();
            let mut next_cmds = CommandSet::new(next_c_rate);
            let _ = next_cmds.load_bytes(&saved);
            self.commands = next_cmds;
        }
        self.commands.max_distance = next.max_distance;
        self.commands.max_ratio = next.max_ratio;
        self.config = next;
    }

    /// Resets live streaming buffers without clearing enrolled commands or unloading the model.
    pub fn reset(&mut self) {
        let s_rate = stream_rate_for(&self.config);
        self.resampler = Resampler::new(self.config.sample_rate, s_rate);
        self.segmenter = Segmenter::new(s_rate, self.config.to_segmenter_config());
        self.teaching = None;
        self.scratch.clear();
    }

    // ---- One-shot transcription ------------------------------------------------------------

    /// Transcribes mono `f32` PCM samples at `sample_rate` Hz into text, word timestamps, and confidences.
    pub fn transcribe(&self, samples: &[f32], sample_rate: u32) -> Result<Transcript, String> {
        let model = self.conformer()?;
        let sr = sample_rate.clamp(4_000, 384_000);
        if sr == SAMPLE_RATE {
            return Ok(model.transcribe_detailed(samples));
        }
        let mut resampled = Vec::new();
        let mut r = Resampler::new(sr, SAMPLE_RATE);
        r.process(samples, &mut resampled);
        r.finish(&mut resampled);
        Ok(model.transcribe_detailed(&resampled))
    }

    /// Transcribes mono `f32` PCM samples at the engine's configured `sample_rate` into plain text.
    pub fn transcribe_clip(&self, samples: &[f32]) -> Result<String, String> {
        let model = self.conformer()?;
        if self.config.sample_rate == SAMPLE_RATE {
            return Ok(model.transcribe(samples));
        }
        let mut resampled = Vec::new();
        let mut r = Resampler::new(self.config.sample_rate, SAMPLE_RATE);
        r.process(samples, &mut resampled);
        r.finish(&mut resampled);
        Ok(model.transcribe(&resampled))
    }

    /// Decodes a RIFF `.wav` file (any sample rate, 8/16/24/32-bit PCM or f32, mono or stereo) and transcribes it.
    pub fn transcribe_wav(&self, wav_bytes: &[u8]) -> Result<Transcript, String> {
        let audio = decode_wav(wav_bytes)?;
        self.transcribe(&audio.samples, audio.sample_rate)
    }

    // ---- Taught voice commands -------------------------------------------------------------

    /// Arms teaching mode so the next live utterance in [`STT::push`] is enrolled under `label`.
    pub fn teach_next(&mut self, label: &str) {
        let trimmed = label.trim();
        if trimmed.is_empty() {
            self.teaching = None;
        } else {
            self.teaching = Some(trimmed.to_string());
        }
    }

    pub fn cancel_teach(&mut self) {
        self.teaching = None;
    }

    pub fn teaching(&self) -> Option<&str> {
        self.teaching.as_deref()
    }

    fn resample_for_commands<'a>(&self, samples: &'a [f32], sample_rate: u32, buf: &'a mut Vec<f32>) -> &'a [f32] {
        let sr = sample_rate.clamp(4_000, 384_000);
        let target = self.commands.sample_rate();
        if sr == target {
            samples
        } else {
            buf.clear();
            Resampler::new(sr, target).process(samples, buf);
            buf.as_slice()
        }
    }

    /// Enrolls a whole PCM clip at `sample_rate` Hz as an example of `label`. Returns `false` if no usable speech.
    pub fn teach(&mut self, label: &str, samples: &[f32], sample_rate: u32) -> bool {
        let label = label.trim();
        if label.is_empty() {
            return false;
        }
        let target = self.commands.sample_rate();
        let sr = sample_rate.clamp(4_000, 384_000);
        if sr == target {
            self.commands.enroll(label, samples)
        } else {
            let mut buf = Vec::new();
            Resampler::new(sr, target).process(samples, &mut buf);
            self.commands.enroll(label, &buf)
        }
    }

    /// Enrolls a `.wav` clip as an example of `label`.
    pub fn teach_wav(&mut self, label: &str, wav_bytes: &[u8]) -> Result<bool, String> {
        let audio = decode_wav(wav_bytes)?;
        Ok(self.teach(label, &audio.samples, audio.sample_rate))
    }

    fn format_match(&self, m: &Match, start_seconds: f64, duration_seconds: f64) -> CommandMatch {
        let labels = self.commands.labels();
        let (runner_up, runner_up_distance) = match m.runner_up {
            Some((idx, d)) => (labels.get(idx).cloned(), d),
            None => (None, f32::INFINITY),
        };
        CommandMatch {
            label: labels.get(m.label).cloned(),
            accepted: m.accepted,
            distance: m.distance,
            runner_up,
            runner_up_distance,
            start_seconds,
            duration_seconds,
        }
    }

    /// Matches a whole PCM clip at `sample_rate` Hz against enrolled commands.
    pub fn recognize_command(&self, samples: &[f32], sample_rate: u32) -> Option<CommandMatch> {
        let sr = sample_rate.clamp(4_000, 384_000);
        let secs = samples.len() as f64 / sr as f64;
        let mut buf = Vec::new();
        let pcm = self.resample_for_commands(samples, sr, &mut buf);
        let m = self.commands.recognize(pcm)?;
        Some(self.format_match(&m, 0.0, secs))
    }

    /// Matches a `.wav` file against enrolled commands.
    pub fn recognize_command_wav(&self, wav_bytes: &[u8]) -> Result<Option<CommandMatch>, String> {
        let audio = decode_wav(wav_bytes)?;
        Ok(self.recognize_command(&audio.samples, audio.sample_rate))
    }

    pub fn command_labels(&self) -> &[String] {
        self.commands.labels()
    }

    pub fn command_count(&self, label: &str) -> usize {
        self.commands.count(label)
    }

    pub fn remove_command(&mut self, label: &str) {
        self.commands.remove(label);
    }

    pub fn clear_commands(&mut self) {
        self.commands.clear();
    }

    pub fn export_commands(&self) -> Vec<u8> {
        self.commands.to_bytes()
    }

    pub fn import_commands(&mut self, bytes: &[u8]) -> bool {
        self.commands.load_bytes(bytes)
    }

    // ---- Live streaming (`push` / `flush`) -------------------------------------------------

    fn handle_utterances(&mut self, found: &[Utterance], utt_rate: u32) -> Result<Vec<SttEvent>, String> {
        let mut out = Vec::with_capacity(found.len());
        for u in found {
            let start = u.start_sample as f64 / utt_rate as f64;
            let duration = u.samples.len() as f64 / utt_rate as f64;

            if let Some(label) = self.teaching.clone() {
                if self.teach(&label, &u.samples, utt_rate) {
                    self.teaching = None;
                    let count = self.commands.count(&label);
                    out.push(SttEvent::Taught {
                        label,
                        count,
                        start_seconds: start,
                        duration_seconds: duration,
                    });
                } else {
                    out.push(SttEvent::Ignored {
                        start_seconds: start,
                        duration_seconds: duration,
                    });
                }
                continue;
            }

            match self.config.mode {
                Mode::Commands => {
                    let mut buf = Vec::new();
                    let pcm = self.resample_for_commands(&u.samples, utt_rate, &mut buf);
                    match self.commands.features(pcm) {
                        None => out.push(SttEvent::Ignored {
                            start_seconds: start,
                            duration_seconds: duration,
                        }),
                        Some(feat) => match self.commands.match_features(&feat) {
                            Some(m) => out.push(SttEvent::Heard(self.format_match(&m, start, duration))),
                            None => out.push(SttEvent::Heard(CommandMatch {
                                label: None,
                                accepted: false,
                                distance: f32::INFINITY,
                                runner_up: None,
                                runner_up_distance: f32::INFINITY,
                                start_seconds: start,
                                duration_seconds: duration,
                            })),
                        },
                    }
                }
                Mode::Dictation => {
                    let model = self.conformer()?;
                    let tr = if utt_rate == SAMPLE_RATE {
                        model.transcribe_detailed(&u.samples)
                    } else {
                        let mut resampled = Vec::new();
                        Resampler::new(utt_rate, SAMPLE_RATE).process(&u.samples, &mut resampled);
                        model.transcribe_detailed(&resampled)
                    };
                    if !tr.text.is_empty() {
                        out.push(SttEvent::Phrase(Phrase {
                            text: tr.text,
                            words: tr.words,
                            confidence: tr.confidence,
                            start_seconds: start,
                            duration_seconds: duration,
                        }));
                    }
                }
                Mode::Both => {
                    if !self.commands.is_empty() && duration <= 3.5 {
                        let mut buf = Vec::new();
                        let pcm = self.resample_for_commands(&u.samples, utt_rate, &mut buf);
                        if let Some(feat) = self.commands.features(pcm) {
                            if let Some(m) = self.commands.match_features(&feat) {
                                let length_ok = self
                                    .commands
                                    .template_frame_range(m.label)
                                    .is_some_and(|(lo, hi)| {
                                        let min_f = (lo as f32 * 0.45) as usize;
                                        let max_f = (hi as f32 * 2.25) as usize + 12;
                                        feat.n >= min_f && feat.n <= max_f
                                    });
                                // When only 1 command is enrolled there is no runner-up ratio check,
                                // so require a tighter match before preempting dictation.
                                let single_ok = m.runner_up.is_some()
                                    || m.distance <= (self.commands.max_distance * 0.80).min(3.7);
                                if m.accepted && length_ok && single_ok {
                                    out.push(SttEvent::Heard(self.format_match(&m, start, duration)));
                                    continue;
                                }
                            }
                        }
                    }
                    let model = self.conformer()?;
                    let tr = if utt_rate == SAMPLE_RATE {
                        model.transcribe_detailed(&u.samples)
                    } else {
                        let mut resampled = Vec::new();
                        Resampler::new(utt_rate, SAMPLE_RATE).process(&u.samples, &mut resampled);
                        model.transcribe_detailed(&resampled)
                    };
                    if !tr.text.is_empty() {
                        out.push(SttEvent::Phrase(Phrase {
                            text: tr.text,
                            words: tr.words,
                            confidence: tr.confidence,
                            start_seconds: start,
                            duration_seconds: duration,
                        }));
                    }
                }
            }
        }
        Ok(out)
    }

    /// Feeds mono PCM samples (`-1..1` at `self.sample_rate()`) in chunks of any size and
    /// returns events for all utterances that completed in this chunk.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<SttEvent>, String> {
        let s_rate = stream_rate_for(&self.config);
        let mut found = Vec::new();
        if self.config.sample_rate == s_rate {
            self.segmenter.push(samples, &mut found);
        } else {
            self.scratch.clear();
            self.resampler.process(samples, &mut self.scratch);
            self.segmenter.push(&self.scratch, &mut found);
        }
        self.handle_utterances(&found, s_rate)
    }

    /// Flushes any utterance currently in progress at the end of a stream.
    pub fn flush(&mut self) -> Result<Vec<SttEvent>, String> {
        let s_rate = stream_rate_for(&self.config);
        let mut found = Vec::new();
        self.segmenter.flush(&mut found);
        self.handle_utterances(&found, s_rate)
    }
}

impl SttConfig {
    fn to_segmenter_config_tuple(&self) -> (u32, u32, u32, u32, u32, u32, u32, u32, u32) {
        (
            self.onset_db.to_bits(),
            self.release_db.to_bits(),
            self.drop_db.to_bits(),
            self.min_level_db.to_bits(),
            self.hang_ms.to_bits(),
            self.pre_roll_ms.to_bits(),
            self.max_ms.to_bits(),
            self.soft_max_ms.to_bits(),
            self.soft_hang_ms.to_bits(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_stt_transcribes_wav_and_handles_commands() {
        let stt = STT::default();
        #[cfg(feature = "embedded-model")]
        {
            let wav = std::fs::read(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/henry.wav"
            ))
            .unwrap();
            let tr = stt.transcribe_wav(&wav).unwrap();
            assert_eq!(tr.text, "and henry might return to england at any moment");
            assert_eq!(tr.words.len(), 9);
            assert!(tr.duration > 3.0 && tr.confidence > 0.7);
        }
    }
}
