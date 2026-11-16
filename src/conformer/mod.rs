//! Free-form speech recognition with a small Conformer-CTC network.
//!
//! The network architecture and weights are NVIDIA's `stt_en_conformer_ctc_small`
//! (13 million parameters, CC-BY-4.0); this module is an independent Rust implementation
//! of its inference. Audio in, lower-case English text + word timestamps & confidences out.

mod mel;
mod net;
mod weights;

pub use mel::{MelFrontEnd, N_MELS, SAMPLE_RATE};
pub use net::Encoder;

use weights::Weights;

const D_MODEL: usize = 176;
const HEADS: usize = 4;
/// Each encoder output frame spans 4 mel frames (4 × 10 ms = 40 ms).
pub const FRAME_SECONDS: f64 = 0.04;
/// SentencePiece marks the start of a word with this character.
const WORD_START: char = '\u{2581}';

pub struct Conformer {
    front_end: MelFrontEnd,
    encoder: Encoder,
    vocab: Vec<String>,
}

/// One recognised word piece and the output frame (40 ms each) where it was emitted.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub id: usize,
    pub frame: usize,
}

/// One recognised word with its start/end timestamps (seconds) and CTC confidence (`0.0..=1.0`).
#[derive(Clone, Debug, PartialEq)]
pub struct WordTiming {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
}

/// Full transcription of an audio clip or utterance.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub words: Vec<WordTiming>,
    pub duration: f64,
    pub confidence: f64,
}

#[derive(Clone, Copy, Debug)]
struct TokenSpan {
    id: usize,
    start_frame: usize,
    end_frame: usize,
    peak_prob: f32,
}

impl Conformer {
    /// Loads a weight file written by `tools/export_conformer.py`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Conformer, String> {
        let mut weights = Weights::from_bytes(bytes)?;
        let filterbank = weights.take("fb", N_MELS * 257)?;
        let window = weights.take("window", 400)?;
        let vocab = std::mem::take(&mut weights.vocab);
        let classes = vocab.len() + 1;
        Ok(Conformer {
            front_end: MelFrontEnd::new(filterbank, &window)?,
            encoder: Encoder::load(&mut weights, D_MODEL, HEADS, classes)?,
            vocab,
        })
    }

    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    /// Normalised log-mel features (`frames × 80`) for 16 kHz samples.
    pub fn features(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        self.front_end.features(samples)
    }

    /// Log-probabilities over word pieces for precomputed features
    /// (`frames_out × (pieces + 1)`, blank last).
    pub fn log_probs_from_features(&self, features: &[f32], frames: usize) -> (Vec<f32>, usize) {
        self.encoder.log_probs(features, frames)
    }

    fn decode_spans(&self, samples: &[f32]) -> Vec<TokenSpan> {
        let (features, frames) = self.features(samples);
        let (log_probs, t) = self.encoder.log_probs(&features, frames);
        let classes = self.encoder.classes;
        let blank = classes - 1;
        let mut spans: Vec<TokenSpan> = Vec::new();
        let mut previous = blank;
        for frame in 0..t {
            let row = &log_probs[frame * classes..(frame + 1) * classes];
            let best = (0..classes)
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .unwrap_or(blank);
            if best != blank {
                let prob = row[best].exp().clamp(0.0, 1.0);
                if best != previous {
                    spans.push(TokenSpan {
                        id: best,
                        start_frame: frame,
                        end_frame: frame + 1,
                        peak_prob: prob,
                    });
                } else if let Some(last) = spans.last_mut() {
                    last.end_frame = frame + 1;
                    last.peak_prob = last.peak_prob.max(prob);
                }
            }
            previous = best;
        }
        spans
    }

    /// Most likely word pieces (CTC greedy decoding: repeats merged, blanks dropped).
    pub fn pieces(&self, samples: &[f32]) -> Vec<Piece> {
        self.decode_spans(samples)
            .into_iter()
            .map(|s| Piece {
                id: s.id,
                frame: s.start_frame,
            })
            .collect()
    }

    /// Joins word pieces into text.
    pub fn text_of(&self, pieces: &[Piece]) -> String {
        let mut text = String::new();
        for piece in pieces {
            let Some(token) = self.vocab.get(piece.id) else {
                continue;
            };
            if token == "<unk>" {
                continue;
            }
            for ch in token.chars() {
                if ch == WORD_START {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                } else {
                    text.push(ch);
                }
            }
        }
        text
    }

    /// Transcribes one utterance of 16 kHz mono audio (samples in `-1..1`) into lower-case text.
    pub fn transcribe(&self, samples: &[f32]) -> String {
        self.text_of(&self.pieces(samples))
    }

    /// Transcribes 16 kHz mono audio into text, per-word timestamps, and CTC confidences.
    pub fn transcribe_detailed(&self, samples: &[f32]) -> Transcript {
        let duration = samples.len() as f64 / SAMPLE_RATE as f64;
        if samples.is_empty() {
            return Transcript {
                text: String::new(),
                words: Vec::new(),
                duration: 0.0,
                confidence: 0.0,
            };
        }

        let spans = self.decode_spans(samples);
        let mut text = String::new();
        struct RawWord {
            word: String,
            start_frame: usize,
            end_frame: usize,
            prob_sum: f64,
            prob_count: usize,
        }
        let mut raw_words: Vec<RawWord> = Vec::new();
        let mut cur: Option<RawWord> = None;

        for span in &spans {
            let Some(token) = self.vocab.get(span.id) else {
                continue;
            };
            if token == "<unk>" {
                continue;
            }
            let mut touched_cur = false;
            for ch in token.chars() {
                if ch == WORD_START {
                    if let Some(done) = cur.take() {
                        if !done.word.is_empty() {
                            raw_words.push(done);
                        }
                    }
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    touched_cur = false;
                } else {
                    let w = cur.get_or_insert(RawWord {
                        word: String::new(),
                        start_frame: span.start_frame,
                        end_frame: span.end_frame,
                        prob_sum: 0.0,
                        prob_count: 0,
                    });
                    w.word.push(ch);
                    w.end_frame = w.end_frame.max(span.end_frame);
                    touched_cur = true;
                    text.push(ch);
                }
            }
            if touched_cur {
                if let Some(w) = cur.as_mut() {
                    w.prob_sum += span.peak_prob as f64;
                    w.prob_count += 1;
                }
            }
        }
        if let Some(done) = cur.take() {
            if !done.word.is_empty() {
                raw_words.push(done);
            }
        }

        let mut words = Vec::with_capacity(raw_words.len());
        let mut prev_end = 0.0f64;
        for (i, rw) in raw_words.iter().enumerate() {
            let next_start_frame = raw_words
                .get(i + 1)
                .map(|nw| nw.start_frame)
                .unwrap_or(usize::MAX);
            // CTC spikes sit on syllable nuclei; give each word a clean non-overlapping span.
            let end_frame = (rw.end_frame + 1)
                .min(next_start_frame.max(rw.end_frame))
                .max(rw.start_frame + 1);
            let mut start = (rw.start_frame as f64 * FRAME_SECONDS)
                .max(prev_end)
                .min(duration);
            let mut end = (end_frame as f64 * FRAME_SECONDS).min(duration);
            if duration > 0.0 && end <= start {
                start = (duration - 0.02).max(prev_end).min(duration);
                end = duration;
            }
            prev_end = end;
            let conf = if rw.prob_count > 0 {
                (rw.prob_sum / rw.prob_count as f64).clamp(0.0, 1.0)
            } else {
                0.0
            };
            words.push(WordTiming {
                word: rw.word.clone(),
                start,
                end,
                confidence: conf,
            });
        }

        let confidence = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.confidence).sum::<f64>() / words.len() as f64
        };

        Transcript {
            text,
            words,
            duration,
            confidence,
        }
    }
}
