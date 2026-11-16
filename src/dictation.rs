//! Live dictation: microphone audio in, phrases of text + word timings out.
//!
//! Audio is resampled to 16 kHz and split into phrases at pauses; each finished phrase
//! is transcribed by the Conformer model. Text appears a moment after the speaker pauses.

use std::sync::Arc;

use crate::conformer::{Conformer, Transcript, WordTiming, SAMPLE_RATE};
use crate::resample::Resampler;
use crate::stream::{Segmenter, SegmenterConfig, Utterance};

/// A transcribed stretch of speech emitted when the speaker pauses.
#[derive(Clone, Debug, PartialEq)]
pub struct Phrase {
    pub text: String,
    pub words: Vec<WordTiming>,
    pub confidence: f64,
    /// Position in the stream, seconds since the first sample pushed.
    pub start_seconds: f64,
    pub duration_seconds: f64,
}

pub struct Dictation {
    model: Arc<Conformer>,
    resampler: Resampler,
    segmenter: Segmenter,
    input_rate: u32,
    scratch: Vec<f32>,
}

/// Speech-detection settings suited to dictation: a phrase ends after 0.6 s of quiet, or
/// at the next brief pause once it is 8 s long, and never runs past 20 s.
pub fn dictation_segmenter() -> SegmenterConfig {
    SegmenterConfig {
        hang_ms: 600.0,
        pre_roll_ms: 300.0,
        max_ms: 20_000.0,
        soft_max_ms: 8_000.0,
        soft_hang_ms: 150.0,
        ..Default::default()
    }
}

impl Dictation {
    /// Creates a `Dictation` stream using a shared [`Conformer`] model.
    pub fn from_conformer(sample_rate: u32, model: Arc<Conformer>) -> Dictation {
        let sample_rate = sample_rate.clamp(4_000, 384_000);
        Dictation {
            model,
            resampler: Resampler::new(sample_rate, SAMPLE_RATE),
            segmenter: Segmenter::new(SAMPLE_RATE, dictation_segmenter()),
            input_rate: sample_rate,
            scratch: Vec::new(),
        }
    }

    /// `sample_rate` is the rate of the audio that will be pushed; `model` is a weight
    /// file written by `tools/export_conformer.py`.
    pub fn new(sample_rate: u32, model: &[u8]) -> Result<Dictation, String> {
        let conformer = Arc::new(Conformer::from_bytes(model)?);
        Ok(Self::from_conformer(sample_rate, conformer))
    }

    /// Creates a `Dictation` stream backed by the built-in embedded Conformer model.
    #[cfg(feature = "embedded-model")]
    pub fn with_embedded_model(sample_rate: u32) -> Result<Dictation, String> {
        Ok(Self::from_conformer(sample_rate, crate::embedded_conformer()?))
    }

    pub fn sample_rate(&self) -> u32 {
        self.input_rate
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

    fn transcribe_all(&self, utterances: &[Utterance]) -> Vec<Phrase> {
        utterances
            .iter()
            .filter_map(|u| {
                let tr = self.model.transcribe_detailed(&u.samples);
                if tr.text.is_empty() {
                    return None;
                }
                Some(Phrase {
                    text: tr.text,
                    words: tr.words,
                    confidence: tr.confidence,
                    start_seconds: u.start_sample as f64 / SAMPLE_RATE as f64,
                    duration_seconds: u.samples.len() as f64 / SAMPLE_RATE as f64,
                })
            })
            .collect()
    }

    /// Feeds microphone samples (mono, `-1..1`) in chunks of any size; returns the phrases
    /// that finished inside this chunk.
    pub fn push(&mut self, samples: &[f32]) -> Vec<Phrase> {
        self.scratch.clear();
        self.resampler.process(samples, &mut self.scratch);
        let mut found = Vec::new();
        self.segmenter.push(&self.scratch, &mut found);
        self.transcribe_all(&found)
    }

    /// Ends the stream: a phrase still in progress is transcribed as it stands.
    pub fn flush(&mut self) -> Vec<Phrase> {
        let mut found = Vec::new();
        self.segmenter.flush(&mut found);
        self.transcribe_all(&found)
    }

    /// Transcribes a whole recording at the input sample rate in one go, returning text + word timings.
    pub fn transcribe(&self, samples: &[f32]) -> Transcript {
        if self.input_rate == SAMPLE_RATE {
            return self.model.transcribe_detailed(samples);
        }
        let mut resampled = Vec::new();
        let mut r = Resampler::new(self.input_rate, SAMPLE_RATE);
        r.process(samples, &mut resampled);
        r.finish(&mut resampled);
        self.model.transcribe_detailed(&resampled)
    }

    /// Transcribes a whole recording at the input sample rate in one go, returning plain text.
    pub fn transcribe_clip(&self, samples: &[f32]) -> String {
        if self.input_rate == SAMPLE_RATE {
            return self.model.transcribe(samples);
        }
        let mut resampled = Vec::new();
        let mut r = Resampler::new(self.input_rate, SAMPLE_RATE);
        r.process(samples, &mut resampled);
        r.finish(&mut resampled);
        self.model.transcribe(&resampled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::decode_wav;

    const SENTENCE: &str = "and henry might return to england at any moment";

    fn file(name: &str) -> Option<Vec<u8>> {
        std::fs::read(format!("{}/{name}", env!("CARGO_MANIFEST_DIR"))).ok()
    }

    fn room_noise(n: usize, seed: &mut u32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                0.0005 * ((*seed >> 8) as f32 / 8_388_608.0 - 1.0)
            })
            .collect()
    }

    #[test]
    fn dictates_real_speech_whole_and_streamed() {
        let (Some(model), Some(clip)) = (
            file("models/conformer_small_int8.bin"),
            file("tests/data/henry.wav"),
        ) else {
            eprintln!("speech model or test clip not present; skipped");
            return;
        };
        let audio = decode_wav(&clip).unwrap();
        assert_eq!(audio.sample_rate, 16_000);
        let speech = audio.samples;

        let whole = Dictation::new(16_000, &model).unwrap();
        assert_eq!(whole.transcribe_clip(&speech), SENTENCE);
        let detailed = whole.transcribe(&speech);
        assert_eq!(detailed.text, SENTENCE);
        assert_eq!(detailed.words.len(), 9);
        assert!(detailed.confidence > 0.7, "confidence {}", detailed.confidence);
        for (i, w) in detailed.words.iter().enumerate() {
            assert!(w.start < w.end && w.end <= detailed.duration + 1e-6, "{w:?}");
            if i > 0 {
                assert!(detailed.words[i - 1].end <= w.start + 1e-9);
            }
        }
        assert_eq!(whole.transcribe_clip(&[]), "");

        // The same sentence twice at a browser's 48 kHz, with pauses, in small chunks.
        let mut speech48 = Vec::new();
        Resampler::new(16_000, 48_000).process(&speech, &mut speech48);
        let mut seed = 5;
        let mut stream = room_noise(48_000, &mut seed);
        for _ in 0..2 {
            stream.extend_from_slice(&speech48);
            stream.extend(room_noise(72_000, &mut seed));
        }
        let mut live = Dictation::new(48_000, &model).unwrap();
        let mut phrases = Vec::new();
        for chunk in stream.chunks(1024) {
            phrases.extend(live.push(chunk));
        }
        phrases.extend(live.flush());
        let texts: Vec<&str> = phrases.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(texts, [SENTENCE, SENTENCE]);
        assert_eq!(phrases[0].words.len(), 9);
        assert!(
            phrases[0].start_seconds > 0.5 && phrases[0].start_seconds < 1.1,
            "{}",
            phrases[0].start_seconds
        );
        assert!(phrases[1].start_seconds > phrases[0].start_seconds + 3.0);
        assert!(!live.speaking());

        // Silence produces nothing, and bad model data is refused.
        let mut quiet = Dictation::new(16_000, &model).unwrap();
        assert!(quiet.push(&room_noise(48_000, &mut seed)).is_empty());
        assert!(quiet.flush().is_empty());
        assert!(Dictation::new(16_000, b"not a model").is_err());
        assert!(Dictation::new(16_000, &model[..model.len() / 2]).is_err());
    }
}
