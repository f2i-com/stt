//! Configuration types for [`crate::STT`].

use crate::stream::SegmenterConfig;

/// Operating mode for streaming audio pushed into [`crate::STT::push`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Free-form English speech-to-text via the Conformer-CTC model.
    Dictation,
    /// Low-latency taught voice commands via Dynamic Time Warping (zero neural network).
    Commands,
    /// Matches taught voice commands on short utterances first; falls back to Conformer dictation.
    Both,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Dictation => "dictation",
            Mode::Commands => "commands",
            Mode::Both => "both",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "dictation" | "dictate" | "transcribe" | "speech" => Some(Mode::Dictation),
            "commands" | "command" | "listener" | "dtw" => Some(Mode::Commands),
            "both" | "hybrid" | "auto" => Some(Mode::Both),
            _ => None,
        }
    }
}

pub const NUMBER_SETTINGS: &[&str] = &[
    "sampleRate",
    "hangMs",
    "preRollMs",
    "maxMs",
    "softMaxMs",
    "softHangMs",
    "onsetDb",
    "releaseDb",
    "dropDb",
    "minLevelDb",
    "maxDistance",
    "maxRatio",
];

/// Full configuration for an [`crate::STT`] engine instance.
#[derive(Clone, Debug, PartialEq)]
pub struct SttConfig {
    /// Input audio sample rate in Hz (`4000..=384000`).
    pub sample_rate: u32,
    /// Streaming recognition mode (`Dictation`, `Commands`, or `Both`).
    pub mode: Mode,
    /// Pause duration required to finish an utterance, in milliseconds.
    pub hang_ms: f32,
    /// Audio kept from before speech onset so soft consonant starts are preserved, in milliseconds.
    pub pre_roll_ms: f32,
    /// Maximum utterance duration before an automatic split, in milliseconds.
    pub max_ms: f32,
    /// Utterance duration after which a brief pause (`soft_hang_ms`) splits the phrase, in milliseconds.
    pub soft_max_ms: f32,
    /// Brief pause duration used once `soft_max_ms` is reached, in milliseconds.
    pub soft_hang_ms: f32,
    /// Level above tracked background floor that triggers speech onset, in dB.
    pub onset_db: f32,
    /// Level above tracked background floor below which speech counts as paused, in dB.
    pub release_db: f32,
    /// Level below utterance peak that also counts as paused, in dB.
    pub drop_db: f32,
    /// Minimum absolute block level to count as speech, in dBFS.
    pub min_level_db: f32,
    /// Maximum DTW distance accepted for a taught command (`lower` = stricter).
    pub max_distance: f32,
    /// Maximum ratio `best_distance / runner_up_distance` accepted for a taught command.
    pub max_ratio: f32,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self::for_mode(Mode::Dictation)
    }
}

impl SttConfig {
    pub fn for_mode(mode: Mode) -> Self {
        match mode {
            Mode::Commands => SttConfig {
                sample_rate: 16_000,
                mode,
                hang_ms: 350.0,
                pre_roll_ms: 250.0,
                max_ms: 5_000.0,
                soft_max_ms: 5_000.0,
                soft_hang_ms: 150.0,
                onset_db: 15.0,
                release_db: 8.0,
                drop_db: 35.0,
                min_level_db: -60.0,
                max_distance: 4.6,
                max_ratio: 0.92,
            },
            Mode::Dictation | Mode::Both => SttConfig {
                sample_rate: 16_000,
                mode,
                hang_ms: 600.0,
                pre_roll_ms: 300.0,
                max_ms: 20_000.0,
                soft_max_ms: 8_000.0,
                soft_hang_ms: 150.0,
                onset_db: 15.0,
                release_db: 8.0,
                drop_db: 35.0,
                min_level_db: -60.0,
                max_distance: 4.6,
                max_ratio: 0.92,
            },
        }
    }

    /// Switches mode and updates timing defaults unless already customized.
    pub fn set_mode(&mut self, mode: Mode) {
        if self.mode == mode {
            return;
        }
        let prev_defaults = Self::for_mode(self.mode);
        let next_defaults = Self::for_mode(mode);
        if (self.hang_ms - prev_defaults.hang_ms).abs() < 1e-4 {
            self.hang_ms = next_defaults.hang_ms;
        }
        if (self.pre_roll_ms - prev_defaults.pre_roll_ms).abs() < 1e-4 {
            self.pre_roll_ms = next_defaults.pre_roll_ms;
        }
        if (self.max_ms - prev_defaults.max_ms).abs() < 1e-4 {
            self.max_ms = next_defaults.max_ms;
        }
        if (self.soft_max_ms - prev_defaults.soft_max_ms).abs() < 1e-4 {
            self.soft_max_ms = next_defaults.soft_max_ms;
        }
        self.mode = mode;
    }

    pub fn sanitize(&mut self) {
        self.sample_rate = self.sample_rate.clamp(4_000, 384_000);
        let clamp_f32 = |v: f32, default: f32, lo: f32, hi: f32| -> f32 {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                default
            }
        };
        self.hang_ms = clamp_f32(self.hang_ms, 600.0, 50.0, 10_000.0);
        self.pre_roll_ms = clamp_f32(self.pre_roll_ms, 300.0, 0.0, 3_000.0);
        self.max_ms = clamp_f32(self.max_ms, 20_000.0, 500.0, 120_000.0);
        self.soft_max_ms = clamp_f32(self.soft_max_ms, 8_000.0, 500.0, self.max_ms);
        self.soft_hang_ms = clamp_f32(self.soft_hang_ms, 150.0, 30.0, self.hang_ms);
        self.onset_db = clamp_f32(self.onset_db, 15.0, 2.0, 60.0);
        self.release_db = clamp_f32(self.release_db, 8.0, 0.5, self.onset_db);
        self.drop_db = clamp_f32(self.drop_db, 35.0, 3.0, 80.0);
        self.min_level_db = clamp_f32(self.min_level_db, -60.0, -100.0, -5.0);
        if !self.max_distance.is_finite() {
            self.max_distance = if self.max_distance.is_sign_positive() {
                1.0e9
            } else {
                0.0
            };
        } else {
            self.max_distance = self.max_distance.max(0.0);
        }
        if !self.max_ratio.is_finite() {
            self.max_ratio = if self.max_ratio.is_sign_positive() {
                1.0e9
            } else {
                0.0
            };
        } else {
            self.max_ratio = self.max_ratio.max(0.0);
        }
    }

    pub fn to_segmenter_config(&self) -> SegmenterConfig {
        SegmenterConfig {
            onset_db: self.onset_db,
            release_db: self.release_db,
            drop_db: self.drop_db,
            min_level_db: self.min_level_db,
            hang_ms: self.hang_ms,
            pre_roll_ms: self.pre_roll_ms,
            max_ms: self.max_ms,
            soft_max_ms: self.soft_max_ms,
            soft_hang_ms: self.soft_hang_ms,
            ..SegmenterConfig::default()
        }
    }
}
