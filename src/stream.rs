//! Live speech detection: turns a continuous microphone stream into separate utterances.
//!
//! The level of every 10 ms block is compared with the background level, taken from the
//! quieter blocks of the last second. An utterance starts when the level stays well
//! above the background, and ends after a short stretch that is either near the
//! background or far below the loudest part of the utterance. A little audio from before
//! the start is kept so soft word onsets are not lost.

use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct SegmenterConfig {
    /// Level above the background that counts as speech starting, dB.
    pub onset_db: f32,
    /// Level above the background under which speech counts as paused, dB.
    pub release_db: f32,
    /// Speech also counts as paused this far below the loudest block of the utterance, dB.
    /// This ends utterances cleanly in rooms whose background level wanders.
    pub drop_db: f32,
    /// Blocks quieter than this are never speech, dB re full scale.
    pub min_level_db: f32,
    /// Blocks quieter than this are digital silence (a muted or gated microphone) and
    /// say nothing about the room, so they do not move the background estimate.
    pub silence_db: f32,
    /// How far back the background estimate looks, ms.
    pub floor_window_ms: f32,
    /// Consecutive loud blocks needed to start (10 ms each).
    pub onset_blocks: usize,
    /// Quiet needed to end an utterance, ms.
    pub hang_ms: f32,
    /// Audio kept from before the start, ms.
    pub pre_roll_ms: f32,
    /// Longest utterance, ms; longer speech is cut here.
    pub max_ms: f32,
    /// Once an utterance is this long, a much shorter pause (`soft_hang_ms`) ends it, so
    /// long speech is split between words rather than at `max_ms`. Dictation uses this;
    /// by default it equals `max_ms` and has no effect.
    pub soft_max_ms: f32,
    pub soft_hang_ms: f32,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        SegmenterConfig {
            onset_db: 15.0,
            release_db: 8.0,
            drop_db: 35.0,
            min_level_db: -60.0,
            silence_db: -85.0,
            floor_window_ms: 1000.0,
            onset_blocks: 3,
            hang_ms: 350.0,
            pre_roll_ms: 250.0,
            max_ms: 5000.0,
            soft_max_ms: 5000.0,
            soft_hang_ms: 150.0,
        }
    }
}

/// One detected utterance.
#[derive(Clone, Debug)]
pub struct Utterance {
    pub samples: Vec<f32>,
    /// Position of the first sample in the stream, in samples since the start.
    pub start_sample: u64,
}

pub struct Segmenter {
    cfg: SegmenterConfig,
    block: usize,
    pending: Vec<f32>,
    pre_roll: VecDeque<f32>,
    pre_roll_len: usize,
    capture: Vec<f32>,
    capture_start: u64,
    position: u64,
    recent: VecDeque<f32>,
    recent_len: usize,
    floor_db: f32,
    loud_run: usize,
    quiet_run: usize,
    speaking: bool,
    peak_db: f32,
    level_db: f32,
    hang_blocks: usize,
    max_samples: usize,
    soft_hang_blocks: usize,
    soft_max_samples: usize,
}

impl Segmenter {
    pub fn new(sample_rate: u32, cfg: SegmenterConfig) -> Segmenter {
        let block = (sample_rate as usize / 100).max(1);
        Segmenter {
            block,
            pending: Vec::with_capacity(block),
            pre_roll: VecDeque::new(),
            pre_roll_len: (sample_rate as f32 * cfg.pre_roll_ms / 1000.0) as usize,
            capture: Vec::new(),
            capture_start: 0,
            position: 0,
            recent: VecDeque::new(),
            recent_len: (cfg.floor_window_ms / 10.0).round().max(1.0) as usize,
            floor_db: cfg.silence_db,
            loud_run: 0,
            quiet_run: 0,
            speaking: false,
            peak_db: -100.0,
            level_db: -100.0,
            hang_blocks: (cfg.hang_ms / 10.0).round().max(1.0) as usize,
            max_samples: (sample_rate as f32 * cfg.max_ms / 1000.0) as usize,
            soft_hang_blocks: (cfg.soft_hang_ms / 10.0).round().max(1.0) as usize,
            soft_max_samples: (sample_rate as f32 * cfg.soft_max_ms / 1000.0) as usize,
            cfg,
        }
    }

    /// Level of the most recent block, dB re full scale.
    pub fn level_db(&self) -> f32 {
        self.level_db
    }

    /// Tracked background level, dB re full scale.
    pub fn floor_db(&self) -> f32 {
        self.floor_db
    }

    pub fn speaking(&self) -> bool {
        self.speaking
    }

    /// Feeds audio; every utterance that completes is appended to `out`.
    pub fn push(&mut self, samples: &[f32], out: &mut Vec<Utterance>) {
        let mut rest = samples;
        while !rest.is_empty() {
            let take = (self.block - self.pending.len()).min(rest.len());
            self.pending.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.pending.len() == self.block {
                let block = std::mem::take(&mut self.pending);
                self.process_block(&block, out);
                self.pending = block;
                self.pending.clear();
            }
        }
    }

    /// Ends the stream: an utterance in progress is emitted as it stands.
    pub fn flush(&mut self, out: &mut Vec<Utterance>) {
        if self.speaking {
            self.finish(out);
        }
    }

    fn finish(&mut self, out: &mut Vec<Utterance>) {
        out.push(Utterance { samples: std::mem::take(&mut self.capture), start_sample: self.capture_start });
        self.speaking = false;
        self.loud_run = 0;
        self.quiet_run = 0;
        self.pre_roll.clear();
    }

    fn process_block(&mut self, block: &[f32], out: &mut Vec<Utterance>) {
        let energy: f32 = block.iter().map(|s| s * s).sum::<f32>() / block.len() as f32;
        let level = 10.0 * (energy + 1e-12).log10();
        self.level_db = level;

        // Background level: the 20th percentile of the last second of block levels. It
        // adapts within a second in both directions, so a fan switching on or off cannot
        // stall detection, and it is not dragged down by one unusually quiet block.
        if level > self.cfg.silence_db {
            self.recent.push_back(level);
            while self.recent.len() > self.recent_len {
                self.recent.pop_front();
            }
            let mut sorted: Vec<f32> = self.recent.iter().cloned().collect();
            sorted.sort_by(|a, b| a.total_cmp(b));
            self.floor_db = sorted[sorted.len() / 5];
        }
        let floor = self.floor_db;

        if self.speaking {
            self.capture.extend_from_slice(block);
            self.peak_db = self.peak_db.max(level);
            if level < (floor + self.cfg.release_db).max(self.peak_db - self.cfg.drop_db) {
                self.quiet_run += 1;
            } else {
                self.quiet_run = 0;
            }
            let hang = if self.capture.len() >= self.soft_max_samples { self.soft_hang_blocks } else { self.hang_blocks };
            if self.quiet_run >= hang || self.capture.len() >= self.max_samples {
                self.finish(out);
            }
        } else {
            self.pre_roll.extend(block.iter().copied());
            while self.pre_roll.len() > self.pre_roll_len {
                self.pre_roll.pop_front();
            }
            if level > floor + self.cfg.onset_db && level > self.cfg.min_level_db {
                self.loud_run += 1;
            } else {
                self.loud_run = 0;
            }
            if self.loud_run >= self.cfg.onset_blocks {
                self.speaking = true;
                self.peak_db = level;
                self.quiet_run = 0;
                self.capture.clear();
                self.capture.extend(self.pre_roll.iter().copied());
                self.capture_start = self.position + block.len() as u64 - self.capture.len() as u64;
            }
        }
        self.position += block.len() as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, amp: f32, seed: &mut u32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                amp * ((*seed >> 8) as f32 / 8_388_608.0 - 1.0)
            })
            .collect()
    }

    #[test]
    fn finds_two_bursts_in_background_noise() {
        let sr = 16_000;
        let mut seed = 7;
        let mut stream = noise(sr, 0.002, &mut seed);
        stream.extend(noise(sr / 2, 0.2, &mut seed));
        stream.extend(noise(sr, 0.002, &mut seed));
        stream.extend(noise(sr / 4, 0.2, &mut seed));
        stream.extend(noise(sr, 0.002, &mut seed));

        // Chunk size must not matter.
        for chunk in [128, 1000, 4096] {
            let mut seg = Segmenter::new(sr as u32, SegmenterConfig::default());
            let mut found = Vec::new();
            for part in stream.chunks(chunk) {
                seg.push(part, &mut found);
            }
            assert_eq!(found.len(), 2, "chunk {chunk}");
            // First burst starts at 1.0 s; the capture begins up to 250 ms earlier.
            let start = found[0].start_sample as f32 / sr as f32;
            assert!((0.74..=1.0).contains(&start), "start {start}");
            let secs = found[0].samples.len() as f32 / sr as f32;
            assert!((0.8..=1.2).contains(&secs), "length {secs}");
            let second = found[1].start_sample as f32 / sr as f32;
            assert!((2.24..=2.5).contains(&second), "second {second}");
            assert!(!seg.speaking());
        }
    }

    #[test]
    fn silence_and_steady_noise_trigger_nothing() {
        let mut seed = 3;
        for amp in [0.0, 0.01, 0.2] {
            let mut seg = Segmenter::new(48_000, SegmenterConfig::default());
            let mut found = Vec::new();
            seg.push(&noise(48_000 * 3, amp, &mut seed), &mut found);
            seg.flush(&mut found);
            assert!(found.is_empty(), "amp {amp}");
        }
    }
}
