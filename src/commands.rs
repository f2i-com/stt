//! Spoken-command recognition by enrolment: the user says each command once or more,
//! and later speech is matched against those recordings via Dynamic Time Warping.
//! Zero neural networks, zero training data.

use crate::dtw::{dtw, DtwOpts, DtwScratch};
use crate::features::{assemble, endpoints, Feat, FeatConfig};
use crate::frontend::{FrontEnd, FrontEndConfig};

/// Shortest stretch accepted as a word, in 10 ms frames. The detected span includes
/// 30 ms of padding at each end and each frame looks at 25 ms of audio, so this asks
/// for roughly 60 ms of actual sound: shorter than any word, longer than a click.
const MIN_FRAMES: usize = 14;

/// Bounds-checked reader over saved bytes.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Match {
    /// Index into [`CommandSet::labels`].
    pub label: usize,
    /// Distance to the closest recording of that command (lower is closer).
    pub distance: f32,
    /// Closest different command and its distance, if there is one.
    pub runner_up: Option<(usize, f32)>,
    /// False when the match is too weak or too ambiguous to act on.
    pub accepted: bool,
}

pub struct CommandSet {
    sample_rate: u32,
    front_end: FrontEnd,
    feat: FeatConfig,
    dtw: DtwOpts,
    labels: Vec<String>,
    templates: Vec<(usize, Feat)>,
    /// A match is rejected when its distance exceeds this.
    pub max_distance: f32,
    /// A match is rejected when `distance / runner-up distance` exceeds this.
    pub max_ratio: f32,
}

impl CommandSet {
    pub fn new(sample_rate: u32) -> CommandSet {
        // The analysed band stops below the Nyquist frequency of slow sample rates.
        let f_hi = (sample_rate as f32 * 0.425).min(6000.0);
        let cfg = FrontEndConfig { f_hi, ..Default::default() };
        CommandSet {
            sample_rate,
            front_end: FrontEnd::new(sample_rate, &cfg),
            feat: FeatConfig { delta_weight: 0.5, ..Default::default() },
            dtw: DtwOpts::default(),
            labels: Vec::new(),
            templates: Vec::new(),
            max_distance: 4.6,
            max_ratio: 0.92,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Number of recordings stored for a command.
    pub fn count(&self, label: &str) -> usize {
        match self.labels.iter().position(|l| l == label) {
            Some(i) => self.templates.iter().filter(|t| t.0 == i).count(),
            None => 0,
        }
    }

    pub fn len(&self) -> usize {
        self.templates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Features of the spoken part of a clip, or `None` if it holds no usable speech.
    pub fn features(&self, samples: &[f32]) -> Option<Feat> {
        let frames = self.front_end.analyze(samples);
        let (start, end) = endpoints(&frames.level_db)?;
        if end - start < MIN_FRAMES {
            return None;
        }
        Some(assemble(&frames, start..end, &self.feat))
    }

    /// Stores a recording of `label`. Returns false if the clip holds no usable speech.
    pub fn enroll(&mut self, label: &str, samples: &[f32]) -> bool {
        let Some(feat) = self.features(samples) else { return false };
        let index = match self.labels.iter().position(|l| l == label) {
            Some(i) => i,
            None => {
                self.labels.push(label.to_string());
                self.labels.len() - 1
            }
        };
        self.templates.push((index, feat));
        true
    }

    /// Forgets every recording of `label`.
    pub fn remove(&mut self, label: &str) {
        if let Some(index) = self.labels.iter().position(|l| l == label) {
            self.labels.remove(index);
            self.templates.retain(|t| t.0 != index);
            for t in self.templates.iter_mut() {
                if t.0 > index {
                    t.0 -= 1;
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.labels.clear();
        self.templates.clear();
    }

    /// Matches a clip against the enrolled commands. `None` if the clip holds no usable
    /// speech or nothing is enrolled.
    pub fn recognize(&self, samples: &[f32]) -> Option<Match> {
        let feat = self.features(samples)?;
        self.match_features(&feat)
    }

    /// Frame-count range `(min_frames, max_frames)` across enrolled templates of `label_idx`.
    pub fn template_frame_range(&self, label_idx: usize) -> Option<(usize, usize)> {
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for &(idx, ref feat) in &self.templates {
            if idx == label_idx {
                lo = lo.min(feat.n);
                hi = hi.max(feat.n);
            }
        }
        if lo <= hi {
            Some((lo, hi))
        } else {
            None
        }
    }

    pub fn match_features(&self, feat: &Feat) -> Option<Match> {
        if self.templates.is_empty() {
            return None;
        }
        let mut scratch = DtwScratch::default();
        let mut best = vec![f32::INFINITY; self.labels.len()];
        for (label, template) in &self.templates {
            let d = dtw(feat, template, self.dtw, &mut scratch);
            if d < best[*label] {
                best[*label] = d;
            }
        }
        let mut order: Vec<usize> = (0..best.len()).collect();
        order.sort_by(|&a, &b| best[a].total_cmp(&best[b]));
        let label = order[0];
        let distance = best[label];
        if !distance.is_finite() {
            return None;
        }
        let runner_up = order.get(1).map(|&i| (i, best[i]));
        let ratio_ok = runner_up.map(|(_, d)| distance <= d * self.max_ratio).unwrap_or(true);
        Some(Match {
            label,
            distance,
            runner_up,
            accepted: distance <= self.max_distance && ratio_ok,
        })
    }

    /// Serialises the enrolled commands (for example to persist them between sessions).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"STTC");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&self.front_end.cfg.f_hi.to_le_bytes());
        out.extend_from_slice(&(self.labels.len() as u32).to_le_bytes());
        for label in &self.labels {
            out.extend_from_slice(&(label.len() as u32).to_le_bytes());
            out.extend_from_slice(label.as_bytes());
        }
        out.extend_from_slice(&(self.templates.len() as u32).to_le_bytes());
        for (label, feat) in &self.templates {
            out.extend_from_slice(&(*label as u32).to_le_bytes());
            out.extend_from_slice(&(feat.n as u32).to_le_bytes());
            out.extend_from_slice(&(feat.dim as u32).to_le_bytes());
            for v in &feat.data {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Replaces the enrolled commands with a set saved by [`CommandSet::to_bytes`].
    /// Fails (leaving the set unchanged) if the data is malformed or was recorded with a
    /// different analysis band.
    pub fn load_bytes(&mut self, bytes: &[u8]) -> bool {
        type Saved = (Vec<String>, Vec<(usize, Feat)>);
        fn parse(bytes: &[u8], f_hi: f32) -> Option<Saved> {
            let mut cur = Cursor { bytes, pos: 0 };
            let magic = cur.take(4)?;
            if (magic != b"STTC" && magic != b"VRCS") || cur.u32()? != 1 {
                return None;
            }
            if (cur.f32()? - f_hi).abs() > 1.0 {
                return None;
            }
            let n_labels = cur.u32()? as usize;
            let mut labels = Vec::new();
            for _ in 0..n_labels {
                let len = cur.u32()? as usize;
                labels.push(String::from_utf8(cur.take(len)?.to_vec()).ok()?);
            }
            let n_templates = cur.u32()? as usize;
            let mut templates = Vec::new();
            for _ in 0..n_templates {
                let label = cur.u32()? as usize;
                let n = cur.u32()? as usize;
                let dim = cur.u32()? as usize;
                if label >= labels.len() || dim == 0 || dim > 512 {
                    return None;
                }
                let raw = cur.take(n.checked_mul(dim)?.checked_mul(4)?)?;
                let data = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                templates.push((label, Feat { n, dim, data }));
            }
            if cur.pos != bytes.len() {
                return None;
            }
            Some((labels, templates))
        }
        match parse(bytes, self.front_end.cfg.f_hi) {
            Some((labels, templates)) => {
                self.labels = labels;
                self.templates = templates;
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic "word": a sequence of tones, each 80 ms, surrounded by silence.
    fn word(sr: u32, tones: &[f32], stretch: f32) -> Vec<f32> {
        let mut x = vec![0.0f32; sr as usize / 5];
        let per = (sr as f32 * 0.08 * stretch) as usize;
        let mut phase = 0.0f32;
        for &hz in tones {
            for _ in 0..per {
                phase += 2.0 * std::f32::consts::PI * hz / sr as f32;
                x.push(0.3 * phase.sin() + 0.1 * (2.3 * phase).sin());
            }
        }
        x.extend(std::iter::repeat(0.0).take(sr as usize / 5));
        let mut seed = 1u32;
        for v in x.iter_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v += 0.0005 * ((seed >> 8) as f32 / 8_388_608.0 - 1.0);
        }
        x
    }

    #[test]
    fn enrolled_words_are_recognised_at_other_speeds() {
        let sr = 16_000;
        let mut set = CommandSet::new(sr);
        let a = [400.0, 1200.0, 700.0, 2000.0];
        let b = [2000.0, 700.0, 1200.0, 400.0];
        let c = [900.0, 900.0, 3000.0, 500.0];
        assert!(set.enroll("alpha", &word(sr, &a, 1.0)));
        assert!(set.enroll("beta", &word(sr, &b, 1.0)));
        assert!(set.enroll("gamma", &word(sr, &c, 1.0)));
        assert_eq!(set.labels(), ["alpha", "beta", "gamma"]);
        for (tones, want) in [(a, 0), (b, 1), (c, 2)] {
            for stretch in [0.8, 1.0, 1.3] {
                let m = set.recognize(&word(sr, &tones, stretch)).unwrap();
                assert_eq!(m.label, want, "stretch {stretch}");
                assert!(m.accepted, "{m:?}");
            }
        }
        assert!(set.recognize(&vec![0.0; 16_000]).is_none());
        assert!(!set.enroll("nothing", &vec![0.0; 16_000]));
    }

    #[test]
    fn save_load_and_remove() {
        let sr = 16_000;
        let mut set = CommandSet::new(sr);
        set.enroll("one", &word(sr, &[500.0, 1500.0, 800.0], 1.0));
        set.enroll("two", &word(sr, &[1800.0, 600.0, 2500.0], 1.0));
        set.enroll("two", &word(sr, &[1800.0, 600.0, 2500.0], 1.2));
        assert_eq!(set.count("two"), 2);
        let bytes = set.to_bytes();

        let mut copy = CommandSet::new(sr);
        assert!(copy.load_bytes(&bytes));
        assert_eq!(copy.labels(), set.labels());
        assert_eq!(copy.len(), 3);
        let probe = word(sr, &[1800.0, 600.0, 2500.0], 0.9);
        assert_eq!(copy.recognize(&probe), set.recognize(&probe));

        assert!(!copy.load_bytes(&bytes[..bytes.len() - 3]));
        assert!(!copy.load_bytes(b"junk"));
        assert_eq!(copy.len(), 3);
        assert!(!CommandSet::new(8_000).load_bytes(&bytes));

        copy.remove("one");
        assert_eq!(copy.labels(), ["two"]);
        assert_eq!(copy.recognize(&probe).unwrap().label, 0);
        copy.clear();
        assert!(copy.recognize(&probe).is_none());
    }
}
