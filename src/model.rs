//! Phone table: for every phone, three successive states (start, middle, end), each
//! described by the average feature vector and its spread. Words are scored by lining
//! their phone states up against the audio.

use crate::features::Feat;
use crate::phones::N_PHONES;

pub const STATES_PER_PHONE: usize = 3;
pub const N_STATES: usize = N_PHONES * STATES_PER_PHONE;

const NEG: f32 = -1.0e30;

#[derive(Clone, Debug)]
pub struct PhoneTable {
    pub dim: usize,
    /// `N_STATES × dim`
    pub mean: Vec<f32>,
    /// `N_STATES × dim`
    pub inv_var: Vec<f32>,
    pub log_norm: Vec<f32>,
    /// Log probability of remaining in a state for another frame, and of leaving it.
    pub log_stay: Vec<f32>,
    pub log_go: Vec<f32>,
}

impl PhoneTable {
    /// Builds a table from per-state means, variances and average state durations (frames).
    pub fn from_stats(dim: usize, mean: Vec<f32>, var: Vec<f32>, frames_per_state: &[f32]) -> PhoneTable {
        assert_eq!(mean.len(), N_STATES * dim);
        assert_eq!(var.len(), N_STATES * dim);
        let inv_var: Vec<f32> = var.iter().map(|v| 1.0 / v.max(1e-6)).collect();
        let log_norm = (0..N_STATES)
            .map(|s| {
                -0.5 * var[s * dim..(s + 1) * dim]
                    .iter()
                    .map(|v| (2.0 * std::f32::consts::PI * v.max(1e-6)).ln())
                    .sum::<f32>()
            })
            .collect();
        let stay: Vec<f32> = frames_per_state.iter().map(|d| 1.0 - 1.0 / d.max(1.05)).collect();
        PhoneTable {
            dim,
            mean,
            inv_var,
            log_norm,
            log_stay: stay.iter().map(|p| p.ln()).collect(),
            log_go: stay.iter().map(|p| (1.0 - p).ln()).collect(),
        }
    }

    #[inline]
    pub fn loglik(&self, state: usize, x: &[f32]) -> f32 {
        let d = self.dim;
        let mean = &self.mean[state * d..(state + 1) * d];
        let inv = &self.inv_var[state * d..(state + 1) * d];
        let mut acc = 0.0f32;
        for k in 0..d {
            let diff = x[k] - mean[k];
            acc += diff * diff * inv[k];
        }
        self.log_norm[state] - 0.5 * acc
    }

    /// Log-likelihood of every state for every frame (`n × N_STATES`).
    pub fn emissions(&self, feat: &Feat) -> Vec<f32> {
        assert_eq!(feat.dim, self.dim);
        let mut out = vec![0.0f32; feat.n * N_STATES];
        for t in 0..feat.n {
            let x = feat.row(t);
            for s in 0..N_STATES {
                out[t * N_STATES + s] = self.loglik(s, x);
            }
        }
        out
    }

    /// Serialises to little-endian bytes: `dim`, then mean, variance and state durations.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"STTP");
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        for v in &self.mean {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.inv_var {
            out.extend_from_slice(&(1.0 / v).to_le_bytes());
        }
        for v in &self.log_go {
            out.extend_from_slice(&(1.0 / v.exp()).to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<PhoneTable> {
        let magic = bytes.get(0..4)?;
        if magic != b"STTP" && magic != b"VRPT" {
            return None;
        }
        let dim = u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?) as usize;
        if dim == 0 || dim > 256 {
            return None;
        }
        let floats: Vec<f32> = bytes
            .get(8..)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let block = N_STATES * dim;
        if floats.len() != 2 * block + N_STATES {
            return None;
        }
        Some(PhoneTable::from_stats(
            dim,
            floats[..block].to_vec(),
            floats[block..2 * block].to_vec(),
            &floats[2 * block..],
        ))
    }
}

/// The three states of a phone, in order.
pub fn phone_states(ph: u8) -> [u16; STATES_PER_PHONE] {
    let base = ph as u16 * STATES_PER_PHONE as u16;
    [base, base + 1, base + 2]
}

/// State sequence for a pronunciation, optionally padded with silence at both ends.
pub fn word_states(phones: &[u8], pad_silence: bool) -> Vec<u16> {
    let mut seq = Vec::with_capacity((phones.len() + 2) * STATES_PER_PHONE);
    if pad_silence {
        seq.extend_from_slice(&phone_states(crate::phones::SIL));
    }
    for &p in phones {
        seq.extend_from_slice(&phone_states(p));
    }
    if pad_silence {
        seq.extend_from_slice(&phone_states(crate::phones::SIL));
    }
    seq
}

/// Reusable buffers for [`align`].
#[derive(Default)]
pub struct AlignScratch {
    prev: Vec<f32>,
    cur: Vec<f32>,
    back: Vec<u8>,
}

/// Best left-to-right alignment of the state sequence `seq` to `n` frames.
///
/// The alignment may start in any of the first `head_skip + 1` states and end in any of
/// the last `tail_skip + 1`, which makes padding silence optional. Returns the total
/// log score, or `None` if the sequence cannot fit. When `path` is given it receives the
/// position in `seq` occupied at every frame.
pub fn align(
    table: &PhoneTable,
    emis: &[f32],
    n: usize,
    seq: &[u16],
    head_skip: usize,
    tail_skip: usize,
    scratch: &mut AlignScratch,
    path: Option<&mut Vec<u16>>,
) -> Option<f32> {
    let len = seq.len();
    if n == 0 || len == 0 {
        return None;
    }
    let want_path = path.is_some();
    scratch.prev.clear();
    scratch.prev.resize(len, NEG);
    scratch.cur.clear();
    scratch.cur.resize(len, NEG);
    if want_path {
        scratch.back.clear();
        scratch.back.resize(n * len, 0);
    }
    for j in 0..=head_skip.min(len - 1) {
        scratch.prev[j] = emis[seq[j] as usize];
    }
    for t in 1..n {
        let row = &emis[t * N_STATES..(t + 1) * N_STATES];
        for j in 0..len {
            let s = seq[j] as usize;
            let stay = scratch.prev[j] + table.log_stay[s];
            let (best, from_prev) = if j > 0 {
                let enter = scratch.prev[j - 1] + table.log_go[seq[j - 1] as usize];
                if enter > stay { (enter, 1u8) } else { (stay, 0u8) }
            } else {
                (stay, 0u8)
            };
            scratch.cur[j] = if best > NEG * 0.5 { best + row[s] } else { NEG };
            if want_path {
                scratch.back[t * len + j] = from_prev;
            }
        }
        std::mem::swap(&mut scratch.prev, &mut scratch.cur);
    }
    let first_end = len - 1 - tail_skip.min(len - 1);
    let (mut end, mut best) = (len - 1, NEG);
    for j in first_end..len {
        if scratch.prev[j] > best {
            best = scratch.prev[j];
            end = j;
        }
    }
    if best <= NEG * 0.5 {
        return None;
    }
    if let Some(path) = path {
        path.clear();
        path.resize(n, 0);
        let mut j = end;
        for t in (0..n).rev() {
            path[t] = j as u16;
            if t > 0 && scratch.back[t * len + j] == 1 {
                j -= 1;
            }
        }
    }
    Some(best)
}

/// A decoded phone with its frame span `[start, end)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhoneSpan {
    pub phone: u8,
    pub start: usize,
    pub end: usize,
}

/// Free phone recognition: the best sequence of phones for the audio with no dictionary.
/// `insertion_penalty` (negative) is paid on entering every phone and controls how
/// readily the decoder splits the audio into more phones.
pub fn phone_loop(table: &PhoneTable, emis: &[f32], n: usize, insertion_penalty: f32) -> Vec<PhoneSpan> {
    if n == 0 {
        return Vec::new();
    }
    let k = STATES_PER_PHONE;
    let mut prev = vec![NEG; N_STATES];
    let mut cur = vec![NEG; N_STATES];
    // back[t][s] = state occupied at t-1.
    let mut back = vec![0u8; n * N_STATES];
    for p in 0..N_PHONES {
        prev[p * k] = emis[p * k] + insertion_penalty;
    }
    for t in 1..n {
        let row = &emis[t * N_STATES..(t + 1) * N_STATES];
        let (mut exit_best, mut exit_state) = (NEG, 0usize);
        for p in 0..N_PHONES {
            let s = p * k + k - 1;
            let v = prev[s] + table.log_go[s];
            if v > exit_best {
                exit_best = v;
                exit_state = s;
            }
        }
        for p in 0..N_PHONES {
            for i in 0..k {
                let s = p * k + i;
                let stay = prev[s] + table.log_stay[s];
                let (enter, from) = if i == 0 {
                    (exit_best + insertion_penalty, exit_state)
                } else {
                    (prev[s - 1] + table.log_go[s - 1], s - 1)
                };
                let (best, src) = if enter > stay { (enter, from) } else { (stay, s) };
                cur[s] = if best > NEG * 0.5 { best + row[s] } else { NEG };
                back[t * N_STATES + s] = src as u8;
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let mut state = (0..N_PHONES)
        .map(|p| p * k + k - 1)
        .max_by(|&a, &b| prev[a].total_cmp(&prev[b]))
        .unwrap_or(k - 1);
    if prev[state] <= NEG * 0.5 {
        state = (0..N_STATES).max_by(|&a, &b| prev[a].total_cmp(&prev[b])).unwrap_or(0);
    }
    let mut spans: Vec<PhoneSpan> = Vec::new();
    let mut end = n;
    for t in (0..n).rev() {
        let before = if t > 0 { back[t * N_STATES + state] as usize } else { usize::MAX };
        let phone_start = state % k == 0 && (t == 0 || before != state);
        if phone_start {
            spans.push(PhoneSpan { phone: (state / k) as u8, start: t, end });
            end = t;
        }
        if t > 0 {
            state = before;
        }
    }
    spans.reverse();
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table in one dimension where phone `p` sits at value `p` in all three states.
    fn toy_table() -> PhoneTable {
        let mean: Vec<f32> = (0..N_STATES).map(|s| (s / STATES_PER_PHONE) as f32).collect();
        PhoneTable::from_stats(1, mean, vec![0.05; N_STATES], &vec![3.0; N_STATES])
    }

    fn feat(values: &[f32]) -> Feat {
        Feat { n: values.len(), dim: 1, data: values.to_vec() }
    }

    #[test]
    fn phone_loop_recovers_a_sequence() {
        let table = toy_table();
        let mut x = Vec::new();
        for (ph, frames) in [(5.0, 6), (12.0, 9), (5.0, 5), (30.0, 7)] {
            x.extend(std::iter::repeat(ph).take(frames));
        }
        let f = feat(&x);
        let spans = phone_loop(&table, &table.emissions(&f), f.n, -2.0);
        let phones: Vec<u8> = spans.iter().map(|s| s.phone).collect();
        assert_eq!(phones, [5, 12, 5, 30]);
        assert_eq!(spans[1], PhoneSpan { phone: 12, start: 6, end: 15 });
        assert_eq!(spans.last().unwrap().end, f.n);
    }

    #[test]
    fn alignment_prefers_the_right_word_and_optional_silence() {
        let table = toy_table();
        let mut x = vec![0.0; 4];
        x.extend(std::iter::repeat(7.0).take(6));
        x.extend(std::iter::repeat(9.0).take(6));
        let f = feat(&x);
        let emis = table.emissions(&f);
        let mut scratch = AlignScratch::default();
        let right = align(&table, &emis, f.n, &word_states(&[7, 9], true), 3, 3, &mut scratch, None).unwrap();
        let wrong = align(&table, &emis, f.n, &word_states(&[9, 7], true), 3, 3, &mut scratch, None).unwrap();
        assert!(right > wrong + 10.0, "{right} vs {wrong}");
        let mut path = Vec::new();
        align(&table, &emis, f.n, &word_states(&[7, 9], true), 3, 3, &mut scratch, Some(&mut path)).unwrap();
        // leading silence used, trailing silence skipped
        assert!(path[0] <= 2 && path[3] <= 2 && path[4] >= 3, "{path:?}");
        assert_eq!(*path.last().unwrap(), 8);
        // too few frames for the states
        assert!(align(&table, &emis, 3, &word_states(&[7, 9], false), 0, 0, &mut scratch, None).is_none());
    }

    #[test]
    fn table_round_trips_through_bytes() {
        let table = toy_table();
        let copy = PhoneTable::from_bytes(&table.to_bytes()).unwrap();
        assert_eq!(copy.dim, 1);
        for (a, b) in table.mean.iter().zip(&copy.mean) {
            assert!((a - b).abs() < 1e-6);
        }
        for (a, b) in table.log_stay.iter().zip(&copy.log_stay) {
            assert!((a - b).abs() < 1e-4);
        }
        assert!(PhoneTable::from_bytes(b"nope").is_none());
    }
}
