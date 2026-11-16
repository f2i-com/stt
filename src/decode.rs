//! Connected-word decoding: finds the sequence of dictionary words whose phone states
//! best explain the audio. There is no grammar; any word may follow any other.

use crate::model::{PhoneTable, N_STATES};

const NEG: f32 = -1.0e30;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WordSpan {
    pub word: usize,
    pub start: usize,
    pub end: usize,
}

/// All words laid end to end as one array of states.
pub struct WordNet {
    states: Vec<u16>,
    /// For every position, the index of its word.
    word_of: Vec<u32>,
    /// First position of every word, plus a final sentinel.
    offsets: Vec<usize>,
    /// Score added on entering each word (insertion penalty and/or how common it is).
    entry: Vec<f32>,
}

impl WordNet {
    pub fn new(words: &[Vec<u16>], entry: &[f32]) -> WordNet {
        assert_eq!(words.len(), entry.len());
        let mut states = Vec::new();
        let mut word_of = Vec::new();
        let mut offsets = Vec::with_capacity(words.len() + 1);
        for (w, seq) in words.iter().enumerate() {
            assert!(!seq.is_empty());
            offsets.push(states.len());
            states.extend_from_slice(seq);
            word_of.extend(std::iter::repeat_n(w as u32, seq.len()));
        }
        offsets.push(states.len());
        WordNet { states, word_of, offsets, entry: entry.to_vec() }
    }

    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Best word sequence for `n` frames of state scores (`n × N_STATES`).
    pub fn decode(&self, table: &PhoneTable, emis: &[f32], n: usize) -> Vec<WordSpan> {
        let total = self.states.len();
        if n == 0 || total == 0 {
            return Vec::new();
        }
        let mut prev = vec![NEG; total];
        let mut cur = vec![NEG; total];
        // Frame at which the word instance occupying each position began.
        let mut prev_start = vec![0u32; total];
        let mut cur_start = vec![0u32; total];
        // Best word ending at each frame: (word, its start frame).
        let mut exits: Vec<(u32, u32)> = vec![(0, 0); n];

        for w in 0..self.len() {
            let pos = self.offsets[w];
            prev[pos] = self.entry[w] + emis[self.states[pos] as usize];
        }
        for t in 1..n {
            let row = &emis[t * N_STATES..(t + 1) * N_STATES];
            let (mut exit_best, mut exit_word, mut exit_start) = (NEG, 0u32, 0u32);
            for w in 0..self.len() {
                let last = self.offsets[w + 1] - 1;
                let v = prev[last] + table.log_go[self.states[last] as usize];
                if v > exit_best {
                    exit_best = v;
                    exit_word = w as u32;
                    exit_start = prev_start[last];
                }
            }
            exits[t - 1] = (exit_word, exit_start);
            for pos in 0..total {
                let s = self.states[pos] as usize;
                let w = self.word_of[pos] as usize;
                let stay = prev[pos] + table.log_stay[s];
                let (enter, enter_start) = if pos == self.offsets[w] {
                    (exit_best + self.entry[w], t as u32)
                } else {
                    (prev[pos - 1] + table.log_go[self.states[pos - 1] as usize], prev_start[pos - 1])
                };
                if enter > stay {
                    cur[pos] = if enter > NEG * 0.5 { enter + row[s] } else { NEG };
                    cur_start[pos] = enter_start;
                } else {
                    cur[pos] = if stay > NEG * 0.5 { stay + row[s] } else { NEG };
                    cur_start[pos] = prev_start[pos];
                }
            }
            std::mem::swap(&mut prev, &mut cur);
            std::mem::swap(&mut prev_start, &mut cur_start);
        }
        let (mut best, mut word, mut start) = (NEG, 0u32, 0u32);
        for w in 0..self.len() {
            let last = self.offsets[w + 1] - 1;
            if prev[last] > best {
                best = prev[last];
                word = w as u32;
                start = prev_start[last];
            }
        }
        if best <= NEG * 0.5 {
            return Vec::new();
        }
        let mut spans = vec![WordSpan { word: word as usize, start: start as usize, end: n }];
        while spans.last().map(|s| s.start > 0).unwrap_or(false) {
            let end = spans.last().unwrap().start;
            let (w, s) = exits[end - 1];
            spans.push(WordSpan { word: w as usize, start: s as usize, end });
        }
        spans.reverse();
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::Feat;
    use crate::model::{word_states, STATES_PER_PHONE};

    #[test]
    fn recovers_a_word_sequence() {
        let mean: Vec<f32> = (0..N_STATES).map(|s| (s / STATES_PER_PHONE) as f32).collect();
        let table = PhoneTable::from_stats(1, mean, vec![0.05; N_STATES], &vec![3.0; N_STATES]);
        // words: 0 = [5 12], 1 = [30], 2 = [12 5], 3 = silence
        let words = vec![
            word_states(&[5, 12], false),
            word_states(&[30], false),
            word_states(&[12, 5], false),
            word_states(&[0], false),
        ];
        let net = WordNet::new(&words, &[-1.0, -1.0, -1.0, 0.0]);
        let mut x = Vec::new();
        for (ph, frames) in [(0.0, 4), (30.0, 5), (5.0, 4), (12.0, 6), (0.0, 3), (12.0, 4), (5.0, 5)] {
            x.extend(std::iter::repeat(ph).take(frames));
        }
        let feat = Feat { n: x.len(), dim: 1, data: x };
        let spans = net.decode(&table, &table.emissions(&feat), feat.n);
        let seq: Vec<usize> = spans.iter().map(|s| s.word).collect();
        assert_eq!(seq, [3, 1, 0, 3, 2]);
        assert_eq!(spans[1], WordSpan { word: 1, start: 4, end: 9 });
        assert_eq!(spans.last().unwrap().end, feat.n);
    }
}
