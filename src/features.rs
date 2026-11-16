//! Feature vectors for matching: cepstra (+ level) with deltas, normalised per utterance,
//! plus energy-based detection of where the speech starts and ends.

use crate::frontend::Frames;

#[derive(Clone, Debug)]
pub struct FeatConfig {
    /// Append the frame level (dB below the utterance peak) as an extra static feature.
    pub energy: bool,
    /// Append first-order time derivatives.
    pub delta: bool,
    /// Subtract the utterance mean of every dimension (removes microphone and voice tilt).
    pub cmn: bool,
    /// Scale every dimension to unit variance over the utterance.
    pub cvn: bool,
    pub delta_weight: f32,
    pub energy_weight: f32,
}

impl Default for FeatConfig {
    fn default() -> Self {
        FeatConfig {
            energy: true,
            delta: true,
            cmn: true,
            cvn: true,
            delta_weight: 1.0,
            energy_weight: 1.0,
        }
    }
}

/// A sequence of feature vectors (`n × dim`, row-major).
#[derive(Clone, Debug, Default)]
pub struct Feat {
    pub n: usize,
    pub dim: usize,
    pub data: Vec<f32>,
}

impl Feat {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

/// Un-normalised static features (cepstra, optionally level) for every frame.
pub fn statics(fr: &Frames, energy: bool) -> Feat {
    let dim = fr.n_cep + energy as usize;
    let mut data = Vec::with_capacity(fr.n * dim);
    for i in 0..fr.n {
        data.extend_from_slice(fr.cep_row(i));
        if energy {
            data.push(fr.level_db[i] * 0.1);
        }
    }
    Feat { n: fr.n, dim, data }
}

/// Appends regression deltas (±2 frames) to every row.
pub fn with_deltas(x: &Feat) -> Feat {
    let (n, d) = (x.n, x.dim);
    let mut data = vec![0.0f32; n * d * 2];
    let at = |t: isize| -> &[f32] { x.row(t.clamp(0, n as isize - 1) as usize) };
    for t in 0..n {
        let row = &mut data[t * d * 2..(t + 1) * d * 2];
        row[..d].copy_from_slice(x.row(t));
        let t = t as isize;
        let (p1, m1, p2, m2) = (at(t + 1), at(t - 1), at(t + 2), at(t - 2));
        for k in 0..d {
            row[d + k] = ((p1[k] - m1[k]) + 2.0 * (p2[k] - m2[k])) / 10.0;
        }
    }
    Feat { n, dim: d * 2, data }
}

/// Mean and standard deviation of every dimension over the rows in `range`.
pub fn stats(x: &Feat, range: std::ops::Range<usize>) -> (Vec<f32>, Vec<f32>) {
    let d = x.dim;
    let count = range.len().max(1) as f32;
    let mut mean = vec![0.0f32; d];
    let mut var = vec![0.0f32; d];
    for i in range.clone() {
        for (k, v) in x.row(i).iter().enumerate() {
            mean[k] += v;
        }
    }
    for m in mean.iter_mut() {
        *m /= count;
    }
    for i in range {
        for (k, v) in x.row(i).iter().enumerate() {
            var[k] += (v - mean[k]) * (v - mean[k]);
        }
    }
    let std = var.iter().map(|v| (v / count).sqrt().max(1e-3)).collect();
    (mean, std)
}

/// Builds the matching features for frames `range` of a recording, normalised over
/// that same range.
pub fn assemble(fr: &Frames, range: std::ops::Range<usize>, cfg: &FeatConfig) -> Feat {
    assemble_with(fr, range.clone(), range, cfg)
}

/// Like [`assemble`], but the normalisation statistics come from `stats_range` (usually
/// the span that contains speech) while features are produced for `range`.
pub fn assemble_with(
    fr: &Frames,
    range: std::ops::Range<usize>,
    stats_range: std::ops::Range<usize>,
    cfg: &FeatConfig,
) -> Feat {
    let range = range.start.min(fr.n)..range.end.min(fr.n);
    let stats_range = stats_range.start.min(fr.n)..stats_range.end.min(fr.n);
    let mut full = statics(fr, cfg.energy);
    if cfg.delta {
        full = with_deltas(&full);
    }
    let d = full.dim;
    let (mean, std) = stats(&full, stats_range);
    let n_static = fr.n_cep + cfg.energy as usize;
    let mut data = Vec::with_capacity(range.len() * d);
    for i in range.clone() {
        for (k, &v) in full.row(i).iter().enumerate() {
            let mut v = v;
            if cfg.cmn {
                v -= mean[k];
            }
            if cfg.cvn {
                v /= std[k];
            }
            let pos = k % n_static;
            if cfg.energy && pos == n_static - 1 {
                v *= cfg.energy_weight;
            }
            if k >= n_static {
                v *= cfg.delta_weight;
            }
            data.push(v);
        }
    }
    Feat { n: range.len(), dim: d, data }
}

/// Finds the frame span `[start, end)` that contains speech, from frame levels in dB.
///
/// The threshold sits a fixed distance under the loudest part of the recording but never
/// closer than a few dB to the background level, so it works for both quiet studio
/// recordings and clips with audible room noise.
pub fn endpoints(level_db: &[f32]) -> Option<(usize, usize)> {
    let n = level_db.len();
    if n < 4 {
        return None;
    }
    let mut sorted = level_db.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = sorted[n / 20];
    let peak = sorted[(n * 97 / 100).min(n - 1)];
    if peak - floor < 6.0 || peak < -70.0 {
        return None;
    }
    let threshold = (peak - 40.0).max((floor + 8.0).min(peak - 15.0));

    // Runs of at least 3 frames above the threshold; gaps up to 250 ms are bridged
    // (stop closures inside a word are silent).
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        if level_db[i] > threshold {
            let start = i;
            while i < n && level_db[i] > threshold {
                i += 1;
            }
            if i - start >= 3 {
                runs.push((start, i));
            }
        } else {
            i += 1;
        }
    }
    if runs.is_empty() {
        return None;
    }
    // Keep the cluster of runs that carries the most energy.
    let mut clusters: Vec<(usize, usize)> = Vec::new();
    for &(s, e) in &runs {
        match clusters.last_mut() {
            Some(last) if s - last.1 <= 25 => last.1 = e,
            _ => clusters.push((s, e)),
        }
    }
    let weight = |&(s, e): &(usize, usize)| -> f32 {
        level_db[s..e].iter().map(|&l| (l - threshold).max(0.0)).sum()
    };
    let best = clusters
        .iter()
        .max_by(|a, b| weight(a).partial_cmp(&weight(b)).unwrap_or(std::cmp::Ordering::Equal))?;
    Some((best.0.saturating_sub(3), (best.1 + 3).min(n)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_find_a_burst_of_sound() {
        let mut level = vec![-70.0f32; 100];
        for l in level[30..60].iter_mut() {
            *l = -20.0;
        }
        // a stop closure inside the word
        for l in level[42..48].iter_mut() {
            *l = -70.0;
        }
        assert_eq!(endpoints(&level), Some((27, 63)));
        assert_eq!(endpoints(&vec![-70.0; 100]), None);
    }

    #[test]
    fn deltas_of_a_ramp_are_constant() {
        let x = Feat { n: 10, dim: 1, data: (0..10).map(|i| i as f32).collect() };
        let d = with_deltas(&x);
        assert_eq!(d.dim, 2);
        for t in 2..8 {
            assert!((d.row(t)[1] - 1.0).abs() < 1e-6);
        }
    }
}
