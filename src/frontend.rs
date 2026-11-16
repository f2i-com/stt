//! Audio front end: samples → 10 ms frames of log mel-band energies and cepstra.
//!
//! Everything is defined in Hz and milliseconds, so recordings at different sample rates
//! (8 kHz telephone audio, 16 kHz, the synthesiser's 24 kHz, a browser's 48 kHz) produce
//! comparable frames as long as the analysed band fits under each Nyquist frequency.

use crate::fft::Fft;

#[derive(Clone, Debug)]
pub struct FrontEndConfig {
    pub win_ms: f32,
    pub hop_ms: f32,
    pub n_mel: usize,
    /// Analysed band, Hz.
    pub f_lo: f32,
    pub f_hi: f32,
    /// Number of cepstral coefficients kept (c1..=c_n; c0 is dropped).
    pub n_cep: usize,
    pub preemph: f32,
    /// Vocal-tract-length warp: >1 analyses a speaker whose formants sit that much higher
    /// than the reference, mapping them back onto the reference bands.
    pub warp: f32,
}

impl Default for FrontEndConfig {
    fn default() -> Self {
        FrontEndConfig {
            win_ms: 25.0,
            hop_ms: 10.0,
            n_mel: 24,
            f_lo: 100.0,
            f_hi: 6000.0,
            n_cep: 12,
            preemph: 0.97,
            warp: 1.0,
        }
    }
}

fn hz_to_mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

fn mel_to_hz(m: f32) -> f32 {
    700.0 * ((m / 1127.0).exp() - 1.0)
}

struct MelFilter {
    first_bin: usize,
    weights: Vec<f32>,
}

/// Per-frame analysis of one recording.
#[derive(Clone, Debug, Default)]
pub struct Frames {
    pub n: usize,
    pub n_mel: usize,
    pub n_cep: usize,
    pub hop_ms: f32,
    /// `n × n_mel` natural-log band energies.
    pub mel: Vec<f32>,
    /// `n × n_cep` cepstra (c1..).
    pub cep: Vec<f32>,
    /// Frame level in dB (un-emphasised signal), relative to full scale.
    pub level_db: Vec<f32>,
    /// Zero crossings per millisecond.
    pub zcr: Vec<f32>,
}

impl Frames {
    pub fn mel_row(&self, i: usize) -> &[f32] {
        &self.mel[i * self.n_mel..(i + 1) * self.n_mel]
    }

    pub fn cep_row(&self, i: usize) -> &[f32] {
        &self.cep[i * self.n_cep..(i + 1) * self.n_cep]
    }
}

pub struct FrontEnd {
    pub cfg: FrontEndConfig,
    pub sample_rate: u32,
    win: usize,
    hop: usize,
    fft: Fft,
    window: Vec<f32>,
    filters: Vec<MelFilter>,
    /// `n_cep × n_mel`
    dct: Vec<f32>,
    /// Centre frequency of every band on the reference (un-warped) axis, Hz.
    pub band_hz: Vec<f32>,
}

impl FrontEnd {
    pub fn new(sample_rate: u32, cfg: &FrontEndConfig) -> FrontEnd {
        let sr = sample_rate as f32;
        let win = (sr * cfg.win_ms / 1000.0).round().max(2.0) as usize;
        let hop = (sr * cfg.hop_ms / 1000.0).round().max(1.0) as usize;
        let nfft = win.next_power_of_two();
        let window = (0..win)
            .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (win - 1) as f32).cos())
            .collect();

        // Triangular filters equally spaced on the mel scale between f_lo and f_hi; the
        // warp factor scales every edge so a shorter vocal tract maps onto the same bands.
        let (m_lo, m_hi) = (hz_to_mel(cfg.f_lo), hz_to_mel(cfg.f_hi));
        let edges: Vec<f32> = (0..cfg.n_mel + 2)
            .map(|k| mel_to_hz(m_lo + (m_hi - m_lo) * k as f32 / (cfg.n_mel + 1) as f32))
            .collect();
        let band_hz = edges[1..=cfg.n_mel].to_vec();
        let nyquist = sr / 2.0;
        let bin_hz = sr / nfft as f32;
        let mut filters = Vec::with_capacity(cfg.n_mel);
        for k in 0..cfg.n_mel {
            let lo = (edges[k] * cfg.warp).min(nyquist);
            let mid = (edges[k + 1] * cfg.warp).min(nyquist);
            let hi = (edges[k + 2] * cfg.warp).min(nyquist);
            let first_bin = (lo / bin_hz).ceil() as usize;
            let last_bin = ((hi / bin_hz).floor() as usize).min(nfft / 2);
            let mut weights = Vec::new();
            for b in first_bin..=last_bin.max(first_bin) {
                let f = b as f32 * bin_hz;
                let w = if f <= mid {
                    if mid > lo { (f - lo) / (mid - lo) } else { 1.0 }
                } else if hi > mid {
                    (hi - f) / (hi - mid)
                } else {
                    0.0
                };
                weights.push(w.clamp(0.0, 1.0));
            }
            // Equal-area normalisation keeps band levels comparable across FFT sizes.
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
            filters.push(MelFilter { first_bin, weights });
        }

        let mut dct = vec![0.0; cfg.n_cep * cfg.n_mel];
        for c in 0..cfg.n_cep {
            for m in 0..cfg.n_mel {
                dct[c * cfg.n_mel + m] = (std::f32::consts::PI * (c + 1) as f32 * (m as f32 + 0.5)
                    / cfg.n_mel as f32)
                    .cos()
                    * (2.0 / cfg.n_mel as f32).sqrt();
            }
        }

        FrontEnd {
            cfg: cfg.clone(),
            sample_rate,
            win,
            hop,
            fft: Fft::new(nfft),
            window,
            filters,
            dct,
            band_hz,
        }
    }

    pub fn hop_samples(&self) -> usize {
        self.hop
    }

    pub fn win_samples(&self) -> usize {
        self.win
    }

    pub fn analyze(&self, samples: &[f32]) -> Frames {
        let n = if samples.len() >= self.win {
            1 + (samples.len() - self.win) / self.hop
        } else {
            0
        };
        let (n_mel, n_cep) = (self.cfg.n_mel, self.cfg.n_cep);
        let nfft = self.fft.len();
        let mut out = Frames {
            n,
            n_mel,
            n_cep,
            hop_ms: self.cfg.hop_ms,
            mel: vec![0.0; n * n_mel],
            cep: vec![0.0; n * n_cep],
            level_db: vec![0.0; n],
            zcr: vec![0.0; n],
        };
        let mut re = vec![0.0f32; nfft];
        let mut im = vec![0.0f32; nfft];
        let mut power = vec![0.0f32; nfft / 2 + 1];
        // Power normalisation so levels do not depend on window length.
        let win_energy: f32 = self.window.iter().map(|w| w * w).sum();
        let win_ms = self.win as f32 * 1000.0 / self.sample_rate as f32;

        for i in 0..n {
            let start = i * self.hop;
            let frame = &samples[start..start + self.win];

            let mut energy = 0.0f32;
            let mut crossings = 0u32;
            for j in 0..self.win {
                energy += frame[j] * frame[j];
                if j > 0 && (frame[j] >= 0.0) != (frame[j - 1] >= 0.0) {
                    crossings += 1;
                }
            }
            out.level_db[i] = 10.0 * (energy / self.win as f32 + 1e-12).log10();
            out.zcr[i] = crossings as f32 / win_ms;

            for j in 0..self.win {
                let prev = if j > 0 {
                    frame[j - 1]
                } else if start > 0 {
                    samples[start - 1]
                } else {
                    0.0
                };
                re[j] = (frame[j] - self.cfg.preemph * prev) * self.window[j];
            }
            re[self.win..].fill(0.0);
            im.fill(0.0);
            self.fft.forward(&mut re, &mut im);
            for (k, p) in power.iter_mut().enumerate() {
                *p = (re[k] * re[k] + im[k] * im[k]) / win_energy;
            }

            let mel = &mut out.mel[i * n_mel..(i + 1) * n_mel];
            for (m, filt) in self.filters.iter().enumerate() {
                let mut acc = 0.0f32;
                for (j, w) in filt.weights.iter().enumerate() {
                    acc += w * power[filt.first_bin + j];
                }
                mel[m] = (acc + 1e-10).ln();
            }
            let mel = &out.mel[i * n_mel..(i + 1) * n_mel];
            let cep = &mut out.cep[i * n_cep..(i + 1) * n_cep];
            for c in 0..n_cep {
                let row = &self.dct[c * n_mel..(c + 1) * n_mel];
                cep[c] = row.iter().zip(mel).map(|(a, b)| a * b).sum();
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(sr: u32, hz: f32, secs: f32) -> Vec<f32> {
        (0..(sr as f32 * secs) as usize)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * hz * i as f32 / sr as f32).sin())
            .collect()
    }

    fn peak_band(fe: &FrontEnd, x: &[f32]) -> f32 {
        let fr = fe.analyze(x);
        let row = fr.mel_row(fr.n / 2);
        let (best, _) = row
            .iter()
            .enumerate()
            .fold((0, f32::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
        fe.band_hz[best]
    }

    #[test]
    fn same_tone_same_band_at_every_sample_rate() {
        let cfg = FrontEndConfig { f_hi: 3800.0, ..Default::default() };
        let mut bands = Vec::new();
        for sr in [8_000, 16_000, 24_000, 48_000] {
            let fe = FrontEnd::new(sr, &cfg);
            bands.push(peak_band(&fe, &tone(sr, 1500.0, 0.3)));
        }
        assert!(bands.iter().all(|&b| (b - bands[0]).abs() < 1.0), "{bands:?}");
        assert!((bands[0] - 1500.0).abs() < 120.0, "{bands:?}");
    }

    #[test]
    fn warp_maps_a_raised_tone_back() {
        let cfg = FrontEndConfig::default();
        let reference = peak_band(&FrontEnd::new(16_000, &cfg), &tone(16_000, 1500.0, 0.3));
        let warped_cfg = FrontEndConfig { warp: 1.2, ..Default::default() };
        let warped = peak_band(&FrontEnd::new(16_000, &warped_cfg), &tone(16_000, 1800.0, 0.3));
        assert!((reference - warped).abs() < 1.0);
    }

    #[test]
    fn frame_count_and_levels() {
        let fe = FrontEnd::new(16_000, &FrontEndConfig::default());
        let fr = fe.analyze(&tone(16_000, 440.0, 1.0));
        assert_eq!(fr.n, 98);
        // A 0.5-amplitude sine is about -9 dB re full scale.
        assert!((fr.level_db[40] + 9.03).abs() < 0.2, "{}", fr.level_db[40]);
        assert!(fe.analyze(&[0.0; 10]).n == 0);
    }
}
