//! Audio features in exactly the form the model was trained on (NVIDIA NeMo's
//! `AudioToMelSpectrogramPreprocessor`): pre-emphasis, 25 ms Hann frames every 10 ms,
//! 512-point power spectrum, 80 mel bands, log, then every band normalised to zero mean
//! and unit variance over the utterance.

use crate::fft::Fft;

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_MELS: usize = 80;
const N_FFT: usize = 512;
const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;
const PREEMPHASIS: f32 = 0.97;
/// Added before the logarithm (2^-24).
const LOG_GUARD: f32 = 5.960_464_5e-8;

pub struct MelFrontEnd {
    /// `N_MELS × BINS` filterbank taken from the checkpoint.
    filterbank: Vec<f32>,
    /// The checkpoint's analysis window, centred in an `N_FFT` frame.
    window: Vec<f32>,
    fft: Fft,
}

impl MelFrontEnd {
    pub fn new(filterbank: Vec<f32>, window: &[f32]) -> Result<MelFrontEnd, String> {
        if filterbank.len() != N_MELS * BINS || window.len() > N_FFT {
            return Err("unexpected filterbank or window size".into());
        }
        let mut padded = vec![0.0f32; N_FFT];
        let offset = (N_FFT - window.len()) / 2;
        padded[offset..offset + window.len()].copy_from_slice(window);
        Ok(MelFrontEnd { filterbank, window: padded, fft: Fft::new(N_FFT) })
    }

    /// Number of feature frames for `n` samples.
    pub fn frames_for(n: usize) -> usize {
        if n <= N_FFT / 2 {
            0
        } else {
            n.div_ceil(HOP)
        }
    }

    /// Normalised log-mel features, `frames × N_MELS`, for 16 kHz samples in -1..1.
    pub fn features(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let n = samples.len();
        let frames = Self::frames_for(n);
        if frames == 0 {
            return (Vec::new(), 0);
        }
        let mut emphasised = Vec::with_capacity(n);
        emphasised.push(samples[0]);
        for i in 1..n {
            emphasised.push(samples[i] - PREEMPHASIS * samples[i - 1]);
        }
        // Frames are centred on multiples of the hop; the ends are mirrored.
        let at = |i: isize| -> f32 {
            let i = if i < 0 { -i } else if i >= n as isize { 2 * (n as isize - 1) - i } else { i };
            emphasised[i.clamp(0, n as isize - 1) as usize]
        };
        let mut out = vec![0.0f32; frames * N_MELS];
        let mut re = vec![0.0f32; N_FFT];
        let mut im = vec![0.0f32; N_FFT];
        let mut power = vec![0.0f32; BINS];
        for t in 0..frames {
            let start = (t * HOP) as isize - (N_FFT / 2) as isize;
            for j in 0..N_FFT {
                re[j] = at(start + j as isize) * self.window[j];
            }
            im.fill(0.0);
            self.fft.forward(&mut re, &mut im);
            for (k, p) in power.iter_mut().enumerate() {
                *p = re[k] * re[k] + im[k] * im[k];
            }
            let row = &mut out[t * N_MELS..(t + 1) * N_MELS];
            for (m, value) in row.iter_mut().enumerate() {
                let filter = &self.filterbank[m * BINS..(m + 1) * BINS];
                let energy: f32 = filter.iter().zip(&power).map(|(f, p)| f * p).sum();
                *value = (energy + LOG_GUARD).ln();
            }
        }
        // Per-band normalisation over the utterance (sample standard deviation).
        for m in 0..N_MELS {
            let mut mean = 0.0f64;
            for t in 0..frames {
                mean += out[t * N_MELS + m] as f64;
            }
            mean /= frames as f64;
            let mut var = 0.0f64;
            for t in 0..frames {
                let d = out[t * N_MELS + m] as f64 - mean;
                var += d * d;
            }
            let std = (var / (frames.max(2) - 1) as f64).sqrt() + 1e-5;
            for t in 0..frames {
                out[t * N_MELS + m] = ((out[t * N_MELS + m] as f64 - mean) / std) as f32;
            }
        }
        (out, frames)
    }
}
