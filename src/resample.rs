//! Streaming sample-rate conversion (for example a browser's 48 kHz microphone to the
//! 16 kHz a speech model expects), by windowed-sinc interpolation.

/// Kernel values stored per input sample; positions in between are interpolated.
const OVERSAMPLE: usize = 256;
/// Zero crossings of the sinc kept on each side.
const CROSSINGS: f64 = 14.0;

pub struct Resampler {
    /// Input samples advanced per output sample.
    step: f64,
    /// Kernel half-width in input samples.
    half: usize,
    /// One side of the symmetric kernel, sampled `OVERSAMPLE` times per input sample.
    table: Vec<f32>,
    buf: Vec<f32>,
    /// Position of the next output sample, as an index into `buf`.
    pos: f64,
    passthrough: bool,
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Resampler {
        let passthrough = in_rate == out_rate;
        // Cut-off as a fraction of the input rate, just below the lower Nyquist frequency.
        let cutoff = 0.5 * (out_rate as f64 / in_rate as f64).min(1.0) * 0.92;
        let half = (CROSSINGS / (2.0 * cutoff)).ceil() as usize;
        let n = half * OVERSAMPLE + 1;
        let mut table = Vec::with_capacity(n + 1);
        for i in 0..n {
            let x = i as f64 / OVERSAMPLE as f64;
            let sinc = if x == 0.0 {
                1.0
            } else {
                let a = 2.0 * std::f64::consts::PI * cutoff * x;
                a.sin() / a
            };
            // Blackman window over the kernel's full width.
            let w = std::f64::consts::PI * x / half as f64;
            let window = 0.42 + 0.5 * w.cos() + 0.08 * (2.0 * w).cos();
            table.push((2.0 * cutoff * sinc * window) as f32);
        }
        table.push(0.0);
        Resampler {
            step: in_rate as f64 / out_rate as f64,
            half,
            table,
            buf: vec![0.0; half],
            pos: half as f64,
            passthrough,
        }
    }

    /// Converts the next chunk of input, appending the output samples to `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.passthrough {
            out.extend_from_slice(input);
            return;
        }
        self.buf.extend_from_slice(input);
        let half = self.half as isize;
        loop {
            let centre = self.pos.floor() as isize;
            if centre + half >= self.buf.len() as isize {
                break;
            }
            let frac = self.pos - centre as f64;
            let mut acc = 0.0f32;
            for k in (centre - half + 1)..=(centre + half) {
                // Distance from the output position to input sample k.
                let d = (frac - (k - centre) as f64).abs() * OVERSAMPLE as f64;
                let i = d as usize;
                let t = (d - i as f64) as f32;
                let h = self.table[i] + t * (self.table[i + 1] - self.table[i]);
                acc += self.buf[k as usize] * h;
            }
            out.push(acc);
            self.pos += self.step;
        }
        // Drop input that no future output can reach.
        let keep_from = (self.pos.floor() as isize - half + 1).max(0) as usize;
        if keep_from > 0 {
            self.buf.drain(..keep_from);
            self.pos -= keep_from as f64;
        }
    }

    /// Flushes the filter tail at the end of a finite one-shot clip.
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        if self.passthrough {
            return;
        }
        let pad = vec![0.0f32; self.half];
        self.process(&pad, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    fn convert(in_rate: u32, x: &[f32], chunk: usize) -> Vec<f32> {
        let mut r = Resampler::new(in_rate, 16_000);
        let mut out = Vec::new();
        for part in x.chunks(chunk) {
            r.process(part, &mut out);
        }
        out
    }

    #[test]
    fn keeps_speech_band_and_removes_what_would_alias() {
        for rate in [48_000, 44_100, 22_050, 8_000] {
            let y = convert(rate, &tone(rate, 1000.0, 1.0), 4096);
            // length follows the rate ratio, minus the kernel's latency
            assert!((y.len() as i64 - 16_000).abs() < 80, "{rate}: {}", y.len());
            let body = &y[2000..12_000];
            assert!((rms(body) - 0.7071).abs() < 0.01, "{rate}: {}", rms(body));
            // the tone is still a 1 kHz tone: compare with a reference at the new rate
            let reference = tone(16_000, 1000.0, 1.0);
            let best = (0..32)
                .map(|lag| {
                    body.iter().zip(&reference[2000 + lag..]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
                })
                .fold(f32::MAX, f32::min);
            assert!(best < 0.2, "{rate}: {best}");
        }
        // 12 kHz cannot exist at 16 kHz; it must not fold back into the speech band
        let y = convert(48_000, &tone(48_000, 12_000.0, 1.0), 4096);
        assert!(rms(&y[2000..12_000]) < 0.003, "{}", rms(&y[2000..12_000]));
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        let x = tone(44_100, 440.0, 0.5);
        let a = convert(44_100, &x, 128);
        let b = convert(44_100, &x, 5000);
        assert_eq!(a, b);
        assert_eq!(convert(16_000, &x, 999), x);
    }
}
