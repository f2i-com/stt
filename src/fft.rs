//! Iterative radix-2 FFT.

pub struct Fft {
    n: usize,
    rev: Vec<u32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Fft {
    /// `n` must be a power of two, at least 2.
    pub fn new(n: usize) -> Fft {
        assert!(n.is_power_of_two() && n >= 2);
        let bits = n.trailing_zeros();
        let rev = (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
        let mut cos = Vec::with_capacity(n / 2);
        let mut sin = Vec::with_capacity(n / 2);
        for k in 0..n / 2 {
            let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
        Fft { n, rev, cos, sin }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// In-place forward transform of `re + i·im`.
    pub fn forward(&self, re: &mut [f32], im: &mut [f32]) {
        let n = self.n;
        debug_assert!(re.len() == n && im.len() == n);
        for i in 0..n {
            let j = self.rev[i] as usize;
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            let mut start = 0;
            while start < n {
                for k in 0..half {
                    let (wr, wi) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
                start += len;
            }
            len <<= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_lands_in_one_bin() {
        let n = 256;
        let fft = Fft::new(n);
        let mut re: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 10.0 * i as f32 / n as f32).sin())
            .collect();
        let mut im = vec![0.0; n];
        fft.forward(&mut re, &mut im);
        let power: Vec<f32> = (0..n / 2).map(|k| re[k] * re[k] + im[k] * im[k]).collect();
        let peak = power.iter().cloned().fold(0.0, f32::max);
        assert!((power[10] - peak).abs() < 1e-3);
        assert!(power.iter().enumerate().all(|(k, &p)| k == 10 || p < peak * 1e-6));
    }
}
