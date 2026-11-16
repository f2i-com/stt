//! Dynamic time warping: the distance between two feature sequences after stretching
//! them in time so that they line up as well as possible.

use crate::features::Feat;

#[derive(Clone, Copy, Debug)]
pub struct DtwOpts {
    /// Half-width of the allowed warping corridor as a fraction of the longer sequence.
    pub band: f32,
    /// Compare frames by squared distance instead of plain Euclidean distance.
    pub squared: bool,
}

impl Default for DtwOpts {
    fn default() -> Self {
        DtwOpts { band: 0.3, squared: false }
    }
}

#[inline]
fn frame_dist(a: &[f32], b: &[f32], squared: bool) -> f32 {
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        acc += d * d;
    }
    if squared {
        acc
    } else {
        acc.sqrt()
    }
}

/// Reusable work buffers so matching many templates does not allocate.
#[derive(Default)]
pub struct DtwScratch {
    prev: Vec<f32>,
    cur: Vec<f32>,
}

/// Length-normalised DTW distance (symmetric step pattern, diagonal steps cost double).
/// Returns `f32::INFINITY` if either sequence is empty.
pub fn dtw(a: &Feat, b: &Feat, opts: DtwOpts, scratch: &mut DtwScratch) -> f32 {
    let (n, m) = (a.n, b.n);
    if n == 0 || m == 0 || a.dim != b.dim {
        return f32::INFINITY;
    }
    let width = ((opts.band * n.max(m) as f32) as usize).max(n.abs_diff(m)) + 2;
    scratch.prev.clear();
    scratch.prev.resize(m + 1, f32::INFINITY);
    scratch.cur.clear();
    scratch.cur.resize(m + 1, f32::INFINITY);
    scratch.prev[0] = 0.0;

    for i in 1..=n {
        // Centre of the corridor for this row.
        let centre = (i * m + n / 2) / n;
        let lo = centre.saturating_sub(width).max(1);
        let hi = (centre + width).min(m);
        scratch.cur.fill(f32::INFINITY);
        let row_a = a.row(i - 1);
        for j in lo..=hi {
            let d = frame_dist(row_a, b.row(j - 1), opts.squared);
            let best = (scratch.prev[j] + d)
                .min(scratch.cur[j - 1] + d)
                .min(scratch.prev[j - 1] + 2.0 * d);
            scratch.cur[j] = best;
        }
        std::mem::swap(&mut scratch.prev, &mut scratch.cur);
    }
    scratch.prev[m] / (n + m) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(values: &[f32]) -> Feat {
        Feat { n: values.len(), dim: 1, data: values.to_vec() }
    }

    #[test]
    fn stretched_copy_is_closer_than_a_different_shape() {
        let mut s = DtwScratch::default();
        let a = seq(&[0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]);
        let stretched = seq(&[0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 0.0]);
        let other = seq(&[3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0]);
        let same = dtw(&a, &stretched, DtwOpts::default(), &mut s);
        let diff = dtw(&a, &other, DtwOpts::default(), &mut s);
        assert!(same < 1e-6, "{same}");
        assert!(diff > 0.5, "{diff}");
        assert_eq!(dtw(&a, &a, DtwOpts::default(), &mut s), 0.0);
        assert!(dtw(&a, &seq(&[]), DtwOpts::default(), &mut s).is_infinite());
    }
}
