//! The Conformer-CTC network: convolutional subsampling to 25 frames per second, a
//! stack of Conformer blocks (feed-forward, self-attention with relative positions,
//! depthwise convolution, feed-forward), and a linear output over word pieces.
//!
//! This follows the architecture of NVIDIA NeMo's `ConformerEncoder`; the arithmetic is
//! plain f32 with loops written so the compiler auto-vectorises with 128-bit SIMD.

use super::mel::N_MELS;
use super::weights::Weights;

const LAYER_NORM_EPS: f32 = 1e-5;
const KERNEL: usize = 31;

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f32; 16];
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for k in 0..16 {
            acc[k] += x[k] * y[k];
        }
    }
    let s0 = (acc[0] + acc[8]) + (acc[4] + acc[12]);
    let s1 = (acc[1] + acc[9]) + (acc[5] + acc[13]);
    let s2 = (acc[2] + acc[10]) + (acc[6] + acc[14]);
    let s3 = (acc[3] + acc[11]) + (acc[7] + acc[15]);
    let mut sum = (s0 + s2) + (s1 + s3);
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    sum
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Linear {
    fn load(weights: &mut Weights, name: &str, rows: usize, cols: usize, bias: bool) -> Result<Linear, String> {
        Ok(Linear {
            w: weights.take(&format!("{name}.w"), rows * cols)?,
            b: if bias { weights.take(&format!("{name}.b"), rows)? } else { vec![0.0; rows] },
            rows,
            cols,
        })
    }

    /// `x` holds `n` vectors of `cols`; returns `n` vectors of `rows`.
    fn apply(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; n * self.rows];
        for t in 0..n {
            let input = &x[t * self.cols..(t + 1) * self.cols];
            let out = &mut y[t * self.rows..(t + 1) * self.rows];
            for (r, o) in out.iter_mut().enumerate() {
                *o = self.b[r] + dot(&self.w[r * self.cols..(r + 1) * self.cols], input);
            }
        }
        y
    }
}

struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    fn load(weights: &mut Weights, name: &str, d: usize) -> Result<LayerNorm, String> {
        Ok(LayerNorm { w: weights.take(&format!("{name}.w"), d)?, b: weights.take(&format!("{name}.b"), d)? })
    }

    fn apply(&self, x: &[f32], n: usize) -> Vec<f32> {
        let d = self.w.len();
        let mut y = vec![0.0f32; n * d];
        for t in 0..n {
            let row = &x[t * d..(t + 1) * d];
            let mean = row.iter().sum::<f32>() / d as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv = 1.0 / (var + LAYER_NORM_EPS).sqrt();
            for k in 0..d {
                y[t * d + k] = (row[k] - mean) * inv * self.w[k] + self.b[k];
            }
        }
        y
    }
}

struct Block {
    ff1_norm: LayerNorm,
    ff1_in: Linear,
    ff1_out: Linear,
    att_norm: LayerNorm,
    att_q: Linear,
    att_k: Linear,
    att_v: Linear,
    att_out: Linear,
    att_pos: Linear,
    /// Per-head biases added to the query for the content and position terms.
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
    conv_norm: LayerNorm,
    conv_in: Linear,
    /// Depthwise kernel `d × KERNEL` with batch normalisation folded in.
    conv_dw: Vec<f32>,
    conv_dw_bias: Vec<f32>,
    conv_out: Linear,
    ff2_norm: LayerNorm,
    ff2_in: Linear,
    ff2_out: Linear,
    out_norm: LayerNorm,
}

pub struct Encoder {
    d: usize,
    heads: usize,
    channels: usize,
    conv0_w: Vec<f32>,
    conv0_b: Vec<f32>,
    conv1: Linear,
    sub_out: Linear,
    blocks: Vec<Block>,
    output: Linear,
    pub classes: usize,
}

fn swish_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= sigmoid(*v);
    }
}

fn add_scaled(x: &mut [f32], y: &[f32], scale: f32) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += scale * b;
    }
}

impl Encoder {
    pub fn load(weights: &mut Weights, d: usize, heads: usize, classes: usize) -> Result<Encoder, String> {
        let channels = d;
        let ff = 4 * d;
        let freq_after = N_MELS / 4;
        let layers = weights.layer_count();
        if layers == 0 {
            return Err("weight file has no encoder layers".into());
        }
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("l{i}");
            blocks.push(Block {
                ff1_norm: LayerNorm::load(weights, &format!("{p}.ff1.norm"), d)?,
                ff1_in: Linear::load(weights, &format!("{p}.ff1.l1"), ff, d, true)?,
                ff1_out: Linear::load(weights, &format!("{p}.ff1.l2"), d, ff, true)?,
                att_norm: LayerNorm::load(weights, &format!("{p}.att.norm"), d)?,
                att_q: Linear::load(weights, &format!("{p}.att.q"), d, d, true)?,
                att_k: Linear::load(weights, &format!("{p}.att.k"), d, d, true)?,
                att_v: Linear::load(weights, &format!("{p}.att.v"), d, d, true)?,
                att_out: Linear::load(weights, &format!("{p}.att.out"), d, d, true)?,
                att_pos: Linear::load(weights, &format!("{p}.att.pos"), d, d, false)?,
                bias_u: weights.take(&format!("{p}.att.u"), d)?,
                bias_v: weights.take(&format!("{p}.att.v"), d)?,
                conv_norm: LayerNorm::load(weights, &format!("{p}.conv.norm"), d)?,
                conv_in: Linear::load(weights, &format!("{p}.conv.pw1"), 2 * d, d, true)?,
                conv_dw: weights.take(&format!("{p}.conv.dw.w"), d * KERNEL)?,
                conv_dw_bias: weights.take(&format!("{p}.conv.dw.b"), d)?,
                conv_out: Linear::load(weights, &format!("{p}.conv.pw2"), d, d, true)?,
                ff2_norm: LayerNorm::load(weights, &format!("{p}.ff2.norm"), d)?,
                ff2_in: Linear::load(weights, &format!("{p}.ff2.l1"), ff, d, true)?,
                ff2_out: Linear::load(weights, &format!("{p}.ff2.l2"), d, ff, true)?,
                out_norm: LayerNorm::load(weights, &format!("{p}.out.norm"), d)?,
            });
        }
        Ok(Encoder {
            d,
            heads,
            channels,
            conv0_w: weights.take("sub.conv0.w", channels * 9)?,
            conv0_b: weights.take("sub.conv0.b", channels)?,
            conv1: Linear::load(weights, "sub.conv1", channels, channels * 9, true)?,
            sub_out: Linear::load(weights, "sub.out", d, channels * freq_after, true)?,
            blocks,
            output: Linear::load(weights, "dec", classes, d, true)?,
            classes,
        })
    }

    /// Output frames for `frames` feature frames (two stride-2 convolutions).
    pub fn output_frames(frames: usize) -> usize {
        let half = |n: usize| if n == 0 { 0 } else { (n - 1) / 2 + 1 };
        half(half(frames))
    }

    /// Log-probabilities over word pieces (blank last), `frames_out × classes`.
    pub fn log_probs(&self, features: &[f32], frames: usize) -> (Vec<f32>, usize) {
        let t = Self::output_frames(frames);
        if t == 0 {
            return (Vec::new(), 0);
        }
        let mut x = self.subsample(features, frames);
        let scale = (self.d as f32).sqrt();
        for v in x.iter_mut() {
            *v *= scale;
        }
        let positions = self.positions(t);
        for block in &self.blocks {
            x = self.block(block, x, &positions, t);
        }
        let mut logits = self.output.apply(&x, t);
        for row in logits.chunks_exact_mut(self.classes) {
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            let norm = max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
            for v in row.iter_mut() {
                *v -= norm;
            }
        }
        (logits, t)
    }

    /// Two 3×3 stride-2 convolutions with ReLU, then a linear layer: `frames × 80`
    /// features become `frames/4 × d`.
    fn subsample(&self, features: &[f32], frames: usize) -> Vec<f32> {
        let c = self.channels;
        let half = |n: usize| (n - 1) / 2 + 1;
        let (t1, f1) = (half(frames), half(N_MELS));
        let (t2, f2) = (half(t1), half(f1));

        // First convolution: one input channel. Output is stored [time][freq][channel].
        let mut a = vec![0.0f32; t1 * f1 * c];
        let mut patch = [0.0f32; 9];
        for ti in 0..t1 {
            for fi in 0..f1 {
                for kh in 0..3 {
                    for kw in 0..3 {
                        let (tt, ff) = ((2 * ti + kh) as isize - 1, (2 * fi + kw) as isize - 1);
                        patch[kh * 3 + kw] = if tt >= 0 && (tt as usize) < frames && ff >= 0 && (ff as usize) < N_MELS {
                            features[tt as usize * N_MELS + ff as usize]
                        } else {
                            0.0
                        };
                    }
                }
                let out = &mut a[(ti * f1 + fi) * c..(ti * f1 + fi + 1) * c];
                for (ch, o) in out.iter_mut().enumerate() {
                    let w = &self.conv0_w[ch * 9..ch * 9 + 9];
                    let mut acc = self.conv0_b[ch];
                    for k in 0..9 {
                        acc += w[k] * patch[k];
                    }
                    *o = acc.max(0.0);
                }
            }
        }

        // Second convolution: gather each 3×3×channels patch, then one matrix product.
        let mut patches = vec![0.0f32; t2 * f2 * 9 * c];
        for ti in 0..t2 {
            for fi in 0..f2 {
                let dst = &mut patches[(ti * f2 + fi) * 9 * c..(ti * f2 + fi + 1) * 9 * c];
                for kh in 0..3 {
                    for kw in 0..3 {
                        let (tt, ff) = ((2 * ti + kh) as isize - 1, (2 * fi + kw) as isize - 1);
                        if tt >= 0 && (tt as usize) < t1 && ff >= 0 && (ff as usize) < f1 {
                            let src = (tt as usize * f1 + ff as usize) * c;
                            dst[(kh * 3 + kw) * c..(kh * 3 + kw + 1) * c].copy_from_slice(&a[src..src + c]);
                        }
                    }
                }
            }
        }
        let mut b = self.conv1.apply(&patches, t2 * f2);
        for v in b.iter_mut() {
            *v = v.max(0.0);
        }
        // `b` is [time][freq][channel]: each frame is already the linear layer's input.
        self.sub_out.apply(&b, t2)
    }

    /// Sinusoidal encodings of the relative positions `t-1, t-2, …, -(t-1)`.
    fn positions(&self, t: usize) -> Vec<f32> {
        let d = self.d;
        let count = 2 * t - 1;
        let mut pe = vec![0.0f32; count * d];
        let rates: Vec<f64> = (0..d / 2).map(|i| (-(2.0 * i as f64) * (10_000.0f64.ln() / d as f64)).exp()).collect();
        for p in 0..count {
            let position = t as f64 - 1.0 - p as f64;
            for (i, rate) in rates.iter().enumerate() {
                let angle = position * rate;
                pe[p * d + 2 * i] = angle.sin() as f32;
                pe[p * d + 2 * i + 1] = angle.cos() as f32;
            }
        }
        pe
    }

    fn block(&self, b: &Block, mut x: Vec<f32>, positions: &[f32], t: usize) -> Vec<f32> {
        let d = self.d;

        // Feed-forward, added at half weight.
        let mut h = b.ff1_in.apply(&b.ff1_norm.apply(&x, t), t);
        swish_in_place(&mut h);
        add_scaled(&mut x, &b.ff1_out.apply(&h, t), 0.5);

        // Self-attention with relative positions.
        let attended = self.attention(b, &b.att_norm.apply(&x, t), positions, t);
        add_scaled(&mut x, &attended, 1.0);

        // Convolution module: expand, gate, depthwise convolution over time, project.
        let expanded = b.conv_in.apply(&b.conv_norm.apply(&x, t), t);
        let pad = KERNEL / 2;
        // Gated activations stored [channel][time] with zero padding at both ends.
        let stride = t + 2 * pad;
        let mut gated = vec![0.0f32; d * stride];
        for ti in 0..t {
            let row = &expanded[ti * 2 * d..(ti + 1) * 2 * d];
            for ch in 0..d {
                gated[ch * stride + pad + ti] = row[ch] * sigmoid(row[d + ch]);
            }
        }
        let mut conv = vec![0.0f32; t * d];
        for ch in 0..d {
            let kernel = &b.conv_dw[ch * KERNEL..(ch + 1) * KERNEL];
            let series = &gated[ch * stride..(ch + 1) * stride];
            for ti in 0..t {
                let v = b.conv_dw_bias[ch] + dot(kernel, &series[ti..ti + KERNEL]);
                conv[ti * d + ch] = v * sigmoid(v);
            }
        }
        add_scaled(&mut x, &b.conv_out.apply(&conv, t), 1.0);

        // Second feed-forward, then the block's output normalisation.
        let mut h = b.ff2_in.apply(&b.ff2_norm.apply(&x, t), t);
        swish_in_place(&mut h);
        add_scaled(&mut x, &b.ff2_out.apply(&h, t), 0.5);
        b.out_norm.apply(&x, t)
    }

    fn attention(&self, b: &Block, x: &[f32], positions: &[f32], t: usize) -> Vec<f32> {
        let (d, heads) = (self.d, self.heads);
        let dk = d / heads;
        let q = b.att_q.apply(x, t);
        let k = b.att_k.apply(x, t);
        let v = b.att_v.apply(x, t);
        let p = b.att_pos.apply(positions, 2 * t - 1);
        let scale = 1.0 / (dk as f32).sqrt();

        let mut context = vec![0.0f32; t * d];
        // Per-head copies so every inner product runs over contiguous memory.
        let mut kh = vec![0.0f32; t * dk];
        let mut vh = vec![0.0f32; t * dk];
        let mut ph = vec![0.0f32; (2 * t - 1) * dk];
        let mut qu = vec![0.0f32; dk];
        let mut qv = vec![0.0f32; dk];
        let mut scores = vec![0.0f32; t];
        for head in 0..heads {
            let off = head * dk;
            for i in 0..t {
                kh[i * dk..(i + 1) * dk].copy_from_slice(&k[i * d + off..i * d + off + dk]);
                vh[i * dk..(i + 1) * dk].copy_from_slice(&v[i * d + off..i * d + off + dk]);
            }
            for i in 0..2 * t - 1 {
                ph[i * dk..(i + 1) * dk].copy_from_slice(&p[i * d + off..i * d + off + dk]);
            }
            for i in 0..t {
                for j in 0..dk {
                    let qi = q[i * d + off + j];
                    qu[j] = qi + b.bias_u[off + j];
                    qv[j] = qi + b.bias_v[off + j];
                }
                // Key j sits at relative position i - j, which is row (t - 1 - i + j) of `ph`.
                let first = t - 1 - i;
                let mut max = f32::MIN;
                for j in 0..t {
                    let content = dot(&qu, &kh[j * dk..(j + 1) * dk]);
                    let position = dot(&qv, &ph[(first + j) * dk..(first + j + 1) * dk]);
                    let s = (content + position) * scale;
                    scores[j] = s;
                    max = max.max(s);
                }
                let mut total = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    total += *s;
                }
                let out = &mut context[i * d + off..i * d + off + dk];
                for j in 0..t {
                    let w = scores[j] / total;
                    if w > 1e-9 {
                        for (o, value) in out.iter_mut().zip(&vh[j * dk..(j + 1) * dk]) {
                            *o += w * value;
                        }
                    }
                }
            }
        }
        b.att_out.apply(&context, t)
    }
}
