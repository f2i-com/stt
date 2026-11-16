//! Minimal, bounds-checked RIFF WAV decoder (PCM 8/16/24/32-bit and 32-bit IEEE float,
//! mono or multi-channel downmixed to mono `f32`) and 16-bit PCM WAV encoder.

#[derive(Clone, Debug, PartialEq)]
pub struct WavAudio {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

impl WavAudio {
    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f64 / self.sample_rate as f64
        }
    }
}

fn u16_at(b: &[u8], i: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(i..i + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(i..i + 4)?.try_into().ok()?))
}

/// Decodes RIFF WAVE bytes into mono `f32` PCM in `-1.0..=1.0` and sample rate in Hz.
pub fn decode_wav(bytes: &[u8]) -> Result<WavAudio, String> {
    if bytes.get(0..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return Err("invalid WAV header (expected RIFF/WAVE)".into());
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32_at(bytes, pos + 4).ok_or("truncated WAV chunk header")? as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(size).min(bytes.len());
        let body = &bytes[body_start..body_end];
        if id == b"fmt " {
            if body.len() < 16 {
                return Err("WAV fmt chunk is too short".into());
            }
            let mut tag = u16_at(body, 0).unwrap();
            let channels = u16_at(body, 2).unwrap();
            let rate = u32_at(body, 4).unwrap();
            let bits = u16_at(body, 14).unwrap();
            if tag == 0xFFFE {
                tag = u16_at(body, 24).ok_or("truncated extensible WAV fmt chunk")?;
            }
            fmt = Some((tag, channels, rate, bits));
        } else if id == b"data" {
            let (tag, channels, rate, bits) = fmt.ok_or("WAV data chunk precedes fmt chunk")?;
            let channels = channels.max(1) as usize;
            let bytes_per = (bits / 8) as usize;
            if bytes_per == 0 || rate == 0 {
                return Err("unsupported WAV sample rate or bit depth".into());
            }
            let frame_bytes = bytes_per * channels;
            let frames = body.len() / frame_bytes;
            let mut samples = Vec::with_capacity(frames);
            for f in 0..frames {
                let mut acc = 0.0f32;
                for c in 0..channels {
                    let o = f * frame_bytes + c * bytes_per;
                    let s = &body[o..o + bytes_per];
                    let v = match (tag, bits) {
                        (1, 8) => (s[0] as f32 - 128.0) / 128.0,
                        (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
                        (1, 24) => {
                            (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8_388_608.0
                        }
                        (1, 32) => {
                            i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0
                        }
                        (3, 32) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
                        _ => {
                            return Err(format!(
                                "unsupported WAV encoding (format tag {tag}, {bits}-bit)"
                            ))
                        }
                    };
                    acc += if v.is_finite() { v } else { 0.0 };
                }
                samples.push(acc / channels as f32);
            }
            return Ok(WavAudio {
                sample_rate: rate,
                samples,
            });
        }
        pos = body_start.saturating_add(size).saturating_add(size & 1);
    }
    Err("WAV file contains no audio data chunk".into())
}

/// Encodes mono `f32` PCM (`-1.0..=1.0`) as a 16-bit little-endian RIFF WAVE file.
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let sr = sample_rate.max(1);
    let data_bytes = (samples.len() as u32).saturating_mul(2);
    let riff_size = 36u32.saturating_add(data_bytes);
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in samples {
        let clamped = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
        let q = (clamped * 32767.0).round() as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_and_decodes_mono_wav() {
        let src = vec![0.0f32, 0.5, -0.5, 0.99, -0.99];
        let wav = encode_wav(&src, 16_000);
        let decoded = decode_wav(&wav).unwrap();
        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.samples.len(), src.len());
        for (a, b) in src.iter().zip(&decoded.samples) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        assert!(decode_wav(b"not a wav").is_err());
    }
}
