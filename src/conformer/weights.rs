//! Reader for the flat weight file written by `tools/export_conformer.py`.

use std::collections::HashMap;

pub struct Weights {
    tensors: HashMap<String, Vec<f32>>,
    pub vocab: Vec<String>,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("size overflow")?;
        let slice = self.bytes.get(self.pos..end).ok_or("weight file is truncated")?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<usize, String> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]) as usize)
    }

    fn u32(&mut self) -> Result<usize, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let raw = self.take(n.checked_mul(4).ok_or("size overflow")?)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

impl Weights {
    pub fn from_bytes(bytes: &[u8]) -> Result<Weights, String> {
        let mut cur = Cursor { bytes, pos: 0 };
        let magic = cur.take(4)?;
        if magic != b"STTW" && magic != b"VRCW" {
            return Err("not an STT weight file".into());
        }
        if cur.u32()? != 1 {
            return Err("unsupported weight file version".into());
        }
        let count = cur.u32()?;
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let name_len = cur.u16()?;
            let name = String::from_utf8(cur.take(name_len)?.to_vec()).map_err(|_| "bad tensor name")?;
            let dtype = cur.u8()?;
            let ndim = cur.u8()? as usize;
            let mut dims = Vec::with_capacity(ndim);
            let mut total = 1usize;
            for _ in 0..ndim {
                let d = cur.u32()?;
                total = total.checked_mul(d).ok_or("size overflow")?;
                dims.push(d);
            }
            let data = match dtype {
                0 => cur.f32s(total)?,
                1 => {
                    // int8 values with one scale per row
                    let rows = *dims.first().ok_or("int8 tensor without rows")?;
                    let scales = cur.f32s(rows)?;
                    let raw = cur.take(total)?;
                    let cols = if rows > 0 { total / rows } else { 0 };
                    let mut data = Vec::with_capacity(total);
                    for (r, &scale) in scales.iter().enumerate() {
                        data.extend(raw[r * cols..(r + 1) * cols].iter().map(|&q| q as i8 as f32 * scale));
                    }
                    data
                }
                _ => return Err(format!("unknown data type in tensor {name}")),
            };
            tensors.insert(name, data);
        }
        let pieces = cur.u32()?;
        let mut vocab = Vec::with_capacity(pieces);
        for _ in 0..pieces {
            let len = cur.u16()?;
            vocab.push(String::from_utf8(cur.take(len)?.to_vec()).map_err(|_| "bad vocabulary entry")?);
        }
        Ok(Weights { tensors, vocab })
    }

    /// Removes and returns a tensor, checking that it has `expect` values.
    pub fn take(&mut self, name: &str, expect: usize) -> Result<Vec<f32>, String> {
        let tensor = self.tensors.remove(name).ok_or_else(|| format!("weight file has no tensor {name}"))?;
        if tensor.len() != expect {
            return Err(format!("tensor {name} has {} values, expected {expect}", tensor.len()));
        }
        Ok(tensor)
    }

    /// Number of layers present (`l0.`, `l1.`, ...).
    pub fn layer_count(&self) -> usize {
        (0..).take_while(|i| self.tensors.contains_key(&format!("l{i}.out.norm.w"))).count()
    }
}
