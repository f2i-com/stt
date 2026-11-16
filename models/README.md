# Speech Model (`conformer_small_int8.bin`)

`conformer_small_int8.bin` (13.7 MB) holds the quantized weights used by **STT** (`stt.computer`) for free-form English dictation. When building with the default `embedded-model` Cargo feature, these weights are compiled directly inside `stt.wasm` so a single `.wasm` file runs with zero external downloads.

## Attribution & License

`conformer_small_int8.bin` is a flat weight conversion of **NVIDIA's `stt_en_conformer_ctc_small`** (Conformer-CTC, 16 layers, 13 million parameters, 1,024 SentencePiece tokens), published at <https://huggingface.co/nvidia/stt_en_conformer_ctc_small> under the [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/) license. NVIDIA trained the checkpoint on its NeMo ASRSET 2.0 collection of English speech. Keep this attribution wherever the weights or embedded WASM binary are distributed.

Changes made during conversion (`tools/export_conformer.py`):
- Linear and convolution weight matrices are quantized to 8-bit signed integers with one `f32` scale factor per output row.
- Depthwise convolution batch normalization is folded into the convolution weights and biases.
- Subsampling convolution layouts are pre-transposed for contiguous Rust SIMD inner products. No weights were retrained.

## Regenerating Weights from Checkpoint

```sh
curl -L -o small.nemo https://huggingface.co/nvidia/stt_en_conformer_ctc_small/resolve/main/stt_en_conformer_ctc_small.nemo
mkdir extracted && tar -xf small.nemo -C extracted
python tools/export_conformer.py extracted models/conformer_small_int8.bin --int8
```

Omit `--int8` to export full-precision `f32` weights (52.7 MB) for `tools/parity_check.py` or `examples/bench.rs`.
