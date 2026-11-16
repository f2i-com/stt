"""Checks the Rust Conformer network against a reference ONNX export of the same model.

    cargo run --release --example transcribe -- --dump check clip.wav
    python tools/parity_check.py <model.onnx> check

`--dump <prefix>` writes the Rust mel features (`<prefix>.feat.bin`) and log-probabilities
(`<prefix>.logp.bin`). This script feeds the same features to the reference ONNX model via
ONNX Runtime and reports the maximum log-probability difference (~1e-4 with f32 weights).

Reference ONNX export published by sherpa-onnx:
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2
"""
import sys

import numpy as np
import onnxruntime as ort


def load(path):
    raw = open(path, "rb").read()
    rows, cols = np.frombuffer(raw[:8], dtype="<u4")
    return np.frombuffer(raw[8:], dtype="<f4").reshape(rows, cols)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    features = load(sys.argv[2] + ".feat.bin")
    port = load(sys.argv[2] + ".logp.bin")
    session = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    reference = session.run(None, {
        names[0]: features.T[None].astype(np.float32),
        names[1]: np.array([features.shape[0]], dtype=np.int64),
    })[0][0]
    n = min(len(reference), len(port))
    diff = np.abs(reference[:n] - port[:n])
    agree = int((reference[:n].argmax(1) == port[:n].argmax(1)).sum())
    print(f"reference {reference.shape}, port {port.shape}")
    print(f"largest difference in log-probability: {diff.max():.5f} (mean {diff.mean():.6f})")
    print(f"frames where the top choice agrees: {agree} of {n}")


if __name__ == "__main__":
    main()
