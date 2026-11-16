"""Converts NVIDIA's `stt_en_conformer_ctc_small` checkpoint into the flat weight file that
`stt` (`stt.computer`) loads.

    python tools/export_conformer.py <extracted .nemo folder> <out.bin> [--int8]

The .nemo file is a tar archive; extract it first. The folder must hold
`model_weights.ckpt` and `model_config.yaml`.

Output layout (little endian):
    "STTW" (or legacy "VRCW") | version u32 | tensor count u32
    per tensor: name length u16 | name | dtype u8 | ndim u8 | dims u32[ndim] | payload
        dtype 0: float32 values
        dtype 1: float32 scale per row (dims[0] of them), then int8 values;
                 value = int8 * scale[row]
    vocabulary: count u32 | per piece: length u16 | UTF-8 bytes

With --int8 every weight matrix is stored as int8 with one scale per output row (about a
quarter of the size). Batch normalisation is folded into the depthwise convolution, and
the two reshapes the Rust code would otherwise do per frame are baked into the layout.

The checkpoint is licensed CC-BY-4.0 by NVIDIA; keep the attribution with any file made
from it.
"""
import argparse
import os
import struct

import numpy as np
import torch
import yaml

BN_EPS = 1e-5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("out")
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()

    sd = torch.load(os.path.join(args.folder, "model_weights.ckpt"), map_location="cpu", weights_only=True)
    sd = {k: v.numpy().astype(np.float32) for k, v in sd.items() if v.dtype == torch.float32}
    # Word pieces in output order (SentencePiece notation: a leading U+2581 starts a word).
    with open(os.path.join(args.folder, "model_config.yaml"), encoding="utf-8") as f:
        vocab = [str(piece) for piece in yaml.safe_load(f)["decoder"]["vocabulary"]]
    assert len(vocab) + 1 == sd["decoder.decoder_layers.0.bias"].shape[0], len(vocab)

    tensors = []  # (name, array, quantise)

    def add(name, array, quantise=False):
        tensors.append((name, np.ascontiguousarray(array, dtype=np.float32), quantise))

    add("fb", sd["preprocessor.featurizer.fb"][0])
    add("window", sd["preprocessor.featurizer.window"])

    # Subsampling: two 3x3 stride-2 convolutions, then a linear layer.
    # Convolution weights go from [out][in][kh][kw] to [out][kh][kw][in], so a patch is read
    # as runs of channels from activations stored [time][freq][channel].
    add("sub.conv0.w", sd["encoder.pre_encode.conv.0.weight"].transpose(0, 2, 3, 1).reshape(176, -1), True)
    add("sub.conv0.b", sd["encoder.pre_encode.conv.0.bias"])
    add("sub.conv1.w", sd["encoder.pre_encode.conv.2.weight"].transpose(0, 2, 3, 1).reshape(176, -1), True)
    add("sub.conv1.b", sd["encoder.pre_encode.conv.2.bias"])
    # The linear layer reads channel * 20 + freq; reorder its columns to freq * 176 + channel.
    out_w = sd["encoder.pre_encode.out.weight"]
    channels = out_w.shape[1] // 20
    add("sub.out.w", out_w.reshape(-1, channels, 20).transpose(0, 2, 1).reshape(out_w.shape[0], -1), True)
    add("sub.out.b", sd["encoder.pre_encode.out.bias"])

    layers = len({k.split(".")[2] for k in sd if k.startswith("encoder.layers.")})
    for i in range(layers):
        p = f"encoder.layers.{i}."
        o = f"l{i}."
        for src, dst in [("norm_feed_forward1", "ff1.norm"), ("norm_self_att", "att.norm"), ("norm_conv", "conv.norm"),
                         ("norm_feed_forward2", "ff2.norm"), ("norm_out", "out.norm")]:
            add(o + dst + ".w", sd[p + src + ".weight"])
            add(o + dst + ".b", sd[p + src + ".bias"])
        for src, dst in [("feed_forward1", "ff1"), ("feed_forward2", "ff2")]:
            add(o + dst + ".l1.w", sd[p + src + ".linear1.weight"], True)
            add(o + dst + ".l1.b", sd[p + src + ".linear1.bias"])
            add(o + dst + ".l2.w", sd[p + src + ".linear2.weight"], True)
            add(o + dst + ".l2.b", sd[p + src + ".linear2.bias"])
        for name in ["q", "k", "v", "out"]:
            add(o + f"att.{name}.w", sd[p + f"self_attn.linear_{name}.weight"], True)
            add(o + f"att.{name}.b", sd[p + f"self_attn.linear_{name}.bias"])
        add(o + "att.pos.w", sd[p + "self_attn.linear_pos.weight"], True)
        add(o + "att.u", sd[p + "self_attn.pos_bias_u"])
        add(o + "att.v", sd[p + "self_attn.pos_bias_v"])
        add(o + "conv.pw1.w", sd[p + "conv.pointwise_conv1.weight"][:, :, 0], True)
        add(o + "conv.pw1.b", sd[p + "conv.pointwise_conv1.bias"])
        # Fold batch normalisation (inference form) into the depthwise convolution.
        gain = sd[p + "conv.batch_norm.weight"] / np.sqrt(sd[p + "conv.batch_norm.running_var"] + BN_EPS)
        dw = sd[p + "conv.depthwise_conv.weight"][:, 0, :] * gain[:, None]
        dw_b = (sd[p + "conv.depthwise_conv.bias"] - sd[p + "conv.batch_norm.running_mean"]) * gain + sd[p + "conv.batch_norm.bias"]
        add(o + "conv.dw.w", dw, True)
        add(o + "conv.dw.b", dw_b)
        add(o + "conv.pw2.w", sd[p + "conv.pointwise_conv2.weight"][:, :, 0], True)
        add(o + "conv.pw2.b", sd[p + "conv.pointwise_conv2.bias"])

    add("dec.w", sd["decoder.decoder_layers.0.weight"][:, :, 0], True)
    add("dec.b", sd["decoder.decoder_layers.0.bias"])

    out = bytearray()
    out += b"STTW" + struct.pack("<II", 1, len(tensors))
    for name, array, quantise in tensors:
        raw = name.encode()
        as_int8 = args.int8 and quantise
        out += struct.pack("<H", len(raw)) + raw + struct.pack("<BB", 1 if as_int8 else 0, array.ndim)
        out += struct.pack(f"<{array.ndim}I", *array.shape)
        if as_int8:
            rows = array.reshape(array.shape[0], -1)
            peak = np.abs(rows).max(axis=1)
            scale = np.where(peak > 0, peak / 127.0, 1.0).astype(np.float32)
            q = np.clip(np.round(rows / scale[:, None]), -127, 127).astype(np.int8)
            out += scale.astype("<f4").tobytes() + q.tobytes()
        else:
            out += array.astype("<f4").tobytes()
    out += struct.pack("<I", len(vocab))
    for piece in vocab:
        raw = piece.encode("utf-8")
        out += struct.pack("<H", len(raw)) + raw
    with open(args.out, "wb") as f:
        f.write(out)
    print(f"{args.out}: {len(tensors)} tensors, {len(vocab)} pieces, {len(out) / 1e6:.2f} MB, {layers} layers")


if __name__ == "__main__":
    main()
