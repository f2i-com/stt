"""Converts a LibriSpeech subset (FLAC) to 16-bit WAV and collects transcripts for
`examples/bench.rs`.

    python tools/librispeech_to_wav.py data/librispeech/LibriSpeech/test-clean data/librispeech_wav

Writes one `<utterance id>.wav` per recording plus `transcripts.tsv` (id, tab, lower-case
text). Requires the `soundfile` package.
"""
import os
import sys

import soundfile as sf


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    rows = []
    count = 0
    for root, _, files in os.walk(src):
        for name in sorted(files):
            path = os.path.join(root, name)
            if name.endswith(".trans.txt"):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        utterance, text = line.strip().split(" ", 1)
                        rows.append((utterance, text.lower()))
            elif name.endswith(".flac"):
                audio, rate = sf.read(path, dtype="int16")
                assert rate == 16000 and audio.ndim == 1, (path, rate, audio.shape)
                sf.write(os.path.join(dst, name[:-5] + ".wav"), audio, rate, subtype="PCM_16")
                count += 1
    rows.sort()
    with open(os.path.join(dst, "transcripts.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for utterance, text in rows:
            f.write(f"{utterance}\t{text}\n")
    print(f"{count} recordings, {len(rows)} transcripts")


if __name__ == "__main__":
    main()
