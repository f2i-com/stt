//! Native command-line front end for STT (`stt.computer`).
//!
//!   cargo run --release --example transcribe -- tests/data/henry.wav
//!   cargo run --release --example transcribe -- --words tests/data/henry.wav
//!   cargo run --release --example transcribe -- --json tests/data/henry.wav
//!   cargo run --release --example transcribe -- --stream tests/data/henry.wav
//!   cargo run --release --example transcribe -- --model custom.bin clip.wav
//!   cargo run --release --example transcribe -- --dump check clip.wav

use std::path::Path;
use std::time::Instant;
use stt::{decode_wav, Conformer, Mode, Resampler, SttConfig, SttEvent, SAMPLE_RATE, STT};

fn write_f32_matrix(path: &str, rows: usize, cols: usize, data: &[f32]) {
    let mut out = Vec::with_capacity(8 + data.len() * 4);
    out.extend_from_slice(&(rows as u32).to_le_bytes());
    out.extend_from_slice(&(cols as u32).to_le_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut show_words = false;
    let mut json_mode = false;
    let mut stream_mode = false;
    let mut model_path: Option<String> = None;
    let mut dump_prefix: Option<String> = None;
    let mut files: Vec<String> = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--words" | "-w" => show_words = true,
            "--json" | "-j" => json_mode = true,
            "--stream" | "-s" => stream_mode = true,
            "--model" => model_path = Some(args.next().expect("--model <weights.bin>")),
            "--dump" => dump_prefix = Some(args.next().expect("--dump <prefix>")),
            "--help" | "-h" => {
                eprintln!(
                    "Usage: cargo run --release --example transcribe -- [--words] [--json] [--stream] [--model weights.bin] [--dump prefix] <audio.wav>..."
                );
                return;
            }
            _ => files.push(arg),
        }
    }

    if files.is_empty() {
        files.push("tests/data/henry.wav".to_string());
    }

    if let Some(prefix) = dump_prefix {
        let wav_bytes = std::fs::read(&files[0]).expect("cannot read WAV file");
        let audio = decode_wav(&wav_bytes).expect("invalid WAV file");
        let mut samples = Vec::new();
        Resampler::new(audio.sample_rate, SAMPLE_RATE).process(&audio.samples, &mut samples);

        let model = match &model_path {
            Some(p) => {
                let bytes = std::fs::read(p).expect("cannot read model file");
                std::sync::Arc::new(Conformer::from_bytes(&bytes).expect("invalid model weights"))
            }
            None => stt::embedded_conformer().expect("embedded model unavailable"),
        };

        let (features, frames) = model.features(&samples);
        let (log_probs, t) = model.log_probs_from_features(&features, frames);
        write_f32_matrix(&format!("{prefix}.feat.bin"), frames, 80, &features);
        write_f32_matrix(
            &format!("{prefix}.logp.bin"),
            t,
            model.vocab().len() + 1,
            &log_probs,
        );
        println!(
            "{frames} feature frames, {t} output frames: {}",
            model.transcribe(&samples)
        );
        return;
    }

    let mut engine = match &model_path {
        Some(p) => {
            let bytes = std::fs::read(p).expect("cannot read model file");
            STT::with_model(SttConfig::default(), &bytes).expect("invalid model weights")
        }
        None => STT::default(),
    };
    engine.warmup().expect("failed to initialize speech model");

    for path_str in &files {
        let path = Path::new(path_str);
        let wav_bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let audio = decode_wav(&wav_bytes)
            .unwrap_or_else(|e| panic!("cannot decode {}: {e}", path.display()));

        if stream_mode {
            engine.set_config(SttConfig {
                sample_rate: audio.sample_rate,
                mode: Mode::Dictation,
                ..SttConfig::default()
            });
            let t0 = Instant::now();
            let mut events = Vec::new();
            for chunk in audio.samples.chunks((audio.sample_rate as usize / 20).max(256)) {
                events.extend(engine.push(chunk).unwrap());
            }
            events.extend(engine.flush().unwrap());
            let elapsed = t0.elapsed().as_secs_f64();
            for ev in events {
                if let SttEvent::Phrase(p) = ev {
                    println!(
                        "[{:5.2}s +{:4.2}s | {:3.0}%] {}",
                        p.start_seconds,
                        p.duration_seconds,
                        p.confidence * 100.0,
                        p.text
                    );
                }
            }
            eprintln!(
                "{}: {:.2}s streamed in {:.1} ms ({:.0}x real time)",
                path.display(),
                audio.duration(),
                elapsed * 1000.0,
                audio.duration() / elapsed.max(1e-9)
            );
            continue;
        }

        let t0 = Instant::now();
        let tr = engine
            .transcribe(&audio.samples, audio.sample_rate)
            .expect("transcription failed");
        let elapsed = t0.elapsed().as_secs_f64();

        if json_mode {
            let words_json: Vec<String> = tr
                .words
                .iter()
                .map(|w| {
                    format!(
                        "{{\"word\":{},\"start\":{:.3},\"end\":{:.3},\"confidence\":{:.3}}}",
                        escape_json(&w.word),
                        w.start,
                        w.end,
                        w.confidence
                    )
                })
                .collect();
            println!(
                "{{\"file\":{},\"text\":{},\"duration\":{:.3},\"confidence\":{:.3},\"words\":[{}]}}",
                escape_json(path_str),
                escape_json(&tr.text),
                tr.duration,
                tr.confidence,
                words_json.join(",")
            );
        } else {
            println!("{}", tr.text);
            if show_words {
                for w in &tr.words {
                    println!(
                        "  {:5.2}s – {:5.2}s  ({:3.0}%)  {}",
                        w.start,
                        w.end,
                        w.confidence * 100.0,
                        w.word
                    );
                }
            }
            eprintln!(
                "({:.2}s @ {} Hz transcribed in {:.1} ms — {:.0}x real time, {:.0}% confidence)",
                tr.duration,
                audio.sample_rate,
                elapsed * 1000.0,
                tr.duration / elapsed.max(1e-9),
                tr.confidence * 100.0
            );
        }
    }
}
