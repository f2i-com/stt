//! Multi-core accuracy & speed benchmark suite for STT (`stt.computer`).
//!
//! Expects public evaluation datasets unpacked under `data/` (or `$STT_DATA`):
//!   cargo run --release --example bench -- librispeech [n=200] [model=path.bin]
//!   cargo run --release --example bench -- arctic [n=200]
//!   cargo run --release --example bench -- commands [n=500]
//!   cargo run --release --example bench -- fsdd [k=3]
//!   cargo run --release --example bench -- live [k=1] [stream]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use stt::{decode_wav, CommandSet, Conformer, Resampler, Segmenter, SegmenterConfig, SAMPLE_RATE};

struct Args(HashMap<String, String>);

impl Args {
    fn parse(iter: impl Iterator<Item = String>) -> Self {
        let mut map = HashMap::new();
        for a in iter {
            match a.split_once('=') {
                Some((k, v)) => map.insert(k.to_string(), v.to_string()),
                None => map.insert(a, "1".to_string()),
            };
        }
        Args(map)
    }

    fn str(&self, key: &str, default: &str) -> String {
        self.0.get(key).cloned().unwrap_or_else(|| default.to_string())
    }

    fn num<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        match self.0.get(key) {
            Some(v) => v.parse().unwrap_or_else(|_| panic!("bad value for {key}: {v}")),
            None => default,
        }
    }

    fn flag(&self, key: &str) -> bool {
        self.0.get(key).map(|v| v != "0" && v != "false").unwrap_or(false)
    }
}

fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(items.len().max(1));
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, R)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= items.len() {
                            break;
                        }
                        local.push((i, f(&items[i])));
                    }
                    local
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    for part in parts {
        for (i, r) in part {
            out[i] = Some(r);
        }
    }
    out.into_iter().map(|o| o.unwrap()).collect()
}

fn pct(num: usize, den: usize) -> f64 {
    100.0 * num as f64 / den.max(1) as f64
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '\''))
        .map(|w| w.trim_matches('\'').to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn data_root() -> PathBuf {
    std::env::var("STT_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data")))
}

fn load_conformer(args: &Args) -> Arc<Conformer> {
    let path = args.str("model", "");
    if path.is_empty() {
        stt::embedded_conformer().expect("embedded model unavailable")
    } else {
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        Arc::new(Conformer::from_bytes(&bytes).unwrap_or_else(|e| panic!("invalid {path}: {e}")))
    }
}

fn read_mono_16k(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let audio = decode_wav(&bytes).unwrap_or_else(|e| panic!("cannot decode {}: {e}", path.display()));
    if audio.sample_rate == SAMPLE_RATE {
        audio.samples
    } else {
        let mut out = Vec::new();
        Resampler::new(audio.sample_rate, SAMPLE_RATE).process(&audio.samples, &mut out);
        out
    }
}

fn score_wer(model: &Conformer, name: &str, items: &[(String, PathBuf)], show: usize) {
    let started = Instant::now();
    let results = par_map(items, |(text, path)| {
        let samples = read_mono_16k(path);
        let heard = model.transcribe(&samples);
        (tokenize(text), tokenize(&heard), samples.len())
    });
    let elapsed = started.elapsed().as_secs_f64();
    let words: usize = results.iter().map(|r| r.0.len()).sum();
    let errors: usize = results.iter().map(|r| edit_distance(&r.0, &r.1)).sum();
    let exact = results.iter().filter(|r| r.0 == r.1).count();
    let audio_secs: f64 = results.iter().map(|r| r.2 as f64 / 16_000.0).sum();
    println!(
        "{name}: WER {:.2}% ({errors} errors / {words} words, {} clips, {:.1}% exact); {:.2} h audio in {:.1} s on all cores",
        pct(errors, words),
        results.len(),
        pct(exact, results.len()),
        audio_secs / 3600.0,
        elapsed
    );
    for r in results.iter().filter(|r| r.0 != r.1).take(show) {
        println!("    said:  {}\n    heard: {}", r.0.join(" "), r.1.join(" "));
    }
}

fn bench_librispeech(args: &Args) {
    let model = load_conformer(args);
    let limit: usize = args.num("n", usize::MAX);
    let show: usize = args.num("show", 0);
    let root = data_root().join("librispeech_wav");
    let index = std::fs::read_to_string(root.join("transcripts.tsv"))
        .expect("expected data/librispeech_wav/transcripts.tsv");
    let items: Vec<(String, PathBuf)> = index
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(id, text)| (text.to_string(), root.join(format!("{id}.wav"))))
        .take(limit)
        .collect();
    score_wer(&model, "LibriSpeech test-clean", &items, show);
}

fn bench_arctic(args: &Args) {
    let model = load_conformer(args);
    let limit: usize = args.num("n", 200);
    let show: usize = args.num("show", 0);
    for speaker in args.str("speakers", "bdl,slt,clb,rms,jmk,awb,ksp").split(',') {
        let root = data_root()
            .join("arctic095")
            .join(speaker)
            .join(format!("cmu_us_{speaker}_arctic"));
        let index = std::fs::read_to_string(root.join("etc").join("txt.done.data"))
            .unwrap_or_else(|e| panic!("cannot read ARCTIC transcripts for {speaker}: {e}"));
        let mut items = Vec::new();
        for line in index.lines() {
            let Some(id) = line.split_whitespace().nth(1) else { continue };
            if !id.starts_with("arctic_b") {
                continue;
            }
            let (Some(q0), Some(q1)) = (line.find('"'), line.rfind('"')) else { continue };
            if q1 <= q0 {
                continue;
            }
            let wav = root.join("wav").join(format!("{id}.wav"));
            if wav.exists() {
                items.push((line[q0 + 1..q1].to_string(), wav));
            }
            if items.len() >= limit {
                break;
            }
        }
        score_wer(&model, &format!("CMU ARCTIC {speaker}"), &items, show);
    }
}

fn bench_speech_commands_dictation(args: &Args) {
    let model = load_conformer(args);
    let limit: usize = args.num("n", usize::MAX);
    let show: usize = args.num("show", 0);
    let root = data_root().join("speech_commands");
    let words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"];
    let mut items = Vec::new();
    for word in words {
        let mut files: Vec<PathBuf> = std::fs::read_dir(root.join(word))
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", root.join(word).display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "wav"))
            .collect();
        files.sort();
        for f in files {
            items.push((word.to_string(), f));
        }
    }
    items.truncate(limit);
    score_wer(&model, "Speech Commands (isolated words)", &items, show);
}

fn bench_fsdd(args: &Args) {
    let k: usize = args.num("k", 3);
    let fsdd = data_root().join("fsdd").join("recordings");
    let names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
    let speakers = ["george", "jackson", "lucas", "nicolas", "theo", "yweweler"];
    let results = par_map(&speakers, |speaker| {
        let mut set = CommandSet::new(8_000);
        set.max_distance = f32::INFINITY;
        set.max_ratio = f32::INFINITY;
        for (d, name) in names.iter().enumerate() {
            for take in 0..k {
                let p = fsdd.join(format!("{d}_{speaker}_{take}.wav"));
                let audio = decode_wav(&std::fs::read(p).unwrap()).unwrap();
                assert!(set.enroll(name, &audio.samples));
            }
        }
        let mut right = 0usize;
        let mut total = 0usize;
        for d in 0..10 {
            for take in k..50 {
                let p = fsdd.join(format!("{d}_{speaker}_{take}.wav"));
                let audio = decode_wav(&std::fs::read(p).unwrap()).unwrap();
                total += 1;
                if let Some(m) = set.recognize(&audio.samples) {
                    if set.labels()[m.label] == names[d] {
                        right += 1;
                    }
                }
            }
        }
        (right, total)
    });
    let right: usize = results.iter().map(|r| r.0).sum();
    let total: usize = results.iter().map(|r| r.1).sum();
    println!(
        "FSDD taught digits ({} speakers, {k} examples/digit): {:.2}% correct ({right}/{total})",
        speakers.len(),
        pct(right, total)
    );
}

fn bench_live_commands(args: &Args) {
    let k: usize = args.num("k", 1);
    let min_words: usize = args.num("minwords", 15);
    let max_speakers: usize = args.num("speakers", 400);
    let streamed = args.flag("stream");

    let root = data_root().join("speech_commands_full");
    let mut map: BTreeMap<String, BTreeMap<String, Vec<PathBuf>>> = BTreeMap::new();
    for dir in std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("cannot list {}: {e}", root.display()))
        .flatten()
    {
        let word = dir.file_name().to_string_lossy().to_string();
        if !dir.path().is_dir() || word.starts_with('_') {
            continue;
        }
        if let Ok(files) = std::fs::read_dir(dir.path()) {
            for path in files.flatten().map(|f| f.path()) {
                if path.extension().is_some_and(|x| x == "wav") {
                    let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                    let speaker = stem.split('_').next().unwrap_or("").to_string();
                    map.entry(speaker).or_default().entry(word.clone()).or_default().push(path);
                }
            }
        }
    }
    for words in map.values_mut() {
        for takes in words.values_mut() {
            takes.sort();
        }
    }

    let chosen: Vec<(&String, &BTreeMap<String, Vec<PathBuf>>)> = map
        .iter()
        .filter(|(_, words)| words.values().filter(|t| t.len() > k).count() >= min_words)
        .take(max_speakers)
        .collect();

    #[derive(Clone, Copy)]
    struct Trial {
        in_vocab: bool,
        correct: bool,
        distance: f32,
        ratio: f32,
        heard: bool,
    }

    let per_speaker: Vec<Vec<Trial>> = par_map(&chosen, |(_, words)| {
        let mut set = CommandSet::new(16_000);
        set.max_distance = f32::INFINITY;
        set.max_ratio = f32::INFINITY;
        let mut tests: Vec<(Option<String>, &PathBuf)> = Vec::new();
        for (word, takes) in words.iter() {
            if takes.len() > k {
                let mut enrolled = 0;
                for take in takes {
                    let ok = enrolled < k
                        && std::fs::read(take)
                            .ok()
                            .and_then(|b| decode_wav(&b).ok())
                            .is_some_and(|a| a.sample_rate == 16_000 && set.enroll(word, &a.samples));
                    if ok {
                        enrolled += 1;
                    } else {
                        tests.push((Some(word.clone()), take));
                    }
                }
            } else {
                for take in takes {
                    tests.push((None, take));
                }
            }
        }

        let labels = set.labels().to_vec();
        let mut trials = Vec::new();
        let mut judge = |truth: &Option<String>, m: Option<stt::Match>| {
            let in_vocab = truth.as_ref().is_some_and(|w| labels.contains(w));
            match m {
                Some(m) => trials.push(Trial {
                    in_vocab,
                    correct: truth.as_deref() == Some(labels[m.label].as_str()),
                    distance: m.distance,
                    ratio: m.runner_up.map(|(_, d)| m.distance / d).unwrap_or(0.0),
                    heard: true,
                }),
                None => trials.push(Trial {
                    in_vocab,
                    correct: false,
                    distance: f32::INFINITY,
                    ratio: 1.0,
                    heard: false,
                }),
            }
        };

        if streamed {
            let mut seg = Segmenter::new(16_000, SegmenterConfig::default());
            let mut found = Vec::new();
            let mut clip_starts = Vec::new();
            let mut pos = 0u64;
            for (_, path) in &tests {
                let clip = std::fs::read(path)
                    .ok()
                    .and_then(|b| decode_wav(&b).ok())
                    .map(|a| a.samples)
                    .unwrap_or_default();
                let mut samples = Vec::with_capacity(clip.len() + 11_200);
                if clip.len() >= 1600 {
                    let quietest = (0..=clip.len() - 1600)
                        .step_by(400)
                        .min_by(|&a, &b| {
                            let e = |s: usize| clip[s..s + 1600].iter().map(|v| v * v).sum::<f32>();
                            e(a).total_cmp(&e(b))
                        })
                        .unwrap_or(0);
                    for _ in 0..7 {
                        samples.extend_from_slice(&clip[quietest..quietest + 1600]);
                    }
                }
                samples.extend_from_slice(&clip);
                clip_starts.push((pos, pos + samples.len() as u64));
                for chunk in samples.chunks(1600) {
                    seg.push(chunk, &mut found);
                }
                pos += samples.len() as u64;
            }
            seg.flush(&mut found);
            let mut matched: Vec<Option<stt::Match>> = vec![None; tests.len()];
            for utt in &found {
                let mid = utt.start_sample + utt.samples.len() as u64 / 2;
                let Some(idx) = clip_starts.iter().position(|&(s, e)| mid >= s && mid < e) else {
                    continue;
                };
                if let Some(m) = set.recognize(&utt.samples) {
                    if matched[idx].as_ref().is_none_or(|prev| m.distance < prev.distance) {
                        matched[idx] = Some(m);
                    }
                }
            }
            for ((truth, _), m) in tests.iter().zip(matched) {
                judge(truth, m);
            }
        } else {
            for (truth, path) in &tests {
                let m = std::fs::read(path)
                    .ok()
                    .and_then(|b| decode_wav(&b).ok())
                    .and_then(|a| set.recognize(&a.samples));
                judge(truth, m);
            }
        }
        trials
    });

    let trials: Vec<_> = per_speaker.into_iter().flatten().collect();
    let known: Vec<_> = trials.iter().filter(|t| t.in_vocab).collect();
    let unknown: Vec<_> = trials.iter().filter(|t| !t.in_vocab).collect();
    println!(
        "Speech Commands taught ({} speakers, k={k}{}): {:.1}% top-1 accuracy ({} clips)",
        chosen.len(),
        if streamed { ", streamed" } else { "" },
        pct(known.iter().filter(|t| t.correct).count(), known.len()),
        known.len()
    );
    let accepted = |t: &&Trial| t.heard && t.distance <= 4.6 && t.ratio <= 0.92;
    let right = known.iter().filter(|t| t.correct && accepted(t)).count();
    let wrong = known.iter().filter(|t| !t.correct && accepted(t)).count();
    let false_accept = unknown.iter().filter(|t| accepted(t)).count();
    println!(
        "  default thresholds (4.6 / 0.92): {:.1}% accepted & right, {:.2}% accepted & wrong, {:.1}% unknown words false-accepted",
        pct(right, known.len()),
        pct(wrong, known.len()),
        pct(false_accept, unknown.len())
    );
}

fn main() {
    let mut argv = std::env::args().skip(1);
    let suite = argv.next().unwrap_or_else(|| "fsdd".to_string());
    let args = Args::parse(argv);
    match suite.as_str() {
        "librispeech" => bench_librispeech(&args),
        "arctic" => bench_arctic(&args),
        "commands" => bench_speech_commands_dictation(&args),
        "fsdd" => bench_fsdd(&args),
        "live" => bench_live_commands(&args),
        other => {
            eprintln!("Unknown benchmark {other:?}; choose: librispeech, arctic, commands, fsdd, live");
            std::process::exit(2);
        }
    }
}
