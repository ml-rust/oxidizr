//! Prep-corpus command implementation.
//!
//! Turns long audio recordings into a packed speech-LM training corpus by
//! driving boostr's [`SpeechCorpusBuilder`]: Silero VAD segments the recording,
//! Whisper transcribes each segment, NeuCodec encodes it, and the base text
//! tokenizer turns the transcript into ids. Nothing in the pipeline is
//! reimplemented here — this file walks the input, prints progress, and writes
//! the three output files.
//!
//! Three files are written, all derived from `--output`:
//!
//! - `corpus.bin` — the packed `u32` stream the trainer memory-maps,
//! - `corpus.layout.json` — the [`SpeechLayout`] sidecar
//!   [`write_speech_corpus`] writes, without which the ids are unreadable,
//! - `corpus.manifest.json` — one record per utterance: source file, sample
//!   range, seconds, transcript, token and frame counts.
//!
//! The manifest is not optional. Every transcript in the corpus is ASR output,
//! and ASR output is wrong often enough that training on it unread is a
//! mistake; the manifest is how a human reads it.
//!
//! # Whisper's dtype
//!
//! Whisper weights are cast to [`DType::F32`] on load, unconditionally, via
//! [`WhisperBundle::from_dir_with_dtype`]. `openai/whisper-large-v3` ships
//! **fp16** weights and numr requires the input and the weight to share a
//! dtype, so an fp16 bundle fed an f32 mel fails inside the first conv. The
//! other two checkpoints load at their stored f32 and the decoded samples are
//! f32, so F32 is the compute dtype end to end. Casting is chosen over probing
//! the checkpoint because numr's `cast` returns the input unchanged when the
//! dtypes already match, so an f32 checkpoint pays nothing for the call.

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use boostr::model::audio::neucodec::{NeuCodecClient, NeuCodecEncoder};
use boostr::model::audio::SileroVad;
use boostr::model::SpeechLayout;
use boostr::runtime::Runtime;
use boostr::{DType, MatmulOps, TypeConversionOps};
use voxr::enhance::EnhanceOptions;
use voxr::{
    check_max_speech_duration, decode_audio_file_mono_at, CorpusOptions, SpeechCorpusBuilder,
    TextTokenizer, Utterance, VadSegmentOptions, WhisperBundle, MAX_UTTERANCE_SECS,
    PRETRAINED_TOKENIZER_NAMES,
};

use crate::speech_corpus::write_speech_corpus;

/// Extensions a directory walk accepts. Everything else in the directory —
/// transcripts, `.DS_Store`, a nested folder — is ignored.
const AUDIO_EXTENSIONS: [&str; 6] = ["flac", "wav", "mp3", "ogg", "m4a", "opus"];

/// Which speech id layout the corpus is packed into.
///
/// Not a detail: the layout decides what every audio id MEANS, and a `.bin` is
/// headerless, so the wrong one trains a base on ids that address different
/// embedding rows with no error anywhere. The sidecar records the choice; see
/// [`crate::speech_corpus`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum LayoutArg {
    /// boostr's own layout, over a derived `SpeechVocab`.
    Native,
    /// The `Multilingual-Expressive-TTS-1.7B` layout.
    ExpressiveTts,
}

/// Prep-corpus command arguments
#[derive(Args, Debug)]
#[command(
    long_about = "Turn long audio recordings into a packed speech-LM training corpus.

Each recording is segmented by the Silero VAD, each segment transcribed by
Whisper and encoded by NeuCodec, and the whole lot packed into one token
stream with a layout sidecar and a reviewable manifest.

Whisper large-v3 runs at roughly five minutes per 30 s of audio on CPU, so
CUDA (build with --features cuda) is the only practical path for a real
corpus. A CPU run is for smoke-testing a handful of clips."
)]
pub struct PrepCorpusArgs {
    /// Audio file, or a directory of audio files (walked non-recursively).
    /// Accepted extensions: flac, wav, mp3, ogg, m4a, opus
    #[arg(short = 'i', long = "input")]
    pub input: String,

    /// Output corpus path (.bin). A `.layout.json` and a `.manifest.json` are
    /// written beside it
    #[arg(short = 'o', long = "output")]
    pub output: String,

    /// Silero VAD checkpoint (silero_vad_16k.safetensors)
    #[arg(long = "vad", required_unless_present = "scripted")]
    pub vad: Option<String>,

    /// Whisper checkpoint directory (config.json + *.safetensors). fp16
    /// checkpoints such as whisper-large-v3 are cast to f32 on load. On CPU
    /// large-v3 costs roughly five minutes per 30 s of audio — build with
    /// --features cuda for a real corpus
    #[arg(long = "whisper", required_unless_present = "scripted")]
    pub whisper: Option<String>,

    /// NeuCodec checkpoint (file, or the directory holding model.safetensors)
    #[arg(long = "neucodec")]
    pub neucodec: String,

    /// Base text tokenizer: a splintr vocabulary name (cl100k_base,
    /// o200k_base, llama3, deepseek_v3) or a path to a tokenizer.json.
    /// Under --layout expressive-tts this MUST be the base checkpoint's own
    /// tokenizer.json: the ids line up with that vocabulary and no other
    #[arg(long = "tokenizer")]
    pub tokenizer: String,

    /// Speech id layout to pack into. expressive-tts matches the
    /// Multilingual-Expressive-TTS-1.7B base, the only layout with a real base
    /// checkpoint today. native is boostr's own layout, for training a speech
    /// LM from scratch on boostr, where nothing constrains the id space
    #[arg(long = "layout", value_enum, default_value = "expressive-tts")]
    pub layout: LayoutArg,

    /// Speaker name, rendered as the literal prefix "{speaker}: " before the
    /// transcript is tokenized. A consistent value is how the base conditions
    /// on speaker identity, so a voice-cloning corpus should set it. Omitted
    /// leaves each transcript exactly as Whisper produced it
    #[arg(long = "speaker")]
    pub speaker: Option<String>,

    /// Whisper language code, e.g. "ms". Omitted lets Whisper auto-detect
    #[arg(long = "language")]
    pub language: Option<String>,

    /// Drop utterances shorter than this, before transcription
    #[arg(long = "min-utterance-secs", default_value = "0.5")]
    pub min_utterance_secs: f32,

    /// Cap on one utterance. Whisper's encoder window is fixed at 30 s, so
    /// this may not exceed it
    #[arg(long = "max-utterance-secs", default_value = "30.0")]
    pub max_utterance_secs: f32,

    /// Silence needed to END an utterance, in milliseconds.
    ///
    /// The VAD default of 100 ms splits on breath, not on sentences: ordinary
    /// speech pauses 100-300 ms mid-sentence, so a 100 ms threshold yields
    /// fragments like "In my" and "just now". 400-500 ms lands utterances on
    /// sentence boundaries, which is what a TTS corpus wants
    #[arg(long = "min-silence-ms", default_value = "100")]
    pub min_silence_ms: u32,

    /// Speech probability above which a chunk counts as speech, 0-1 exclusive.
    ///
    /// Raise it for noisy recordings, lower it if quiet speech is being dropped
    #[arg(long = "vad-threshold", default_value = "0.5")]
    pub vad_threshold: f32,

    /// Audio kept either side of a detected utterance, in milliseconds.
    ///
    /// Too little clips consonants at the edges; too much drags in neighbouring
    /// speech
    #[arg(long = "speech-pad-ms", default_value = "30")]
    pub speech_pad_ms: u32,

    /// Pad the packed stream to a multiple of this many tokens
    #[arg(long = "pad-to-multiple")]
    pub pad_to_multiple: Option<usize>,

    /// TSV mapping each recording to the exact text that was read, turning this
    /// into a SCRIPTED run: no VAD, no Whisper, one utterance per file.
    ///
    /// Columns `file` (basename, with or without `.wav`) and `text`; any other
    /// columns are ignored, so a recorder's own manifest can be passed as-is.
    ///
    /// Use this whenever the text is known, which for prompted recording is
    /// always. ASR over 52 prompted takes scored 31.6% WER against the prompts
    /// that produced them, mostly because `--language` forces one language on a
    /// code-switching corpus and translates the rest: "Before we go further,
    /// save what you have right now" came back in Malay. That pairs English
    /// audio with Malay text, and nothing downstream can notice.
    ///
    /// Skips loading the VAD and Whisper checkpoints entirely, so `--vad` and
    /// `--whisper` are not required with it.
    #[arg(long = "scripted", value_name = "TSV")]
    pub scripted: Option<String>,

    /// Clean up each recording before segmenting it: rumble filter, spectral
    /// denoise, tone shaping, then loudness normalization under a peak ceiling.
    ///
    /// Leave this off for audio that is already treated. Turn it on for raw
    /// recordings — a reference cut from an untreated take was measured to
    /// clone at -36 LUFS against -14 LUFS from the same segment enhanced,
    /// because the model inherits the level and the noise floor of what it is
    /// given.
    ///
    /// Runs over the whole recording, before the VAD, never per utterance: the
    /// denoiser reads its noise profile from the pauses between phrases, and
    /// one gain per recording keeps the relative loudness between utterances
    /// that per-clip normalization would erase.
    #[arg(long = "enhance")]
    pub enhance: bool,

    /// Loudness target for --enhance, in LUFS.
    #[arg(
        long = "enhance-target-lufs",
        default_value = "-18.0",
        requires = "enhance"
    )]
    pub enhance_target_lufs: f64,

    /// Low-shelf gain for --enhance, in dB. Restores the weight the rumble
    /// filter and the denoiser take off the bottom.
    #[arg(long = "enhance-bass-db", default_value = "3.0", requires = "enhance")]
    pub enhance_bass_db: f64,

    /// Skip the denoiser while keeping the rest of --enhance. Use when the
    /// recording is quiet but clean, or when a gate must not touch the signal.
    #[arg(long = "enhance-no-denoise", requires = "enhance")]
    pub enhance_no_denoise: bool,

    /// Override target device: "gpu" or "cpu" (default gpu when built with
    /// the cuda feature)
    #[arg(long = "target-device")]
    pub target_device: Option<String>,
}

/// Everything decided before a checkpoint is touched, so a bad argument costs
/// no model load.
struct Plan<'a> {
    files: Vec<PathBuf>,
    output: &'a Path,
    /// Absent in a scripted run, which loads neither checkpoint.
    vad_path: Option<&'a Path>,
    whisper_dir: Option<&'a Path>,
    /// `file stem -> exact transcript`, present only for a scripted run.
    scripted: Option<HashMap<String, String>>,
    neucodec_path: &'a Path,
    tokenizer: TextTokenizer<'a>,
    /// Chosen here, resolved into a [`SpeechLayout`] only by the builder: the
    /// native layout's `SpeechVocab` is derived from the loaded tokenizer and
    /// encoder, neither of which exists yet at plan time.
    layout: LayoutArg,
    opts: CorpusOptions<'a>,
}

/// One utterance as a human reviews it.
#[derive(Serialize)]
struct ManifestEntry {
    source: String,
    start_sample: usize,
    end_sample: usize,
    start_secs: f64,
    end_secs: f64,
    text: String,
    text_tokens: usize,
    frames: usize,
}

/// The manifest written beside the corpus.
#[derive(Serialize)]
struct Manifest {
    corpus: String,
    sample_rate: usize,
    total_tokens: usize,
    utterances: Vec<ManifestEntry>,
}

/// Path of the manifest beside a corpus: `foo.bin` -> `foo.manifest.json`.
///
/// Derived the same way [`crate::speech_corpus::layout_sidecar_path`] derives
/// the layout sidecar, so the three files share one stem.
fn manifest_path(output: &Path) -> PathBuf {
    output.with_extension("manifest.json")
}

/// Decide where the base text tokenizer comes from.
///
/// A value ending in `.json` names a file; anything else is a splintr
/// vocabulary name. A `.json` value naming no file is a hard error rather than
/// a fall-through to the name branch: the fall-through reports "unknown
/// pretrained tokenizer name /path/to/tokenizer.json", which hides a typo in a
/// path behind an error about vocabularies.
fn resolve_tokenizer(value: &str) -> Result<TextTokenizer<'_>> {
    let path = Path::new(value);
    let is_json = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
    if !is_json {
        return Ok(TextTokenizer::Pretrained(value));
    }
    if !path.is_file() {
        return Err(anyhow::anyhow!(
            "--tokenizer '{value}' ends in .json but names no file. Pass a tokenizer.json \
             path that exists, or one of the vocabulary names: {}.",
            PRETRAINED_TOKENIZER_NAMES.join(", ")
        ));
    }
    Ok(TextTokenizer::JsonFile(path))
}

/// Whether a path carries one of [`AUDIO_EXTENSIONS`].
fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            AUDIO_EXTENSIONS
                .iter()
                .any(|known| ext.eq_ignore_ascii_case(known))
        })
}

/// The recordings to prepare, in a stable order.
///
/// A file is used as-is, whatever its extension: naming one explicitly is a
/// statement that it is audio. A directory is walked NON-recursively and
/// filtered to [`AUDIO_EXTENSIONS`], then sorted by filename so two runs over
/// the same directory pack utterances in the same order.
fn collect_inputs(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if !input.is_dir() {
        return Err(anyhow::anyhow!(
            "--input '{}' is neither a file nor a directory.",
            input.display()
        ));
    }

    // An entry that cannot be read is an ERROR, not something to skip. Silently
    // dropping one means a recording quietly missing from the corpus, and the
    // only symptom is a training set that is smaller than the operator thinks.
    let entries = fs::read_dir(input)
        .with_context(|| format!("Failed to read directory '{}'", input.display()))?;
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| {
            format!("Failed to read an entry of directory '{}'", input.display())
        })?;
        let path = entry.path();
        if path.is_file() && is_audio_file(&path) {
            files.push(path);
        }
    }
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    if files.is_empty() {
        return Err(anyhow::anyhow!(
            "--input '{}' holds no audio file. The walk is non-recursive and accepts \
             these extensions: {}.",
            input.display(),
            AUDIO_EXTENSIONS.join(", ")
        ));
    }
    Ok(files)
}

/// Refuse an utterance cap Whisper cannot transcribe, before anything loads.
///
/// The rule is boostr's own [`check_max_speech_duration`]; only the wording is
/// ours, so the error names the flag the operator typed rather than the option
/// field it lands in.
fn check_max_utterance_secs(secs: f32) -> Result<()> {
    check_max_speech_duration(secs).with_context(|| {
        format!(
            "--max-utterance-secs {secs} is not usable: every segment is transcribed by \
             Whisper, whose encoder window is fixed at {MAX_UTTERANCE_SECS} s, so the cap \
             must be greater than zero and at most {MAX_UTTERANCE_SECS}."
        )
    })
}

/// Turn the parsed arguments into a checked plan.
fn plan(args: &PrepCorpusArgs) -> Result<Plan<'_>> {
    check_max_utterance_secs(args.max_utterance_secs)?;
    // The segmenter rejects a non-finite or out-of-range threshold, but only
    // once the checkpoints have loaded. Catch it here instead of after a 3 GB
    // Whisper load.
    if !args.vad_threshold.is_finite() || args.vad_threshold <= 0.0 || args.vad_threshold > 1.0 {
        return Err(anyhow::anyhow!(
            "--vad-threshold {} is not usable: it must be in (0, 1]",
            args.vad_threshold
        ));
    }

    let files = collect_inputs(Path::new(&args.input))?;

    // Read the transcripts at PLAN time, before any checkpoint is touched: a
    // malformed TSV should cost nothing, and a recording with no transcript
    // must be reported before a multi-gigabyte load, not after.
    let scripted = match &args.scripted {
        Some(path) => {
            let map = read_scripted(Path::new(path))?;
            let missing: Vec<String> = files
                .iter()
                .filter_map(|f| f.file_stem().map(|s| s.to_string_lossy().into_owned()))
                .filter(|stem| !map.contains_key(stem))
                .collect();
            if !missing.is_empty() {
                return Err(anyhow::anyhow!(
                    "--scripted {} has no transcript for {} recording(s): {}{}",
                    path,
                    missing.len(),
                    missing
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", "),
                    if missing.len() > 5 { ", ..." } else { "" }
                ));
            }
            Some(map)
        }
        None => None,
    };
    let tokenizer = resolve_tokenizer(&args.tokenizer)?;

    let defaults = CorpusOptions::default();
    let opts = CorpusOptions {
        vad: VadSegmentOptions {
            max_speech_duration_s: args.max_utterance_secs,
            min_silence_duration_ms: args.min_silence_ms,
            threshold: args.vad_threshold,
            speech_pad_ms: args.speech_pad_ms,
            ..defaults.vad
        },
        language: args.language.as_deref(),
        speaker: args.speaker.as_deref(),
        min_utterance_secs: args.min_utterance_secs,
        pad_to_multiple: args.pad_to_multiple,
        enhance: args.enhance.then(|| EnhanceOptions {
            target_lufs: args.enhance_target_lufs,
            bass_boost_db: args.enhance_bass_db,
            denoise: if args.enhance_no_denoise {
                None
            } else {
                EnhanceOptions::default().denoise
            },
            ..EnhanceOptions::default()
        }),
        ..defaults
    };

    Ok(Plan {
        files,
        output: Path::new(&args.output),
        vad_path: args.vad.as_deref().map(Path::new),
        whisper_dir: args.whisper.as_deref().map(Path::new),
        scripted,
        neucodec_path: Path::new(&args.neucodec),
        tokenizer,
        layout: args.layout,
        opts,
    })
}

/// Read a `--scripted` TSV into `file stem -> transcript`.
///
/// Columns `file` and `text` are required; everything else is ignored so a
/// recorder's own manifest can be passed unchanged. The `.wav` suffix is
/// optional in the `file` column, and matching is by stem, so a manifest that
/// names `C07_take3.wav` matches the recording of that name.
fn read_scripted(path: &Path) -> Result<HashMap<String, String>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading --scripted {}", path.display()))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("--scripted {} is empty", path.display()))?
        .split('\t')
        .collect();
    let col = |name: &str| {
        header.iter().position(|h| h.trim() == name).ok_or_else(|| {
            anyhow::anyhow!(
                "--scripted {} has no `{name}` column; found [{}]",
                path.display(),
                header.join(", ")
            )
        })
    };
    let (fi, ti) = (col("file")?, col("text")?);

    let mut out = HashMap::new();
    for (n, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split('\t').collect();
        let (Some(file), Some(t)) = (cells.get(fi), cells.get(ti)) else {
            return Err(anyhow::anyhow!(
                "--scripted {} line {} has {} column(s), needs at least {}",
                path.display(),
                n + 2,
                cells.len(),
                fi.max(ti) + 1
            ));
        };
        let stem = Path::new(file.trim())
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if stem.is_empty() || t.trim().is_empty() {
            continue;
        }
        out.insert(stem, t.trim().to_string());
    }
    if out.is_empty() {
        return Err(anyhow::anyhow!(
            "--scripted {} lists no usable rows",
            path.display()
        ));
    }
    Ok(out)
}

/// Load the three checkpoints and the tokenizer into one builder.
fn load_builder<R, C>(
    plan: &Plan<'_>,
    device: &R::Device,
    client: &C,
) -> Result<SpeechCorpusBuilder<R>>
where
    R: Runtime<DType = DType>,
    R::Client: NeuCodecClient<R>,
    C: TypeConversionOps<R>,
{
    // A scripted run loads neither the VAD nor Whisper: it has the transcript
    // and each recording is one utterance, so there is nothing for them to do
    // and no reason to spend the load time or the memory.
    if plan.scripted.is_some() {
        println!("  NeuCodec:  {}", plan.neucodec_path.display());
        let codec = NeuCodecEncoder::<R>::from_safetensors(plan.neucodec_path, device)?;
        println!("  Tokenizer: {:?}", plan.tokenizer);
        println!("  Layout:    {}", layout_name(plan.layout));
        println!("  Mode:      scripted (no VAD, no ASR)");
        return match plan.layout {
            LayoutArg::ExpressiveTts => {
                SpeechCorpusBuilder::new_scripted_expressive_tts(codec, plan.tokenizer)
                    .map_err(Into::into)
            }
            LayoutArg::Native => Err(anyhow::anyhow!(
                "--scripted currently supports --layout expressive-tts only: the native \
                 layout derives its SpeechVocab from the loaded models, and a scripted \
                 run does not load them"
            )),
        };
    }

    let vad_path = plan
        .vad_path
        .ok_or_else(|| anyhow::anyhow!("--vad is required without --scripted"))?;
    let whisper_dir = plan
        .whisper_dir
        .ok_or_else(|| anyhow::anyhow!("--whisper is required without --scripted"))?;

    println!("  VAD:       {}", vad_path.display());
    let vad = SileroVad::<R>::from_safetensors(vad_path, device)?;

    println!("  Whisper:   {}", whisper_dir.display());
    let whisper = WhisperBundle::<R>::from_dir_with_dtype(whisper_dir, device, client, DType::F32)?;

    println!("  NeuCodec:  {}", plan.neucodec_path.display());
    let codec = NeuCodecEncoder::<R>::from_safetensors(plan.neucodec_path, device)?;

    println!("  Tokenizer: {:?}", plan.tokenizer);
    println!("  Layout:    {}", layout_name(plan.layout));
    Ok(match plan.layout {
        LayoutArg::Native => SpeechCorpusBuilder::new(vad, whisper, codec, plan.tokenizer)?,
        LayoutArg::ExpressiveTts => {
            SpeechCorpusBuilder::new_expressive_tts(vad, whisper, codec, plan.tokenizer)?
        }
    })
}

/// The layout's name as the sidecar and boostr's errors spell it.
const fn layout_name(layout: LayoutArg) -> &'static str {
    match layout {
        LayoutArg::Native => "boostr-native",
        LayoutArg::ExpressiveTts => "expressive-tts-1.7b",
    }
}

/// Turns one decoded recording (path, mono samples) into its utterances.
///
/// A named type: the closure signature is the backend seam, and clippy's
/// `type_complexity` lint fires on it inline.
type UtterancesOf<'a> = dyn Fn(&Path, &[f32]) -> Result<Vec<Utterance>> + 'a;

/// Decode every recording, prepare its utterances, then pack and write once.
///
/// Runtime-free: the two model-touching steps arrive as closures, so this — the
/// whole of the walk, the reporting, and the writing — exists once rather than
/// per backend.
fn drive(
    plan: &Plan<'_>,
    layout: &SpeechLayout,
    utterances_of: &UtterancesOf<'_>,
    pack: &dyn Fn(&[Utterance]) -> Result<Vec<u32>>,
) -> Result<()> {
    let sample_rate = plan.opts.sample_rate;
    let decode_rate = u32::try_from(sample_rate)
        .with_context(|| format!("sample rate {sample_rate} does not fit in u32"))?;

    let mut utterances: Vec<Utterance> = Vec::new();
    let mut sources: Vec<String> = Vec::new();
    let mut total_speech_secs = 0.0_f64;

    println!("\nPreparing {} recording(s)...", plan.files.len());
    for (index, file) in plan.files.iter().enumerate() {
        let name = file
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| file.display().to_string());

        let samples = decode_audio_file_mono_at(file, decode_rate)
            .with_context(|| format!("Failed to decode '{}'", file.display()))?;

        let prepared = utterances_of(file, &samples)
            .with_context(|| format!("Failed to prepare utterances from '{}'", file.display()))?;

        let speech_secs: f64 = prepared
            .iter()
            .map(|utterance| utterance.segment.duration_secs(sample_rate))
            .sum();
        total_speech_secs += speech_secs;

        println!(
            "  [{}/{}] {} — {} utterance(s), {:.1} s of speech",
            index + 1,
            plan.files.len(),
            name,
            prepared.len(),
            speech_secs
        );

        let source = file.display().to_string();
        for utterance in prepared {
            sources.push(source.clone());
            utterances.push(utterance);
        }
    }

    if utterances.is_empty() {
        return Err(anyhow::anyhow!(
            "no utterance survived across {} recording(s): every segment was shorter than \
             --min-utterance-secs {} or transcribed to nothing. Nothing was written.",
            plan.files.len(),
            plan.opts.min_utterance_secs
        ));
    }

    println!("\n  Packing {} utterance(s)...", utterances.len());
    let tokens = pack(&utterances)?;

    if let Some(parent) = plan.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory '{}'", parent.display()))?;
    }

    println!("  Writing corpus...");
    write_speech_corpus(plan.output, &tokens, layout)?;
    write_manifest(plan, &sources, &utterances, tokens.len())?;

    println!("\n✅ Corpus complete!");
    println!("   Files:      {}", plan.files.len());
    println!("   Utterances: {}", utterances.len());
    println!(
        "   Speech:     {:.1} s ({:.2} h)",
        total_speech_secs,
        total_speech_secs / 3600.0
    );
    println!("   Tokens:     {}", tokens.len());
    println!("\nOutput files:");
    println!("  - {}", plan.output.display());
    println!(
        "  - {}",
        crate::speech_corpus::layout_sidecar_path(plan.output).display()
    );
    println!("  - {}", manifest_path(plan.output).display());
    println!("\nNext step: read the manifest before training on it.");

    Ok(())
}

/// Write the reviewable record of every utterance in the corpus.
fn write_manifest(
    plan: &Plan<'_>,
    sources: &[String],
    utterances: &[Utterance],
    total_tokens: usize,
) -> Result<()> {
    let sample_rate = plan.opts.sample_rate;
    let rate = sample_rate as f64;
    let entries: Vec<ManifestEntry> = sources
        .iter()
        .zip(utterances.iter())
        .map(|(source, utterance)| ManifestEntry {
            source: source.clone(),
            start_sample: utterance.segment.start,
            end_sample: utterance.segment.end,
            start_secs: utterance.segment.start as f64 / rate,
            end_secs: utterance.segment.end as f64 / rate,
            text: utterance.text.clone(),
            text_tokens: utterance.text_tokens.len(),
            frames: utterance.frames.len(),
        })
        .collect();

    let manifest = Manifest {
        corpus: plan.output.display().to_string(),
        sample_rate,
        total_tokens,
        utterances: entries,
    };

    let path = manifest_path(plan.output);
    let file = fs::File::create(&path)
        .with_context(|| format!("Failed to create manifest '{}'", path.display()))?;
    serde_json::to_writer_pretty(&file, &manifest)
        .with_context(|| format!("Failed to write manifest '{}'", path.display()))?;
    file.sync_all()
        .with_context(|| format!("Failed to flush manifest '{}'", path.display()))?;
    Ok(())
}

/// Run the prep-corpus command
pub fn run_prep_corpus(args: PrepCorpusArgs) -> Result<()> {
    println!("🎙️  Oxidizr Speech Corpus Preparation");
    println!("====================================\n");

    let plan = plan(&args)?;
    println!("Input:      {}", args.input);
    println!("Recordings: {}", plan.files.len());
    println!("Output:     {}", plan.output.display());
    println!(
        "Language:   {}",
        plan.opts.language.unwrap_or("auto-detect")
    );
    println!("Layout:     {}", layout_name(plan.layout));
    println!("Speaker:    {}", plan.opts.speaker.unwrap_or("none"));
    println!(
        "Utterance:  {:.2} s .. {:.2} s\n",
        plan.opts.min_utterance_secs, plan.opts.vad.max_speech_duration_s
    );

    #[cfg(feature = "cuda")]
    {
        let use_cpu = args
            .target_device
            .as_deref()
            .is_some_and(|dev| dev.eq_ignore_ascii_case("cpu"));
        if !use_cpu {
            return run_with_runtime::<boostr::CudaRuntime>(&plan);
        }
    }

    println!(
        "  Whisper large-v3 on CPU runs at roughly five minutes per 30 s of audio.\n  \
         CUDA (--features cuda) is the only practical path for a corpus of any size."
    );
    run_with_runtime::<boostr::CpuRuntime>(&plan)
}

/// Load the models on `R` and run the corpus over them.
///
/// The two model-touching steps reach [`drive`] as closures, so the walk, the
/// reporting and the writing stay backend-free.
fn run_with_runtime<R>(plan: &Plan<'_>) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: NeuCodecClient<R> + MatmulOps<R> + TypeConversionOps<R>,
{
    println!("Using runtime: {}", R::name());

    let device = R::default_device();
    let client = R::default_client(&device);
    let builder = load_builder::<R, _>(plan, &device, &client)?;
    let opts = &plan.opts;

    drive(
        plan,
        builder.layout(),
        &|file: &Path, samples: &[f32]| match &plan.scripted {
            // Scripted: the transcript is known, so the whole recording is one
            // utterance and neither the segmenter nor the recognizer runs.
            // `read_scripted` already proved every file has an entry.
            Some(map) => {
                let stem = file
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let text = map.get(&stem).ok_or_else(|| {
                    anyhow::anyhow!("no transcript for {stem} (checked at plan time)")
                })?;
                builder
                    .scripted_utterance(&client, samples, text, opts)
                    .map(|u| vec![u])
                    .map_err(anyhow::Error::from)
            }
            None => builder
                .utterances(&client, samples, opts)
                .map_err(anyhow::Error::from),
        },
        &|utterances: &[Utterance]| builder.pack(utterances, opts).map_err(anyhow::Error::from),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The subcommand's arguments behind a parser, so the tests exercise the
    /// real clap surface — defaults and value-enum errors included — rather
    /// than a struct literal that would define its own defaults.
    #[derive(Parser, Debug)]
    struct Cli {
        #[command(flatten)]
        args: PrepCorpusArgs,
    }

    /// Parse the required flags plus `extra`.
    fn parse(extra: &[&str]) -> std::result::Result<PrepCorpusArgs, clap::Error> {
        parse_in("recording.wav", extra)
    }

    /// Same, but with a caller-chosen `--input`, for the tests that need the
    /// path to exist on disk so `plan()` can walk it.
    fn parse_in(input: &str, extra: &[&str]) -> std::result::Result<PrepCorpusArgs, clap::Error> {
        let mut argv = vec![
            "prep-corpus",
            "--input",
            input,
            "--output",
            "corpus.bin",
            "--vad",
            "silero_vad_16k.safetensors",
            "--whisper",
            "whisper-large-v3",
            "--neucodec",
            "neucodec",
            "--tokenizer",
            "cl100k_base",
        ];
        argv.extend_from_slice(extra);
        Cli::try_parse_from(argv).map(|cli| cli.args)
    }

    /// A fresh directory under the system temp dir. `tempfile` is not a
    /// dependency of this crate, and `tests/speech_corpus.rs` already builds
    /// its scratch directories this way.
    fn scratch_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "oxidizr-prep-corpus-{}-{}-{}",
            std::process::id(),
            name,
            n
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, b"not really audio").unwrap();
        path
    }

    fn names(files: &[PathBuf]) -> Vec<String> {
        files
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn a_bare_name_is_a_pretrained_vocabulary() {
        let tokenizer = resolve_tokenizer("cl100k_base").expect("a vocabulary name must resolve");
        assert!(
            matches!(tokenizer, TextTokenizer::Pretrained("cl100k_base")),
            "{tokenizer:?}"
        );
    }

    #[test]
    fn an_existing_json_path_is_a_tokenizer_file() {
        let dir = scratch_dir("tokenizer-json");
        let path = touch(&dir, "tokenizer.json");
        let value = path.to_string_lossy().into_owned();

        let tokenizer = resolve_tokenizer(&value).expect("an existing tokenizer.json must resolve");

        assert!(
            matches!(tokenizer, TextTokenizer::JsonFile(p) if p == path),
            "{tokenizer:?}"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_json_path_is_not_read_as_a_vocabulary_name() {
        // The failure mode this guards: a typo'd path reported as "unknown
        // pretrained tokenizer name", which sends the operator to the wrong
        // place entirely.
        let message = resolve_tokenizer("/no/such/dir/tokenizer.json")
            .expect_err("a .json naming no file must fail")
            .to_string();

        assert!(message.contains("/no/such/dir/tokenizer.json"), "{message}");
    }

    #[test]
    fn a_directory_yields_only_audio_files_sorted() {
        let dir = scratch_dir("mixed");
        touch(&dir, "b_second.wav");
        touch(&dir, "a_first.flac");
        touch(&dir, "c_third.opus");
        touch(&dir, "notes.txt");
        touch(&dir, "transcript.json");
        touch(&dir, "cover.png");
        fs::create_dir(dir.join("nested.wav")).unwrap();

        let files = collect_inputs(&dir).expect("a directory of audio must walk");

        assert_eq!(
            names(&files),
            vec![
                "a_first.flac".to_string(),
                "b_second.wav".to_string(),
                "c_third.opus".to_string(),
            ]
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn extension_matching_ignores_case() {
        let dir = scratch_dir("case");
        touch(&dir, "LOUD.WAV");
        touch(&dir, "quiet.Mp3");

        let files = collect_inputs(&dir).expect("upper-case extensions are still audio");

        assert_eq!(
            names(&files),
            vec!["LOUD.WAV".to_string(), "quiet.Mp3".to_string()]
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_empty_directory_is_an_error() {
        let dir = scratch_dir("empty");

        let message = collect_inputs(&dir)
            .expect_err("an empty directory must not silently prepare nothing")
            .to_string();

        assert!(message.contains("no audio file"), "{message}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_directory_without_audio_is_an_error() {
        let dir = scratch_dir("no-audio");
        touch(&dir, "readme.md");
        touch(&dir, "labels.csv");

        let message = collect_inputs(&dir)
            .expect_err("a directory of non-audio must not prepare nothing")
            .to_string();

        assert!(message.contains("no audio file"), "{message}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_single_file_is_used_as_given() {
        let dir = scratch_dir("single");
        let path = touch(&dir, "recording.flac");

        let files = collect_inputs(&path).expect("a named file must be used");

        assert_eq!(files, vec![path]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_named_file_bypasses_the_extension_filter() {
        // Naming a file explicitly says it is audio; the extension list only
        // exists to filter a directory walk.
        let dir = scratch_dir("odd-extension");
        let path = touch(&dir, "recording.aiff");

        let files = collect_inputs(&path).expect("an explicitly named file must be used");

        assert_eq!(files, vec![path]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_input_is_an_error() {
        let message = collect_inputs(Path::new("/no/such/dir/recordings"))
            .expect_err("a missing input must fail")
            .to_string();

        assert!(message.contains("/no/such/dir/recordings"), "{message}");
    }

    #[test]
    fn the_whisper_window_is_the_upper_bound() {
        check_max_utterance_secs(MAX_UTTERANCE_SECS).expect("the window itself must be accepted");
        check_max_utterance_secs(12.5).expect("a shorter cap must be accepted");

        let message = check_max_utterance_secs(MAX_UTTERANCE_SECS + 15.0)
            .expect_err("a cap past Whisper's window must be rejected")
            .to_string();
        assert!(message.contains("max-utterance-secs"), "{message}");

        let zero = check_max_utterance_secs(0.0)
            .expect_err("a zero cap must be rejected")
            .to_string();
        assert!(zero.contains("max-utterance-secs"), "{zero}");
    }

    #[test]
    fn the_default_layout_is_the_base_checkpoints_own() {
        // expressive-tts is the only layout with a real base checkpoint, so a
        // corpus prepared without thinking about layouts is the one that base
        // can read.
        let args = parse(&[]).expect("the required flags alone must parse");
        assert_eq!(args.layout, LayoutArg::ExpressiveTts);
    }

    #[test]
    fn both_layouts_parse() {
        assert_eq!(
            parse(&["--layout", "native"])
                .expect("native must parse")
                .layout,
            LayoutArg::Native
        );
        assert_eq!(
            parse(&["--layout", "expressive-tts"])
                .expect("expressive-tts must parse")
                .layout,
            LayoutArg::ExpressiveTts
        );
    }

    #[test]
    fn an_unknown_layout_is_rejected_with_the_accepted_ones() {
        let message = parse(&["--layout", "mamba"])
            .expect_err("an unknown layout must not fall back to a default")
            .to_string();

        assert!(message.contains("mamba"), "{message}");
        assert!(message.contains("native"), "{message}");
        assert!(message.contains("expressive-tts"), "{message}");
    }

    #[test]
    fn the_speaker_reaches_the_corpus_options() {
        let dir = scratch_dir("speaker");
        let path = touch(&dir, "recording.wav");
        let input = path.to_string_lossy().into_owned();

        let args = parse_in(&input, &["--speaker", "Rahman"]).expect("a speaker name must parse");
        let planned = plan(&args).expect("a named audio file must plan");

        assert_eq!(planned.opts.speaker, Some("Rahman"));

        // And absent by default: the native layout has never prefixed anything.
        let bare = parse_in(&input, &[]).expect("the required flags alone must parse");
        assert_eq!(plan(&bare).unwrap().opts.speaker, None);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_manifest_sits_beside_the_corpus() {
        assert_eq!(
            manifest_path(Path::new("data/speech/corpus.bin")),
            PathBuf::from("data/speech/corpus.manifest.json")
        );
        // Same derivation as the layout sidecar: replace the final extension,
        // or append when there is none.
        assert_eq!(
            manifest_path(Path::new("corpus")),
            PathBuf::from("corpus.manifest.json")
        );
    }

    #[test]
    fn the_three_output_files_share_one_stem() {
        let output = Path::new("out/ms-speech.bin");
        assert_eq!(
            crate::speech_corpus::layout_sidecar_path(output),
            PathBuf::from("out/ms-speech.layout.json")
        );
        assert_eq!(
            manifest_path(output),
            PathBuf::from("out/ms-speech.manifest.json")
        );
    }

    /// The VAD knobs must actually reach `VadSegmentOptions`. `min_silence_ms`
    /// is the one that decides whether utterances are sentences or fragments.
    #[test]
    fn the_vad_flags_reach_the_segment_options() {
        let dir = scratch_dir("vad-flags");
        let path = touch(&dir, "recording.wav");
        let input = path.to_string_lossy().into_owned();

        let args = parse_in(
            &input,
            &[
                "--min-silence-ms",
                "450",
                "--vad-threshold",
                "0.6",
                "--speech-pad-ms",
                "80",
            ],
        )
        .expect("the vad flags must parse");
        let planned = plan(&args).expect("a named audio file must plan");

        assert_eq!(planned.opts.vad.min_silence_duration_ms, 450);
        assert_eq!(planned.opts.vad.threshold, 0.6);
        assert_eq!(planned.opts.vad.speech_pad_ms, 80);

        // Defaults match the VAD's own, so omitting them changes nothing.
        let bare = parse_in(&input, &[]).expect("required flags alone must parse");
        let bare = plan(&bare).expect("plan");
        assert_eq!(bare.opts.vad.min_silence_duration_ms, 100);
        assert_eq!(bare.opts.vad.threshold, 0.5);
        assert_eq!(bare.opts.vad.speech_pad_ms, 30);
    }

    /// An out-of-range threshold is refused BEFORE a 3 GB Whisper load.
    #[test]
    fn a_nonsensical_vad_threshold_is_refused_up_front() {
        let dir = scratch_dir("vad-threshold");
        let path = touch(&dir, "recording.wav");
        let input = path.to_string_lossy().into_owned();

        // A negative value needs the `=` form: clap reads a bare `-0.2` as a flag.
        for bad in [
            "--vad-threshold=0",
            "--vad-threshold=1.5",
            "--vad-threshold=-0.2",
            "--vad-threshold=nan",
        ] {
            let args =
                parse_in(&input, &[bad]).expect("clap accepts any f32; the range check is ours");
            assert!(
                plan(&args).is_err(),
                "--vad-threshold {bad} must be refused"
            );
        }
    }
}
