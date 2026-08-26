//! End-to-end: pack speech records with boostr, write them as a corpus, load
//! them back through the training data loader, and check the audio-only loss
//! mask lands on exactly the right positions.
//!
//! The mask expectation here is computed INDEPENDENTLY from the packed stream
//! using the layout's arithmetic constants, never with the helper the loader
//! uses. A test that re-derived the mask with the loader's own helper would pass
//! for any consistent-but-wrong shift.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use boostr::model::{
    pack_records, CodecVocab, SpeechLayout, SpeechRecord, SpeechVocab, ALL_SPECIAL_TOKENS,
};
use boostr::{CpuDevice, CpuRuntime};

use oxidizr::data::{write_tokens_to_file, LitDataLoader};
use oxidizr::speech_corpus::{
    check_corpus_layout, layout_sidecar_path, read_corpus_layout, write_speech_corpus, CorpusLayout,
};

// Layout constants, restated here so the expectations below do not depend on
// the same accessors the production mask uses.
const TEXT_VOCAB: usize = 16;
const CONTROL_REGION: usize = 16;
const NUM_CODEBOOKS: usize = 2;
const CODEBOOK_SIZE: usize = 4;
const AUDIO_BASE: u32 = (TEXT_VOCAB + CONTROL_REGION) as u32;
const AUDIO_END: u32 = AUDIO_BASE + (NUM_CODEBOOKS * CODEBOOK_SIZE) as u32;

fn toy_vocab() -> SpeechVocab {
    let codec = CodecVocab::new(NUM_CODEBOOKS, CODEBOOK_SIZE).unwrap();
    SpeechVocab::with_sequential_specials(TEXT_VOCAB, CONTROL_REGION, &ALL_SPECIAL_TOKENS, codec)
        .unwrap()
}

/// A layout that is valid but NOT equal to [`toy_vocab`]: one more text id, so
/// every control and audio id sits one higher.
fn other_vocab() -> SpeechVocab {
    let codec = CodecVocab::new(NUM_CODEBOOKS, CODEBOOK_SIZE).unwrap();
    SpeechVocab::with_sequential_specials(
        TEXT_VOCAB + 1,
        CONTROL_REGION,
        &ALL_SPECIAL_TOKENS,
        codec,
    )
    .unwrap()
}

/// Two utterances with known text ids and known codec frames.
fn packed_stream(vocab: &SpeechVocab) -> Vec<u32> {
    let frames_a: Vec<Vec<usize>> = vec![vec![0, 1], vec![2, 3]];
    let frames_b: Vec<Vec<usize>> = vec![vec![3, 0], vec![1, 2], vec![0, 0]];
    let speaker: Vec<u32> = vec![7, 8];
    let text_a: Vec<u32> = vec![1, 2, 3];
    let text_b: Vec<u32> = vec![4, 5];

    let records = vec![
        SpeechRecord {
            speaker: Some(speaker.as_slice()),
            style: None,
            text: &text_a,
            frames: &frames_a,
        },
        SpeechRecord {
            speaker: None,
            style: None,
            text: &text_b,
            frames: &frames_b,
        },
    ];
    pack_records(vocab, &records).unwrap()
}

/// Independent expectation: `1.0` where the TARGET (the id at `s + 1`) is an
/// audio id, computed straight from the packed stream and the id-range
/// arithmetic, with no call into the loader's masking code.
fn expected_mask(tokens: &[u32], seq_len: usize) -> Vec<f32> {
    (0..seq_len)
        .map(|s| {
            let target = tokens[s + 1];
            if (AUDIO_BASE..AUDIO_END).contains(&target) {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn scratch_dir(name: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "oxidizr-speech-corpus-{}-{}-{}",
        std::process::id(),
        name,
        n
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn mask_of(path: &Path, vocab: SpeechVocab, seq_len: usize) -> Vec<f32> {
    let mut loader = LitDataLoader::<CpuRuntime>::from_file(path, 1, seq_len, CpuDevice::new())
        .unwrap()
        .try_with_speech_layout(SpeechLayout::Native(vocab))
        .unwrap();
    let batch = loader.next_batch().unwrap();
    batch
        .loss_mask
        .as_ref()
        .expect("speech vocab configured, so the loader must emit a mask")
        .contiguous()
        .unwrap()
        .to_vec()
}

#[test]
fn packed_corpus_masks_exactly_the_audio_targets() {
    let vocab = toy_vocab();
    let tokens = packed_stream(&vocab);
    // One row covering the whole stream: targets are `tokens[1..]`.
    let seq_len = tokens.len() - 1;

    let dir = scratch_dir("mask");
    let bin = dir.join("train.bin");
    write_speech_corpus(&bin, &tokens, &SpeechLayout::Native(vocab.clone())).unwrap();
    assert!(layout_sidecar_path(&bin).exists());
    assert_eq!(
        layout_sidecar_path(&bin),
        bin.with_file_name("train.layout.json")
    );

    let mask = mask_of(&bin, vocab, seq_len);
    let expected = expected_mask(&tokens, seq_len);
    assert_eq!(mask, expected);

    // The expectation is only meaningful if the stream actually mixes both
    // kinds of target, and if the masked-in count is the audio the records
    // carry: (2 + 3) frames x 2 codes per frame.
    let ones = mask.iter().filter(|v| **v == 1.0).count();
    assert_eq!(ones, 10);
    assert!(mask.contains(&0.0));

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn sidecar_roundtrips_the_layout() {
    let vocab = toy_vocab();
    let tokens = packed_stream(&vocab);

    let dir = scratch_dir("roundtrip");
    let bin = dir.join("train.bin");
    write_speech_corpus(&bin, &tokens, &SpeechLayout::Native(vocab.clone())).unwrap();

    let read_back = read_corpus_layout(&bin).unwrap();
    assert_eq!(
        read_back,
        Some(CorpusLayout::Native {
            vocab: vocab.clone()
        })
    );
    check_corpus_layout(&bin, &SpeechLayout::Native(vocab.clone())).unwrap();

    // The exact on-disk shape: a `layout` tag naming the variant, and the whole
    // vocabulary under `vocab`.
    let json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(layout_sidecar_path(&bin)).unwrap()).unwrap();
    assert_eq!(json["layout"], "native");
    assert_eq!(
        json["vocab"],
        serde_json::to_value(&vocab).unwrap(),
        "{json}"
    );
    assert_eq!(json.as_object().unwrap().len(), 2, "{json}");

    // The `.bin` payload is unchanged by the sidecar: still a bare u32 array.
    let bytes = fs::metadata(&bin).unwrap().len() as usize;
    assert_eq!(bytes, tokens.len() * std::mem::size_of::<u32>());

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn layout_mismatch_is_rejected() {
    let packed_with = toy_vocab();
    let tokens = packed_stream(&packed_with);

    let dir = scratch_dir("mismatch");
    let bin = dir.join("train.bin");
    write_speech_corpus(&bin, &tokens, &SpeechLayout::Native(packed_with.clone())).unwrap();

    let err = check_corpus_layout(&bin, &SpeechLayout::Native(other_vocab()))
        .expect_err("a corpus packed under a different layout must not load");
    let msg = err.to_string();
    assert!(msg.contains("speech layout mismatch"), "{msg}");
    assert!(msg.contains("packed layout"), "{msg}");
    assert!(msg.contains("configured layout"), "{msg}");

    // The loader must refuse it too, not just the free function.
    let loader = LitDataLoader::<CpuRuntime>::from_file(&bin, 1, 4, CpuDevice::new()).unwrap();
    assert!(loader
        .try_with_speech_layout(SpeechLayout::Native(other_vocab()))
        .is_err());

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn missing_sidecar_loads_fine() {
    let vocab = toy_vocab();
    let tokens = packed_stream(&vocab);
    let seq_len = tokens.len() - 1;

    let dir = scratch_dir("no-sidecar");
    let bin = dir.join("train.bin");
    // Written the old way: no sidecar, exactly like every existing text corpus.
    write_tokens_to_file(&tokens, &bin).unwrap();
    assert!(!layout_sidecar_path(&bin).exists());

    assert_eq!(read_corpus_layout(&bin).unwrap(), None);
    check_corpus_layout(&bin, &SpeechLayout::Native(vocab.clone())).unwrap();
    check_corpus_layout(&bin, &SpeechLayout::expressive_tts()).unwrap();

    // And masking still works, because a missing sidecar is not a mismatch.
    let mask = mask_of(&bin, vocab, seq_len);
    assert_eq!(mask, expected_mask(&tokens, seq_len));

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_expressive_tts_sidecar_records_the_ids_that_give_audio_its_meaning() {
    let layout = SpeechLayout::expressive_tts();
    let text: Vec<u32> = vec![9, 10];
    let frames: Vec<Vec<usize>> = vec![vec![0], vec![65_535]];
    let record = SpeechRecord {
        speaker: None,
        style: None,
        text: &text,
        frames: &frames,
    };
    let tokens = layout.pack_records(&[record]).unwrap();

    let dir = scratch_dir("expressive-roundtrip");
    let bin = dir.join("train.bin");
    write_speech_corpus(&bin, &tokens, &layout).unwrap();

    assert_eq!(
        read_corpus_layout(&bin).unwrap(),
        Some(CorpusLayout::ExpressiveTts {
            audio_base: 151_670,
            codebook_size: 65_536,
            vocab_size: 217_208,
        })
    );
    check_corpus_layout(&bin, &layout).unwrap();

    // The exact on-disk shape. The three numbers are recorded, not implied by
    // the tag: they are what decides which codebook entry an id names.
    let json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(layout_sidecar_path(&bin)).unwrap()).unwrap();
    assert_eq!(json["layout"], "expressive_tts");
    assert_eq!(json["audio_base"].as_u64().unwrap(), 151_670);
    assert_eq!(json["codebook_size"].as_u64().unwrap(), 65_536);
    assert_eq!(json["vocab_size"].as_u64().unwrap(), 217_208);
    assert_eq!(json.as_object().unwrap().len(), 4, "{json}");

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_legacy_bare_vocab_sidecar_loads_as_native() {
    // The pre-existing on-disk format: a bare SpeechVocab object with no
    // `layout` key, written when native was the only layout. Hand-written here
    // exactly as such a file looks on disk.
    let legacy = r#"{
        "text_vocab_size": 16,
        "control_region_size": 16,
        "specials": {},
        "codec": {
            "num_codebooks": 2,
            "codebook_size": 4,
            "codes_per_codebook": [1, 1]
        }
    }"#;
    let expected = SpeechVocab::new(16, 16, BTreeMap::new(), CodecVocab::new(2, 4).unwrap())
        .expect("the hand-written layout must be a valid vocabulary");

    let dir = scratch_dir("legacy-sidecar");
    let bin = dir.join("train.bin");
    write_tokens_to_file(&[1_u32, 2, 3, 4], &bin).unwrap();
    fs::write(layout_sidecar_path(&bin), legacy).unwrap();

    assert_eq!(
        read_corpus_layout(&bin).unwrap(),
        Some(CorpusLayout::Native {
            vocab: expected.clone()
        })
    );
    check_corpus_layout(&bin, &SpeechLayout::Native(expected)).unwrap();

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_corpus_packed_under_another_layout_names_both_layouts() {
    let vocab = toy_vocab();
    let tokens = packed_stream(&vocab);

    let dir = scratch_dir("cross-layout");
    let bin = dir.join("train.bin");
    write_speech_corpus(&bin, &tokens, &SpeechLayout::Native(vocab)).unwrap();

    let msg = check_corpus_layout(&bin, &SpeechLayout::expressive_tts())
        .expect_err("a native corpus must not train as the expressive-tts base")
        .to_string();

    assert!(msg.contains("boostr-native"), "{msg}");
    assert!(msg.contains("expressive-tts-1.7b"), "{msg}");
    assert!(msg.contains("packed layout"), "{msg}");
    assert!(msg.contains("configured layout"), "{msg}");

    fs::remove_dir_all(&dir).unwrap();
}

/// A sidecar matching NEITHER shape must name both, not leak whichever serde
/// attempt happened to run last.
#[test]
fn a_sidecar_matching_neither_shape_names_both() {
    let dir = scratch_dir("sidecar-garbage");
    let bin = dir.join("corpus.bin");
    fs::write(&bin, []).unwrap();

    for garbage in [r#"{}"#, r#"{"nonsense":1}"#, r#"[1,2,3]"#, r#"not json"#] {
        fs::write(layout_sidecar_path(&bin), garbage).unwrap();
        let Err(err) = read_corpus_layout(&bin) else {
            panic!("{garbage} is neither shape and must not parse");
        };
        let msg = err.to_string();
        assert!(msg.contains("layout"), "must name the tagged shape: {msg}");
        assert!(
            msg.contains("text_vocab_size") || msg.contains("SpeechVocab"),
            "must name the legacy shape too: {msg}"
        );
    }

    fs::remove_dir_all(&dir).unwrap();
}

/// A TAGGED sidecar with a typo'd field must NOT fall through and be misread as
/// the legacy bare-vocab form. It cannot today, because `SpeechVocab` carries
/// `deny_unknown_fields` and the two field sets are disjoint — but that is a
/// property of another crate's derive, so pin it here rather than trust it.
#[test]
fn a_corrupt_tagged_sidecar_does_not_fall_through_to_legacy() {
    let dir = scratch_dir("sidecar-typo");
    let bin = dir.join("corpus.bin");
    fs::write(&bin, []).unwrap();

    for typo in [
        // `audio_base` misspelled.
        r#"{"layout":"expressive_tts","audio_bass":151670,"codebook_size":65536,"vocab_size":217208}"#,
        // Right names, wrong type.
        r#"{"layout":"expressive_tts","audio_base":"151670","codebook_size":65536,"vocab_size":217208}"#,
        // Unknown tag value.
        r#"{"layout":"mamba","audio_base":151670,"codebook_size":65536,"vocab_size":217208}"#,
    ] {
        fs::write(layout_sidecar_path(&bin), typo).unwrap();
        let Err(err) = read_corpus_layout(&bin) else {
            panic!("a corrupt tagged sidecar must not parse: {typo}");
        };
        // The failure must be reported, not silently downgraded to Native.
        assert!(
            !err.to_string().is_empty(),
            "the error must say something: {typo}"
        );
    }

    fs::remove_dir_all(&dir).unwrap();
}
