//! End-to-end: pack speech records with boostr, write them as a corpus, load
//! them back through the training data loader, and check the audio-only loss
//! mask lands on exactly the right positions.
//!
//! The mask expectation here is computed INDEPENDENTLY from the packed stream
//! using the layout's arithmetic constants, never with the helper the loader
//! uses. A test that re-derived the mask with the loader's own helper would pass
//! for any consistent-but-wrong shift.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use boostr::model::{pack_records, CodecVocab, SpeechRecord, SpeechVocab, ALL_SPECIAL_TOKENS};
use boostr::{CpuDevice, CpuRuntime};

use oxidizr::data::{write_tokens_to_file, LitDataLoader};
use oxidizr::speech_corpus::{
    check_corpus_layout, layout_sidecar_path, read_corpus_layout, write_speech_corpus,
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
        .try_with_speech_vocab(vocab)
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
    write_speech_corpus(&bin, &tokens, &vocab).unwrap();
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
    write_speech_corpus(&bin, &tokens, &vocab).unwrap();

    let read_back = read_corpus_layout(&bin).unwrap();
    assert_eq!(read_back, Some(vocab.clone()));
    check_corpus_layout(&bin, &vocab).unwrap();

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
    write_speech_corpus(&bin, &tokens, &packed_with).unwrap();

    let err = check_corpus_layout(&bin, &other_vocab())
        .expect_err("a corpus packed under a different layout must not load");
    let msg = err.to_string();
    assert!(msg.contains("speech layout mismatch"), "{msg}");
    assert!(msg.contains("packed layout"), "{msg}");
    assert!(msg.contains("configured layout"), "{msg}");

    // The loader must refuse it too, not just the free function.
    let loader = LitDataLoader::<CpuRuntime>::from_file(&bin, 1, 4, CpuDevice::new()).unwrap();
    assert!(loader.try_with_speech_vocab(other_vocab()).is_err());

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
    check_corpus_layout(&bin, &vocab).unwrap();

    // And masking still works, because a missing sidecar is not a mismatch.
    let mask = mask_of(&bin, vocab, seq_len);
    assert_eq!(mask, expected_mask(&tokens, seq_len));

    fs::remove_dir_all(&dir).unwrap();
}
