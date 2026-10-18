//! A `tensorboard` build writes the logged steps and evals to an event file beside `steps.json`.
//!
//! A minimal reader below splits the TFRecord stream, checks both checksums
//! of every record, and decodes the `Event` and `Summary` fields.

#![cfg(feature = "tensorboard")]

mod common;

use common::{
    build_model, eval_loaders, event_files, holdout_eval, scratch_dir, tiny_config, train_parts,
    trainer,
};
use oxidizr::data::DataPosition;
use oxidizr::fit::TuiSink;
use oxidizr::metrics::{masked_crc32c, tags, FILE_VERSION};
use oxidizr::run_record::{StepRecord, STEPS_FILE};

/// One decoded event.
#[derive(Debug, Default)]
struct Event {
    step: u64,
    version: Option<String>,
    scalars: Vec<(String, f32)>,
}

fn varint(bytes: &[u8], at: &mut usize) -> u64 {
    let (mut value, mut shift) = (0u64, 0);
    loop {
        let byte = bytes[*at];
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return value;
        }
        shift += 7;
    }
}

/// The `(field, payload)` pairs of one protobuf message.
fn fields(bytes: &[u8]) -> Vec<(u64, Vec<u8>)> {
    let mut at = 0;
    let mut out = Vec::new();
    while at < bytes.len() {
        let key = varint(bytes, &mut at);
        let len = match key & 7 {
            0 => {
                let value = varint(bytes, &mut at);
                out.push((key >> 3, value.to_le_bytes().to_vec()));
                continue;
            }
            1 => 8,
            2 => varint(bytes, &mut at) as usize,
            5 => 4,
            wire => panic!("unexpected wire type {wire}"),
        };
        out.push((key >> 3, bytes[at..at + len].to_vec()));
        at += len;
    }
    out
}

fn decode(data: &[u8]) -> Event {
    let mut event = Event::default();
    for (field, payload) in fields(data) {
        match field {
            2 => event.step = u64::from_le_bytes(payload.try_into().expect("8 bytes")),
            3 => event.version = Some(String::from_utf8(payload).expect("utf-8")),
            5 => {
                for (_, value) in fields(&payload) {
                    let value = fields(&value);
                    let tag = String::from_utf8(value[0].1.clone()).expect("utf-8 tag");
                    let number = f32::from_le_bytes(value[1].1.clone().try_into().expect("4"));
                    event.scalars.push((tag, number));
                }
            }
            _ => {}
        }
    }
    event
}

/// Every record of an event file, after both checksums of each record pass.
fn read_events(bytes: &[u8]) -> Vec<Event> {
    let mut at = 0;
    let mut events = Vec::new();
    while at < bytes.len() {
        let len_bytes = &bytes[at..at + 8];
        let len = u64::from_le_bytes(len_bytes.try_into().expect("8")) as usize;
        let len_crc = u32::from_le_bytes(bytes[at + 8..at + 12].try_into().expect("4"));
        assert_eq!(len_crc, masked_crc32c(len_bytes), "length checksum");
        let data = &bytes[at + 12..at + 12 + len];
        let data_crc =
            u32::from_le_bytes(bytes[at + 12 + len..at + 16 + len].try_into().expect("4"));
        assert_eq!(data_crc, masked_crc32c(data), "data checksum");
        events.push(decode(data));
        at += 16 + len;
    }
    events
}

fn scalar(event: &Event, tag: &str) -> Option<f32> {
    event
        .scalars
        .iter()
        .find_map(|(t, v)| (t == tag).then_some(*v))
}

#[test]
fn logged_steps_and_evals_reach_the_event_file() {
    let dir = scratch_dir("tensorboard-events");
    let mut cfg = tiny_config(&dir, 3);
    cfg.trainer.log_interval = 1;
    cfg.trainer.eval = Some(holdout_eval(40, 2, 1));
    let (mut train, evaluator) = eval_loaders(&cfg);
    let model = build_model(&cfg);
    let mut parts = train_parts(&model);
    {
        let layout = train.row_layout().expect("row layout");
        let mut t = trainer(&cfg, &mut parts, layout, 0, DataPosition::default());
        t.attach_eval(evaluator);
        t.fit_with_sink(&model, &mut train, &mut TuiSink::hidden())
            .expect("fit");
    }

    let files = event_files(&dir);
    assert_eq!(files.len(), 1, "{files:?}");
    let events = read_events(&std::fs::read(&files[0]).expect("read events"));
    assert_eq!(events[0].version.as_deref(), Some(FILE_VERSION));

    let records: Vec<StepRecord> = std::fs::read_to_string(dir.join(STEPS_FILE))
        .expect("steps.json")
        .lines()
        .map(|l| serde_json::from_str(l).expect("a step line"))
        .collect();
    let train_events: Vec<&Event> = events
        .iter()
        .filter(|e| scalar(e, tags::LOSS).is_some())
        .collect();
    assert_eq!(train_events.len(), records.len());
    for (event, record) in train_events.iter().zip(&records) {
        assert_eq!(event.step, record.step as u64);
        assert_eq!(scalar(event, tags::LOSS), Some(record.loss as f32));
        assert_eq!(
            scalar(event, tags::GRAD_NORM),
            Some(record.grad_norm as f32)
        );
        assert_eq!(
            scalar(event, tags::LEARNING_RATE),
            Some(record.learning_rate as f32)
        );
        assert_eq!(
            scalar(event, tags::TOKENS_PER_SEC),
            Some(record.tokens_per_sec as f32)
        );
        assert_eq!(scalar(event, tags::MFU), None, "no peak_tflops, no MFU");
    }

    // Evals after steps 2 and 3, the last step.
    let eval_steps: Vec<u64> = events
        .iter()
        .filter(|e| scalar(e, tags::EVAL_LOSS).is_some())
        .map(|e| e.step)
        .collect();
    assert_eq!(eval_steps, vec![2, 3]);
    for event in events
        .iter()
        .filter(|e| scalar(e, tags::EVAL_LOSS).is_some())
    {
        let loss = scalar(event, tags::EVAL_LOSS).expect("eval loss");
        let ppl = scalar(event, tags::EVAL_PERPLEXITY).expect("eval perplexity");
        assert!(loss.is_finite() && ppl > 1.0, "{event:?}");
    }
    let _ = std::fs::remove_dir_all(dir);
}
