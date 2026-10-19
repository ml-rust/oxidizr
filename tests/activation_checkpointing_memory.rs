//! `trainer.activation_checkpointing: all` lowers the per-layer peak memory.
//!
//! A counting global allocator tracks live heap bytes. The test reads the
//! peak over one forward and backward pass at 2 and 4 layers, under `none`
//! and under `all`. The CPU tensors come from the global allocator, so the
//! peak covers every activation and gradient.
//!
//! `#[global_allocator]` is process-wide, so this test has its own binary
//! and one `#[test]`, and no other test allocates during a measurement.

mod common;

use boostr::autograd::{backward_wrt, var_reshape, var_sum};
use boostr::runtime::cpu::ParallelismConfig;
use boostr::tensor::TensorId;
use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor};
use common::{build_model, tiny_config};
use oxidizr::config::ActivationCheckpointing;
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Live heap bytes.
static CURRENT: AtomicUsize = AtomicUsize::new(0);
/// The largest value of `CURRENT` since the last [`window_start`].
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// `System`, with live and peak byte counts.
struct Counting;

fn grow(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

fn shrink(bytes: usize) {
    CURRENT.fetch_sub(bytes, Ordering::SeqCst);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        shrink(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = System.realloc(ptr, layout, new_size);
        if !new.is_null() {
            if new_size >= layout.size() {
                grow(new_size - layout.size());
            } else {
                shrink(layout.size() - new_size);
            }
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Rows per batch. Many tokens per layer make activations dominate parameters.
const BATCH: usize = 4;
/// Tokens per row.
const SEQ: usize = 64;

/// Reset the peak to the live count, and return the live count.
fn window_start() -> usize {
    let now = CURRENT.load(Ordering::SeqCst);
    PEAK.store(now, Ordering::SeqCst);
    now
}

/// Peak bytes above the start over one forward and backward pass.
fn peak_bytes(num_layers: usize, policy: &ActivationCheckpointing) -> usize {
    let mut cfg = tiny_config(Path::new("unused"), 1);
    cfg.model.num_layers = num_layers;
    cfg.model.max_seq_len = SEQ;
    let model = build_model(&cfg);
    let device = CpuDevice::new();
    // One worker thread: no parallel scratch buffers overlap a measurement.
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let vocab = cfg.model.vocab_size;
    let ids: Vec<i64> = (0..BATCH * SEQ).map(|i| ((i * 7) % vocab) as i64).collect();
    let ids = Tensor::<CpuRuntime>::from_slice(&ids, &[BATCH, SEQ], &device).expect("token ids");
    let wanted: Vec<TensorId> = model
        .trainable_parameters()
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    let start = window_start();
    {
        let out = model.forward_with(&client, &ids, policy).expect("forward");
        let logits = &out.logits;
        let flat = var_reshape(logits, &[logits.tensor().numel()]).expect("flatten");
        let loss = var_sum(&flat, &[0], false, &client).expect("sum");
        let grads = backward_wrt(&loss, &wanted, &client).expect("backward");
        assert!(
            wanted.iter().all(|id| grads.get(*id).is_some()),
            "every parameter gets a gradient"
        );
    }
    PEAK.load(Ordering::SeqCst) - start
}

#[test]
fn recompute_lowers_the_per_layer_peak() {
    let none = ActivationCheckpointing::None;
    let all = ActivationCheckpointing::All;
    // One untimed pass first, so lazy one-time allocations miss the measurements.
    peak_bytes(2, &none);
    peak_bytes(2, &all);

    let growth_none = peak_bytes(4, &none).saturating_sub(peak_bytes(2, &none));
    let growth_all = peak_bytes(4, &all).saturating_sub(peak_bytes(2, &all));
    eprintln!("peak growth from 2 to 4 layers: none {growth_none} B, all {growth_all} B");
    assert!(growth_none > 0, "two more layers must raise the plain peak");
    // Under `none`, each extra layer keeps its whole graph until backward.
    // Under `all`, it keeps only its input and its parameter gradients.
    // Half of the plain growth leaves a wide margin for allocator noise and
    // still fails when recompute keeps a full layer graph per layer.
    assert!(
        growth_all < growth_none / 2,
        "recompute must at least halve the per-layer peak growth: \
         none {growth_none} B, all {growth_all} B"
    );
}
