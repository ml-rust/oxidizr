//! Data-parallel rank loaders split each global batch of one shared row order.
//!
//! Rank `k` of `W` ranks at batch size `B` reads slots `[k * B, (k + 1) * B)`
//! of each global batch. The rank windows in rank order must equal the
//! window of one loader at batch size `B * W`.

use boostr::{CpuDevice, CpuRuntime};
use oxidizr::data::{Batch, DataPosition, LitDataLoader, PrefetchDataLoader, RowOrder, TargetMask};

const SEQ_LEN: usize = 3;
const WIDTH: usize = SEQ_LEN + 1;
const SEED: u64 = 21;

/// `rows` rows where every token is its own stream offset, plus a `tail`
/// shorter than a row.
fn offsets(rows: usize, tail: usize) -> Vec<u32> {
    (0..(rows * WIDTH + tail) as u32).collect()
}

/// A shuffled loader over `tokens` that scores targets at or above 8.
fn loader(tokens: &[u32], batch_size: usize) -> LitDataLoader<CpuRuntime> {
    LitDataLoader::<CpuRuntime>::new(tokens.to_vec(), batch_size, SEQ_LEN, CpuDevice::new())
        .with_row_order(RowOrder::Shuffled { seed: SEED })
        .with_target_mask(TargetMask::PrefixLm {
            target_base: 8,
            always_score: Vec::new(),
        })
}

/// [`loader`] as rank `rank` of `world` ranks.
fn rank_loader(
    tokens: &[u32],
    batch_size: usize,
    rank: usize,
    world: usize,
) -> LitDataLoader<CpuRuntime> {
    loader(tokens, batch_size)
        .with_rank(rank, world)
        .expect("valid rank")
}

/// Host copies of one batch: inputs, targets, mask, scored count.
#[derive(Debug, PartialEq)]
struct HostBatch {
    inputs: Vec<i64>,
    targets: Vec<i64>,
    mask: Vec<f32>,
    scored: usize,
}

fn host(batch: &Batch<CpuRuntime>) -> HostBatch {
    let mask = batch.loss_mask.as_ref().expect("loss mask present");
    HostBatch {
        inputs: batch.inputs.contiguous().expect("contiguous").to_vec(),
        targets: batch.targets.contiguous().expect("contiguous").to_vec(),
        mask: mask.contiguous().expect("contiguous").to_vec(),
        scored: batch.scored,
    }
}

/// Rows of a batch, in batch order.
fn rows(batch: &HostBatch) -> Vec<usize> {
    batch
        .inputs
        .chunks(SEQ_LEN)
        .map(|row| row[0] as usize / WIDTH)
        .collect()
}

/// The rank batches of one step concatenated in rank order.
fn concat(parts: Vec<HostBatch>) -> HostBatch {
    let mut out = HostBatch {
        inputs: Vec::new(),
        targets: Vec::new(),
        mask: Vec::new(),
        scored: 0,
    };
    for part in parts {
        out.inputs.extend(part.inputs);
        out.targets.extend(part.targets);
        out.mask.extend(part.mask);
        out.scored += part.scored;
    }
    out
}

fn step(ranks: &mut [LitDataLoader<CpuRuntime>]) -> Vec<HostBatch> {
    ranks
        .iter_mut()
        .map(|l| host(&l.next_batch().expect("rank batch")))
        .collect()
}

#[test]
fn rank_windows_concatenate_to_the_global_window_across_epochs() {
    // 29 rows, 3 ranks x 2: 4 global batches per epoch, 5 rows dropped.
    let tokens = offsets(29, 3);
    let (batch, world) = (2, 3);
    let mut global = loader(&tokens, batch * world);
    let mut ranks: Vec<_> = (0..world)
        .map(|rank| rank_loader(&tokens, batch, rank, world))
        .collect();
    let layout = ranks[0].row_layout().expect("layout");
    assert_eq!(layout.batch_size(), batch * world);
    assert_eq!(layout.batches_per_epoch(), 4);
    let single = global.row_layout().expect("global layout");
    assert_eq!(layout.batch_size(), single.batch_size());
    assert_eq!(layout.batches_per_epoch(), single.batches_per_epoch());

    for n in 0..4 * layout.batches_per_epoch() + 2 {
        let parts = step(&mut ranks);
        for part in &parts {
            assert_eq!(part.inputs.len(), batch * SEQ_LEN, "step {n}");
            assert_eq!(part.mask.len(), batch * SEQ_LEN, "step {n}");
        }
        let expected = host(&global.next_batch().expect("global batch"));
        assert_eq!(concat(parts), expected, "step {n}");
        for rank in &ranks {
            assert_eq!(rank.position(), global.position(), "step {n}");
        }
    }
}

#[test]
fn every_rank_runs_the_same_steps_and_drops_the_same_rows_each_epoch() {
    // 17 rows, 2 ranks x 3: 2 global batches per epoch, 5 rows dropped.
    let tokens = offsets(17, 0);
    let mut ranks: Vec<_> = (0..2)
        .map(|rank| rank_loader(&tokens, 3, rank, 2))
        .collect();
    let per_epoch = ranks[1].row_layout().expect("layout").batches_per_epoch();
    assert_eq!(per_epoch, 2);
    for epoch in 0..3 {
        let mut seen = Vec::new();
        for _ in 0..per_epoch {
            for part in step(&mut ranks) {
                seen.extend(rows(&part));
            }
            let positions: Vec<DataPosition> = ranks.iter().map(|l| l.position()).collect();
            assert!(
                positions.iter().all(|p| *p == positions[0]),
                "{positions:?}"
            );
            assert_eq!(positions[0].epoch, epoch);
        }
        let mut unique = seen.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            12,
            "epoch {epoch}: 12 distinct rows, 5 dropped"
        );
    }
}

#[test]
fn every_rank_resumes_at_one_global_position() {
    let tokens = offsets(23, 1);
    let (batch, world) = (2, 2);
    let mut uninterrupted: Vec<_> = (0..world)
        .map(|rank| rank_loader(&tokens, batch, rank, world))
        .collect();
    let layout = uninterrupted[0].row_layout().expect("layout");
    // 7 steps: epoch 1, two global batches in.
    for _ in 0..7 {
        step(&mut uninterrupted);
    }
    let position = layout.advance(DataPosition::default(), 7);
    assert_eq!(position, DataPosition { epoch: 1, row: 8 });

    let mut resumed: Vec<_> = (0..world)
        .map(|rank| {
            let mut l = rank_loader(&tokens, batch, rank, world);
            l.seek(position).expect("seek inside the shard");
            l
        })
        .collect();
    for rank in &uninterrupted {
        assert_eq!(rank.position(), position);
    }
    for n in 0..8 {
        assert_eq!(step(&mut resumed), step(&mut uninterrupted), "step {n}");
    }
}

#[test]
fn prefetch_rank_loader_matches_the_plain_rank_loader() {
    let tokens = offsets(26, 2);
    let (batch, world) = (2, 3);
    let start = DataPosition { epoch: 2, row: 6 };
    for rank in 0..world {
        let mut plain = rank_loader(&tokens, batch, rank, world);
        plain.seek(start).expect("seek");
        let mut base = rank_loader(&tokens, batch, rank, world);
        base.seek(start).expect("seek");
        let mut prefetch = PrefetchDataLoader::new(base, 2).expect("prefetch loader");
        for n in 0..10 {
            let a = host(&plain.next_batch().expect("plain batch"));
            let b = host(&prefetch.next_batch().expect("prefetch batch"));
            assert_eq!(a, b, "rank {rank} step {n}");
        }
    }
}

#[test]
fn world_one_is_the_legacy_loader_bit_for_bit() {
    let tokens = offsets(19, 2);
    let mut legacy = loader(&tokens, 3);
    let mut ranked = rank_loader(&tokens, 3, 0, 1);
    assert_eq!(ranked.row_layout().ok(), legacy.row_layout().ok());
    for n in 0..20 {
        let a = host(&legacy.next_batch().expect("legacy batch"));
        let b = host(&ranked.next_batch().expect("ranked batch"));
        assert_eq!(a.inputs, b.inputs, "step {n}");
        assert_eq!(a.targets, b.targets, "step {n}");
        let bits = |m: &[f32]| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.mask), bits(&b.mask), "step {n}");
        assert_eq!(a.scored, b.scored, "step {n}");
        assert_eq!(legacy.position(), ranked.position(), "step {n}");
    }
}

#[test]
fn rank_outside_the_world_is_an_error() {
    let tokens = offsets(8, 0);
    for (rank, world) in [(1, 1), (3, 2), (0, 0)] {
        let Err(err) = loader(&tokens, 2).with_rank(rank, world) else {
            panic!("rank {rank} of world {world} must be refused");
        };
        let text = err.to_string();
        assert!(
            text.contains("is not below world size") || text.contains("world size is 0"),
            "{text}"
        );
    }
}

#[test]
fn ranks_split_only_the_training_range_of_a_holdout() {
    // Rows [0, 14) train and [14, 20) are held out.
    let tokens = offsets(20, 0);
    let ranged = |batch_size: usize| {
        loader(&tokens, batch_size)
            .with_row_range(0, 14)
            .expect("training range")
    };
    let mut global = ranged(4);
    let mut ranks: Vec<_> = (0..2)
        .map(|rank| ranged(2).with_rank(rank, 2).expect("valid rank"))
        .collect();
    assert_eq!(ranks[1].row_range(), (0, 14));
    assert_eq!(ranks[1].total_rows(), 20);
    for n in 0..9 {
        let union = concat(step(&mut ranks));
        assert!(rows(&union).iter().all(|r| *r < 14), "step {n}");
        assert_eq!(
            union,
            host(&global.next_batch().expect("global batch")),
            "step {n}"
        );
    }
}
