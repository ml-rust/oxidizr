//! The in-process launcher: one scoped thread per rank, one shared abort, and the first error.

use super::trainer_hooks::AbortHook;
use crate::oom_guard::panic_message;
use anyhow::{anyhow, bail, Result};
use boostr::runtime::{Communicator, LocalCommunicator, NoOpCommunicator};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// One communicator per rank, and the call that aborts every rank of the group.
pub struct RankGroup {
    comms: Vec<Arc<dyn Communicator>>,
    abort: AbortHook,
}

impl RankGroup {
    /// One rank and no collective: a `NoOpCommunicator` and an abort that does nothing.
    pub fn single() -> Self {
        Self {
            comms: vec![Arc::new(NoOpCommunicator) as Arc<dyn Communicator>],
            abort: Arc::new(|| {}),
        }
    }

    /// A group of `comms`, where `comms[i]` is rank `i`.
    ///
    /// `abort` must make every pending and later collective of every rank
    /// return an error. The launcher calls it once, on the first rank error.
    ///
    /// # Errors
    /// Returns an error when `comms` is empty, or a communicator's rank or
    /// world size does not match its slot.
    pub fn new(comms: Vec<Arc<dyn Communicator>>, abort: AbortHook) -> Result<Self> {
        let world = comms.len();
        if world == 0 {
            bail!("a rank group needs at least one communicator. Pass one per rank");
        }
        for (slot, comm) in comms.iter().enumerate() {
            if comm.rank() != slot || comm.world_size() != world {
                bail!(
                    "communicator {slot} is rank {} of world size {}, but the group holds {world} \
                     communicators in rank order. Pass communicator i as rank i of the same group",
                    comm.rank(),
                    comm.world_size()
                );
            }
        }
        Ok(Self { comms, abort })
    }

    /// An in-process CPU group of `world` ranks whose waits give up after `timeout`.
    ///
    /// The abort calls [`LocalCommunicator::abort`], which aborts every rank.
    ///
    /// # Errors
    /// Returns an error when `world` is 0.
    pub fn local(world: usize, timeout: Duration) -> Result<Self> {
        let ranks: Vec<Arc<LocalCommunicator>> = LocalCommunicator::with_timeout(world, timeout)
            .into_iter()
            .map(Arc::new)
            .collect();
        let Some(first) = ranks.first().map(Arc::clone) else {
            bail!("a local rank group needs a world size of at least 1, got 0");
        };
        let comms = ranks
            .into_iter()
            .map(|comm| comm as Arc<dyn Communicator>)
            .collect();
        Self::new(comms, Arc::new(move || first.abort()))
    }

    /// An NCCL group with one rank per client, on each client's compute stream.
    ///
    /// Rank `i` uses `clients[i]`. Every client must be on its own GPU. The
    /// abort calls `NcclCommunicator::abort` on every rank, since NCCL peers
    /// learn of an abort only when their transport reports it.
    ///
    /// # Errors
    /// Returns an error when NCCL init fails.
    #[cfg(feature = "nccl")]
    pub fn nccl(clients: &[boostr::CudaClient]) -> Result<Self> {
        use boostr::runtime::NcclCommunicator;
        let streams = clients.iter().map(|c| Arc::clone(c.stream_arc())).collect();
        let ranks: Vec<Arc<NcclCommunicator>> = NcclCommunicator::from_streams(streams)
            .map_err(|e| {
                anyhow!(
                    "NCCL init of {} ranks failed: {e}. Check that every id in `--gpus` names a \
                     different visible GPU",
                    clients.len()
                )
            })?
            .into_iter()
            .map(Arc::new)
            .collect();
        let peers = ranks.clone();
        let abort: AbortHook = Arc::new(move || {
            for comm in &peers {
                if let Err(e) = comm.abort() {
                    eprintln!("Warning: NCCL abort of rank {} failed: {e}", comm.rank());
                }
            }
        });
        let comms = ranks
            .into_iter()
            .map(|comm| comm as Arc<dyn Communicator>)
            .collect();
        Self::new(comms, abort)
    }

    /// Ranks in the group.
    pub fn world_size(&self) -> usize {
        self.comms.len()
    }
}

/// What one rank's body receives from [`run_ranks`].
#[derive(Clone)]
pub struct RankSlot {
    pub rank: usize,
    pub world: usize,
    /// This rank's communicator.
    pub comm: Arc<dyn Communicator>,
    /// Records this rank as failed, then aborts the group. Pass it to
    /// `LitTrainer::with_abort_hook`, so the first error is this rank's.
    pub abort: AbortHook,
}

/// No failure recorded for a rank.
const NO_TICKET: u64 = u64::MAX;

/// The order in which ranks failed, and the one group abort.
struct FailureLog {
    aborted: AtomicBool,
    next: AtomicU64,
    tickets: Vec<AtomicU64>,
    abort: AbortHook,
}

impl FailureLog {
    /// Record `rank`'s first failure, then abort the group once.
    fn fail(&self, rank: usize) {
        if let Some(ticket) = self.tickets.get(rank) {
            let order = self.next.fetch_add(1, Ordering::SeqCst);
            let _ = ticket.compare_exchange(NO_TICKET, order, Ordering::SeqCst, Ordering::SeqCst);
        }
        if !self.aborted.swap(true, Ordering::SeqCst) {
            (self.abort)();
        }
    }

    fn ticket(&self, rank: usize) -> u64 {
        self.tickets
            .get(rank)
            .map_or(NO_TICKET, |t| t.load(Ordering::SeqCst))
    }
}

/// Records a panic of its rank and aborts the group while the thread unwinds.
struct PanicGuard<'a> {
    log: &'a FailureLog,
    rank: usize,
}

impl Drop for PanicGuard<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.log.fail(self.rank);
        }
    }
}

/// Run `body` once per rank of `group` and return each rank's value in rank order.
///
/// A group of 1 runs `body` on the calling thread and returns its error unchanged.
/// A larger group runs one scoped thread per rank. The first error or panic of
/// any rank aborts the group, so peers blocked in a collective return. The
/// launcher joins every thread, and a panic becomes an error.
///
/// # Errors
/// Returns the error of the rank that failed first, with its rank as context.
/// Later errors of the other ranks follow from the abort and are dropped.
pub fn run_ranks<T, F>(group: &RankGroup, body: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(RankSlot) -> Result<T> + Sync,
{
    let world = group.world_size();
    let log = Arc::new(FailureLog {
        aborted: AtomicBool::new(false),
        next: AtomicU64::new(0),
        tickets: (0..world).map(|_| AtomicU64::new(NO_TICKET)).collect(),
        abort: Arc::clone(&group.abort),
    });
    let slot = |rank: usize, comm: &Arc<dyn Communicator>| {
        let log = Arc::clone(&log);
        RankSlot {
            rank,
            world,
            comm: Arc::clone(comm),
            abort: Arc::new(move || log.fail(rank)),
        }
    };
    if let [comm] = group.comms.as_slice() {
        return body(slot(0, comm)).map(|value| vec![value]);
    }
    let outcomes: Vec<std::thread::Result<Result<T>>> = std::thread::scope(|s| {
        let handles: Vec<_> = group
            .comms
            .iter()
            .enumerate()
            .map(|(rank, comm)| {
                let slot = slot(rank, comm);
                let (log, body) = (&*log, &body);
                let spawned = std::thread::Builder::new()
                    .name(format!("oxidizr-rank-{rank}"))
                    .spawn_scoped(s, move || {
                        let _guard = PanicGuard { log, rank };
                        let result = body(slot);
                        if result.is_err() {
                            log.fail(rank);
                        }
                        result
                    });
                if spawned.is_err() {
                    // The started ranks would wait for this one forever.
                    log.fail(rank);
                }
                spawned
            })
            .collect();
        handles
            .into_iter()
            .enumerate()
            .map(|(rank, handle)| match handle {
                Ok(handle) => handle.join(),
                Err(e) => Ok(Err(anyhow!("the thread of rank {rank} did not start: {e}"))),
            })
            .collect()
    });
    first_failure(world, &log, outcomes)
}

/// The values of every rank, or the error of the rank that failed first.
fn first_failure<T>(
    world: usize,
    log: &FailureLog,
    outcomes: Vec<std::thread::Result<Result<T>>>,
) -> Result<Vec<T>> {
    let mut values = Vec::with_capacity(world);
    let mut failures = Vec::new();
    for (rank, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            Ok(Ok(value)) => values.push(value),
            Ok(Err(e)) => failures.push((log.ticket(rank), rank, e)),
            Err(payload) => failures.push((
                log.ticket(rank),
                rank,
                anyhow!("panic: {}", panic_message(payload.as_ref())),
            )),
        }
    }
    match failures
        .into_iter()
        .min_by_key(|(ticket, rank, _)| (*ticket, *rank))
    {
        None => Ok(values),
        Some((_, rank, error)) => {
            Err(error.context(format!("rank {rank} of {world} failed first")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::DType;
    use std::time::Instant;

    /// One f32 sum on `comm`, as a step's collective.
    fn sum_one(comm: &dyn Communicator) -> Result<f32> {
        let mut value = [1.0f32];
        // SAFETY: `value` is a live host buffer of one f32.
        unsafe {
            comm.all_reduce(
                value.as_mut_ptr() as u64,
                1,
                DType::F32,
                boostr::runtime::ReduceOp::Sum,
            )
        }
        .map_err(|e| anyhow!("{e}"))?;
        Ok(value[0])
    }

    #[test]
    fn every_rank_returns_its_value_in_rank_order() {
        let group = RankGroup::local(3, Duration::from_secs(30)).expect("group");
        let got = run_ranks(&group, |slot| Ok((slot.rank, sum_one(&*slot.comm)?))).expect("run");
        assert_eq!(got, vec![(0, 3.0), (1, 3.0), (2, 3.0)]);
    }

    #[test]
    fn a_group_of_one_runs_on_the_calling_thread() {
        let caller = std::thread::current().id();
        let got = run_ranks(&RankGroup::single(), |slot| {
            assert_eq!((slot.rank, slot.world), (0, 1));
            Ok(std::thread::current().id())
        })
        .expect("run");
        assert_eq!(got, vec![caller]);
        let err = run_ranks(&RankGroup::single(), |_| -> Result<()> { bail!("plain") })
            .expect_err("error");
        assert_eq!(format!("{err:#}"), "plain");
    }

    #[test]
    fn the_first_error_wins_and_peers_do_not_wait() {
        let group = RankGroup::local(2, Duration::from_secs(60)).expect("group");
        let start = Instant::now();
        let err = run_ranks(&group, |slot| {
            if slot.rank == 1 {
                (slot.abort)();
                bail!("rank one broke");
            }
            sum_one(&*slot.comm)
        })
        .expect_err("rank 1 fails");
        let text = format!("{err:#}");
        assert!(text.contains("rank 1 of 2 failed first"), "{text}");
        assert!(text.contains("rank one broke"), "{text}");
        assert!(start.elapsed() < Duration::from_secs(30), "rank 0 waited");
    }

    #[test]
    fn a_panic_is_an_error_and_aborts_the_group() {
        let group = RankGroup::local(2, Duration::from_secs(60)).expect("group");
        let err = run_ranks(&group, |slot| {
            if slot.rank == 0 {
                panic!("rank zero panicked here");
            }
            sum_one(&*slot.comm)
        })
        .expect_err("rank 0 panics");
        let text = format!("{err:#}");
        assert!(text.contains("rank 0 of 2 failed first"), "{text}");
        assert!(text.contains("rank zero panicked here"), "{text}");
    }

    #[test]
    fn a_group_out_of_rank_order_is_an_error() {
        let mut comms: Vec<Arc<dyn Communicator>> = LocalCommunicator::group(2)
            .into_iter()
            .map(|c| Arc::new(c) as Arc<dyn Communicator>)
            .collect();
        comms.reverse();
        let Err(err) = RankGroup::new(comms, Arc::new(|| {})) else {
            panic!("reversed ranks must be refused");
        };
        assert!(
            err.to_string().contains("communicator 0 is rank 1"),
            "{err}"
        );
        assert!(RankGroup::new(Vec::new(), Arc::new(|| {})).is_err());
        assert!(RankGroup::local(0, Duration::from_secs(1)).is_err());
    }
}
