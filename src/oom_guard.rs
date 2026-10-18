//! Out-of-memory guard around the forward and backward passes of a training step.

use crate::prelude::{Error, Result};
use std::any::Any;
use std::fmt;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

/// The pass of a training step that [`oom_guard`] wraps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pass {
    Forward,
    Backward,
}

impl fmt::Display for Pass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Pass::Forward => "forward",
            Pass::Backward => "backward",
        })
    }
}

/// Runs `f` and turns a device allocation panic into a training error.
///
/// The error names `pass`, the requested size when the panic states it, and
/// `batch_size`. A panic that is not an allocation error resumes unwinding.
pub(crate) fn oom_guard<T>(
    pass: Pass,
    batch_size: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let message = panic_message(payload.as_ref());
            if !is_allocation_error(&message) {
                resume_unwind(payload);
            }
            Err(Error::TrainingError {
                reason: out_of_memory_text(pass, &size_detail(&message), batch_size),
            })
        }
    }
}

/// The text of a panic payload, or `Unknown panic` for a payload of another type.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "Unknown panic".to_string()
    }
}

fn is_allocation_error(message: &str) -> bool {
    message.contains("can't allocate buffer") || message.contains("out of memory")
}

/// ` (N.N MB)` for a `size: <bytes>` field in `message`, else empty.
fn size_detail(message: &str) -> String {
    let Some(start) = message.find("size:") else {
        return String::new();
    };
    message[start + 5..]
        .split_whitespace()
        .next()
        .and_then(|size| size.parse::<u64>().ok())
        .map(|bytes| format!(" ({:.1} MB)", bytes as f64 / 1_000_000.0))
        .unwrap_or_default()
}

fn out_of_memory_text(pass: Pass, size_info: &str, batch_size: usize) -> String {
    format!(
        "GPU out of memory during {} pass{}\n\n\
        The model requires more VRAM than available. Try:\n\
        • Halve batch_size (current: {}) and double gradient_accumulation \
          - peak memory now tracks batch_size alone, and the effective \
          batch size is unchanged\n\
        • Reduce max_seq_len in config\n\
        • Use a smaller model configuration\n\
        • Free GPU memory from other processes (check with: nvidia-smi)",
        pass, size_info, batch_size
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reason(result: Result<()>) -> String {
        match result {
            Err(Error::TrainingError { reason }) => reason,
            other => panic!("expected a training error, got {other:?}"),
        }
    }

    #[test]
    fn an_allocation_panic_becomes_the_out_of_memory_error() {
        let result = oom_guard(Pass::Backward, 8, || -> Result<()> {
            panic!("can't allocate buffer of size: 2500000 bytes")
        });
        let text = reason(result);
        assert!(
            text.starts_with("GPU out of memory during backward pass (2.5 MB)\n\n"),
            "{text}"
        );
        assert!(text.contains("• Halve batch_size (current: 8) and double gradient_accumulation - peak memory now tracks batch_size alone, and the effective batch size is unchanged\n"), "{text}");
        assert!(text.ends_with("(check with: nvidia-smi)"), "{text}");
    }

    #[test]
    fn a_panic_without_a_size_names_the_pass_only() {
        let result = oom_guard(Pass::Forward, 2, || -> Result<()> {
            panic!("{}", String::from("CUDA out of memory"))
        });
        let text = reason(result);
        assert!(
            text.starts_with("GPU out of memory during forward pass\n\n"),
            "{text}"
        );
    }

    #[test]
    fn a_result_passes_through_unchanged() {
        assert_eq!(oom_guard(Pass::Forward, 1, || Ok(7)).ok(), Some(7));
        let inner = oom_guard(Pass::Forward, 1, || -> Result<()> {
            Err(Error::TrainingError {
                reason: "inner".to_string(),
            })
        });
        assert_eq!(reason(inner), "inner");
    }

    #[test]
    fn another_panic_keeps_unwinding() {
        let outcome = catch_unwind(|| {
            let _ = oom_guard(Pass::Forward, 1, || -> Result<()> {
                panic!("index out of bounds")
            });
        });
        let payload = outcome.expect_err("the panic must resume");
        assert_eq!(panic_message(payload.as_ref()), "index out of bounds");
    }
}
