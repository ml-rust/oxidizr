//! GPU memory of this process, read in-process through NVML.
//!
//! Only a `cuda` build reads NVML. It loads `libnvidia-ml` at run time on the
//! first query. A missing library or driver reads as unavailable, never as
//! an error. Other builds always report unavailable.
//!
//! NVML lists processes by host PID. In a container without the host PID
//! namespace, this process is not listed and holds 0 bytes.

/// GPU memory of this process and of the GPUs it uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuMemory {
    /// Bytes this process holds over every GPU that lists it. `None` when a
    /// GPU lists it without a usage figure, as Windows WDDM does.
    pub process_bytes: Option<u64>,
    /// GPUs that list this process.
    pub devices: usize,
    /// Used bytes of those GPUs, every process included.
    pub device_used_bytes: u64,
    /// Total bytes of those GPUs.
    pub device_total_bytes: u64,
}

const MIB: u64 = 1024 * 1024;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The `vram` value of a step line: this process's MiB, or `n/a`.
pub fn process_mib_text(memory: Option<GpuMemory>) -> String {
    match memory.and_then(|m| m.process_bytes) {
        Some(bytes) => (bytes / MIB).to_string(),
        None => "n/a".to_string(),
    }
}

/// The `vram_device` value of a step line: `<used>/<total>` MiB, or `n/a`.
///
/// `n/a` also when no GPU lists this process.
pub fn device_mib_text(memory: Option<GpuMemory>) -> String {
    match memory.filter(|m| m.devices > 0) {
        Some(m) => format!(
            "{}/{}",
            m.device_used_bytes / MIB,
            m.device_total_bytes / MIB
        ),
        None => "n/a".to_string(),
    }
}

/// One console line of `memory`, in GiB. `None` when NVML is unavailable.
pub fn summary_text(memory: Option<GpuMemory>) -> Option<String> {
    let m = memory?;
    let process = m.process_bytes.map_or_else(
        || "n/a".to_string(),
        |b| format!("{:.2} GiB", b as f64 / GIB),
    );
    if m.devices == 0 {
        return Some(format!("this process: {process} (no GPU lists it)"));
    }
    Some(format!(
        "this process: {process} | {} GPU(s): {:.2} of {:.2} GiB used",
        m.devices,
        m.device_used_bytes as f64 / GIB,
        m.device_total_bytes as f64 / GIB
    ))
}

/// The GPU memory of this process. `None` when NVML is unavailable.
#[cfg(feature = "cuda")]
pub fn gpu_memory() -> Option<GpuMemory> {
    use nvml_wrapper::enums::device::UsedGpuMemory;

    let nvml = nvml_handle()?;
    let pid = std::process::id();
    let count = nvml.device_count().ok()?;
    let mut memory = GpuMemory {
        process_bytes: Some(0),
        ..GpuMemory::default()
    };
    for index in 0..count {
        let Ok(device) = nvml.device_by_index(index) else {
            continue;
        };
        let Ok(processes) = device.running_compute_processes() else {
            continue;
        };
        let Some(entry) = processes.iter().find(|p| p.pid == pid) else {
            continue;
        };
        memory.process_bytes = match (memory.process_bytes, &entry.used_gpu_memory) {
            (Some(sum), UsedGpuMemory::Used(bytes)) => Some(sum + bytes),
            _ => None,
        };
        if let Ok(info) = device.memory_info() {
            memory.devices += 1;
            memory.device_used_bytes += info.used;
            memory.device_total_bytes += info.total;
        }
    }
    Some(memory)
}

/// The GPU memory of this process. A build without `cuda` has no NVML.
#[cfg(not(feature = "cuda"))]
pub fn gpu_memory() -> Option<GpuMemory> {
    None
}

/// The process-wide NVML handle. `None` when the library or driver is missing.
#[cfg(feature = "cuda")]
fn nvml_handle() -> Option<&'static nvml_wrapper::Nvml> {
    static NVML: std::sync::OnceLock<Option<nvml_wrapper::Nvml>> = std::sync::OnceLock::new();
    NVML.get_or_init(|| nvml_wrapper::Nvml::init().ok())
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_memory_reads_n_a() {
        assert_eq!(process_mib_text(None), "n/a");
        assert_eq!(device_mib_text(None), "n/a");
        assert_eq!(summary_text(None), None);
    }

    #[test]
    fn known_memory_reads_whole_mib() {
        let memory = GpuMemory {
            process_bytes: Some(1536 * MIB + 7),
            devices: 2,
            device_used_bytes: 4096 * MIB,
            device_total_bytes: 24576 * MIB,
        };
        assert_eq!(process_mib_text(Some(memory)), "1536");
        assert_eq!(device_mib_text(Some(memory)), "4096/24576");
        let line = summary_text(Some(memory)).expect("a line");
        assert!(
            line.contains("1.50 GiB") && line.contains("2 GPU(s)"),
            "{line}"
        );
    }

    #[test]
    fn a_process_no_gpu_lists_has_no_device_figure() {
        let memory = GpuMemory {
            process_bytes: Some(0),
            ..GpuMemory::default()
        };
        assert_eq!(process_mib_text(Some(memory)), "0");
        assert_eq!(device_mib_text(Some(memory)), "n/a");
        let unknown = GpuMemory {
            process_bytes: None,
            devices: 1,
            ..GpuMemory::default()
        };
        assert_eq!(process_mib_text(Some(unknown)), "n/a");
    }

    /// Any build answers without an error. A CPU build always answers `None`.
    #[test]
    fn a_query_never_fails() {
        let memory = gpu_memory();
        #[cfg(not(feature = "cuda"))]
        assert_eq!(memory, None);
        let _ = process_mib_text(memory);
    }
}
