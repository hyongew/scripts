#!/usr/bin/env python3
"""Find a practical llama.cpp configuration for one GGUF model.

Steps:
* read basic model metadata from the GGUF header;
* require llama-fit-params to find a viable context and placement;
* optionally verify the resulting context with a temporary llama-server load;
* optionally run llama-bench at a realistic prompt length and choose batch settings;
* optionally compare speculative-decoding configurations with official SPEED-Bench.

The memory policy is deliberately stricter than llama.cpp's automatic fitter:

* dense models must keep the model on the GPUs (0.5 GiB aggregate VRAM reserve);
* MoE models may place weights in host RAM; available RAM is reported after a
  real load but is not used as a pass/fail threshold.

A successful target-context server load is the authority, especially for
hybrid, multimodal, or MoE models.
The optional local SPEED-Bench step requires the third-party ``datasets``
package; the rest of the script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import csv
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


GIB = 1024 ** 3
PROBE_TIMEOUT_SECONDS = 300
BENCH_TIMEOUT_SECONDS = 1800
DEFAULT_VRAM_BUFFER_GIB = 0.25
DEFAULT_COMPUTE_BUFFER_GIB = 1.0
DEFAULT_BANDWIDTH_EFFICIENCY = 0.85
DEFAULT_CACHE_RAM_GIB = 2.0
FIT_CONTEXTS = (65536, 98304, 131072, 163840, 196608, 228376, 262144)
BENCH_PROMPT = 4096
BENCH_GENERATION = 128
BENCH_REPETITIONS = 2
SPEED_BENCH_OSL = 512
SPEED_BENCH_CONTEXT = 65536
SPEED_BENCH_LIMIT = 2
# SPEED-Bench setup:
# 1. Open https://huggingface.co/datasets/nvidia/SPEED-Bench.
# 2. Download the test-00000-of-00001.parquet file from the throughput_2k
#    folder, preserving the folder name as below.
# 3. Set the DATASETS_DIR environment variable to your generic datasets root.
#    The script automatically looks below it for the SPEED-Bench folder.
#
# Expected layout:
#   SPEED-Bench/throughput_2k/test-00000-of-00001.parquet
DATASETS_DIR_ENV = "DATASETS_DIR"
SPEED_BENCH_TIMEOUT_SECONDS = 1800
SPEED_BENCH_REQUEST_TIMEOUT_SECONDS = 600
SPEED_BENCH_SPLIT = "throughput_2k"
SPEED_BENCH_INPUT_TOKENS = 32768
GGUF_MAGIC = b"GGUF"

# GGUF scalar types. Strings and arrays are handled separately.
GGUF_WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
               10: 8, 11: 8, 12: 8}
GGUF_SIGNED = {1, 3, 5, 11}
GGUF_FLOAT = {6, 12}
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_smi_path: str | None = None



class FitError(RuntimeError):
    pass


@dataclass
class GPU:
    index: int
    uuid: str
    name: str
    total_gib: float
    free_gib: float


@dataclass
class BandwidthEstimate:
    per_gpu_gbps: dict[int, float]
    pipeline_gbps: float
    sequential_gbps: float


@dataclass
class Metadata:
    values: dict[str, Any]
    architecture: str | None
    experts: int
    context_length: int | None
    total_layers: int | None
    sampling: dict[str, Any]
    notes: list[str] = field(default_factory=list)


@dataclass
class Selection:
    server: Path | None
    bench: Path | None
    fit_params: Path
    model: Path
    mmproj: Path | None
    draft: Path | None
    draft_spec_type: str | None
    cache_k: str
    cache_v: str
    cache_ram_gib: float
    verify_server: bool
    embedded_mtp: bool = False


@dataclass
class BenchRow:
    prompt: int
    generation: int
    batch: int
    ubatch: int
    depth: int
    tps: float
    stddev: float


@dataclass
class SpeedSample:
    id: str
    category: str
    turns: list[str]


@dataclass
class SpeedRequestResult:
    id: str
    category: str
    ok: bool
    turns: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str | None
    draft_n: int
    draft_n_accepted: int
    prompt_ms: float | None
    predicted_ms: float | None
    prompt_per_second: float | None
    predicted_per_second: float | None
    error: str | None


@dataclass
class Probe:
    context: int
    gpu_used_gib: dict[int, float]
    gpu_kv_gib: dict[int, float]
    gpu_model_gib: dict[int, float]
    host_model_gib: float
    available_ram_after_gib: float
    log: str
    log_path: Path | None
    ready: bool
    fits_policy: bool
    reason: str = ""


def clean_path(raw: str) -> str:
    value = raw.strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'`":
        value = value[1:-1].strip()
    return value


def ask_path(label: str, suffix: str | None = None, optional: bool = False,
             allow_embedded_mtp: bool = False) -> Path | None | Literal["embedded-mtp"]:
    while True:
        hints = []
        if optional:
            hints.append("blank for none")
        if allow_embedded_mtp:
            hints.append("1 for embedded MTP")
        extra = f" ({' / '.join(hints)})" if hints else ""
        raw = clean_path(input(f"{label}{extra}: "))
        if optional and not raw:
            return None
        if allow_embedded_mtp and raw == "1":
            return "embedded-mtp"
        path = Path(raw).expanduser()
        if path.is_file() and (suffix is None or path.suffix.lower() == suffix.lower()):
            return path.resolve()
        wanted = f" {suffix} file" if suffix else " file"
        print(f"Enter an existing{wanted}.")


def ask_yes_no(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{label} [{suffix}]: ").strip().lower()
    return default if not raw else raw in ("y", "yes")


def find_tool(name: str, prompt: bool = True) -> Path | None:
    candidates: list[Path] = []
    found = shutil.which(name) or shutil.which(f"{name}.exe")
    if found:
        candidates.append(Path(found))
    base = Path.cwd()
    for directory in (base, base / "build" / "bin" / "Release",
                      base / "llama.cpp" / "build" / "bin" / "Release"):
        candidates.extend((directory / name, directory / f"{name}.exe"))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    if not prompt:
        return None
    return ask_path(f"Path to {name}", optional=True)


def run_capture(command: list[str], timeout: int = 30,
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              errors="replace", check=False, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FitError(f"Could not run {' '.join(command)}: {exc}") from exc


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, seconds = divmod(total, 60)
    return f"{minutes:02d}:{seconds:02d}"


def run_capture_timed(command: list[str], label: str, timeout: int = 30,
                      env: dict[str, str] | None = None,
                      display_started: float | None = None,
                      show_finished: bool = True,
                      show_timer: bool = True,
                      elapsed_callback: Callable[[float], None] | None = None
                      ) -> subprocess.CompletedProcess[str]:
    """Run a command while showing elapsed time without blocking its output pipes."""
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, errors="replace", env=env)
    except OSError as exc:
        raise FitError(f"Could not run {' '.join(command)}: {exc}") from exc

    result: dict[str, Any] = {}

    def collect() -> None:
        try:
            result["stdout"], result["stderr"] = process.communicate()
        except Exception as exc:  # communicate errors need to reach the caller thread
            result["error"] = exc

    worker = threading.Thread(target=collect, daemon=True)
    worker.start()
    started = time.monotonic()
    display_started = started if display_started is None else display_started
    if elapsed_callback:
        elapsed_callback(time.monotonic() - display_started)
    if show_timer:
        print(f"{label} elapsed time: {format_elapsed(time.monotonic() - display_started)}",
              end="", flush=True)
    while worker.is_alive():
        command_elapsed = time.monotonic() - started
        if command_elapsed >= timeout:
            process.kill()
            worker.join(timeout=5)
            total_elapsed = time.monotonic() - display_started
            if elapsed_callback:
                elapsed_callback(total_elapsed)
            if show_timer:
                print(f"\r{label} elapsed time: {format_elapsed(total_elapsed)} (timed out)")
            raise FitError(f"{' '.join(command)} timed out after {timeout} seconds")
        worker.join(timeout=min(1.0, timeout - command_elapsed))
        total_elapsed = time.monotonic() - display_started
        if elapsed_callback:
            elapsed_callback(total_elapsed)
        if show_timer and worker.is_alive():
            print(f"\r{label} elapsed time: {format_elapsed(total_elapsed)}", end="", flush=True)
    worker.join()
    total_elapsed = time.monotonic() - display_started
    if elapsed_callback:
        elapsed_callback(total_elapsed)
    suffix = " (finished)" if show_finished else ""
    if show_timer:
        print(f"\r{label} elapsed time: {format_elapsed(total_elapsed)}{suffix}")
    if "error" in result:
        raise FitError(f"Could not run {' '.join(command)}: {result['error']}") from result["error"]
    return subprocess.CompletedProcess(command, process.returncode,
                                       result.get("stdout", ""), result.get("stderr", ""))


def run_text(command: list[str], timeout: int = 30) -> str:
    result = run_capture(command, timeout)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise FitError(f"Command failed: {' '.join(command)}\n{detail}")
    return result.stdout


def existing_llama_server_pids() -> list[int]:
    """Find llama-server processes without requiring an extra Python package."""
    if os.name == "nt":
        command = ["tasklist", "/FI", "IMAGENAME eq llama-server.exe",
                   "/FO", "CSV", "/NH"]
    else:
        command = ["pgrep", "-x", "llama-server"]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=10, errors="replace", check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode not in (0, 1):
        return []
    pids: list[int] = []
    if os.name == "nt":
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) >= 2 and row[0].lower() == "llama-server.exe":
                try:
                    pids.append(int(row[1]))
                except ValueError:
                    pass
    else:
        for line in result.stdout.splitlines():
            try:
                pids.append(int(line.strip()))
            except ValueError:
                pass
    return sorted(set(pids))


def stop_existing_llama_servers(pids: list[int]) -> list[int]:
    """Stop explicitly approved llama-server PIDs and return survivors."""
    for pid in pids:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, text=True, check=False)
        else:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
    time.sleep(0.5)
    return existing_llama_server_pids()


def format_command(command: list[str], short_program: bool = False) -> str:
    display = list(command)
    if short_program and display:
        display[0] = Path(display[0]).stem
    return subprocess.list2cmdline(display) if os.name == "nt" else shlex.join(display)


def cuda_env() -> dict[str, str]:
    """Keep CUDA ordinals aligned with the indices reported by nvidia-smi."""
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def save_log(text: str, prefix: str = "llama-fit-") -> Path | None:
    try:
        handle = tempfile.NamedTemporaryFile("w", prefix=prefix, suffix=".log",
                                             delete=False, encoding="utf-8",
                                             errors="replace")
    except OSError:
        return None
    with handle:
        handle.write(text)
    return Path(handle.name)


def smi() -> str:
    global _smi_path
    if _smi_path is None:
        _smi_path = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe") or ""
        if not _smi_path:
            raise FitError("nvidia-smi was not found on PATH.")
    return _smi_path


def detect_gpus() -> list[GPU]:
    query = "index,uuid,name,memory.total,memory.free"
    output = run_text([smi(), f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    result: list[GPU] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            result.append(GPU(int(parts[0]), parts[1], parts[2],
                              float(parts[3]) / 1024, float(parts[4]) / 1024))
        except ValueError:
            continue
    if not result:
        raise FitError("No NVIDIA GPUs were reported by nvidia-smi.")
    return result


def free_shares(gpus: list[GPU]) -> dict[int, float]:
    total = sum(gpu.free_gib for gpu in gpus)
    if total <= 0:
        return {gpu.index: 1 / len(gpus) for gpu in gpus}
    return {gpu.index: gpu.free_gib / total for gpu in gpus}


def runtime_bandwidth_gbps(gpus: list[GPU]) -> BandwidthEstimate | None:
    """Derive theoretical peak bandwidth through NVIDIA's NVML DLL.

    NVML reports memory clock in MHz and bus width in bits. GDDR transfers
    data on both clock edges, hence the factor of two. This avoids a model-name
    database and also follows board-specific clocks where NVML exposes them.

    The returned pipeline estimate assumes that layer placement follows the
    same free-VRAM ratios used by tensor_split(). It is a rough bound only:
    inter-GPU transfers, unequal layer costs, and CPU-offloaded tensors are
    not represented here.
    """
    if os.name != "nt":
        return None
    dll_candidates = ["nvml.dll"]
    windows_dir = os.environ.get("WINDIR", r"C:\Windows")
    dll_candidates.append(str(Path(windows_dir) / "System32" / "nvml.dll"))
    program_files = os.environ.get("ProgramW6432")
    if program_files:
        dll_candidates.append(str(Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvml.dll"))
    nvml = None
    for candidate in dll_candidates:
        try:
            nvml = ctypes.WinDLL(candidate)
            break
        except OSError:
            continue
    if nvml is None:
        return None
    try:
        init = getattr(nvml, "nvmlInit_v2", None) or getattr(nvml, "nvmlInit")
        shutdown = getattr(nvml, "nvmlShutdown", None)
        get_handle = (getattr(nvml, "nvmlDeviceGetHandleByIndex_v2", None)
                      or getattr(nvml, "nvmlDeviceGetHandleByIndex"))
        init.restype = ctypes.c_int
        if init() != 0:
            return None
        get_handle.restype = ctypes.c_int
        get_handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        get_clock = nvml.nvmlDeviceGetMaxClockInfo
        get_clock.restype = ctypes.c_int
        get_clock.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        get_width = nvml.nvmlDeviceGetMemoryBusWidth
        get_width.restype = ctypes.c_int
        get_width.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        values: dict[int, float] = {}
        for gpu in gpus:
            handle = ctypes.c_void_p()
            clock_mhz = ctypes.c_uint()
            bus_width = ctypes.c_uint()
            if (get_handle(gpu.index, ctypes.byref(handle)) != 0
                    or get_clock(handle, 2, ctypes.byref(clock_mhz)) != 0
                    or get_width(handle, ctypes.byref(bus_width)) != 0):
                return None
            values[gpu.index] = clock_mhz.value * 2 * (bus_width.value / 8) / 1000
        if not values:
            return None
        shares = free_shares(gpus)
        stage_times = [shares[index] / values[index] for index in values]
        return BandwidthEstimate(
            per_gpu_gbps=values,
            # Steady-state pipeline throughput is limited by its slowest stage.
            pipeline_gbps=1 / max(stage_times),
            # Single-stream latency crosses every stage serially.
            sequential_gbps=1 / sum(stage_times),
        )
    except (AttributeError, OSError):
        return None
    finally:
        if nvml is not None and 'shutdown' in locals() and shutdown is not None:
            try:
                shutdown()
            except OSError:
                pass


def available_ram_gib() -> float:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise FitError("Windows could not report available system memory.")
        return status.ullAvailPhys / GIB
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass
    raise FitError("Could not determine available system RAM.")


def _read_exact(handle: Any, count: int, filename: str) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise FitError(f"{filename} ends inside its GGUF metadata header.")
    return data


def read_gguf_header(path: Path) -> dict[str, Any]:
    """Read GGUF metadata and tensor names without reading tensor data."""
    values: dict[str, Any] = {}
    filename = path.name
    with path.open("rb") as handle:
        if _read_exact(handle, 4, filename) != GGUF_MAGIC:
            raise FitError(f"{filename} does not start with GGUF magic.")
        version = int.from_bytes(_read_exact(handle, 4, filename), "little")
        if version not in (2, 3):
            raise FitError(f"{filename} uses unsupported GGUF version {version}.")

        def count() -> int:
            return int.from_bytes(_read_exact(handle, 8, filename), "little")

        def read_string() -> str:
            return _read_exact(handle, count(), filename).decode("utf-8", "replace")

        def read_value(kind: int) -> Any:
            if kind == 8:
                return read_string()
            if kind == 9:
                item_kind = int.from_bytes(_read_exact(handle, 4, filename), "little")
                items = count()
                if item_kind == 8:
                    for _ in range(items):
                        handle.seek(count(), os.SEEK_CUR)
                elif item_kind in GGUF_WIDTHS:
                    handle.seek(GGUF_WIDTHS[item_kind] * items, os.SEEK_CUR)
                else:
                    raise FitError(f"{filename} has unsupported GGUF array type {item_kind}.")
                return None
            width = GGUF_WIDTHS.get(kind)
            if width is None:
                raise FitError(f"{filename} has unsupported GGUF value type {kind}.")
            raw = _read_exact(handle, width, filename)
            if kind in GGUF_FLOAT:
                import struct
                return struct.unpack("<f" if kind == 6 else "<d", raw)[0]
            if kind == 7:
                return bool(raw[0])
            return int.from_bytes(raw, "little", signed=kind in GGUF_SIGNED)

        values["__tensor_count__"] = count()
        metadata_count = count()
        for _ in range(metadata_count):
            key = read_string()
            kind = int.from_bytes(_read_exact(handle, 4, filename), "little")
            value = read_value(kind)
            # Token arrays and merges are enormous and are irrelevant to sizing.
            if value is not None and not key.startswith(("tokenizer.ggml.tokens",
                                                          "tokenizer.ggml.merges",
                                                          "tokenizer.ggml.token_type")):
                values[key] = value
        tensor_names: list[str] = []
        for _ in range(values["__tensor_count__"]):
            tensor_names.append(read_string())
            dimensions = int.from_bytes(_read_exact(handle, 4, filename), "little")
            handle.seek(8 * dimensions + 4 + 8, os.SEEK_CUR)
        values["__tensor_names__"] = tensor_names
    return values


def meta_for(values: dict[str, Any], architecture: str | None, suffix: str) -> Any:
    keys = []
    if architecture:
        keys.append(f"{architecture}.{suffix}")
    keys += [key for key in values if key.endswith(f".{suffix}")]
    for key in keys:
        if key in values:
            return values[key]
    return None


def inspect_model(path: Path) -> Metadata:
    values = read_gguf_header(path)
    architecture = values.get("general.architecture")
    if not isinstance(architecture, str):
        architecture = None
    experts_value = meta_for(values, architecture, "expert_count")
    experts = int(experts_value) if isinstance(experts_value, (int, float)) else 0
    context_value = meta_for(values, architecture, "context_length")
    context_length = int(context_value) if isinstance(context_value, (int, float)) else None
    block_count = meta_for(values, architecture, "block_count")
    total_layers = int(block_count) if isinstance(block_count, (int, float)) else None
    notes: list[str] = []
    sampling = {key: value for key, value in values.items()
                if key.startswith("general.sampling.")}
    if experts > 1:
        notes.append(f"The GGUF declares {experts} experts: host offload is allowed by policy.")
    if not values.get("general.architecture"):
        notes.append("No general.architecture metadata was found; suffix matching was used.")
    return Metadata(values, architecture, experts, context_length,
                    total_layers, sampling, notes)


DRAFT_SPEC_OPTIONS = (
    ("draft-simple", "standalone autoregressive draft model"),
    ("draft-eagle3", "EAGLE3 draft model"),
    ("draft-dflash", "DFlash block-diffusion draft model"),
    ("draft-dspark", "DSpark draft model"),
)


def infer_draft_spec_type(path: Path) -> str | None:
    """Detect an MTP sidecar; other draft types require an explicit choice."""
    metadata = inspect_model(path)
    names = [name for name in metadata.values.get("__tensor_names__", [])
             if isinstance(name, str)]
    # MTP sidecars commonly store these at the root (``nextn.*``), while
    # other architectures may place them below a tensor namespace.
    nextn_names = [name for name in names
                   if re.search(r"(?:^|\.)nextn\.", name, re.IGNORECASE)]
    block_ids = {match.group(1) for name in names
                 if (match := re.match(r"^blk\.(\d+)\.", name))}
    if nextn_names and (len(block_ids) == 1 or len(names) <= 64):
        return "draft-mtp"
    return None


def choose_draft_spec_type(automatic: bool = False) -> str:
    """Ask which llama.cpp speculative draft implementation a non-MTP GGUF uses."""
    if automatic:
        spec_type, description = DRAFT_SPEC_OPTIONS[0]
        print(f"\nAutomatic mode: using {spec_type} for the non-MTP draft ({description}).")
        return spec_type
    print("\nDraft GGUF type")
    for number, (spec_type, description) in enumerate(DRAFT_SPEC_OPTIONS, start=1):
        print(f"  {number}. {spec_type} ({description})")
    while True:
        choice = input("Choice [1]: ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(DRAFT_SPEC_OPTIONS):
            return DRAFT_SPEC_OPTIONS[int(choice) - 1][0]
        print(f"Enter a number from 1 to {len(DRAFT_SPEC_OPTIONS)}.")


def tensor_split(gpus: list[GPU], separator: str = ",") -> str:
    shares = free_shares(gpus)
    return separator.join(f"{shares[gpu.index]:.6f}" for gpu in gpus)


def option(help_text: str, *names: str) -> str | None:
    for name in names:
        if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", help_text):
            return name
    return None


def require_options(help_text: str, names: list[str]) -> None:
    missing = [name for name in names if option(help_text, name) is None]
    if missing:
        raise FitError("This llama.cpp build lacks required option(s): " + ", ".join(missing))


def fit_target_mib(vram_buffer_gib: float) -> int:
    """Convert a per-device GiB margin to a 256-MiB-aligned value."""
    return max(256, round(vram_buffer_gib * 1024 / 256) * 256)


def fit_target(gpus: list[GPU], vram_buffer_gib: float) -> str:
    """Return one per-device margin for each detected GPU."""
    target_mib = fit_target_mib(vram_buffer_gib)
    return ",".join([str(target_mib)] * len(gpus))


def benchmark_fit_target(gpus: list[GPU], vram_buffer_gib: float) -> str:
    """llama-bench accepts one 256-MiB-aligned margin and broadcasts it."""
    return str(fit_target_mib(vram_buffer_gib))


def memory_plan(selection: Selection, metadata: Metadata, gpus: list[GPU],
                vram_buffer_gib: float) -> dict[str, Any]:
    vram_budget = max(0.0, sum(gpu.free_gib for gpu in gpus) - vram_buffer_gib)
    model_gib = selection.model.stat().st_size / GIB
    draft_gib = selection.draft.stat().st_size / GIB if selection.draft else 0.0
    mmproj_gib = selection.mmproj.stat().st_size / GIB if selection.mmproj else 0.0
    dense_fixed_vram = model_gib + draft_gib + DEFAULT_COMPUTE_BUFFER_GIB
    dense_fits = dense_fixed_vram <= vram_budget
    fits_policy = dense_fits if metadata.experts <= 1 else True
    return {
        "vram_budget": vram_budget, "model_gib": model_gib,
        "draft_gib": draft_gib, "mmproj_gib": mmproj_gib,
        "dense_fixed_vram": dense_fixed_vram,
        "fits_policy": fits_policy,
        "policy": "MoE placement requires a real load" if metadata.experts > 1 else "dense all-GPU",
    }


def benchmark_command(selection: Selection, gpus: list[GPU], help_text: str,
                      metadata: Metadata) -> list[str]:
    require_options(help_text, ["-m", "-p", "-n", "-b", "-ub", "-r", "-o"])
    command = [str(selection.bench), "-m", str(selection.model), "-p", str(BENCH_PROMPT),
               "-n", str(BENCH_GENERATION), "-b", "1024,2048", "-ub", "256,512",
               "-r", str(BENCH_REPETITIONS), "-o", "json"]
    for flag, value in (("-ctk", selection.cache_k), ("-ctv", selection.cache_v)):
        if option(help_text, flag):
            command += [flag, value]
    # llama-bench has no separate "fit on" switch; -fitt is its equivalent
    # and leaves split mode, device selection, and GPU-layer count automatic.
    fit_flag = option(help_text, "-fitt", "--fit-target")
    if fit_flag:
        command += [fit_flag, benchmark_fit_target(gpus, DEFAULT_VRAM_BUFFER_GIB)]
    return command


def parse_benchmark(text: str) -> list[BenchRow]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise FitError("llama-bench did not return JSON results.")
    try:
        records = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise FitError(f"Could not parse llama-bench JSON: {exc}") from exc
    rows: list[BenchRow] = []
    for record in records:
        prompt = int(record.get("n_prompt", 0))
        generation = int(record.get("n_gen", 0))
        if prompt == 0 and generation == 0:
            continue
        rows.append(BenchRow(prompt, generation, int(record.get("n_batch", 0)),
                             int(record.get("n_ubatch", 0)), int(record.get("n_depth", 0)),
                             float(record.get("avg_ts", 0.0)),
                             float(record.get("stddev_ts", 0.0))))
    if not rows:
        raise FitError("llama-bench returned no usable rows.")
    return rows


def run_benchmark(selection: Selection, gpus: list[GPU], metadata: Metadata,
                  automatic: bool = False) -> list[BenchRow]:
    assert selection.bench is not None
    help_text = run_text([str(selection.bench), "--help"])
    command = benchmark_command(selection, gpus, help_text, metadata)
    existing = existing_llama_server_pids()
    if existing:
        print("\nExisting llama-server process(es) found: "
              + ", ".join(str(pid) for pid in existing))
        if not (automatic or ask_yes_no("Stop existing llama-server process(es)?", True)):
            print("llama-bench skipped because an existing llama-server is still running.")
            return []
        remaining = stop_existing_llama_servers(existing)
        if remaining:
            print("Could not stop llama-server process(es): "
                  + ", ".join(str(pid) for pid in remaining))
            print("llama-bench skipped.")
            return []
    else:
        print("\nNo existing llama-server process found.")
    if not (automatic or ask_yes_no("Run llama-bench at 4096 prompt tokens?", True)):
        return []
    print("Running llama-bench; its output is JSON and may take a few minutes...")
    print(format_command(command, short_program=True))
    result = run_capture_timed(command, "llama-bench",
                               timeout=BENCH_TIMEOUT_SECONDS, env=cuda_env())
    log_path = save_log(result.stdout + "\n" + result.stderr, "llama-bench-")
    if result.returncode:
        raise FitError(f"llama-bench failed (log: {log_path})\n{result.stderr[-3000:]}")
    rows = parse_benchmark(result.stdout)
    print(f"Benchmark log: {log_path}")
    print("  phase  batch ubatch  depth       t/s   stddev")
    for row in rows:
        phase = f"pp{row.prompt}" if row.prompt else f"tg{row.generation}"
        print(f"  {phase:6} {row.batch:5} {row.ubatch:6} {row.depth:6}"
              f" {row.tps:9.2f} {row.stddev:8.2f}")
    return rows


def choose_bench(rows: list[BenchRow]) -> tuple[int, int] | None:
    prompts = [row for row in rows if row.prompt > 0]
    if not prompts:
        return None
    winner = max(prompts, key=lambda row: row.tps)
    generations = [row for row in rows if row.generation > 0
                   and row.batch == winner.batch and row.ubatch == winner.ubatch]
    gen_note = f", generation {generations[0].tps:.2f} t/s" if generations else ""
    print(f"Recommended batch settings: -b {winner.batch} -ub {winner.ubatch}"
          f" (prefill {winner.tps:.2f} t/s{gen_note}).")
    return winner.batch, winner.ubatch


def fitted_args_keep_dense_on_gpu(output: str, metadata: Metadata) -> bool:
    match = re.search(r"(?:^|\s)-ngl\s+(-?\d+)(?:\s|$)", output)
    if not match:
        return False
    ngl = int(match.group(1))
    all_layers = ngl == -1 or (metadata.total_layers is not None and ngl >= metadata.total_layers)
    no_cpu_override = not re.search(r"=CPU(?:\s|,|$)", output, re.IGNORECASE)
    return all_layers and no_cpu_override


def run_fit_params(selection: Selection, gpus: list[GPU],
                   metadata: Metadata) -> int | None:
    help_text = run_text([str(selection.fit_params), "--help"])
    require_options(help_text, ["-m", "-c"])
    base_command = [str(selection.fit_params), "-m", str(selection.model)]
    for flag, value in (("-fa", "on"), ("-ctk", selection.cache_k),
                        ("-ctv", selection.cache_v)):
        if option(help_text, flag):
            base_command += [flag, value]
    for flag, value in (("-fit", "on"), ("-fitt", fit_target(gpus, DEFAULT_VRAM_BUFFER_GIB)),
                        ("-dev", ",".join(f"CUDA{gpu.index}" for gpu in gpus))):
        supported = option(help_text, flag, "--fit-target" if flag == "-fitt" else flag)
        if supported:
            base_command += [supported, value]
    verbose_flag = option(help_text, "-v", "--verbose")
    if verbose_flag:
        base_command += [verbose_flag]

    contexts = [context for context in reversed(FIT_CONTEXTS)
                if metadata.context_length is None or context <= metadata.context_length]
    if metadata.experts <= 1:
        print(f"\nTesting GPU memory layer fit (Total layers: "
              f"{metadata.total_layers if metadata.total_layers is not None else 'unknown'}):")
    else:
        print("\nTesting projected memory fit:")

    started = time.monotonic()
    last_elapsed_report = 0

    def report_elapsed(seconds: float) -> None:
        nonlocal last_elapsed_report
        elapsed = int(seconds)
        if elapsed != last_elapsed_report:
            print(f"\r\033[2Kllama-fit-params elapsed time: {format_elapsed(elapsed)}",
                  end="", flush=True)
            last_elapsed_report = elapsed

    try:
        for context in contexts:
            command = base_command + ["-c", str(context)]
            print(format_command(command, short_program=True))
            result = run_capture_timed(command, "llama-fit-params",
                                       timeout=PROBE_TIMEOUT_SECONDS, env=cuda_env(),
                                       display_started=started, show_finished=False,
                                       show_timer=False,
                                       elapsed_callback=report_elapsed)
            save_log(result.stdout + "\n" + result.stderr, f"llama-fit-params-{context}-")
            output = result.stdout.strip()
            if result.returncode or not output:
                print(f"\r\033[2Kllama-fit-params attempt at context {context} "
                      "produced no usable output")
                continue
            print(f"\r\033[2K{output}")
            if metadata.experts <= 1 and not fitted_args_keep_dense_on_gpu(output, metadata):
                continue
            return context
        return None
    finally:
        print(f"\r\033[2Kllama-fit-params elapsed time: "
              f"{format_elapsed(time.monotonic() - started)} "
              "(finished)")


def server_command(selection: Selection, gpus: list[GPU], help_text: str,
                   context: int, metadata: Metadata,
                   batch: tuple[int, int] | None = None,
                   spec_type: str | None = None,
                   draft_device: int | None = None) -> list[str]:
    require_options(help_text, ["--model", "--ctx-size", "--host", "--port"])
    command = [str(selection.server), "--model", str(selection.model),
               "--ctx-size", str(context), "--host", "127.0.0.1", "--port", str(free_port())]
    fit_flag = option(help_text, "--fit", "-fit")
    if not fit_flag:
        raise FitError("This llama-server build does not support --fit.")
    command += [fit_flag, "on"]
    for flag, value in (("--cache-type-k", selection.cache_k),
                        ("--cache-type-v", selection.cache_v)):
        short = {"--cache-type-k": "-ctk", "--cache-type-v": "-ctv"}[flag]
        supported = option(help_text, flag, short)
        if supported:
            command += [supported, value]
    if batch:
        for flag, value in (("--batch-size", str(batch[0])),
                            ("--ubatch-size", str(batch[1]))):
            short = {"--batch-size": "-b", "--ubatch-size": "-ub"}[flag]
            supported = option(help_text, flag, short)
            if supported:
                command += [supported, value]
    if selection.mmproj:
        mmproj_flag = option(help_text, "--mmproj")
        if not mmproj_flag:
            raise FitError("This llama-server build does not support --mmproj.")
        command += [mmproj_flag, str(selection.mmproj)]
        no_offload = option(help_text, "--no-mmproj-offload")
        if not no_offload:
            raise FitError("This build cannot keep mmproj on host RAM; the memory plan would be inaccurate.")
        command.append(no_offload)
    if spec_type is None and (selection.draft or selection.embedded_mtp):
        spec_type = selection.draft_spec_type
        if spec_type is None:
            raise FitError("A draft GGUF or embedded MTP was supplied but no draft speculative type was selected.")
    if spec_type is not None:
        spec_flag = option(help_text, "--spec-type")
        if not spec_flag:
            raise FitError("This llama-server build does not support --spec-type.")
        spec_types = {item.strip() for item in spec_type.split(",")}
        if any(item.startswith("draft-") for item in spec_types):
            if selection.draft:
                draft_flag = option(help_text, "--model-draft")
                if not draft_flag:
                    raise FitError("This llama-server build does not support --model-draft.")
                command += [draft_flag, str(selection.draft)]
                if draft_device is not None:
                    draft_device_flag = option(help_text, "--spec-draft-device", "-devd",
                                                "--device-draft")
                    if not draft_device_flag:
                        raise FitError("This llama-server build does not support --spec-draft-device.")
                    command += [draft_device_flag, f"CUDA{draft_device}"]
            elif "draft-mtp" not in spec_types:
                raise FitError("A draft speculative type was requested but no draft GGUF or embedded MTP was supplied.")
            if "draft-mtp" in spec_types:
                for flag, value in (
                                    ("--spec-draft-n-max", "3"),
                                    ):
                    supported = option(help_text, flag)
                    if supported:
                        command += [supported, value]
        if spec_type != "none":
            command += [spec_flag, spec_type]
        if "ngram-mod" in spec_types:
            for flag, value in (("--spec-ngram-mod-n-match", "24"),
                                ("--spec-ngram-mod-n-min", "48"),
                                ("--spec-ngram-mod-n-max", "64")):
                supported = option(help_text, flag)
                if supported:
                    command += [supported, value]
    return command


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def health(port: int) -> bool:
    try:
        with DIRECT_OPENER.open(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def query_process_vram(pid: int, gpus: list[GPU]) -> dict[int, float]:
    try:
        output = run_text([smi(), "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                           "--format=csv,noheader,nounits"], timeout=10)
    except FitError:
        return {}
    uuid_to_index = {gpu.uuid: gpu.index for gpu in gpus}
    used: dict[int, float] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or parts[1] != str(pid) or parts[2].upper() == "N/A":
            continue
        try:
            used[uuid_to_index[parts[0]]] = float(parts[2]) / 1024
        except (ValueError, KeyError):
            pass
    return used


KV_PATTERN = re.compile(r"CUDA(\d+).*?KV buffer size\s*=\s*([0-9.]+)\s*(MiB|GiB)", re.I)
MODEL_PATTERN = re.compile(r"CUDA(\d+).*?model buffer size\s*=\s*([0-9.]+)\s*(MiB|GiB)", re.I)
# CPU_Mapped is a file-backed mapping used while loading an otherwise GPU-
# resident model; it is not evidence that model layers spilled to CPU RAM.
HOST_MODEL_PATTERN = re.compile(
    r"(?:CPU(?![_A-Za-z])|Host).*?model buffer size\s*=\s*([0-9.]+)\s*(MiB|GiB)", re.I)


def parse_probe_log(log: str, gpus: list[GPU]) -> tuple[dict[int, float], dict[int, float],
                                                       dict[int, float], float]:
    kv = {gpu.index: 0.0 for gpu in gpus}
    model = {gpu.index: 0.0 for gpu in gpus}
    for match in KV_PATTERN.finditer(log):
        if int(match.group(1)) in kv:
            value = float(match.group(2)) / 1024 if match.group(3).lower() == "mib" else float(match.group(2))
            kv[int(match.group(1))] += value
    for match in MODEL_PATTERN.finditer(log):
        if int(match.group(1)) in model:
            value = float(match.group(2)) / 1024 if match.group(3).lower() == "mib" else float(match.group(2))
            model[int(match.group(1))] += value
    host = 0.0
    for match in HOST_MODEL_PATTERN.finditer(log):
        host += float(match.group(1)) / 1024 if match.group(2).lower() == "mib" else float(match.group(1))
    return kv, model, host, sum(kv.values())


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def drain_output(output_queue: queue.Queue[str], lines: list[str]) -> None:
    while True:
        try:
            lines.append(output_queue.get_nowait())
        except queue.Empty:
            return


def start_benchmark_server(command: list[str], label: str,
                           display_started: float | None = None) -> tuple[
        subprocess.Popen[str], queue.Queue[str], list[str], threading.Thread]:
    """Start a temporary server and wait for its health endpoint."""
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors="replace", bufsize=1, env=cuda_env())
    except OSError as exc:
        raise FitError(f"Could not start llama-server: {exc}") from exc

    output_queue: queue.Queue[str] = queue.Queue()
    lines: list[str] = []

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    port = int(command[command.index("--port") + 1])
    started = time.monotonic()
    display_started = started if display_started is None else display_started
    last_displayed = -1
    print(f"{label} elapsed time: {format_elapsed(time.monotonic() - display_started)}",
          end="", flush=True)
    ready = False
    while time.monotonic() - started < PROBE_TIMEOUT_SECONDS:
        drain_output(output_queue, lines)
        if process.poll() is not None:
            break
        if health(port):
            ready = True
            break
        elapsed = int(time.monotonic() - display_started)
        if elapsed != last_displayed:
            print(f"\r{label} elapsed time: {format_elapsed(elapsed)}",
                  end="", flush=True)
            last_displayed = elapsed
        time.sleep(0.5)
    elapsed = time.monotonic() - display_started
    if not ready:
        drain_output(output_queue, lines)
        print(f"\r{label} elapsed time: {format_elapsed(elapsed)} (failed)")
        stop_process(process)
        thread.join(timeout=5)
        drain_output(output_queue, lines)
        log_path = save_log("".join(lines), "speed-server-startup-")
        raise FitError("temporary llama-server did not become ready "
                       f"(log: {log_path})")
    print(f"\r{label} elapsed time: {format_elapsed(elapsed)} (ready)")
    return process, output_queue, lines, thread


# SPEED-Bench client functionality below is sourced from speed_bench.py.
# It is intentionally local-dataset-only so llama-fit does not need to clone
# llama.cpp or contact the Hugging Face Hub while running the benchmark.
def load_speed_bench_samples(dataset_dir: Path, bench: str,
                             sample_limit: int) -> list[SpeedSample]:
    if not dataset_dir.is_dir():
        raise FitError("SPEED-Bench dataset directory does not exist: "
                       f"{dataset_dir}. Check the {DATASETS_DIR_ENV} "
                       "environment variable and ensure it contains a SPEED-Bench folder.")
    bench_dirs = [dataset_dir / bench]
    if dataset_dir.name.lower() == bench.lower():
        bench_dirs.insert(0, dataset_dir)
    parquet_files: list[Path] = []
    for bench_dir in bench_dirs:
        parquet_files = sorted(bench_dir.glob("*.parquet")) if bench_dir.is_dir() else []
        if parquet_files:
            break
    if not parquet_files:
        raise FitError(f"No parquet files found for {bench!r} below {dataset_dir}. "
                       f"Expected {bench}/test-00000-of-00001.parquet.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise FitError("The local SPEED-Bench loader requires the 'datasets' package. "
                       "Install it with: python -m pip install datasets") from exc
    try:
        dataset = load_dataset(
            "parquet",
            data_files={"test": [str(path) for path in parquet_files]},
            split="test",
            # Keep the generated Arrow cache beside the manually downloaded
            # dataset so later runs remain local and do not use HF's default
            # user cache or contact the Hub.
            cache_dir=str(dataset_dir / ".datasets-cache"),
        )
    except Exception as exc:
        raise FitError(f"Could not read local SPEED-Bench data for {bench!r}: {exc}") from exc

    samples: list[SpeedSample] = []
    samples_per_category: dict[str, int] = {}
    for row_raw in dataset:
        row = dict(row_raw)
        category_raw = row.get("category")
        turns_raw = row.get("turns")
        question_id = row.get("question_id")
        if not isinstance(category_raw, str) or not category_raw.strip():
            continue
        if not isinstance(question_id, str) or not question_id.strip():
            continue
        if not isinstance(turns_raw, list):
            continue
        turns = [str(turn).strip() for turn in turns_raw if turn and str(turn).strip()]
        if not turns:
            continue
        category = category_raw.strip()
        if sample_limit > 0 and samples_per_category.get(category, 0) >= sample_limit:
            continue
        samples.append(SpeedSample(question_id.strip(), category, turns))
        samples_per_category[category] = samples_per_category.get(category, 0) + 1
    if not samples:
        raise FitError(f"No usable samples found in local SPEED-Bench split {bench!r}.")
    return samples


def speed_completion_response(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any],
                                                               str | None, str]:
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}
    finish_reason: str | None = None
    content = ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            content = message["content"]
        elif isinstance(choice.get("text"), str):
            content = choice["text"]
    return usage, timings, finish_reason, content


def speed_request(endpoint: str, model: str | None,
                  messages: list[dict[str, str]], osl: int,
                  timeout: float) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {"messages": messages, "max_tokens": osl,
                               "stream": False, "temperature": 0}
    if model:
        payload["model"] = model
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with DIRECT_OPENER.open(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace").replace("\n", "\\n")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(str(exc)) from exc
    if status != 200:
        detail = body[:500].decode("utf-8", errors="replace").replace("\n", "\\n")
        raise RuntimeError(f"HTTP {status}: {detail}")
    try:
        return json.loads(body.decode("utf-8")), time.perf_counter() - started
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON response: {exc}") from exc


def run_speed_sample(sample: SpeedSample, endpoint: str, model: str | None,
                     osl: int, timeout: float) -> SpeedRequestResult:
    messages: list[dict[str, str]] = []
    total_latency = 0.0
    prompt_tokens = completion_tokens = total_tokens = 0
    draft_n = draft_accepted = 0
    prompt_ms = predicted_ms = 0.0
    prompt_speed = predicted_speed = None
    finish_reason: str | None = None
    try:
        for turn in sample.turns:
            messages.append({"role": "user", "content": turn})
            data, latency = speed_request(endpoint, model, messages, osl, timeout)
            total_latency += latency
            usage, timings, finish_reason, assistant_text = speed_completion_response(data)
            turn_prompt = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
            turn_completion = int(usage.get("completion_tokens")
                                  or timings.get("predicted_n") or 0)
            turn_total = int(usage.get("total_tokens")
                             or (turn_prompt + turn_completion))
            prompt_tokens += turn_prompt
            completion_tokens += turn_completion
            total_tokens += turn_total
            draft_n += int(timings.get("draft_n") or 0)
            draft_accepted += int(timings.get("draft_n_accepted") or 0)
            prompt_ms += float(timings.get("prompt_ms") or 0)
            predicted_ms += float(timings.get("predicted_ms") or 0)
            if len(sample.turns) == 1:
                if isinstance(timings.get("prompt_per_second"), (int, float)):
                    prompt_speed = float(timings["prompt_per_second"])
                if isinstance(timings.get("predicted_per_second"), (int, float)):
                    predicted_speed = float(timings["predicted_per_second"])
            messages.append({"role": "assistant", "content": assistant_text})
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        if len(sample.turns) > 1:
            prompt_speed = prompt_tokens / (prompt_ms / 1000) if prompt_ms > 0 else None
            predicted_speed = (completion_tokens / (predicted_ms / 1000)
                               if predicted_ms > 0 else None)
        return SpeedRequestResult(
            sample.id, sample.category, True, len(sample.turns), total_latency,
            prompt_tokens, completion_tokens, total_tokens, finish_reason, draft_n,
            draft_accepted, prompt_ms or None, predicted_ms or None, prompt_speed,
            predicted_speed, None)
    except Exception as exc:
        return SpeedRequestResult(
            sample.id, sample.category, False, len(sample.turns), total_latency,
            0, 0, 0, None, 0, 0, None, None, None, None, str(exc))


def summarize_speed_results(category: str,
                            results: list[SpeedRequestResult]) -> dict[str, Any]:
    successful = [result for result in results if result.ok]
    prompt_speeds = [result.prompt_per_second for result in successful
                     if result.prompt_per_second is not None]
    predicted_speeds = [result.predicted_per_second for result in successful
                        if result.predicted_per_second is not None]
    accepted = sum(result.draft_n_accepted for result in successful)
    proposed = sum(result.draft_n for result in successful)
    return {
        "category": category,
        "requests": len(successful),
        "turns": sum(result.turns for result in successful),
        "failed": len(results) - len(successful),
        "avg_prompt_t_s": statistics.mean(prompt_speeds) if prompt_speeds else None,
        "avg_pred_t_s": statistics.mean(predicted_speeds) if predicted_speeds else None,
        "avg_latency": statistics.mean(result.latency_s for result in successful)
        if successful else None,
        "draft_n": proposed,
        "accepted": accepted,
        "accept_rate": accepted / proposed if proposed else None,
    }


def run_local_speed_bench(dataset_dir: Path, split: str, port: int,
                          sample_limit: int, config_name: str) -> dict[str, Any]:
    samples = load_speed_bench_samples(dataset_dir, split, sample_limit)
    endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
    label = f"SPEED-Bench {config_name}"
    print(f"SPEED-Bench loaded {len(samples)} local samples from {split}")
    started = time.monotonic()
    print(f"{label} elapsed time: {format_elapsed(0)}", end="", flush=True)
    results: list[SpeedRequestResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(run_speed_sample, sample, endpoint, None,
                                    SPEED_BENCH_OSL, SPEED_BENCH_REQUEST_TIMEOUT_SECONDS)
                   for sample in samples]
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(
                pending, timeout=0.5,
                return_when=concurrent.futures.FIRST_COMPLETED)
            results.extend(future.result() for future in done)
            print(f"\r{label} elapsed time: {format_elapsed(time.monotonic() - started)}",
                  end="", flush=True)
    print(f"\r{label} elapsed time: {format_elapsed(time.monotonic() - started)} "
          "(finished)")
    categories = list(dict.fromkeys(sample.category for sample in samples))
    summary = [summarize_speed_results(category,
                                       [result for result in results
                                        if result.category == category])
               for category in categories]
    overall = summarize_speed_results("overall", results)
    failed = [result for result in results if not result.ok]
    if failed:
        print(f"SPEED-Bench: {len(failed)} samples failed; first error: {failed[0].error}")
    return overall


def run_speed_bench_case(selection: Selection, gpus: list[GPU], metadata: Metadata,
                         context: int, batch: tuple[int, int] | None,
                         dataset_dir: Path,
                         split: str, config_name: str,
                         spec_type: str, sample_limit: int) -> dict[str, Any]:
    """Run one local-dataset SPEED-Bench configuration against a temporary server."""
    assert selection.server is not None
    help_text = run_text([str(selection.server), "--help"])
    command = server_command(selection, gpus, help_text, context, metadata,
                             batch=batch, spec_type=spec_type)
    port = int(command[command.index("--port") + 1])
    test_started = time.monotonic()
    print(f"\nSPEED-Bench: {config_name} ({split})")
    print(format_command(command, short_program=True))
    process: subprocess.Popen[str] | None = None
    output_queue: queue.Queue[str] | None = None
    lines: list[str] = []
    thread: threading.Thread | None = None
    try:
        process, output_queue, lines, thread = start_benchmark_server(
            command, f"{config_name} server", display_started=test_started)
        overall = run_local_speed_bench(dataset_dir, split, port,
                                         sample_limit, config_name)
        drain_output(output_queue, lines)
        generation = overall.get("avg_pred_t_s")
        acceptance = overall.get("accept_rate")
        completion = (f"Completed {config_name} ({split}): token generation "
                      f"{generation:.2f} t/s" if generation is not None else
                      f"Completed {config_name} ({split}): token generation unavailable")
        if acceptance is not None:
            completion += f"; acceptance {acceptance * 100:.1f}%"
        print(completion)
        return overall
    finally:
        if process is not None:
            stop_process(process)
        if thread is not None:
            thread.join(timeout=5)
        if output_queue is not None:
            drain_output(output_queue, lines)
        server_log = save_log("".join(lines), f"speed-server-{split}-")
        if server_log is not None:
            print(f"  server log: {server_log}")


def print_speed_bench_table(split: str, input_tokens: int,
                            results: list[tuple[str, dict[str, Any]]]) -> None:
    base_summary = next((summary for name, summary in results if name == "base"), None)
    base_decode = base_summary.get("avg_pred_t_s") if base_summary else None
    rows: list[list[str]] = []
    for config_name, summary in results:
        prompt = summary.get("avg_prompt_t_s")
        decode = summary.get("avg_pred_t_s")
        latency = summary.get("avg_latency")
        acceptance = summary.get("accept_rate")
        speedup = (decode / base_decode) if decode and base_decode else None
        rows.append([
            config_name,
            f"{prompt:.2f}" if prompt is not None else "n/a",
            f"{decode:.2f}" if decode is not None else "n/a",
            f"{latency:.2f}s" if latency is not None else "n/a",
            f"{acceptance * 100:.1f}%" if acceptance is not None else "n/a",
            f"{speedup:.2f}x" if speedup is not None else "n/a",
        ])
    headers = ["configuration", "pp t/s", "tg t/s", "latency", "accept", "tg vs base"]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print(f"\nSPEED-Bench summary ({split}; input {input_tokens} tokens, "
          f"output {SPEED_BENCH_OSL} tokens):")
    print("  " + "  ".join(header.ljust(widths[index])
                              for index, header in enumerate(headers)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(cell.ljust(widths[index])
                              for index, cell in enumerate(row)))


def run_speed_bench(selection: Selection, gpus: list[GPU], metadata: Metadata,
                    context: int, batch: tuple[int, int] | None,
                    sample_limit: int) -> None:
    """Compare base, ngram, draft/MTP, and draft/MTP+ngram configurations."""
    dataset_root_raw = os.environ.get(DATASETS_DIR_ENV, "").strip()
    if not dataset_root_raw:
        raise FitError(f"Set the {DATASETS_DIR_ENV} environment variable "
                       "to your generic datasets directory before running this "
                       "optional step.")
    dataset_dir = Path(dataset_root_raw).expanduser() / "SPEED-Bench"
    if context < SPEED_BENCH_CONTEXT:
        print(f"SPEED-Bench skipped: the fitted context ({context}) is below the "
              f"required benchmark context ({SPEED_BENCH_CONTEXT}).")
        return
    split = SPEED_BENCH_SPLIT
    input_tokens = SPEED_BENCH_INPUT_TOKENS
    configs = [("base", "none"), ("base + ngram-mod", "ngram-mod")]
    if selection.draft or selection.embedded_mtp:
        draft_type = selection.draft_spec_type
        if draft_type is None:
            raise FitError("A draft GGUF or embedded MTP was supplied but no draft speculative type was selected.")
        draft_label = "embedded MTP" if selection.embedded_mtp else "drafter"
        configs += [(f"base + {draft_label}", draft_type),
                    (f"base + {draft_label} + ngram-mod", f"{draft_type},ngram-mod")]
    print("\nSPEED-Bench will test: " + ", ".join(name for name, _ in configs))
    results: list[tuple[str, dict[str, Any]]] = []
    for config_name, spec_type in configs:
        try:
            summary = run_speed_bench_case(
                selection, gpus, metadata, SPEED_BENCH_CONTEXT, batch, dataset_dir, split,
                config_name, spec_type, sample_limit)
            results.append((config_name, summary))
        except FitError as exc:
            print(f"SPEED-Bench {config_name} skipped: {exc}")
    if results:
        print_speed_bench_table(split, input_tokens, results)


def probe_has_cuda_oom(log: str) -> bool:
    """Recognize the common CUDA allocation failures in a server startup log."""
    lowered = log.lower()
    return ("cuda malloc failed" in lowered or "cudamalloc failed" in lowered
            or "out of memory" in lowered or "out-of-memory" in lowered)


def probe_failure_reason(log: str) -> str:
    if probe_has_cuda_oom(log):
        if "draft model" in log.lower() or "failed to load draft" in log.lower():
            return "CUDA out of memory while loading the draft model"
        return "CUDA out of memory during server startup"
    return "server did not become ready"


def draft_device_free_from_log(log: str) -> dict[int, float]:
    """Return the latest reported free MiB for each device in a probe log."""
    result: dict[int, float] = {}
    draft_start = log.lower().rfind("loading draft model")
    if draft_start >= 0:
        log = log[draft_start:]
    pattern = re.compile(r"using device CUDA(\d+).*?-\s*([\d.]+)\s*MiB free")
    for match in pattern.finditer(log):
        result[int(match.group(1))] = float(match.group(2))
    return result


def next_lower_fit_context(context: int, native: int | None) -> int | None:
    candidates = [candidate for candidate in FIT_CONTEXTS
                  if candidate < context and (native is None or candidate <= native)]
    return max(candidates) if candidates else None


def probe_with_context_retries(selection: Selection, gpus: list[GPU], metadata: Metadata,
                               context: int, vram_buffer_gib: float,
                               automatic: bool = False) -> tuple[Probe, int]:
    """Probe once, then retry lower fit contexts after failed loads."""
    dense = metadata.experts <= 1
    result = probe(selection, gpus, metadata, context, vram_buffer_gib)
    oom = probe_has_cuda_oom(result.log)
    if result.fits_policy:
        return result, context
    # Dense models are required to fit entirely on the GPUs.  A failed startup
    # can still be caused by the requested context, so keep walking the fit
    # ladder even when the log does not contain an explicit CUDA OOM.
    if not dense and not result.ready and not oom:
        return result, context

    draft_device: int | None = None
    if selection.draft:
        logged_free = draft_device_free_from_log(result.log)
        allowed_devices = {gpu.index for gpu in gpus}
        candidates = {index: free for index, free in logged_free.items()
                      if index in allowed_devices}
        draft_device = (max(candidates, key=candidates.get) if candidates
                        else max(gpus, key=lambda gpu: gpu.free_gib).index)
        if oom:
            print(f"\nCUDA OOM detected during draft loading; preferred retry device: "
                  f"CUDA{draft_device}.")
        else:
            message = ("Probe did not meet the memory policy"
                       if result.ready else "Probe did not become ready")
            print(f"\n{message}; preferred retry device: CUDA{draft_device}.")
        prompt = "Retry with explicit draft placement and lower contexts?"
        retry_context = context
    else:
        prompt = "Retry at lower contexts?"
        retry_context = next_lower_fit_context(context, metadata.context_length)
        if retry_context is None:
            print("\nNo lower fit context remains.")
            return result, context
        print(f"\nProbe failed; next context is {retry_context}.")

    if not (automatic or ask_yes_no(prompt, True)):
        return result, context

    while True:
        placement = f" with draft on CUDA{draft_device}" if draft_device is not None else ""
        print(f"\nRetrying context {retry_context}{placement}...")
        try:
            result = probe(selection, gpus, metadata, retry_context,
                           vram_buffer_gib, draft_device=draft_device)
        except FitError as exc:
            print(f"Retry placement unavailable: {exc}")
            return result, retry_context
        if result.fits_policy:
            return result, retry_context
        if not dense and not result.ready and not probe_has_cuda_oom(result.log):
            return result, retry_context
        lower_context = next_lower_fit_context(retry_context, metadata.context_length)
        if lower_context is None:
            print("No lower fit context remains.")
            return result, retry_context
        print(f"Probe still fails; stepping down to context {lower_context}.")
        retry_context = lower_context


def probe(selection: Selection, gpus: list[GPU], metadata: Metadata, context: int,
          vram_buffer_gib: float,
          draft_device: int | None = None) -> Probe:
    assert selection.server is not None
    help_text = run_text([str(selection.server), "--help"])
    command = server_command(selection, gpus, help_text, context, metadata,
                             draft_device=draft_device)
    port = int(command[command.index("--port") + 1])
    before_gpu_free = {gpu.index: gpu.free_gib for gpu in gpus}
    env = cuda_env()
    started = time.monotonic()
    print(f"\nVerifying {context} tokens with a temporary llama-server load...")
    print(f"elapsed time: {format_elapsed(0)}", end="", flush=True)
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors="replace", bufsize=1, env=env)
    except OSError as exc:
        print(f"\relapsed time: {format_elapsed(time.monotonic() - started)} (failed)")
        raise FitError(f"Could not start llama-server: {exc}") from exc
    lines: list[str] = []
    output_queue: queue.Queue[str] = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    ready = False
    probe_status = "failed"
    try:
        while time.monotonic() - started < PROBE_TIMEOUT_SECONDS:
            while True:
                try:
                    lines.append(output_queue.get_nowait())
                except queue.Empty:
                    break
            if process.poll() is not None:
                break
            if health(port):
                ready = True
                break
            print(f"\relapsed time: {format_elapsed(time.monotonic() - started)}",
                  end="", flush=True)
            time.sleep(0.5)
        thread.join(timeout=5)
        while True:
            try:
                lines.append(output_queue.get_nowait())
            except queue.Empty:
                break
        log = "".join(lines)
        log_path = save_log(log, f"llama-probe-{context}-")
        after_ram = available_ram_gib()
        if not ready:
            return Probe(context, {}, {}, {}, 0.0, after_ram, log,
                         log_path, False, False, probe_failure_reason(log))
        time.sleep(1)
        while True:
            try:
                lines.append(output_queue.get_nowait())
            except queue.Empty:
                break
        used = query_process_vram(process.pid, gpus)
        if len(used) != len(gpus):
            try:
                after_gpus = {gpu.index: gpu.free_gib for gpu in detect_gpus()}
                used = {index: max(0.0, before_gpu_free[index] - after_gpus.get(index, before_gpu_free[index]))
                        for index in before_gpu_free}
            except FitError:
                pass
        kv, model, host_model_gib, _ = parse_probe_log(log, gpus)
        shares = free_shares(gpus)
        budgets = {gpu.index: gpu.free_gib - vram_buffer_gib * shares[gpu.index]
                   for gpu in gpus}
        vram_ok = bool(used) and all(
            used.get(index, 0.0) <= budgets[index] for index in budgets)
        allowed_host_model_gib = (selection.mmproj.stat().st_size / GIB if selection.mmproj else 0.0) + 0.5
        dense_no_spill = metadata.experts > 1 or host_model_gib <= allowed_host_model_gib
        fits = vram_ok and dense_no_spill
        reason = "fits policy" if fits else (
            f"VRAM/model-placement policy exceeded (GPU used {sum(used.values()):.2f} GiB, "
            f"host model {host_model_gib:.2f} GiB)")
        probe_status = "ready" if fits else "ready; policy failed"
        return Probe(context, used, kv, model, host_model_gib, after_ram,
                     log, log_path, True, fits, reason)
    finally:
        stop_process(process)
        if not ready:
            probe_status = "failed"
        print(f"\relapsed time: {format_elapsed(time.monotonic() - started)} "
              f"({probe_status})")


def real_workload(url: str, prompt_file: Path) -> None:
    try:
        prompt = prompt_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise FitError(f"Could not read prompt file {prompt_file}: {exc}") from exc
    payload = json.dumps({"prompt": prompt, "n_predict": 128, "temperature": 0,
                          "cache_prompt": False}).encode()
    request = urllib.request.Request(url.rstrip("/") + "/completion", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with DIRECT_OPENER.open(request, timeout=600) as response:
            result = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise FitError(f"Workload request failed: {exc}") from exc
    timings = result.get("timings", {})
    print(f"\nReal workload: prompt {timings.get('prompt_n', '?')} @ "
          f"{timings.get('prompt_per_second', 0):.1f} t/s | generation "
          f"{timings.get('predicted_n', '?')} @ "
          f"{timings.get('predicted_per_second', 0):.2f} t/s")


def print_plan(selection: Selection, metadata: Metadata, gpus: list[GPU], plan: dict[str, Any]) -> None:
    print("\nModel metadata")
    print(f"  model: {selection.model}")
    if selection.embedded_mtp:
        print(f"  draft: embedded MTP; mode: {selection.draft_spec_type}")
    elif selection.draft:
        print(f"  draft: {selection.draft}; mode: {selection.draft_spec_type}")
    print(f"  architecture: {metadata.architecture or 'unknown'}; experts: {metadata.experts or 'not declared'}")
    print(f"  native context: {metadata.context_length or 'not declared'}")
    for note in metadata.notes:
        print(f"  note: {note}")
    if metadata.sampling:
        print("  model sampling metadata: " + ", ".join(
            f"{key.removeprefix('general.sampling.')}={value}" for key, value in metadata.sampling.items()))

    print("\nMemory precheck")
    print(f"  free VRAM: {sum(gpu.free_gib for gpu in gpus):.2f} GiB; usable after 0.5 GiB buffer: "
          f"{plan['vram_budget']:.2f} GiB")
    print(f"  lower-bound files: model {plan['model_gib']:.2f} GiB + draft {plan['draft_gib']:.2f} GiB + "
          f"mmproj {plan['mmproj_gib']:.2f} GiB")
    print(f"  policy: {plan['policy']}")
    if metadata.experts <= 1:
        if not plan["fits_policy"]:
            print("  Model is unable to fit onto the GPUs under the dense all-GPU policy.")


def print_config(selection: Selection, gpus: list[GPU], metadata: Metadata, context: int,
                 batch: tuple[int, int] | None, bench_rows: list[BenchRow],
                 probe_result: Probe | None = None) -> None:
    args = ["-c", str(context), "-ctk", selection.cache_k, "-ctv", selection.cache_v]
    if batch:
        args += ["-b", str(batch[0]), "-ub", str(batch[1])]
    prompts = [row for row in bench_rows if row.prompt > 0]
    preferred_prompt = max(prompts, key=lambda row: row.tps) if prompts else None
    generations = [row for row in bench_rows if row.generation > 0]
    if preferred_prompt:
        matching = [row for row in generations
                    if row.batch == preferred_prompt.batch
                    and row.ubatch == preferred_prompt.ubatch]
        preferred_generation = max(matching or generations, key=lambda row: row.tps) \
            if generations else None
    else:
        preferred_generation = max(generations, key=lambda row: row.tps) if generations else None
    print("\nRecommended flags:")
    print("  " + " ".join(args))
    print(f"Estimated pp speed: {preferred_prompt.tps:.2f} t/s"
          if preferred_prompt else "Estimated pp speed: unavailable")
    print(f"Estimated tg speed: {preferred_generation.tps:.2f} t/s"
          if preferred_generation else "Estimated tg speed: unavailable")
    if metadata.experts > 1 and probe_result is not None:
        print(f"RAM available after load: {probe_result.available_ram_after_gib:.2f} GiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", help="running llama-server base URL for a real workload")
    parser.add_argument("--prompt-file", type=Path, help="prompt file used with --server-url")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-speed-bench", action="store_true")
    parser.add_argument("--speed-bench-limit", type=int, default=SPEED_BENCH_LIMIT,
                        help="samples per category for the optional SPEED-Bench step")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.server_url is None) != (args.prompt_file is None):
        raise FitError("--server-url and --prompt-file must be supplied together.")
    print("llama-fit — metadata, benchmark, and memory-policy planner")
    gpus = detect_gpus()
    print("\nEnter the path to the following GGUF files:")
    model = ask_path("Main model", ".gguf")
    mmproj = ask_path("Multimodal projector", ".gguf", optional=True)
    draft_choice = ask_path("Draft model", ".gguf", optional=True,
                            allow_embedded_mtp=True)
    embedded_mtp = draft_choice == "embedded-mtp"
    draft = draft_choice if isinstance(draft_choice, Path) else None
    print("\nKV cache format")
    print("  1. f16 / f16 (quality-first default)")
    print("  2. f16 / q8_0")
    print("  3. q8_0 / q8_0")
    print("  4. q8_0 / q5_1")
    print("  5. q8_0 / q4_0")
    print("  6. q5_1 / q5_1")
    print("  7. q4_0 / q4_0")
    while True:
        choice = input("Choice [1]: ").strip() or "1"
        if choice in ("1", "2", "3", "4", "5", "6"):
            cache_k, cache_v = (("f16", "f16"), ("f16", "q8_0"), ("q8_0", "q8_0"),
                                ("q8_0", "q5_1"), ("q8_0", "q4_0"), ("q5_1", "q5_1"),
                                ("q4_0", "q4_0"))[int(choice) - 1]
            break
    automatic = ask_yes_no("Automatically run full script?", True)
    cache_ram = DEFAULT_CACHE_RAM_GIB
    bandwidth_estimate = runtime_bandwidth_gbps(gpus)
    if bandwidth_estimate is not None:
        per_gpu = ", ".join(f"GPU {index}: {value:.0f} GB/s"
                             for index, value in bandwidth_estimate.per_gpu_gbps.items())
        planning_bandwidth = bandwidth_estimate.sequential_gbps * DEFAULT_BANDWIDTH_EFFICIENCY
        print(f"\nRuntime-derived GPU bandwidth: {per_gpu}")
        print(f"Estimated average bandwidth at {DEFAULT_BANDWIDTH_EFFICIENCY:.0%} efficiency: "
              f"{planning_bandwidth:.0f} GB/s")
    metadata = inspect_model(model)
    draft_spec_type = None
    if embedded_mtp:
        draft_spec_type = "draft-mtp"
        print("\nEmbedded MTP selected; using draft-mtp automatically.")
    elif draft:
        detected_draft_type = infer_draft_spec_type(draft)
        if detected_draft_type == "draft-mtp":
            draft_spec_type = detected_draft_type
            print("\nMTP sidecar detected; using draft-mtp automatically.")
        else:
            draft_spec_type = choose_draft_spec_type(automatic)
    server = find_tool("llama-server", prompt=not args.skip_probe and not automatic)
    fit_params = find_tool("llama-fit-params", prompt=not automatic)
    if fit_params is None:
        raise FitError("llama-fit-params is required. Put it on PATH or provide its path.")
    selection = Selection(server, None, fit_params, model, mmproj, draft, draft_spec_type,
                          cache_k, cache_v, cache_ram, not args.skip_probe,
                          embedded_mtp=embedded_mtp)
    plan = memory_plan(selection, metadata, gpus, DEFAULT_VRAM_BUFFER_GIB)
    if server is None and not args.skip_probe:
        print("No llama-server found; skipping live verification.")
        selection.verify_server = False
    # Fit first using a fresh memory snapshot. Any benchmark output is kept
    # out of this decision so the fitter sees the machine's normal state.
    try:
        gpus = detect_gpus()
        plan = memory_plan(selection, metadata, gpus, DEFAULT_VRAM_BUFFER_GIB)
    except FitError as exc:
        print(f"Warning: could not refresh free memory ({exc}); retaining the initial snapshot.")
    print_plan(selection, metadata, gpus, plan)
    if not plan["fits_policy"]:
        return 2
    fitted_context = run_fit_params(selection, gpus, metadata)
    if fitted_context is None:
        raise FitError("llama-fit-params could not find a viable context from FIT_CONTEXTS.")
    context = fitted_context
    draft_fit_context: int | None = None
    if (selection.draft or selection.embedded_mtp) and context >= 1024:
        draft_context = next_lower_fit_context(context, metadata.context_length)
        if draft_context is not None:
            draft_fit_context = context
            context = draft_context
    probe_result: Probe | None = None
    if selection.verify_server and server is not None and context >= 1024:
        context_note = (f" (one fit step below {draft_fit_context} to reserve VRAM for speculative decoding)"
                        if draft_fit_context is not None else "")
        print(f"\nFitted context limit: {context}{context_note}")
        if automatic or ask_yes_no("Verify with a real load?", True):
            probe_result, context = probe_with_context_retries(
                selection, gpus, metadata, context,
                DEFAULT_VRAM_BUFFER_GIB, automatic)
            print(f"\nProbe result: {'PASS' if probe_result.fits_policy else 'FAIL'} — "
                  f"{probe_result.reason}")
            print(f"  VRAM used: {sum(probe_result.gpu_used_gib.values()):.2f} GiB")
            if metadata.experts > 1:
                print(f"  RAM available after load: {probe_result.available_ram_after_gib:.2f} GiB")
            print(f"  Full probe log: {probe_result.log_path}")
            if not probe_result.fits_policy:
                print("\nThe real-load verification did not pass the memory policy; stopping.")
                return 2
    bench_rows: list[BenchRow] = []
    if not args.skip_bench:
        bench = find_tool("llama-bench", prompt=not automatic)
        if bench is not None:
            selection.bench = bench
            bench_rows = run_benchmark(selection, gpus, metadata, automatic)
        else:
            print("llama-bench was not found; skipping throughput benchmark.")
    batch = choose_bench(bench_rows)
    if (not args.skip_speed_bench and server is not None and context >= 1024
            and (probe_result is None or probe_result.fits_policy)):
        if automatic or ask_yes_no("\nRun SPEED-Bench speculative-decoding comparison?", True):
            if args.speed_bench_limit < 1:
                raise FitError("--speed-bench-limit must be at least 1.")
            run_speed_bench(selection, gpus, metadata, context, batch,
                            args.speed_bench_limit)
    if args.server_url and args.prompt_file:
        real_workload(args.server_url, args.prompt_file)
    if context < 1024:
        raise FitError("llama-fit-params returned no usable context.")
    print_config(selection, gpus, metadata, context, batch, bench_rows, probe_result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FitError, EOFError, KeyboardInterrupt) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(130 if isinstance(exc, (EOFError, KeyboardInterrupt)) else 1)
