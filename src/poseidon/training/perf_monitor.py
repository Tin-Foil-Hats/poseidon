                 
import time, csv, os, threading, queue
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import psutil

try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

try:
    import torch
    import lightning.pytorch as L
except Exception:
    torch = None
    L = None

                               
def _now_s():
    return time.time()

def _safe_ratio(a, b):
    return (a / b) if b else 0.0

                                   
class GPUSampler(threading.Thread):
    def __init__(self, interval=0.5, device_index=0):
        super().__init__(daemon=True)
        self.interval = interval
        self.device_index = device_index
        self.q = queue.Queue()
        self._stop = threading.Event()
        self._ok = False
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._ok = True
            except Exception:
                self._ok = False

    def run(self):
        if not self._ok:
            return
        while not self._stop.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(self.h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.h)
            self.q.put({
                "ts": _now_s(),
                "gpu_util": float(util.gpu),             
                "mem_util": 100.0 * mem.used / mem.total,     
                "mem_used_MB": mem.used / (1024**2),
                "mem_total_MB": mem.total / (1024**2),
                "pstate": pynvml.nvmlDeviceGetPerformanceState(self.h),
            })
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()

    def drain(self) -> List[Dict[str, Any]]:
        out = []
        try:
            while True:
                out.append(self.q.get_nowait())
        except queue.Empty:
            pass
        return out

                                                
class IOSampler(threading.Thread):
    def __init__(self, interval=0.5, pid=None):
        super().__init__(daemon=True)
        self.interval = interval
        self.proc = psutil.Process(pid or os.getpid())
        self.q = queue.Queue()
        self._stop = threading.Event()

    def run(self):
        last = self.proc.io_counters()
        last_ts = _now_s()
        while not self._stop.is_set():
            time.sleep(self.interval)
            now = self.proc.io_counters()
            now_ts = _now_s()
            dt = now_ts - last_ts
            rBps = (now.read_bytes  - last.read_bytes)  / dt
            wBps = (now.write_bytes - last.write_bytes) / dt
            self.q.put({
                "ts": now_ts,
                "read_MBps": rBps / (1024**2),
                "write_MBps": wBps / (1024**2),
            })
            last, last_ts = now, now_ts

    def stop(self):
        self._stop.set()

    def drain(self) -> List[Dict[str, Any]]:
        out = []
        try:
            while True:
                out.append(self.q.get_nowait())
        except queue.Empty:
            pass
        return out

                                    
@dataclass
class BatchTiming:
    ts: float
    epoch: int
    step: int
    data_ms: float
    compute_ms: float
    batch_size: Optional[int]
    gpu_util_avg: Optional[float]
    gpu_mem_util_avg: Optional[float]
    io_r_MBps_avg: Optional[float]
    io_w_MBps_avg: Optional[float]

class PerfCallback(L.Callback):
    """Lightning callback: splits data time vs compute time, samples GPU + IO, writes CSV."""
    def __init__(self, csv_path="perf_log.csv", gpu_index=0, sample_interval=0.5):
        super().__init__()
        self.csv_path = csv_path
        self.gpu = GPUSampler(interval=sample_interval, device_index=gpu_index)
        self.io  = IOSampler(interval=sample_interval)
        self._t_data_start = None
        self._t_forward_start = None
        self._rows: List[BatchTiming] = []
        self._header_written = False

                           
    def on_fit_start(self, trainer, pl_module):
        self.gpu.start()
        self.io.start()

    def on_fit_end(self, trainer, pl_module):
        self.gpu.stop(); self.io.stop()
        self._flush_csv(force=True)

                             
    def on_before_batch_transfer(self, trainer, pl_module, batch):
        self._t_data_start = _now_s()

    def on_after_batch_transfer(self, trainer, pl_module, batch):
                           
        if self._t_data_start is not None:
            self._data_ms = 1000.0 * (_now_s() - self._t_data_start)
        else:
            self._data_ms = float("nan")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._t_forward_start = _now_s()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        compute_ms = 1000.0 * (_now_s() - self._t_forward_start) if self._t_forward_start else float("nan")
                                  
        g = self.gpu.drain()
        i = self.io.drain()
        gpu_util = sum(d["gpu_util"] for d in g)/len(g) if g else None
        gpu_memu = sum(d["mem_util"] for d in g)/len(g) if g else None
        io_r = sum(d["read_MBps"]  for d in i)/len(i) if i else None
        io_w = sum(d["write_MBps"] for d in i)/len(i) if i else None
                                                       
        try:
            bs = len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch)
        except Exception:
            bs = None

        row = BatchTiming(
            ts=_now_s(),
            epoch=trainer.current_epoch,
            step=trainer.global_step,
            data_ms=self._data_ms,
            compute_ms=compute_ms,
            batch_size=bs,
            gpu_util_avg=gpu_util,
            gpu_mem_util_avg=gpu_memu,
            io_r_MBps_avg=io_r,
            io_w_MBps_avg=io_w,
        )
        self._rows.append(row)

                                  
        if trainer.is_global_zero:
            trainer.logger.log_metrics({
                "perf/data_ms": row.data_ms,
                "perf/compute_ms": row.compute_ms,
                "perf/gpu_util%": row.gpu_util_avg or 0.0,
                "perf/io_read_MBps": row.io_r_MBps_avg or 0.0,
                "perf/io_write_MBps": row.io_w_MBps_avg or 0.0,
            }, step=trainer.global_step)

                            
        if (len(self._rows) % 20) == 0:
            self._flush_csv()

    def _flush_csv(self, force=False):
        if not self._rows and not force:
            return
        write_header = not os.path.exists(self.csv_path) or not self._header_written
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(self._rows[0]).keys()))
            if write_header:
                w.writeheader()
                self._header_written = True
            for r in self._rows:
                w.writerow(asdict(r))
        self._rows.clear()

                                                               
class TorchOperatorProfiler:
    """
    Usage:
        with TorchOperatorProfiler("trace.json", wait=2, warmup=2, active=4):
            trainer.fit(...)
    Open trace.json in chrome://tracing or TensorBoard.
    """
    def __init__(self, trace_path: str, wait=1, warmup=1, active=3, repeat=1):
        import torch
        self.torch = torch
        self.trace_path = trace_path
        self.wait, self.warmup, self.active, self.repeat = wait, warmup, active, repeat

    def __enter__(self):
        self.prof = self.torch.profiler.profile(
            schedule=self.torch.profiler.schedule(wait=self.wait, warmup=self.warmup, active=self.active, repeat=self.repeat),
            on_trace_ready=self.torch.profiler.tensorboard_trace_handler(os.path.dirname(self.trace_path) or "."),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            activities=[self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA]
        )
        self.prof.__enter__()
        return self.prof

    def __exit__(self, exc_type, exc, tb):
        self.prof.__exit__(exc_type, exc, tb)
