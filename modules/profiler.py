#profiler.py
import threading
import time
from typing import Dict, Optional

from loguru import logger
import psutil

# Mapping of thread id to human readable tag
_thread_tags: Dict[int, str] = {}
# Last CPU time measurement per thread id
_last_cpu_times: Dict[int, tuple[float, float]] = {}
# Last inference duration per tag
_last_inference: Dict[str, float] = {}

_process = psutil.Process()


def register_thread(tag: str) -> None:
    """Register current thread with a tag for profiling."""
    _thread_tags[threading.get_ident()] = tag


def log_inference(tag: str, duration: float) -> None:
    """Record a YOLOv8 inference duration for the given tag."""
    _last_inference[tag] = duration


def profile_predict(model, tag: str, *args, **kwargs):
    """Wrap YOLOv8 ``predict`` and log inference duration."""
    start = time.time()
    res = model.predict(*args, **kwargs)
    log_inference(tag, time.time() - start)
    return res


def _calc_cpu_percent(tid: int, cpu_time: float, now: float) -> float:
    last = _last_cpu_times.get(tid)
    if not last:
        _last_cpu_times[tid] = (cpu_time, now)
        return 0.0
    diff = cpu_time - last[0]
    interval = now - last[1]
    _last_cpu_times[tid] = (cpu_time, now)
    if interval <= 0:
        return 0.0
    return (diff / interval) * 100.0 / psutil.cpu_count()


def _collect_stats() -> Dict[str, tuple[float, float, Optional[float]]]:
    """Return stats for registered threads."""
    mem = _process.memory_info().rss / (1024 * 1024)
    now = time.time()
    stats = {}
    for th in _process.threads():
        tid = th.id
        tag = _thread_tags.get(tid)
        if not tag:
            continue
        cpu_time = th.user_time + th.system_time
        cpu_pct = _calc_cpu_percent(tid, cpu_time, now)
        inf = _last_inference.get(tag)
        stats[tag] = (cpu_pct, mem, inf)
    return stats


def log_resource_usage(tag: str) -> None:
    """Immediately log resource usage for the given tag."""
    stats = _collect_stats().get(tag)
    if not stats:
        logger.info(f"[Profiler] {tag} not registered")
        return
    cpu, mem, inf = stats
    msg = f"[Profiler] {tag} CPU: {cpu:.1f}%, RAM: {mem:.0f}MB"
    if inf is not None:
        msg += f", Last YOLOv8 Inference: {inf:.2f}s"
    logger.info(msg)


class Profiler(threading.Thread):
    """Background profiler thread."""

    def __init__(self, interval: int = 5):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True

    def run(self) -> None:
        while self.running:
            stats = _collect_stats()
            for tag, (cpu, mem, inf) in stats.items():
                msg = f"[Profiler] {tag} CPU: {cpu:.1f}%, RAM: {mem:.0f}MB"
                if inf is not None:
                    msg += f", Last YOLOv8 Inference: {inf:.2f}s"
                logger.info(msg)
            time.sleep(self.interval)


_profiler: Optional[Profiler] = None


def start_profiler(cfg: dict) -> None:
    """Start the background profiler if enabled in config."""
    global _profiler
    if not cfg.get("enable_profiling"):
        return
    if _profiler and _profiler.is_alive():
        return
    interval = int(cfg.get("profiling_interval", 5))
    _profiler = Profiler(interval)
    _profiler.start()
    logger.info(f"Profiler started with interval={interval}s")
