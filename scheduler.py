"""
scheduler.py  -  Pre-emptive Priority Scheduler with Aging and Round Robin
==========================================================================
OPERATING SYSTEM CONCEPTS DEMONSTRATED IN THIS FILE
---------------------------------------------------
1. PRE-EMPTIVE PRIORITY SCHEDULING
   Every job has a priority number. LOWER NUMBER = HIGHER PRIORITY.
       0 = Emergency detection      (highest)
       1 = Signal control
       2 = Density analysis
       3 = Logging / Dashboard      (lowest)
   The CPU is simulated in small time slices called "quanta". Before every
   quantum the scheduler picks the READY job with the best priority. If a
   higher-priority job arrives while a lower-priority job is half-finished,
   the lower-priority job is PRE-EMPTED (kicked off the CPU).

2. AGING
   A job that keeps waiting slowly gets a better priority (its number goes
   down). This prevents STARVATION (a low-priority job never running).
   Aging never lifts a job above priority 1, so it can never beat the
   emergency task.

3. ROUND ROBIN
   When several READY jobs have the SAME effective priority, they take turns:
   the job that ran least recently goes first. Each job gets one quantum, then
   goes to the back of the line.

Time in this file is measured in simulation "ticks" (1 tick = 1 simulated
second). Each tick the CPU can execute `quanta_per_tick` quanta.
"""

from __future__ import annotations

import itertools
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# Priority levels (lower number = more important)
PRIORITY_EMERGENCY = 0
PRIORITY_SIGNAL = 1
PRIORITY_DENSITY = 2
PRIORITY_LOGGING = 3

PRIORITY_LABELS = {
    0: "0 - Emergency",
    1: "1 - Signal control",
    2: "2 - Density analysis",
    3: "3 - Logging/Dashboard",
}


@dataclass
class Job:
    """One schedulable unit of work (like a process / PCB in an OS)."""

    job_id: int
    name: str
    base_priority: int
    burst: int                      # total quanta needed to finish
    action: Callable[[], None]      # code executed when the job completes
    release_tick: int
    effective_priority: int = field(init=False)
    remaining: int = field(init=False)
    status: str = "READY"           # READY / RUNNING / PREEMPTED / DONE
    waiting: int = 0                # ticks spent waiting in the ready queue
    last_run: int = -1              # dispatch sequence number (for Round Robin)
    ran_tick: int = -1
    preemptions: int = 0
    aged_steps: int = 0
    finish_tick: Optional[int] = None

    def __post_init__(self):
        self.effective_priority = self.base_priority
        self.remaining = self.burst


class PriorityScheduler:
    """Simulated single-CPU pre-emptive priority scheduler."""

    def __init__(self, quanta_per_tick: int = 4, aging_interval: int = 3,
                 aging_enabled: bool = True, aging_floor: int = 1,
                 log_fn: Optional[Callable[[str, str], None]] = None):
        self.quanta_per_tick = quanta_per_tick
        self.aging_interval = aging_interval      # ticks waited per priority step
        self.aging_enabled = aging_enabled
        self.aging_floor = aging_floor            # best priority aging can reach
        self.log_fn = log_fn or (lambda cat, msg: None)

        self.ready: List[Job] = []                # the READY QUEUE
        self.finished: deque = deque(maxlen=40)
        self.trace: deque = deque(maxlen=300)     # dispatch history (Gantt-like)
        self.stats: Dict[str, dict] = {}          # per-task statistics

        self._ids = itertools.count(1)
        self._dispatch_clock = 0
        self._last_job: Optional[Job] = None
        self._lock = threading.Lock()             # protects the ready queue

        self.context_switches = 0
        self.preemptions = 0
        self.rr_rotations = 0
        self.aging_promotions = 0
        self.missed_releases = 0

    # ------------------------------------------------------------------ API
    def submit(self, name: str, priority: int, burst: int,
               action: Callable[[], None], tick: int) -> Optional[Job]:
        """Add a job to the ready queue.

        A periodic task is NOT released again while its previous instance is
        still pending (like a real-time overrun); that is counted as a
        'missed release'.
        """
        with self._lock:
            st = self._stat(name, priority)
            if any(j.name == name for j in self.ready):
                st["missed"] += 1
                self.missed_releases += 1
                return None
            job = Job(next(self._ids), name, priority, burst, action, tick)
            self.ready.append(job)
            return job

    def run_tick(self, now: int) -> List[str]:
        """Run one simulation tick = `quanta_per_tick` CPU quanta, then age."""
        executed: List[str] = []
        for _ in range(self.quanta_per_tick):
            with self._lock:
                job = self._pick()
            if job is None:
                break
            self._dispatch(job, now)
            executed.append(job.name)
        with self._lock:
            self._age(now)
        return executed

    # ------------------------------------------------------------- internals
    def _stat(self, name: str, priority: int) -> dict:
        return self.stats.setdefault(name, {
            "base_priority": priority, "runs": 0, "total_wait": 0,
            "max_wait": 0, "preemptions": 0, "missed": 0, "aged": 0})

    def _pick(self) -> Optional[Job]:
        """Choose the next job.

        sort key = (effective priority, last_run, job_id)
          * lowest priority number wins            -> PRIORITY SCHEDULING
          * tie -> least recently run wins         -> ROUND ROBIN
        """
        if not self.ready:
            return None
        return min(self.ready, key=lambda j: (j.effective_priority, j.last_run, j.job_id))

    def _dispatch(self, job: Job, now: int) -> None:
        prev = self._last_job
        if prev is not None and prev is not job and prev.remaining > 0:
            self.context_switches += 1
            if job.effective_priority < prev.effective_priority:
                # A more important job took the CPU from an unfinished one.
                prev.status = "PREEMPTED"
                prev.preemptions += 1
                self.preemptions += 1
                self.stats[prev.name]["preemptions"] += 1
                self.log_fn("SCHEDULER",
                            f"PRE-EMPT: {job.name} (prio {job.effective_priority}) took the CPU "
                            f"from {prev.name} (prio {prev.effective_priority}, "
                            f"{prev.remaining} quanta left)")
            else:
                prev.status = "READY"
                self.rr_rotations += 1       # equal priority -> Round Robin turn

        job.status = "RUNNING"
        job.ran_tick = now
        self._dispatch_clock += 1
        job.last_run = self._dispatch_clock
        job.remaining -= 1
        self.trace.append({"tick": now, "job": job.name,
                           "priority": job.effective_priority,
                           "remaining": job.remaining})

        if job.remaining == 0:
            job.action()                      # the job's real work happens here
            job.status = "DONE"
            job.finish_tick = now
            with self._lock:
                if job in self.ready:
                    self.ready.remove(job)
                self.finished.append(job)
                st = self._stat(job.name, job.base_priority)
                st["runs"] += 1
                st["total_wait"] += job.waiting
                st["max_wait"] = max(st["max_wait"], job.waiting)
            self._last_job = None
        else:
            self._last_job = job

    def _age(self, now: int) -> None:
        """AGING: every job that waited this tick gets one step closer to a
        better priority after `aging_interval` ticks of waiting."""
        for job in self.ready:
            if job.ran_tick == now:
                continue
            job.waiting += 1
            if (self.aging_enabled and job.base_priority > self.aging_floor
                    and job.effective_priority > self.aging_floor
                    and job.waiting % self.aging_interval == 0):
                old = job.effective_priority
                job.effective_priority -= 1
                job.aged_steps += 1
                self.aging_promotions += 1
                self._stat(job.name, job.base_priority)["aged"] += 1
                self.log_fn("SCHEDULER",
                            f"AGING: {job.name} waited {job.waiting} ticks -> "
                            f"priority {old} -> {job.effective_priority}")

    # ----------------------------------------------------------- reporting
    def table(self) -> List[dict]:
        """Rows for the 'task scheduling table' shown on the dashboard."""
        with self._lock:
            rows = []
            for j in sorted(self.ready, key=lambda x: (x.effective_priority, x.job_id)):
                rows.append({
                    "Task": j.name,
                    "Base priority": j.base_priority,
                    "Current priority": j.effective_priority,
                    "Status": j.status,
                    "Waiting (ticks)": j.waiting,
                    "Quanta left": j.remaining,
                    "Pre-empted": j.preemptions,
                })
            for j in list(self.finished)[-6:][::-1]:
                rows.append({
                    "Task": j.name,
                    "Base priority": j.base_priority,
                    "Current priority": j.effective_priority,
                    "Status": "DONE",
                    "Waiting (ticks)": j.waiting,
                    "Quanta left": 0,
                    "Pre-empted": j.preemptions,
                })
            return rows

    def task_stats(self) -> List[dict]:
        with self._lock:
            out = []
            for name, st in sorted(self.stats.items(), key=lambda kv: (kv[1]["base_priority"], kv[0])):
                runs = st["runs"]
                out.append({
                    "Task": name,
                    "Base priority": st["base_priority"],
                    "Completed runs": runs,
                    "Avg wait (ticks)": round(st["total_wait"] / runs, 2) if runs else 0.0,
                    "Max wait (ticks)": st["max_wait"],
                    "Pre-empted": st["preemptions"],
                    "Aging steps": st["aged"],
                    "Skipped releases": st["missed"],
                })
            return out

    def summary(self) -> dict:
        return {
            "context_switches": self.context_switches,
            "preemptions": self.preemptions,
            "rr_rotations": self.rr_rotations,
            "aging_promotions": self.aging_promotions,
            "missed_releases": self.missed_releases,
            "ready_queue_length": len(self.ready),
        }


# =====================================================================
# Stand-alone demonstrations (used by the "OS Concepts" page and tests)
# =====================================================================
def preemption_demo() -> dict:
    """A long low-priority job is running; an emergency job arrives at tick 3
    and immediately takes the CPU."""
    s = PriorityScheduler(quanta_per_tick=1, aging_enabled=False)
    events: List[str] = []
    s.log_fn = lambda cat, msg: events.append(msg)
    done = []
    s.submit("LOG_WRITER", PRIORITY_LOGGING, 5, lambda: done.append("LOG_WRITER"), 1)
    for t in range(1, 9):
        if t == 3:
            s.submit("EMERGENCY_DETECT", PRIORITY_EMERGENCY, 2,
                     lambda: done.append("EMERGENCY"), t)
        s.run_tick(t)
    return {"trace": [(e["tick"], e["job"]) for e in s.trace],
            "preemptions": s.preemptions, "events": events, "finish_order": done}


def round_robin_demo() -> dict:
    """Three equal-priority jobs share the CPU one quantum at a time."""
    s = PriorityScheduler(quanta_per_tick=1, aging_enabled=False)
    for n in ("DENSITY_J1", "DENSITY_J2", "DENSITY_J3"):
        s.submit(n, PRIORITY_DENSITY, 3, lambda: None, 1)
    for t in range(1, 10):
        s.run_tick(t)
    return {"trace": [(e["tick"], e["job"]) for e in s.trace],
            "rotations": s.rr_rotations}


def aging_demo(aging_enabled: bool, ticks: int = 30, aging_interval: int = 3) -> dict:
    """One priority-3 job competes with a NEW priority-1 job released every
    tick on a 1-quantum CPU. Without aging it starves; with aging it runs."""
    s = PriorityScheduler(quanta_per_tick=1, aging_interval=aging_interval,
                          aging_enabled=aging_enabled)
    ran_at: List[int] = []
    low = s.submit("LOW_PRIORITY_LOGGER", PRIORITY_LOGGING, 1,
                   lambda: ran_at.append(1), 1)
    for t in range(1, ticks + 1):
        s.submit("HIGH_PRIORITY_FLOOD", PRIORITY_SIGNAL, 1, lambda: None, t)
        s.run_tick(t)
    finish_tick = low.finish_tick if low else None
    return {"aging_enabled": aging_enabled, "ticks_simulated": ticks,
            "low_job_finished_at_tick": finish_tick,
            "low_job_final_priority": low.effective_priority if low else None,
            "low_job_waiting": low.waiting if low else None,
            "starved": finish_tick is None}
