"""
traffic_controller.py  -  Signal control, synchronization and deadlock safety
=============================================================================
This module owns the SHARED SIGNAL DATA and the rules for changing it.

The model
---------
One crossing with four approach roads (our four "junctions"):
    J1 North Road, J2 East Road, J3 South Road, J4 West Road
Two phases (compatible directions that may be green together):
    NS = {J1, J3}      EW = {J2, J4}
Any road of NS CONFLICTS with any road of EW.

OPERATING SYSTEM CONCEPTS DEMONSTRATED IN THIS FILE
---------------------------------------------------
* SYNCHRONIZATION (mutual exclusion)
    - one Lock per signal  (a "resource")  -> self.locks[j]
    - one RLock for the controller's own variables -> self.state_lock
  The dashboard thread reads signal data while the controller thread changes
  it. Locks make sure the reader never sees a half-updated (unsafe) state.

* DEADLOCK PREVENTION
  Deadlock needs four conditions (Coffman): mutual exclusion, hold-and-wait,
  no pre-emption, circular wait. We break two of them:
    1. CIRCULAR WAIT is impossible because every thread takes locks in ONE
       global order (J1 < J2 < J3 < J4) -> `ordered_locks()`.
    2. HOLD-AND-WAIT is limited because each acquire has a TIMEOUT; if a lock
       cannot be obtained, everything already held is released and the
       operation is retried on the next tick.
  Lock hierarchy:  state_lock  ->  junction locks (ascending id).
  `deadlock_demo()` at the bottom shows a real deadlock (unordered) and the
  fix (ordered) with two real threads.

* SAFE EMERGENCY PRE-EMPTION
  An emergency vehicle can cut a normal green short, but the controller still
  follows  GREEN -> YELLOW -> ALL-RED -> GREEN  so signals are never unsafe.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

RED, YELLOW, GREEN = "RED", "YELLOW", "GREEN"

JUNCTIONS: List[str] = ["J1", "J2", "J3", "J4"]
JUNCTION_NAMES = {"J1": "North Road", "J2": "East Road",
                  "J3": "South Road", "J4": "West Road"}
PHASE_GROUPS = {"NS": ["J1", "J3"], "EW": ["J2", "J4"]}
GROUP_OF = {j: g for g, members in PHASE_GROUPS.items() for j in members}
# Conflict table: which junctions must NOT be non-red while j is green
CONFLICTS = {j: sorted(k for k in JUNCTIONS if GROUP_OF[k] != GROUP_OF[j]) for j in JUNCTIONS}

# Timing rules (in simulated seconds)
MIN_GREEN = 6
MAX_GREEN = 25
YELLOW_TIME = 3
ALL_RED_TIME = 1
EMERGENCY_HOLD = 8
GREEN_PER_VEHICLE = 0.8       # extra green seconds per queued vehicle


def other_group(group: str) -> str:
    return "EW" if group == "NS" else "NS"


class LockTimeout(Exception):
    """Raised when a lock cannot be obtained in time (deadlock avoidance)."""


class TrafficController:
    def __init__(self, log_fn: Optional[Callable[[str, str], None]] = None,
                 lock_timeout: float = 1.0):
        self.log_fn = log_fn or (lambda cat, msg: None)
        self.lock_timeout = lock_timeout
        self.locks: Dict[str, threading.Lock] = {j: threading.Lock() for j in JUNCTIONS}
        self.state_lock = threading.RLock()
        self.reset()

    # ----------------------------------------------------------------- reset
    def reset(self) -> None:
        with self.state_lock:
            self.signals: Dict[str, str] = {j: RED for j in JUNCTIONS}
            self.phase = "NS"
            self.stage = GREEN
            self.timer = 10
            self.next_group: Optional[str] = None
            self.pending: deque = deque()          # emergencies waiting
            self.emergency: Optional[dict] = None  # emergency being handled
            self.emergency_history: List[dict] = []
            self.green_history: List[dict] = []
            self.phase_changes = 0
            self.conflict_violations = 0
            self.conflict_denials = 0
            self.lock_acquisitions = 0
            self.lock_timeouts = 0
            self.emergencies_served = 0
            self._write_signals_locked()

    # ----------------------------------------------------------- lock helpers
    @contextmanager
    def ordered_locks(self, junctions: List[str]):
        """Acquire several junction locks in ASCENDING ORDER with a timeout.

        Ordered acquisition makes a circular wait impossible; the timeout
        stops hold-and-wait from blocking forever.
        """
        acquired: List[str] = []
        try:
            for j in sorted(junctions):
                if not self.locks[j].acquire(timeout=self.lock_timeout):
                    self.lock_timeouts += 1
                    raise LockTimeout(f"could not lock {j}")
                self.lock_acquisitions += 1
                acquired.append(j)
            yield
        finally:
            for j in reversed(acquired):
                self.locks[j].release()

    def _write_signals_locked(self) -> None:
        """Write signal colours from (phase, stage). Caller holds state_lock."""
        with self.ordered_locks(JUNCTIONS):
            for j in JUNCTIONS:
                if self.stage == GREEN:
                    self.signals[j] = GREEN if GROUP_OF[j] == self.phase else RED
                elif self.stage == YELLOW:
                    self.signals[j] = YELLOW if GROUP_OF[j] == self.phase else RED
                else:
                    self.signals[j] = RED

    # --------------------------------------------------------------- safety
    def _violations_now(self) -> List[Tuple[str, str]]:
        """Pairs of conflicting roads that are BOTH not red (unsafe)."""
        bad = []
        for a in JUNCTIONS:
            for b in CONFLICTS[a]:
                if a < b and self.signals[a] != RED and self.signals[b] != RED:
                    bad.append((a, b))
        return bad

    def request_green(self, junction: str) -> Tuple[bool, str]:
        """Ask for permission to turn `junction` green (safety check only).

        Used by tests/UI to prove conflicts are refused. The real signal
        change is made only by the controller state machine (single writer).
        """
        with self.state_lock:
            with self.ordered_locks(JUNCTIONS):
                blockers = [k for k in CONFLICTS[junction] if self.signals[k] != RED]
                if blockers:
                    self.conflict_denials += 1
                    msg = (f"DENIED green for {junction}: conflicting "
                           f"{', '.join(blockers)} not RED")
                    self.log_fn("SAFETY", msg)
                    return False, msg
            return True, f"{junction} may be green (no conflicting road is active)"

    # ------------------------------------------------------------- emergency
    def register_emergency(self, junction: str, kind: str, now: int) -> None:
        with self.state_lock:
            item = {"junction": junction, "kind": kind, "group": GROUP_OF[junction],
                    "arrived_tick": now, "status": "WAITING", "green_tick": None,
                    "cleared_tick": None}
            self.pending.append(item)
            self.emergency_history.append(item)
            self.log_fn("EMERGENCY", f"{kind} detected on {JUNCTION_NAMES[junction]} ({junction}) "
                                     f"- highest priority request registered")

    def _update_emergency_locked(self, now: int) -> None:
        if self.emergency is None and self.pending:
            self.emergency = self.pending.popleft()
        e = self.emergency
        if e is None:
            return
        g = e["group"]
        # Another emergency from the same road group can ride the same green.
        for p in list(self.pending):
            if p["group"] == g and e["status"] == "SERVING":
                self.pending.remove(p)
                p["status"] = "SERVING"
                p["green_tick"] = now
                self.timer = max(self.timer, EMERGENCY_HOLD + 1)
                self.log_fn("EMERGENCY", f"{p['kind']} on {p['junction']} joins the active "
                                         f"emergency green")
        if e["status"] == "WAITING":
            if self.stage == GREEN and self.phase == g:
                e["status"] = "SERVING"
                e["green_tick"] = now
                self.timer = max(self.timer, EMERGENCY_HOLD + 1)
                self.log_fn("EMERGENCY", f"{e['junction']} already GREEN - holding green for "
                                         f"{e['kind']}")
            elif self.stage == GREEN:
                self.timer = 1                       # end this green now ...
                self.next_group = g                  # ... and serve emergency group next
                e["status"] = "PREEMPTING"
                self.log_fn("EMERGENCY", f"Pre-empting {self.phase} green. Safe sequence: "
                                         f"YELLOW({YELLOW_TIME}s) -> ALL-RED -> {g} GREEN")
            else:
                self.next_group = g                  # YELLOW/ALL-RED already running
                e["status"] = "PREEMPTING"
                self.log_fn("EMERGENCY", f"Transition in progress; next green goes to {g}")
        elif e["status"] == "PREEMPTING":
            self.next_group = g

    def _finish_emergency_locked(self, now: int) -> None:
        e = self.emergency
        if e is not None:
            e["status"] = "CLEARED"
            e["cleared_tick"] = now
            self.emergencies_served += 1
            self.log_fn("EMERGENCY", f"{e['kind']} on {e['junction']} has passed - "
                                     f"normal control resumes")
        self.emergency = None
        self.next_group = None

    # ----------------------------------------------------------- state machine
    @staticmethod
    def compute_green_time(demand: int) -> int:
        """Longer green for denser traffic (clamped to MIN..MAX)."""
        return int(max(MIN_GREEN, min(MAX_GREEN, round(MIN_GREEN + GREEN_PER_VEHICLE * demand))))

    def step(self, now: int, queue_lengths: Dict[str, int]) -> None:
        """Advance the controller by one tick (called by the SIGNAL_CONTROL job)."""
        with self.state_lock:
            self._update_emergency_locked(now)
            self.timer -= 1
            if self.timer <= 0:
                self._advance_locked(now, queue_lengths)
            try:
                self._write_signals_locked()
            except LockTimeout:
                self.log_fn("SAFETY", "Lock timeout while writing signals - retry next tick")
            bad = self._violations_now()
            if bad:
                self.conflict_violations += len(bad)
                self.log_fn("SAFETY", f"CONFLICT VIOLATION detected: {bad}")

    def _advance_locked(self, now: int, queue_lengths: Dict[str, int]) -> None:
        if self.stage == GREEN:
            if self.emergency and self.emergency["status"] == "SERVING" \
                    and self.emergency["group"] == self.phase:
                self._finish_emergency_locked(now)
            if self.next_group is None:
                self.next_group = other_group(self.phase)    # round-robin of phases
            self.stage, self.timer = YELLOW, YELLOW_TIME
            self.log_fn("SIGNAL", f"{self.phase} roads: GREEN -> YELLOW ({YELLOW_TIME}s)")
        elif self.stage == YELLOW:
            self.stage, self.timer = "ALL_RED", ALL_RED_TIME
            self.log_fn("SIGNAL", f"{self.phase} roads: YELLOW -> RED, all-red clearance "
                                  f"({ALL_RED_TIME}s)")
        else:  # ALL_RED -> grant green to the next group
            target = self.next_group or other_group(self.phase)
            try:
                self._grant_group_locked(target, now, queue_lengths)
            except LockTimeout:
                self.timer = 1                               # no hold-and-wait: retry
                self.log_fn("SAFETY", "Lock timeout while granting green - retrying")

    def _grant_group_locked(self, group: str, now: int, queue_lengths: Dict[str, int]) -> None:
        members = PHASE_GROUPS[group]
        with self.ordered_locks(JUNCTIONS):
            for j in members:                                # safety rule before GREEN
                for k in CONFLICTS[j]:
                    if self.signals[k] != RED:
                        self.conflict_denials += 1
                        self.log_fn("SAFETY", f"Green for {j} DENIED, {k} is {self.signals[k]}")
                        self.timer = 1
                        return
        e = self.emergency
        demand = max(queue_lengths.get(j, 0) for j in members)
        if e and e["group"] == group and e["status"] in ("PREEMPTING", "WAITING", "SERVING"):
            duration = EMERGENCY_HOLD
            e["status"], e["green_tick"] = "SERVING", now
            reason = f"emergency ({e['kind']})"
        else:
            duration = self.compute_green_time(demand)
            reason = f"density: busiest queue = {demand} vehicles"
        self.phase, self.stage, self.timer = group, GREEN, duration
        self.next_group = None
        self.phase_changes += 1
        self.green_history.append({"tick": now, "group": group, "duration": duration,
                                   "demand": demand, "reason": reason})
        self.log_fn("SIGNAL", f"{group} roads ({', '.join(members)}): GREEN for {duration}s "
                              f"[{reason}]")

    # ---------------------------------------------------------------- reading
    def snapshot(self) -> dict:
        """Consistent copy of the shared signal data (readers take the locks)."""
        with self.state_lock:
            try:
                with self.ordered_locks(JUNCTIONS):
                    signals = dict(self.signals)
            except LockTimeout:
                signals = dict(self.signals)
            if self.stage == GREEN:
                to_green_other = self.timer + YELLOW_TIME + ALL_RED_TIME
            elif self.stage == YELLOW:
                to_green_other = self.timer + ALL_RED_TIME
            else:
                to_green_other = self.timer
            countdown = {}
            for j in JUNCTIONS:
                if GROUP_OF[j] == self.phase and self.stage in (GREEN, YELLOW):
                    countdown[j] = max(self.timer, 0)
                else:
                    countdown[j] = max(to_green_other, 0)
            e = dict(self.emergency) if self.emergency else None
            return {"signals": signals, "countdown": countdown, "phase": self.phase,
                    "stage": self.stage, "timer": self.timer, "emergency": e,
                    "phase_changes": self.phase_changes,
                    "conflict_violations": self.conflict_violations,
                    "conflict_denials": self.conflict_denials,
                    "lock_acquisitions": self.lock_acquisitions,
                    "lock_timeouts": self.lock_timeouts,
                    "emergencies_served": self.emergencies_served,
                    "pending_emergencies": len(self.pending),
                    "green_history": list(self.green_history[-20:])}


# =====================================================================
# DEADLOCK DEMONSTRATION (two real threads, two real locks)
# =====================================================================
def deadlock_demo(timeout: float = 0.8) -> dict:
    """Run the classic two-lock deadlock, then the ordered-lock fix.

    Scenario: Thread A needs locks (J1, J2) and thread B needs (J2, J1).
      * UNSAFE  : each thread takes its FIRST lock, waits until the other has
                  done the same (barrier), then asks for its SECOND lock.
                  Each holds what the other needs -> circular wait. We use a
                  timeout only so the demo can report the deadlock and exit.
      * SAFE    : both threads take locks in the global order J1 -> J2.
                  The second thread simply waits its turn: no deadlock.
    """
    def run(ordered: bool) -> dict:
        l1, l2 = threading.Lock(), threading.Lock()
        barrier = threading.Barrier(2, timeout=3)
        result: Dict[str, str] = {}
        started = time.perf_counter()

        def worker(name: str, first: threading.Lock, second: threading.Lock,
                   first_name: str, second_name: str):
            if ordered:   # global ordering rule: always J1 before J2
                if first_name > second_name:
                    first, second = second, first
                    first_name, second_name = second_name, first_name
            first.acquire()
            try:
                if not ordered:
                    barrier.wait()          # make sure both hold their first lock
                time.sleep(0.05)
                if second.acquire(timeout=timeout):
                    result[name] = (f"OK - got {first_name} then {second_name}"
                                    + ("" if ordered else " (only AFTER the other thread timed "
                                       "out and released its lock)"))
                    second.release()
                else:
                    result[name] = (f"STUCK - holds {first_name}, waiting for {second_name} "
                                    f"(deadlock detected by timeout)")
            finally:
                first.release()

        ta = threading.Thread(target=worker, args=("Thread A", l1, l2, "J1", "J2"))
        tb = threading.Thread(target=worker, args=("Thread B", l2, l1, "J2", "J1"))
        ta.start(); tb.start(); ta.join(); tb.join()
        deadlocked = any(v.startswith("STUCK") for v in result.values())
        return {"results": result, "deadlock": deadlocked,
                "seconds": round(time.perf_counter() - started, 2)}

    return {"unsafe_unordered": run(False), "safe_ordered": run(True)}


def lost_update_demo(threads: int = 4, increments: int = 2000) -> dict:
    """Race-condition demo: a shared counter updated with and without a lock."""
    def run(use_lock: bool) -> int:
        box = {"n": 0}
        lock = threading.Lock()

        def work():
            for _ in range(increments):
                if use_lock:
                    with lock:
                        v = box["n"]
                        time.sleep(0)           # yield the CPU mid-update
                        box["n"] = v + 1
                else:
                    v = box["n"]
                    time.sleep(0)               # another thread may run here
                    box["n"] = v + 1

        ts = [threading.Thread(target=work) for _ in range(threads)]
        for t in ts: t.start()
        for t in ts: t.join()
        return box["n"]

    expected = threads * increments
    unlocked, locked = run(False), run(True)
    return {"expected": expected, "without_lock": unlocked, "with_lock": locked,
            "lost_updates_without_lock": expected - unlocked}
