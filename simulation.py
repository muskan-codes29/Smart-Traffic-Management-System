"""
simulation.py  -  Vehicle generation, IPC queues, threads and the whole engine
==============================================================================
This file glues everything together:

    VehicleGenerator (PRODUCER thread) --queue--> intake in step()
    UI / emergency button ---------------queue--> EMERGENCY_DETECT job
    every module ------------------------queue--> LOGGER job -> log file

OPERATING SYSTEM CONCEPTS DEMONSTRATED IN THIS FILE
---------------------------------------------------
* THREADS         : a producer thread (vehicle arrivals) and a controller
                    thread (runs the scheduler) work at the same time.
* IPC             : modules never call each other's internals to pass events.
                    They put messages in thread-safe `queue.Queue` objects
                    (arrival queues, emergency queue, log queue).
* SCHEDULING      : each tick, work is released as JOBS to the scheduler
                    (scheduler.py) which decides what runs on the "CPU".
* SYNCHRONIZATION : `sim_lock` protects the shared vehicle counters that the
                    dashboard thread reads while the controller thread updates.
"""

from __future__ import annotations

import csv
import math
import os
import queue
import random
import threading
import time
from collections import deque
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional

from scheduler import (PRIORITY_DENSITY, PRIORITY_EMERGENCY, PRIORITY_LOGGING,
                       PRIORITY_SIGNAL, PriorityScheduler)
from traffic_controller import (GREEN, GROUP_OF, JUNCTION_NAMES, JUNCTIONS,
                                TrafficController)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "sample_traffic.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")

ROAD_CAPACITY = 40          # vehicles a road can hold in the queue
DISCHARGE_PER_TICK = 1      # vehicles that leave a GREEN road each second
DENSITY_LOW = 30.0          # density % below this  -> Low
DENSITY_MEDIUM = 60.0       # density % below this  -> Medium, else High

# How often (in ticks) each periodic job is released to the scheduler
DENSITY_PERIOD = 4
LOGGER_PERIOD = 3
DASHBOARD_PERIOD = 4
DEFAULT_QUANTA = 5          # CPU time slices available per tick
AGING_INTERVAL = 2          # ticks a job must wait for one aging step

DEFAULT_SCENARIOS = {
    "Normal": {"J1": 0.25, "J2": 0.22, "J3": 0.25, "J4": 0.20},
    "High traffic": {"J1": 0.90, "J2": 0.85, "J3": 0.90, "J4": 0.80},
}


def load_scenarios(path: str = DATA_FILE) -> Dict[str, Dict[str, float]]:
    """Read the sample simulated traffic data (CSV); fall back to defaults."""
    scenarios: Dict[str, Dict[str, float]] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scenarios.setdefault(row["scenario"], {})[row["junction"]] = \
                    float(row["arrival_rate_per_second"])
    except (OSError, KeyError, ValueError):
        pass
    return scenarios or DEFAULT_SCENARIOS


def density_level(density_pct: float) -> str:
    if density_pct < DENSITY_LOW:
        return "Low"
    if density_pct < DENSITY_MEDIUM:
        return "Medium"
    return "High"


def poisson(rng: random.Random, lam: float) -> int:
    """Number of arrivals in one second (Knuth's algorithm, no numpy needed)."""
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


class VehicleGenerator:
    """PRODUCER: creates simulated vehicle arrivals and sends them via queues."""

    def __init__(self, rates: Dict[str, float], arrival_queues: Dict[str, queue.Queue], seed: int):
        self.rates = dict(rates)
        self.queues = arrival_queues
        self.rng = random.Random(seed)

    def produce(self, tick: int) -> None:
        for j in JUNCTIONS:
            n = poisson(self.rng, self.rates.get(j, 0.0))
            if n:
                self.queues[j].put({"type": "ARRIVAL", "junction": j, "count": n, "tick": tick})


class TrafficSimulation:
    def __init__(self, scenario: str = "Normal", seed: int = 42, quanta_per_tick: int = 4,
                 speed: float = 1.0, threaded: bool = False, log_to_file: bool = False,
                 log_dir: str = LOG_DIR):
        self.scenarios = load_scenarios()
        self.scenario = scenario if scenario in self.scenarios else next(iter(self.scenarios))
        self.seed = seed
        self.quanta_per_tick = quanta_per_tick
        self.speed = speed
        self.threaded = threaded
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self.sim_lock = threading.RLock()
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self._running = threading.Event()
        self._last_view = time.time()
        self.started_once = False
        self.log_file_path: Optional[str] = None
        self._build()

    # ------------------------------------------------------------- building
    def _build(self) -> None:
        self.tick = 0
        self.vehicles = {j: 0 for j in JUNCTIONS}
        self.total_arrived = {j: 0 for j in JUNCTIONS}
        self.total_passed = {j: 0 for j in JUNCTIONS}
        self.total_dropped = {j: 0 for j in JUNCTIONS}
        # IPC: thread-safe message queues
        self.arrival_queues = {j: queue.Queue() for j in JUNCTIONS}
        self.emergency_queue: queue.Queue = queue.Queue()
        self.log_queue: queue.Queue = queue.Queue()
        self.log_entries: deque = deque(maxlen=1500)
        self.history: deque = deque(maxlen=300)
        self.analysis = {j: {"vehicles": 0, "density": 0.0, "level": "Low", "tick": 0}
                         for j in JUNCTIONS}
        self.controller = TrafficController(log_fn=self._log)
        self.scheduler = PriorityScheduler(quanta_per_tick=self.quanta_per_tick,
                                           aging_interval=AGING_INTERVAL, log_fn=self._log)
        self.generator = VehicleGenerator(self.scenarios[self.scenario], self.arrival_queues, self.seed)
        self.signal_changes_log: List[dict] = []
        self.log_counts: Dict[str, int] = {}
        self._open_log_file()
        self._log("SYSTEM", f"Simulation initialised (scenario '{self.scenario}', seed {self.seed}). "
                            f"NOTE: all traffic data is simulated.")
        self._log("SIGNAL", "NS roads (J1, J3): GREEN, EW roads (J2, J4): RED")
        self.flush_logs()

    def _open_log_file(self) -> None:
        self.log_file_path = None
        if self.log_to_file:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_file_path = os.path.join(self.log_dir, f"traffic_log_{stamp}.txt")
            except OSError:
                self.log_file_path = None

    # ------------------------------------------------------------- logging
    def _log(self, category: str, message: str) -> None:
        """Any module may call this; it only PUTS a message on the log queue (IPC)."""
        self.log_queue.put({"tick": self.tick, "time": datetime.now().strftime("%H:%M:%S"),
                            "category": category, "message": message})

    def flush_logs(self) -> int:
        """LOGGER job body: drain the log queue into memory (+ file)."""
        batch = []
        while True:
            try:
                batch.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        for entry in batch:
            self.log_entries.append(entry)
            self.log_counts[entry["category"]] = self.log_counts.get(entry["category"], 0) + 1
        if batch and self.log_file_path:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    for e in batch:
                        f.write(f"[tick {e['tick']:>4}] {e['time']} {e['category']:<9} {e['message']}\n")
            except OSError:
                pass
        return len(batch)

    # ------------------------------------------------------------ job bodies
    def _job_emergency_detect(self) -> None:
        while True:
            try:
                ev = self.emergency_queue.get_nowait()
            except queue.Empty:
                return
            self.controller.register_emergency(ev["junction"], ev["kind"], self.tick)

    def _job_signal_control(self) -> None:
        queues = {j: self.analysis[j]["vehicles"] for j in JUNCTIONS}
        self.controller.step(self.tick, queues)

    def _job_density(self, j: str) -> None:
        v = self.vehicles[j]
        d = round(100.0 * v / ROAD_CAPACITY, 1)
        level = density_level(d)
        if level != self.analysis[j]["level"]:
            self._log("DENSITY", f"{JUNCTION_NAMES[j]} ({j}) density {self.analysis[j]['level']} "
                                 f"-> {level} ({v} vehicles, {d}%)")
        self.analysis[j] = {"vehicles": v, "density": d, "level": level, "tick": self.tick}

    def _job_dashboard(self) -> None:
        row = {"tick": self.tick}
        for j in JUNCTIONS:
            row[f"{j}_vehicles"] = self.vehicles[j]
            row[f"{j}_density"] = round(100.0 * self.vehicles[j] / ROAD_CAPACITY, 1)
        self.history.append(row)

    # ----------------------------------------------------------- one tick
    def step(self) -> None:
        """Advance the simulation by one second."""
        with self.sim_lock:
            self.tick += 1
            if not self.threaded:
                self.generator.produce(self.tick)       # single-thread mode (tests)

            # 1) INTAKE: consume arrival events sent by the producer (IPC)
            for j in JUNCTIONS:
                arrived = 0
                while True:
                    try:
                        ev = self.arrival_queues[j].get_nowait()
                    except queue.Empty:
                        break
                    arrived += ev["count"]
                if arrived:
                    room = ROAD_CAPACITY - self.vehicles[j]
                    accepted = min(arrived, max(room, 0))
                    self.vehicles[j] += accepted
                    self.total_arrived[j] += accepted
                    self.total_dropped[j] += arrived - accepted
                    self._log("ARRIVAL", f"{arrived} vehicle(s) arrived at {JUNCTION_NAMES[j]} "
                                         f"({j}); queue = {self.vehicles[j]}")

            # 2) DISCHARGE: vehicles leave roads that currently show GREEN
            for j in JUNCTIONS:
                if self.controller.signals[j] == GREEN and self.vehicles[j] > 0:
                    out = min(self.vehicles[j], DISCHARGE_PER_TICK)
                    self.vehicles[j] -= out
                    self.total_passed[j] += out

            # 3) RELEASE periodic jobs and let the scheduler run the CPU
            s, t = self.scheduler, self.tick
            s.submit("EMERGENCY_DETECT", PRIORITY_EMERGENCY, 1, self._job_emergency_detect, t)
            s.submit("SIGNAL_CONTROL", PRIORITY_SIGNAL, 1, self._job_signal_control, t)
            if t % DENSITY_PERIOD == 0:
                for j in JUNCTIONS:
                    s.submit(f"DENSITY_{j}", PRIORITY_DENSITY, 2, partial(self._job_density, j), t)
            if t % LOGGER_PERIOD == 0:
                s.submit("LOGGER", PRIORITY_LOGGING, 1, self.flush_logs, t)
            if t % DASHBOARD_PERIOD == 0:
                s.submit("DASHBOARD_UPDATE", PRIORITY_LOGGING, 1, self._job_dashboard, t)
            s.run_tick(t)

    # ---------------------------------------------------------- user actions
    def trigger_emergency(self, junction: str, kind: str = "Ambulance") -> None:
        """Called from the UI thread: sends an event to the emergency queue (IPC)."""
        self.emergency_queue.put({"junction": junction, "kind": kind})
        self._log("EMERGENCY", f"{kind} request sent to emergency queue for {junction}")

    def set_scenario(self, name: str) -> None:
        if name in self.scenarios and name != self.scenario:
            with self.sim_lock:
                self.scenario = name
                self.generator.rates = dict(self.scenarios[name])
                self._log("SYSTEM", f"Traffic scenario changed to '{name}'")

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.25, float(speed))

    # ------------------------------------------------------- thread control
    @property
    def interval(self) -> float:
        return 1.0 / self.speed

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def status(self) -> str:
        if self.is_running:
            return "RUNNING"
        return "PAUSED" if self.started_once and self.tick > 0 else "STOPPED"

    def touch(self) -> None:
        """UI calls this on every refresh: proves someone is watching."""
        self._last_view = time.time()

    def start(self) -> None:
        self.touch()
        self.started_once = True
        if self.threaded and not any(t.is_alive() for t in self._threads):
            self._stop.clear()
            self._threads = [
                threading.Thread(target=self._controller_loop, name="controller-thread", daemon=True),
                threading.Thread(target=self._producer_loop, name="vehicle-producer-thread", daemon=True),
            ]
            for t in self._threads:
                t.start()
        if self.threaded:
            if self.tick == 0:
                self._log("SYSTEM", "Simulation started")
            else:
                self._log("SYSTEM", "Simulation resumed")
        self._running.set()

    def pause(self) -> None:
        if self.is_running:
            self._running.clear()
            self._log("SYSTEM", "Simulation paused")

    def shutdown(self) -> None:
        self._running.clear()
        self._stop.set()
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=3)
        self._threads = []

    def reset(self) -> None:
        """Stop everything and rebuild all state from scratch."""
        self.shutdown()
        with self.sim_lock:
            self.started_once = False
            self._build()
            self._log("SYSTEM", "Simulation reset to initial state")
            self.flush_logs()

    def _controller_loop(self) -> None:
        while not self._stop.is_set():
            if self._running.wait(timeout=0.1):
                if time.time() - self._last_view > 120:        # nobody watching
                    self._running.clear()
                    self._log("SYSTEM", "Auto-paused: dashboard not viewed for 2 minutes")
                    continue
                self.step()
                self._stop.wait(self.interval)

    def _producer_loop(self) -> None:
        while not self._stop.is_set():
            if self._running.wait(timeout=0.1):
                self.generator.produce(self.tick)
                self._stop.wait(self.interval)

    # ------------------------------------------------------------- reading
    def snapshot(self, log_lines: int = 200) -> dict:
        """Consistent copy of everything the UI needs."""
        self.touch()
        with self.sim_lock:
            ctrl = self.controller.snapshot()
            junctions = {}
            levels = []
            for j in JUNCTIONS:
                v = self.vehicles[j]
                d = round(100.0 * v / ROAD_CAPACITY, 1)
                levels.append(d)
                junctions[j] = {
                    "name": JUNCTION_NAMES[j], "group": GROUP_OF[j], "vehicles": v,
                    "density": d, "level": density_level(d),
                    "signal": ctrl["signals"][j], "countdown": ctrl["countdown"][j],
                    "arrived": self.total_arrived[j], "passed": self.total_passed[j],
                    "dropped": self.total_dropped[j],
                }
            avg = sum(levels) / len(levels)
            return {
                "tick": self.tick, "status": self.status, "scenario": self.scenario,
                "speed": self.speed, "junctions": junctions,
                "overall": {"level": density_level(avg), "avg_density": round(avg, 1),
                            "total_vehicles": sum(self.vehicles.values())},
                "controller": ctrl,
                "history": list(self.history),
                "logs": list(self.log_entries)[-log_lines:],
                "scheduler_table": self.scheduler.table(),
                "task_stats": self.scheduler.task_stats(),
                "scheduler_summary": self.scheduler.summary(),
                "trace": list(self.scheduler.trace)[-24:],
                "queue_sizes": {"arrival": sum(q.qsize() for q in self.arrival_queues.values()),
                                "emergency": self.emergency_queue.qsize(),
                                "log": self.log_queue.qsize()},
                "log_file": self.log_file_path,
            }
