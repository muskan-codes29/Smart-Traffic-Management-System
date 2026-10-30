"""
test_suite.py  -  Tests that run the REAL simulation and report MEASURED results
================================================================================
Nothing here is hard-coded: every number in the report comes from stepping
`TrafficSimulation` (single-thread mode, fixed random seed) or from running the
real thread demos. Run from the command line with:

    python test_suite.py

or press "Run all tests" on the Testing page of the dashboard.
A JSON copy of the last run is saved to logs/test_report.json.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List

from scheduler import aging_demo, preemption_demo, round_robin_demo
from simulation import LOG_DIR, TrafficSimulation
from traffic_controller import (GREEN, GROUP_OF, JUNCTIONS, RED, YELLOW, deadlock_demo,
                                lost_update_demo)

SEED = 7


def _new(scenario: str) -> TrafficSimulation:
    return TrafficSimulation(scenario=scenario, seed=SEED, threaded=False, log_to_file=False)


def _result(name: str, goal: str, passed: bool, checks: List[dict], measured: dict) -> dict:
    return {"name": name, "goal": goal, "passed": passed, "checks": checks, "measured": measured}


def _check(label: str, ok: bool, detail: str) -> dict:
    return {"check": label, "result": "PASS" if ok else "FAIL", "detail": detail}


# --------------------------------------------------------------------- tests
def test_normal_operation() -> dict:
    sim = _new("Normal")
    seen_colors = set()
    for _ in range(120):
        sim.step()
        seen_colors.update(sim.controller.signals.values())
    sim.flush_logs()
    c = sim.controller
    arrived, passed = sum(sim.total_arrived.values()), sum(sim.total_passed.values())
    max_q = max(sim.vehicles.values())
    checks = [
        _check("Signals cycle through all three colours", seen_colors == {RED, YELLOW, GREEN},
               f"colours seen: {sorted(seen_colors)}"),
        _check("Phases alternate (at least 4 green phases in 120 s)", c.phase_changes >= 4,
               f"{c.phase_changes} phase changes"),
        _check("Queues stay under road capacity", max_q < 40, f"largest queue at the end = {max_q}"),
        _check("Most arriving vehicles get through", passed >= 0.8 * arrived,
               f"{passed} of {arrived} vehicles passed"),
    ]
    return _result("1. Normal traffic operation", "Low arrival rates; signals alternate and queues stay short.",
                   all(x["result"] == "PASS" for x in checks), checks,
                   {"ticks": 120, "vehicles_arrived": arrived, "vehicles_passed": passed,
                    "phase_changes": c.phase_changes,
                    "avg_green_seconds": round(sum(g["duration"] for g in c.green_history) / len(c.green_history), 1)})


def test_high_density() -> dict:
    normal, high = _new("Normal"), _new("High traffic")
    for _ in range(150):
        normal.step(); high.step()
    avg = lambda s: sum(g["duration"] for g in s.controller.green_history) / len(s.controller.green_history)
    snap = high.snapshot()
    checks = [
        _check("Green time is longer under heavy traffic", avg(high) > avg(normal),
               f"avg green: normal {avg(normal):.1f}s vs high {avg(high):.1f}s"),
        _check("Green time never exceeds the safe maximum (25 s)",
               max(g["duration"] for g in high.controller.green_history) <= 25,
               f"longest green = {max(g['duration'] for g in high.controller.green_history)} s"),
        _check("Overall status reaches High", snap["overall"]["level"] == "High",
               f"overall = {snap['overall']['level']} (avg density {snap['overall']['avg_density']}%)"),
        _check("No unsafe signal combination occurred", high.controller.conflict_violations == 0,
               f"violations = {high.controller.conflict_violations}"),
    ]
    return _result("2. High traffic density", "Heavy arrivals; green time adapts to the queue length.",
                   all(x["result"] == "PASS" for x in checks), checks,
                   {"ticks": 150, "avg_green_normal_s": round(avg(normal), 1),
                    "avg_green_high_s": round(avg(high), 1),
                    "queues_at_end": {j: v["vehicles"] for j, v in snap["junctions"].items()},
                    "vehicles_dropped_full_roads": sum(high.total_dropped.values())})


def test_emergency() -> dict:
    sim = _new("Normal")
    for _ in range(14):
        sim.step()
    c = sim.controller
    # choose a road whose group is currently RED so the controller must pre-empt
    target = next(j for j in JUNCTIONS if c.signals[j] == RED)
    before_phase = c.phase
    sim.trigger_emergency(target, "Ambulance")
    old_road = next(j for j in JUNCTIONS if GROUP_OF[j] == before_phase)   # was GREEN
    t0, green_at = sim.tick, None
    sequence = [c.signals[old_road]]                  # colour just before detection
    for _ in range(25):
        sim.step()
        sequence.append(c.signals[old_road])
        if c.signals[target] == GREEN and green_at is None:
            green_at = sim.tick - t0
    for _ in range(20):                               # let the emergency clear
        sim.step()
    sim.flush_logs()
    comp = [sequence[0]]                              # compress repeated colours
    for s_ in sequence[1:]:
        if s_ != comp[-1]:
            comp.append(s_)
    emerg_logs = [e["message"] for e in sim.log_entries if e["category"] == "EMERGENCY"]
    checks = [
        _check("Emergency road received GREEN", green_at is not None,
               f"{target} turned GREEN {green_at} s after detection"),
        _check("Pre-empted road went GREEN -> YELLOW -> RED (never skipped YELLOW)",
               comp[:3] == [GREEN, YELLOW, RED],
               f"colour sequence of {old_road}: {' > '.join(comp)}"),
        _check("Green arrived within YELLOW(3 s) + ALL-RED(1 s) + 2 s",
               green_at is not None and green_at <= 6,
               f"{green_at} s after detection"),
        _check("Emergency was cleared afterwards", c.emergencies_served == 1,
               f"emergencies served = {c.emergencies_served}"),
        _check("No unsafe signal combination", c.conflict_violations == 0,
               f"violations = {c.conflict_violations}"),
    ]
    return _result("3. Emergency vehicle arrival", "Ambulance appears on a RED road; controller pre-empts safely.",
                   all(x["result"] == "PASS" for x in checks), checks,
                   {"emergency_road": target, "seconds_until_green": green_at,
                    "old_green_road_sequence": comp, "emergency_log": emerg_logs})


def test_conflict_prevention() -> dict:
    sim = _new("High traffic")
    violations_each_tick = 0
    for _ in range(300):
        sim.step()
        violations_each_tick += len(sim.controller._violations_now())
    c = sim.controller
    # Direct conflict attempt: ask for green on a road conflicting with the active one
    sim.step()
    while GREEN not in c.signals.values():
        sim.step()
    green_now = next(j for j in JUNCTIONS if c.signals[j] == GREEN)
    conflicting = next(j for j in JUNCTIONS if GROUP_OF[j] != GROUP_OF[green_now])
    same_group = next(j for j in JUNCTIONS if GROUP_OF[j] == GROUP_OF[green_now] and j != green_now)
    ok_conf, msg_conf = c.request_green(conflicting)
    ok_same, msg_same = c.request_green(same_group)
    checks = [
        _check("0 conflicting greens in 300 ticks (checked every tick)", violations_each_tick == 0 and c.conflict_violations == 0,
               f"violations counted = {violations_each_tick + c.conflict_violations}"),
        _check("Request for a conflicting green is DENIED", not ok_conf, msg_conf),
        _check("Request for a compatible road is allowed", ok_same, msg_same),
    ]
    return _result("4. Signal conflict prevention", "Directions NS and EW must never be active together.",
                   all(x["result"] == "PASS" for x in checks), checks,
                   {"ticks_checked": 300, "denials_recorded": c.conflict_denials,
                    "active_green_road": green_now, "conflict_attempt": conflicting})


def test_scheduling_and_sync() -> dict:
    sim = _new("High traffic")
    sim.step()
    # emergency job must be the FIRST thing dispatched every tick
    first_each_tick: Dict[int, str] = {}
    for _ in range(100):
        sim.step()
    for e in sim.scheduler.trace:
        first_each_tick.setdefault(e["tick"], e["job"])
    emergency_first = all(v == "EMERGENCY_DETECT" for v in first_each_tick.values())
    s = sim.scheduler.summary()
    stats = {r["Task"]: r for r in sim.scheduler.task_stats()}
    all_ran = all(r["Completed runs"] > 0 for r in stats.values())
    worst_wait = max(r["Max wait (ticks)"] for r in stats.values())
    pre, rr, ag = preemption_demo(), round_robin_demo(), (aging_demo(False), aging_demo(True))
    race = lost_update_demo()
    rr_order = [name for _, name in rr["trace"]]
    checks = [
        _check("Emergency job is dispatched first in every traced tick", emergency_first,
               f"{len(first_each_tick)} traced ticks, all led by EMERGENCY_DETECT" if emergency_first else "order broken"),
        _check("Pre-emption occurred in the live scheduler", s["preemptions"] > 0,
               f"{s['preemptions']} pre-emptions, {s['context_switches']} context switches"),
        _check("Pre-emption demo: emergency finished before the long low-priority job",
               pre["finish_order"][0] == "EMERGENCY", f"finish order = {pre['finish_order']}"),
        _check("Aging: starvation without aging, completion with aging",
               ag[0]["starved"] and not ag[1]["starved"],
               f"no aging -> low job finished at tick {ag[0]['low_job_finished_at_tick']}; "
               f"aging -> finished at tick {ag[1]['low_job_finished_at_tick']}"),
        _check("Round Robin: equal-priority jobs alternate",
               rr_order[:6] == ["DENSITY_J1", "DENSITY_J2", "DENSITY_J3"] * 2, f"order = {rr_order[:6]}"),
        _check("No task starved in the live run", all_ran, f"longest wait = {worst_wait} ticks"),
        _check("Lock protects shared counter (no lost updates)", race["with_lock"] == race["expected"],
               f"expected {race['expected']}: with lock = {race['with_lock']}, "
               f"without lock = {race['without_lock']}"),
    ]
    return _result("5. Scheduling and synchronization", "Priorities, pre-emption, aging, Round Robin, locks.",
                   all(x["result"] == "PASS" for x in checks), checks,
                   {"scheduler_summary": s, "aging_without": ag[0], "aging_with": ag[1],
                    "round_robin_order": rr_order, "race_demo": race,
                    "preemption_trace": pre["trace"]})


def test_deadlock_avoidance() -> dict:
    d = deadlock_demo()
    sim = _new("High traffic")
    for _ in range(100):
        sim.step()
    c = sim.controller
    checks = [
        _check("Unordered locking deadlocks (demonstrated)", d["unsafe_unordered"]["deadlock"],
               "; ".join(f"{k}: {v}" for k, v in d["unsafe_unordered"]["results"].items())),
        _check("Ordered locking completes with no deadlock", not d["safe_ordered"]["deadlock"],
               "; ".join(f"{k}: {v}" for k, v in d["safe_ordered"]["results"].items())),
        _check("Controller ran 100 ticks with zero lock time-outs", c.lock_timeouts == 0,
               f"{c.lock_acquisitions} ordered lock acquisitions, {c.lock_timeouts} time-outs"),
    ]
    return _result("6. Deadlock avoidance", "Ordered, time-limited lock acquisition.",
                   all(x["result"] == "PASS" for x in checks), checks,
                   {"unsafe_seconds": d["unsafe_unordered"]["seconds"],
                    "safe_seconds": d["safe_ordered"]["seconds"],
                    "lock_acquisitions": c.lock_acquisitions})


def test_reset() -> dict:
    sim = _new("High traffic")
    for _ in range(40):
        sim.step()
    sim.trigger_emergency("J2", "Fire truck")
    for _ in range(15):
        sim.step()
    before = {"tick": sim.tick, "vehicles": sum(sim.vehicles.values()),
              "phase_changes": sim.controller.phase_changes,
              "log_entries": len(sim.log_entries)}
    sim.reset()
    c = sim.controller
    after = {"tick": sim.tick, "vehicles": sum(sim.vehicles.values()),
             "phase_changes": c.phase_changes, "log_entries": len(sim.log_entries),
             "signals": dict(c.signals), "emergency": c.emergency,
             "queued_events": sum(q.qsize() for q in sim.arrival_queues.values()) + sim.emergency_queue.qsize(),
             "ready_jobs": len(sim.scheduler.ready)}
    checks = [
        _check("Clock back to 0", after["tick"] == 0, f"tick {before['tick']} -> {after['tick']}"),
        _check("All vehicle counts cleared", after["vehicles"] == 0, f"{before['vehicles']} -> {after['vehicles']}"),
        _check("Signal controller restarted", after["phase_changes"] == 0 and c.phase == "NS" and c.stage == GREEN,
               f"phase_changes {before['phase_changes']} -> {after['phase_changes']}, phase={c.phase}"),
        _check("Emergency and queues cleared", after["emergency"] is None and after["queued_events"] == 0
               and after["ready_jobs"] == 0, f"events queued = {after['queued_events']}, ready jobs = {after['ready_jobs']}"),
        _check("Log restarted with a fresh header", after["log_entries"] <= 3 and after["log_entries"] < before["log_entries"],
               f"log lines {before['log_entries']} -> {after['log_entries']}"),
        _check("Simulation stopped (needs Start)", sim.status == "STOPPED", f"status = {sim.status}"),
    ]
    return _result("7. Reset functionality", "Reset returns everything to the initial state.",
                   all(x["result"] == "PASS" for x in checks), checks,
                   {"before": before, "after": {k: v for k, v in after.items() if k != "signals"}})


ALL_TESTS: List[Callable[[], dict]] = [test_normal_operation, test_high_density, test_emergency,
                                       test_conflict_prevention, test_scheduling_and_sync,
                                       test_deadlock_avoidance, test_reset]


def run_all(save: bool = True) -> List[dict]:
    results = []
    for fn in ALL_TESTS:
        started = time.perf_counter()
        try:
            r = fn()
        except Exception as exc:  # a crashing test is reported as FAIL, never hidden
            r = _result(fn.__name__, "test crashed", False,
                        [_check("Test ran without error", False, f"{type(exc).__name__}: {exc}")], {})
        r["seconds"] = round(time.perf_counter() - started, 2)
        results.append(r)
    if save:
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, "test_report.json"), "w", encoding="utf-8") as f:
                json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
                           "results": results}, f, indent=2, default=str)
        except OSError:
            pass
    return results


if __name__ == "__main__":
    out = run_all()
    for r in out:
        print(f"\n[{'PASS' if r['passed'] else 'FAIL'}] {r['name']}  ({r['seconds']} s)")
        for c in r["checks"]:
            print(f"    {c['result']}  {c['check']}\n          -> {c['detail']}")
    n_ok = sum(r["passed"] for r in out)
    print(f"\n{n_ok}/{len(out)} tests passed")
    raise SystemExit(0 if n_ok == len(out) else 1)
