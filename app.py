"""
app.py  -  Streamlit dashboard for the Smart Traffic Management System (SIMULATION)
==================================================================================
Run with:   streamlit run app.py

This file only DRAWS things. All logic lives in:
    scheduler.py           (priority scheduling, aging, round robin)
    traffic_controller.py  (signals, locks, deadlock prevention, emergency rules)
    simulation.py          (threads, queues, vehicle generation)
    test_suite.py          (tests that use the real simulation)

IMPORTANT: every number on screen comes from a simulation. There is no
connection to real cameras, sensors or road signals.
"""

from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

import test_suite
from scheduler import PRIORITY_LABELS, aging_demo, preemption_demo, round_robin_demo
from simulation import DENSITY_LOW, DENSITY_MEDIUM, ROAD_CAPACITY, TrafficSimulation
from traffic_controller import (GREEN, JUNCTION_NAMES, JUNCTIONS, RED, YELLOW, deadlock_demo,
                                lost_update_demo)

# ----------------------------------------------------------------- appearance
NAVY, CARD, CARD2 = "#0b1630", "#13234a", "#1b2f5e"
C_GREEN, C_YELLOW, C_RED = "#22c55e", "#facc15", "#ef4444"
SIGNAL_COLOR = {GREEN: C_GREEN, YELLOW: C_YELLOW, RED: C_RED}
LEVEL_COLOR = {"Low": C_GREEN, "Medium": C_YELLOW, "High": C_RED}
LINE_COLORS = {"J1": "#38bdf8", "J2": "#f472b6", "J3": "#a78bfa", "J4": "#fb923c"}
CAT_COLOR = {"SIGNAL": "#38bdf8", "ARRIVAL": "#94a3b8", "EMERGENCY": C_RED, "SCHEDULER": "#a78bfa",
             "SAFETY": C_YELLOW, "DENSITY": "#fb923c", "SYSTEM": C_GREEN}

CSS = f"""
<style>
.block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
.hdr {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-bottom:8px; }}
.hdr h1 {{ font-size:1.6rem; margin:0 12px 0 0; }}
.chip {{ background:{CARD}; border:1px solid {CARD2}; border-radius:999px; padding:4px 12px;
        font-size:.82rem; white-space:nowrap; }}
.sim-note {{ background:{CARD}; border-left:4px solid {C_YELLOW}; padding:6px 12px; border-radius:6px;
            font-size:.82rem; margin-bottom:10px; color:#cbd5e1; }}
.card {{ background:{CARD}; border:1px solid {CARD2}; border-radius:14px; padding:12px 14px; }}
.card-title {{ font-weight:700; font-size:1rem; margin-bottom:6px; }}
.card-row {{ display:flex; gap:14px; align-items:center; }}
.lamp-box {{ background:#050b1a; border-radius:10px; padding:6px; display:flex; flex-direction:column;
            gap:5px; border:1px solid #223; }}
.lamp {{ width:22px; height:22px; border-radius:50%; opacity:.18; }}
.lamp.red {{ background:{C_RED}; }} .lamp.yellow {{ background:{C_YELLOW}; }} .lamp.green {{ background:{C_GREEN}; }}
.lamp.on {{ opacity:1; box-shadow:0 0 12px currentColor; }}
.lamp.red.on {{ box-shadow:0 0 14px {C_RED}; }} .lamp.yellow.on {{ box-shadow:0 0 14px {C_YELLOW}; }}
.lamp.green.on {{ box-shadow:0 0 14px {C_GREEN}; }}
.sig {{ font-weight:800; font-size:1.25rem; letter-spacing:.5px; }}
.big {{ font-size:1.9rem; font-weight:800; line-height:1.1; }}
.sub {{ font-size:.75rem; color:#94a3b8; }}
.bar-bg {{ background:#0a1330; border-radius:6px; height:9px; margin-top:6px; overflow:hidden; }}
.bar-fg {{ height:9px; border-radius:6px; }}
.lvl {{ border-radius:6px; padding:1px 8px; font-size:.75rem; font-weight:700; color:#06101f; }}
.alert {{ background:linear-gradient(90deg,#7f1d1d,#b91c1c); border:1px solid {C_RED}; border-radius:12px;
         padding:10px 16px; margin:6px 0 12px 0; font-weight:600; animation:pulse 1.2s infinite; }}
@keyframes pulse {{ 0%{{box-shadow:0 0 0 0 rgba(239,68,68,.6);}} 70%{{box-shadow:0 0 0 12px rgba(239,68,68,0);}}
                   100%{{box-shadow:0 0 0 0 rgba(239,68,68,0);}} }}
.logbox {{ background:#07102a; border:1px solid {CARD2}; border-radius:10px; padding:8px 10px;
          font-family:Consolas,monospace; font-size:.78rem; max-height:330px; overflow-y:auto; }}
.logline {{ padding:1px 0; border-bottom:1px dashed #12204a; }}
.tag {{ font-weight:700; }}
.note {{ background:{CARD}; border-radius:10px; padding:10px 14px; border:1px solid {CARD2}; }}
</style>
"""


# -------------------------------------------------------------------- helpers
def show_df(df: pd.DataFrame, height: int | None = None) -> None:
    """st.dataframe that works on old and new Streamlit versions."""
    kwargs = {"hide_index": True}
    if height:
        kwargs["height"] = height
    try:
        st.dataframe(df, use_container_width=True, **kwargs)
    except Exception:
        st.dataframe(df, **kwargs)


def show_fig(fig: Figure) -> None:
    try:
        st.pyplot(fig, use_container_width=True)
    except Exception:
        st.pyplot(fig)


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def run_live(render, *args) -> None:
    """Run `render` as a Streamlit fragment that refreshes itself every second
    while the simulation is running (so only this block re-draws)."""
    engine = args[0]
    every = 1.0 if engine.is_running else None
    frag = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    if frag is None:
        render(*args)
    else:
        frag(run_every=every)(render)(*args)


def lamp_html(signal: str) -> str:
    return "<div class='lamp-box'>" + "".join(
        f"<div class='lamp {c.lower()}{' on' if signal == c else ''}'></div>"
        for c in (RED, YELLOW, GREEN)) + "</div>"


def level_chip(level: str) -> str:
    return f"<span class='lvl' style='background:{LEVEL_COLOR[level]}'>{level}</span>"


# ---------------------------------------------------------------- dashboard
def junction_card(j: str, d: dict) -> str:
    sig = d["signal"]
    if sig == RED:
        timer = f"<div class='big'>{d['countdown']}s</div><div class='sub'>until next green (est.)</div>"
    else:
        timer = f"<div class='big'>{d['countdown']}s</div><div class='sub'>remaining</div>"
    width = min(100, d["density"])
    return (
        f"<div class='card'><div class='card-title'>{j} &middot; {d['name']} "
        f"<span class='sub'>({d['group']} phase)</span></div>"
        f"<div class='card-row'>{lamp_html(sig)}<div>"
        f"<div class='sig' style='color:{SIGNAL_COLOR[sig]}'>{sig}</div>{timer}</div></div>"
        f"<div style='margin-top:10px;display:flex;justify-content:space-between'>"
        f"<span>Vehicles waiting: <b>{d['vehicles']}</b></span>{level_chip(d['level'])}</div>"
        f"<div class='bar-bg'><div class='bar-fg' style='width:{width}%;background:{LEVEL_COLOR[d['level']]}'></div></div>"
        f"<div class='sub'>Density {d['density']}% of road capacity ({ROAD_CAPACITY}) &middot; "
        f"passed {d['passed']} &middot; arrived {d['arrived']}</div></div>"
    )


def intersection_svg(snap: dict) -> str:
    """Small top-view picture of the crossing with live lights and queues."""
    emer = snap["controller"]["emergency"]
    pos = {  # lamp position, queue start, queue direction
        "J1": ((180, 128), (172, 118), (0, -13)),
        "J3": ((180, 232), (188, 242), (0, 13)),
        "J2": ((232, 180), (242, 172), (13, 0)),
        "J4": ((128, 180), (118, 188), (-13, 0)),
    }
    out = [f"<svg viewBox='0 0 360 360' width='100%' style='max-width:340px;background:{CARD};"
           f"border-radius:14px;border:1px solid {CARD2}'>",
           "<rect x='150' y='0' width='60' height='360' fill='#2b3a5c'/>",
           "<rect x='0' y='150' width='360' height='60' fill='#2b3a5c'/>",
           "<rect x='150' y='150' width='60' height='60' fill='#34466d'/>",
           "<line x1='180' y1='0' x2='180' y2='148' stroke='#64748b' stroke-dasharray='8 8'/>",
           "<line x1='180' y1='212' x2='180' y2='360' stroke='#64748b' stroke-dasharray='8 8'/>",
           "<line x1='0' y1='180' x2='148' y2='180' stroke='#64748b' stroke-dasharray='8 8'/>",
           "<line x1='212' y1='180' x2='360' y2='180' stroke='#64748b' stroke-dasharray='8 8'/>"]
    for j in JUNCTIONS:
        (lx, ly), (qx, qy), (dx, dy) = pos[j]
        d = snap["junctions"][j]
        color = SIGNAL_COLOR[d["signal"]]
        ring = ""
        if emer and emer["junction"] == j:
            ring = f"<circle cx='{lx}' cy='{ly}' r='19' fill='none' stroke='{C_RED}' stroke-width='3' stroke-dasharray='4 3'/>"
        out.append(ring + f"<circle cx='{lx}' cy='{ly}' r='11' fill='{color}' stroke='#050b1a' stroke-width='2'/>")
        for i in range(min(d["vehicles"], 8)):
            x, y = qx + dx * i, qy + dy * i
            w, h = (10, 8) if dx else (8, 10)
            out.append(f"<rect x='{x - w / 2}' y='{y - h / 2}' width='{w}' height='{h}' rx='2' fill='#e2e8f0'/>")
        label_pos = {"J1": (180, 18), "J3": (180, 352), "J2": (300, 140), "J4": (60, 140)}[j]
        out.append(f"<text x='{label_pos[0]}' y='{label_pos[1]}' fill='#cbd5e1' font-size='12' "
                   f"text-anchor='middle'>{j} {d['name'].split()[0]}: {d['vehicles']}</text>")
    out.append("</svg>")
    return "".join(out)


def style_axes(ax) -> None:
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color(CARD2)
    ax.tick_params(colors="#cbd5e1", labelsize=8)
    ax.yaxis.label.set_color("#cbd5e1")
    ax.title.set_color("#e6edf7")


def bar_chart(snap: dict) -> Figure:
    fig = Figure(figsize=(5, 3.3), dpi=100, facecolor=CARD)
    ax = fig.add_subplot(111)
    style_axes(ax)
    labels = [f"{j}\n{snap['junctions'][j]['name'].split()[0]}" for j in JUNCTIONS]
    vals = [snap["junctions"][j]["vehicles"] for j in JUNCTIONS]
    cols = [SIGNAL_COLOR[snap["junctions"][j]["signal"]] for j in JUNCTIONS]
    bars = ax.bar(labels, vals, color=cols, edgecolor="#050b1a")
    ax.axhline(ROAD_CAPACITY, color="#94a3b8", linestyle="--", linewidth=1)
    ax.text(len(JUNCTIONS) - 0.5, ROAD_CAPACITY + 0.5, "road capacity", color="#94a3b8", fontsize=7, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, str(v), ha="center", color="#e6edf7", fontsize=9)
    ax.set_ylim(0, ROAD_CAPACITY + 6)
    ax.set_ylabel("Vehicles waiting")
    ax.set_title("Vehicle count per junction (bar colour = signal)", fontsize=10)
    fig.tight_layout()
    return fig


def density_chart(snap: dict) -> Figure:
    fig = Figure(figsize=(5, 3.3), dpi=100, facecolor=CARD)
    ax = fig.add_subplot(111)
    style_axes(ax)
    ax.axhspan(0, DENSITY_LOW, color=C_GREEN, alpha=0.10)
    ax.axhspan(DENSITY_LOW, DENSITY_MEDIUM, color=C_YELLOW, alpha=0.10)
    ax.axhspan(DENSITY_MEDIUM, 100, color=C_RED, alpha=0.10)
    hist = snap["history"]
    if len(hist) >= 2:
        x = [r["tick"] for r in hist]
        for j in JUNCTIONS:
            ax.plot(x, [r[f"{j}_density"] for r in hist], color=LINE_COLORS[j], linewidth=1.8, label=j)
        ax.legend(loc="upper left", fontsize=7, ncol=4, facecolor=CARD, edgecolor=CARD2, labelcolor="#e6edf7")
        ax.set_xlabel("simulation time (s)", color="#cbd5e1", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Press Start to collect data", transform=ax.transAxes, ha="center", color="#94a3b8")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Density (%)")
    ax.set_title("Traffic density over time (green / yellow / red zones)", fontsize=10)
    fig.tight_layout()
    return fig


def log_html(entries: list, limit: int) -> str:
    lines = []
    for e in reversed(entries[-limit:]):
        col = CAT_COLOR.get(e["category"], "#e2e8f0")
        lines.append(f"<div class='logline'><span class='sub'>t={e['tick']:>4} {e['time']}</span> "
                     f"<span class='tag' style='color:{col}'>{e['category']:<9}</span> "
                     f"{html.escape(e['message'])}</div>")
    return "<div class='logbox'>" + ("".join(lines) or "<span class='sub'>No events yet.</span>") + "</div>"


def emergency_banner(snap: dict, engine: TrafficSimulation) -> None:
    e = snap["controller"]["emergency"]
    stage_txt = {"WAITING": "detected - preparing to clear the junction",
                 "PREEMPTING": "pre-empting: YELLOW then ALL-RED before green",
                 "SERVING": "GREEN corridor active - vehicle passing"}
    if e:
        more = snap["controller"]["pending_emergencies"]
        extra = f" (+{more} more queued)" if more else ""
        html_block(f"<div class='alert'>&#128680; EMERGENCY ALERT: {html.escape(e['kind'])} approaching on "
                   f"{JUNCTION_NAMES[e['junction']]} ({e['junction']}) &mdash; "
                   f"{stage_txt.get(e['status'], e['status'])}{extra}</div>")
    elif snap["queue_sizes"]["emergency"]:
        html_block("<div class='alert'>&#128680; Emergency request is waiting in the queue &mdash; "
                   "press Start so the EMERGENCY_DETECT task can run.</div>")


def render_dashboard(engine: TrafficSimulation) -> None:
    snap = engine.snapshot()
    ctrl, overall = snap["controller"], snap["overall"]
    status_col = {"RUNNING": C_GREEN, "PAUSED": C_YELLOW, "STOPPED": "#94a3b8"}[snap["status"]]
    html_block(
        f"<div class='hdr'><h1>&#128678; Smart Traffic Management System</h1>"
        f"<span class='chip' style='border-color:{status_col};color:{status_col}'>&#9679; {snap['status']}</span>"
        f"<span class='chip'>Time: {snap['tick']} s</span>"
        f"<span class='chip'>Scenario: {html.escape(snap['scenario'])}</span>"
        f"<span class='chip'>Active phase: {ctrl['phase']} &middot; {ctrl['stage'].replace('_', '-')} &middot; {max(ctrl['timer'], 0)} s</span>"
        f"<span class='chip' style='background:{LEVEL_COLOR[overall['level']]};color:#06101f;font-weight:700'>"
        f"Overall traffic: {overall['level']} ({overall['avg_density']}%)</span></div>"
        f"<div class='sim-note'>SIMULATION ONLY &mdash; all vehicles, signals and emergencies on this page are "
        f"generated by software. Nothing is connected to real cameras, sensors or road signals.</div>")
    emergency_banner(snap, engine)

    cols = st.columns(4)
    for col, j in zip(cols, JUNCTIONS):
        with col:
            html_block(junction_card(j, snap["junctions"][j]))
    st.write("")

    c1, c2, c3 = st.columns([1.0, 1.25, 1.25])
    with c1:
        html_block(intersection_svg(snap))
    with c2:
        show_fig(bar_chart(snap))
    with c3:
        show_fig(density_chart(snap))

    l1, l2 = st.columns([1.15, 1])
    with l1:
        st.markdown("**Live event log** (newest first)")
        html_block(log_html(snap["logs"], 14))
    with l2:
        st.markdown("**Task scheduling table** (lower priority number = more important)")
        rows = snap["scheduler_table"]
        show_df(pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["Task", "Base priority", "Current priority", "Status", "Waiting (ticks)"]), height=300)


# ------------------------------------------------------- OS concepts page
def render_os_live(engine: TrafficSimulation) -> None:
    snap = engine.snapshot()
    s, ctrl, q = snap["scheduler_summary"], snap["controller"], snap["queue_sizes"]
    m = st.columns(6)
    m[0].metric("Pre-emptions", s["preemptions"])
    m[1].metric("Aging promotions", s["aging_promotions"])
    m[2].metric("Round-Robin turns", s["rr_rotations"])
    m[3].metric("Context switches", s["context_switches"])
    m[4].metric("Lock acquisitions", ctrl["lock_acquisitions"])
    m[5].metric("Lock time-outs", ctrl["lock_timeouts"])
    m = st.columns(5)
    m[0].metric("Ready-queue length", s["ready_queue_length"])
    m[1].metric("IPC: arrival msgs queued", q["arrival"])
    m[2].metric("IPC: emergency msgs queued", q["emergency"])
    m[3].metric("IPC: log msgs queued", q["log"])
    m[4].metric("Conflict violations", ctrl["conflict_violations"],
                help="Must always stay 0: conflicting directions were never active together.")

    a, b = st.columns([1.1, 1])
    with a:
        st.markdown("**Ready queue / recent tasks**")
        rows = snap["scheduler_table"]
        show_df(pd.DataFrame(rows) if rows else pd.DataFrame({"info": ["Press Start"]}), height=330)
    with b:
        st.markdown("**CPU dispatch trace** (last 24 time slices, newest last)")
        tr = snap["trace"]
        show_df(pd.DataFrame(tr) if tr else pd.DataFrame({"info": ["Press Start"]}), height=330)
    st.markdown("**Per-task statistics** (waiting time, pre-emptions, aging)")
    ts = snap["task_stats"]
    show_df(pd.DataFrame(ts) if ts else pd.DataFrame({"info": ["Press Start"]}))


def demo_state(key: str, label: str, fn) -> None:
    if st.button(label, key=f"btn_{key}"):
        st.session_state[key] = fn()
    return st.session_state.get(key)


def page_os(engine: TrafficSimulation) -> None:
    st.header("Operating System concepts - live")
    st.caption("The numbers below come from the running simulation. Start it from the sidebar to watch them change.")
    run_live(render_os_live, engine)
    st.divider()
    st.subheader("Explanations and hands-on demos")
    tabs = st.tabs(["Priority + Pre-emption", "Aging", "Round Robin", "Synchronization", "IPC",
                    "Deadlock", "Logging"])

    with tabs[0]:
        st.markdown(
            "**Pre-emptive priority scheduling.** Each system task has a priority "
            "(lower number = more important):\n\n"
            + "\n".join(f"- `{v}`" for v in PRIORITY_LABELS.values()) +
            "\n\nThe simulated CPU runs a few time slices per second. If an Emergency task arrives while a "
            "lower-priority task is half-done, the lower task is **pre-empted**.\n\n"
            "*Code: `scheduler.py` -> `PriorityScheduler._pick()` and `_dispatch()`*")
        r = demo_state("demo_pre", "Run pre-emption demo", preemption_demo)
        if r:
            st.write("A 5-slice LOG_WRITER starts at t=1. EMERGENCY_DETECT (2 slices) arrives at t=3.")
            show_df(pd.DataFrame(r["trace"], columns=["tick", "job on CPU"]))
            st.success(f"Pre-emptions: {r['preemptions']}  |  finish order: {' -> '.join(r['finish_order'])}")

    with tabs[1]:
        st.markdown(
            "**Aging** stops starvation: a task that keeps waiting has its priority number reduced by 1 every "
            "few ticks, down to priority 1 (it can never beat the Emergency task).\n\n"
            "The demo floods a 1-slice CPU with a new priority-1 task every tick while one priority-3 task waits.\n\n"
            "*Code: `scheduler.py` -> `PriorityScheduler._age()`*")
        r = demo_state("demo_aging", "Run aging demo (with vs without)",
                       lambda: {"off": aging_demo(False), "on": aging_demo(True)})
        if r:
            df = pd.DataFrame([
                {"Aging": "OFF", "Low job finished at tick": r["off"]["low_job_finished_at_tick"] or "never (starved)",
                 "Ticks waited": r["off"]["low_job_waiting"], "Final priority": r["off"]["low_job_final_priority"]},
                {"Aging": "ON", "Low job finished at tick": r["on"]["low_job_finished_at_tick"] or "never (starved)",
                 "Ticks waited": r["on"]["low_job_waiting"], "Final priority": r["on"]["low_job_final_priority"]}])
            show_df(df)

    with tabs[2]:
        st.markdown(
            "**Round Robin.** When tasks have the **same** priority (for example DENSITY_J1..J4), the one that "
            "ran least recently goes next, so they take turns.\n\n*Code: the tie-breaker `last_run` in "
            "`PriorityScheduler._pick()`*")
        r = demo_state("demo_rr", "Run Round Robin demo", round_robin_demo)
        if r:
            show_df(pd.DataFrame(r["trace"], columns=["tick", "job on CPU"]))
            st.info("Order: " + " -> ".join(n.replace("DENSITY_", "") for _, n in r["trace"]))

    with tabs[3]:
        st.markdown(
            "**Synchronization.** The dashboard thread reads signal data while the controller thread changes it. "
            "Each signal has a `Lock` and the controller has an `RLock`, so a reader never sees a half-changed "
            "(unsafe) state.\n\n*Code: `traffic_controller.py` -> `ordered_locks()`, `snapshot()`*\n\n"
            "Demo: 4 threads add 1 to a shared counter 2000 times each.")
        r = demo_state("demo_race", "Run race-condition demo", lost_update_demo)
        if r:
            show_df(pd.DataFrame([{"Expected": r["expected"], "WITHOUT lock": r["without_lock"],
                                   "WITH lock": r["with_lock"], "Updates lost without lock": r["lost_updates_without_lock"]}]))

    with tabs[4]:
        st.markdown(
            "**Inter-Process Communication.** Modules do not call each other's internals; they send messages "
            "through thread-safe `queue.Queue` objects:\n\n"
            "- vehicle-producer thread -> **arrival queues** -> intake in the controller thread\n"
            "- dashboard emergency button -> **emergency queue** -> EMERGENCY_DETECT task\n"
            "- every module -> **log queue** -> LOGGER task\n\n"
            "The live queue sizes are shown at the top of this page. *Code: `simulation.py`*")

    with tabs[5]:
        st.markdown(
            "**Deadlock awareness.** Two threads each need locks J1 and J2. If one takes J1 then J2 and the other "
            "takes J2 then J1, each can end up holding what the other needs (circular wait).\n\n"
            "**Our design prevents it:** (1) every thread takes locks in the same global order J1 < J2 < J3 < J4, "
            "(2) every acquire has a time-out and everything is released on failure, (3) the controller is the "
            "only writer of signal colours.\n\n*Code: `traffic_controller.py` -> `ordered_locks()`, `deadlock_demo()`*")
        r = demo_state("demo_dead", "Run deadlock demo (takes about 1 second)", deadlock_demo)
        if r:
            for title, key in (("UNSAFE: opposite lock order", "unsafe_unordered"),
                               ("SAFE: same global lock order", "safe_ordered")):
                res = r[key]
                (st.error if res["deadlock"] else st.success)(
                    f"{title} -> {'DEADLOCK detected' if res['deadlock'] else 'no deadlock'} ({res['seconds']} s)")
                for k, v in sorted(res["results"].items()):
                    st.write(f"- {k}: {v}")

    with tabs[6]:
        st.markdown(
            "**Logging.** Signal changes, vehicle arrivals, scheduling decisions (pre-emption, aging) and "
            "emergency events are all recorded. Modules put messages on the log queue; the LOGGER task "
            "(lowest priority) writes them to memory and to a text file in `logs/`. See the *Event Log* page.")


# ------------------------------------------------------------ event log page
def render_log(engine: TrafficSimulation) -> None:
    snap = engine.snapshot(log_lines=1500)
    cats = st.session_state.get("log_cats") or list(CAT_COLOR)
    n = st.session_state.get("log_n", 60)
    entries = [e for e in snap["logs"] if e["category"] in cats]
    st.caption(f"{len(entries)} matching events (showing newest {n}). Log file: {snap['log_file'] or 'not enabled'}")
    html_block(log_html(entries, n).replace("max-height:330px", "max-height:620px"))


def page_log(engine: TrafficSimulation) -> None:
    st.header("Event log")
    c1, c2 = st.columns([3, 1])
    c1.multiselect("Categories", list(CAT_COLOR), default=list(CAT_COLOR), key="log_cats")
    c2.slider("Lines", 20, 300, 60, key="log_n")
    run_live(render_log, engine)
    snap = engine.snapshot(log_lines=1500)
    if snap["logs"]:
        st.download_button("Download full log (CSV)", pd.DataFrame(snap["logs"]).to_csv(index=False),
                           file_name="traffic_log.csv", mime="text/csv")


# ----------------------------------------------------------------- tests page
def page_tests() -> None:
    st.header("Testing")
    st.markdown(
        "Each test builds a **fresh simulation** (fixed random seed), runs it, and reports what was "
        "*actually measured*. Nothing is pre-written. Results are also saved to `logs/test_report.json`.")
    if st.button("Run all tests", type="primary", key="btn_tests"):
        with st.spinner("Running simulations..."):
            st.session_state["test_results"] = test_suite.run_all()
    results = st.session_state.get("test_results")
    if not results:
        st.info("Press **Run all tests**. It takes a couple of seconds.")
        return
    ok = sum(r["passed"] for r in results)
    st.metric("Tests passed", f"{ok} / {len(results)}")
    for r in results:
        icon = "PASS" if r["passed"] else "FAIL"
        with st.expander(f"[{icon}] {r['name']}  -  {r['goal']}", expanded=not r["passed"]):
            show_df(pd.DataFrame(r["checks"]))
            st.caption(f"Measured values (run took {r['seconds']} s):")
            st.code(json.dumps(r["measured"], indent=2, default=str), language="json")


# ------------------------------------------------------------------ about page
def page_about() -> None:
    st.header("About this project")
    st.markdown(
        """
**Smart Traffic Management System - simulation prototype**
Noida Institute of Engineering & Technology (NIET), Greater Noida · B.Tech CSE (AI), 2nd year · Operating Systems PBL

**What it is.** A software simulation of one crossing with four approach roads (J1 North, J2 East, J3 South,
J4 West). North/South can be green together; East/West can be green together; a North/South road and an
East/West road are *conflicting* and are never active at the same time.

**What it is not.** It is not connected to real cameras, sensors or signals. All traffic is random data produced
from the rates in `data/sample_traffic.csv`.

| File | Purpose |
|---|---|
| `app.py` | This dashboard (Streamlit). Only draws things. |
| `traffic_controller.py` | Signal rules, locks, deadlock prevention, emergency pre-emption. |
| `scheduler.py` | Pre-emptive priority scheduler with aging and Round Robin. |
| `simulation.py` | Threads, IPC queues, vehicle generator, one-second simulation step. |
| `test_suite.py` | Tests that run the real simulation and print measured results. |
| `data/sample_traffic.csv` | Sample simulated traffic (arrival rates per scenario). |
| `logs/` | Log files written while the simulation runs. |

**Signal timing rules.** Green = 6 s + 0.8 s per waiting vehicle on the busiest road of the phase (max 25 s).
Yellow = 3 s, then 1 s all-red. An emergency vehicle cuts the current green short, but the safe sequence
GREEN → YELLOW → ALL-RED → GREEN is always followed.

**Limitations.** One crossing only; no turning movements, pedestrians or real vehicle physics; a single
simulated CPU; arrival data is random, not measured; the scheduler is a teaching model, not a real OS scheduler.
        """)


# ----------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="Smart Traffic Management System", page_icon="🚦", layout="wide")
    html_block(CSS)
    if "engine" not in st.session_state:
        st.session_state["engine"] = TrafficSimulation(threaded=True, log_to_file=True)
    engine: TrafficSimulation = st.session_state["engine"]
    engine.touch()

    sb = st.sidebar
    sb.title("🚦 Traffic Control")
    page = sb.radio("Navigate", ["Dashboard", "OS Concepts & Scheduler", "Event Log", "Testing", "About"],
                    key="nav")
    sb.markdown("**Simulation controls**")
    c1, c2, c3 = sb.columns(3)
    if c1.button("Start", key="btn_start", type="primary"):
        engine.start()
    if c2.button("Pause", key="btn_pause"):
        engine.pause()
    if c3.button("Reset", key="btn_reset"):
        engine.reset()
    sb.caption(f"Status: **{engine.status}**")

    scenario = sb.selectbox("Traffic scenario", list(engine.scenarios), key="scenario",
                            index=list(engine.scenarios).index(engine.scenario))
    if scenario != engine.scenario:
        engine.set_scenario(scenario)
    speed = sb.select_slider("Speed (x real time)", options=[0.5, 1.0, 2.0, 3.0], value=engine.speed, key="speed")
    engine.set_speed(speed)

    sb.markdown("**Emergency vehicle**")
    road = sb.selectbox("Approaching on", JUNCTIONS, format_func=lambda j: f"{j} - {JUNCTION_NAMES[j]}", key="em_road")
    kind = sb.selectbox("Vehicle type", ["Ambulance", "Fire truck"], key="em_kind")
    if sb.button("Simulate emergency vehicle", key="btn_emergency"):
        engine.trigger_emergency(road, kind)
        if not engine.is_running:
            sb.warning("Simulation is not running. The request is queued and will be handled after Start.")
    sb.caption("All data is simulated. No real sensors or signals.")

    if page == "Dashboard":
        run_live(render_dashboard, engine)
    elif page == "OS Concepts & Scheduler":
        page_os(engine)
    elif page == "Event Log":
        page_log(engine)
    elif page == "Testing":
        page_tests()
    else:
        page_about()


main()
