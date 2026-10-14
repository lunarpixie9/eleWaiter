"""EleWaiter - Streamlit UI.   Run:  streamlit run ui/app.py"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import streamlit as st

from elewaiter import charts
from elewaiter.benchmark import ALGORITHMS, run_once
from elewaiter.buildings import custom_hospital
from elewaiter.dispatchers import DEFAULT_WEIGHTS, load_tuned_weights
from elewaiter.metrics import EMERGENCY_WAIT_TARGET
from elewaiter.optimizer import genetic_algorithm
from elewaiter.traffic import BASE_RATE_PER_LIFT, SCENARIOS

DISPLAY_NAMES = {
    "EleWaiter AI": "EleWaiter AI (A* planner)",
    "AI (no emergency logic)": "AI without emergency rules (A* planner)",
}


def display_method(name):
    return DISPLAY_NAMES.get(name, name)

st.set_page_config(page_title="EleWaiter", page_icon="🛗", layout="wide",
                   initial_sidebar_state="expanded")
st.title("🛗 EleWaiter")
st.caption("Simulate hospital lift decisions and check emergency waiting times.")
st.info("Choose a situation on the left, then press **Run simulation**. "
        "This is a simulation, not a live lift control system.")
with st.expander("Which algorithms are included, and which are needed?"):
    st.markdown(r"""
**Used automatically for hospital dispatch:** **EleWaiter AI (A\* planner)** combines urgency rules,
predicted lift space, a wait/travel/fairness cost score, A\* assignment of several waiting calls,
reassignment as conditions change, an emergency lane, and a guard against very long waits.
Lifts serve planned stops using LOOK-style direction control.

**Use for comparison:** FCFS, Nearest, Collective (LOOK-ETA), and Priority-Nearest show how
simpler methods perform on the same trips. **AI without emergency rules** tests the value of
the emergency features. These are useful for evaluation, but are not required to dispatch calls.

**Optional tuning:** Genetic Algorithm (GA) and Simulated Annealing (SA) can tune the AI's
cost weights before a run. Neither is required to use the app or runs on every lift call.

**Where is A\*?** It is inside EleWaiter AI, not a separate method to select. Its search is
bounded; the AI falls back to a simpler assignment if the search limit is reached.
""")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Set up your hospital")
    situation = st.radio("Hospital situation",
                         ["Normal traffic (no emergency)", "With emergency"], index=1)
    st.caption("Normal traffic has staff and visitor trips. With emergency adds urgent lift trips.")

    num_floors = int(st.number_input("Number of floors", min_value=3, max_value=30, value=10))
    num_lifts = int(st.number_input("Number of lifts", min_value=1, max_value=20, value=3))
    capacity = int(st.number_input("Max capacity per lift (spaces)", min_value=3,
                                   max_value=40, value=12,
                                   help="One space is one unit of lift capacity. An ordinary rider uses 1; a stretcher with staff uses 3 spaces total by default."))
    floor_options = list(range(2, num_floors + 1))
    er_floor = st.selectbox("Emergency room (ER) floor", floor_options, index=floor_options.index(2))
    icu_floor = st.selectbox("ICU floor", floor_options, index=floor_options.index(max(3, num_floors - 1)))
    ot_floor = st.selectbox("Operating theatre floor", floor_options, index=floor_options.index(num_floors))
    imaging_floor = st.selectbox("Imaging / labs floor", floor_options, index=floor_options.index(3))
    st.caption("Floor 1 is the lobby. Departments can share a floor; use at least two clinical floors for lift transfers.")

    transfer_space = 3
    if situation == "With emergency":
        transfer = st.selectbox("Emergency transfer load",
                                ["Stretcher + staff (3 spaces)", "Stretcher + team (5 spaces)",
                                 "Staff response only (1 space)"],
                                help="The patient, stretcher and staff must fit in one lift.")
        transfer_space = {"Stretcher + staff (3 spaces)": 3,
                          "Stretcher + team (5 spaces)": 5,
                          "Staff response only (1 space)": 1}[transfer]
        st.caption("Every emergency trip is reported as urgent and requires a lift.")
        st.caption("The stretcher has no separate size setting: 3 or 5 spaces cover the whole patient, stretcher and staff group.")

    layout = (num_floors, num_lifts, capacity, er_floor, icu_floor, ot_floor, imaging_floor)
    problem = None
    try:
        cfg = custom_hospital(*layout)
    except ValueError as exc:
        cfg, problem = None, str(exc)
    if transfer_space > capacity:
        problem = "The selected stretcher/team needs more spaces than this lift can hold."
    if problem:
        st.error(problem)

    with st.expander("Optional test settings"):
        length = st.selectbox("How long should the test run?", ["5 minutes", "3 minutes", "10 minutes"])
        duration = {"3 minutes": 180, "5 minutes": 300, "10 minutes": 600}[length]
        seed = st.number_input("Repeatable trip pattern", 0, 99999, 2000)
        compare = st.checkbox("Compare with other methods", value=False)
        defaults = ["EleWaiter AI", "Collective (LOOK-ETA)", "Priority-Nearest"]
        algos = ["EleWaiter AI"]
        if compare:
            comparison = st.selectbox("Which methods should we compare?",
                                      ["AI vs common methods", "All methods", "Choose methods"])
            algos = {"AI vs common methods": defaults, "All methods": ALGORITHMS}.get(comparison)
            if algos is None:
                algos = st.multiselect("Lift methods", ALGORITHMS, default=defaults,
                                       format_func=display_method)
    st.caption("Lift method: EleWaiter AI (includes A*).")
    run = st.button("Run simulation", type="primary", width="stretch", disabled=bool(problem))

if run:
    with st.spinner("Simulating..."):
        emergency_rate = 0.0 if situation == "Normal traffic (no emergency)" else None
        st.session_state["results"] = [run_once(cfg, "emergency_surge", a, int(seed), float(duration), 1.0,
                                                emergency_rate=emergency_rate, alert_coverage=1.0,
                                                transfer_space=transfer_space) for a in (algos or ["EleWaiter AI"])]
        st.session_state["meta"] = (situation, layout, transfer_space, duration, seed, tuple(algos))
        st.session_state["result_cfg"] = cfg
if problem:
    st.stop()
if "results" not in st.session_state or "result_cfg" not in st.session_state:
    st.stop()
results = st.session_state["results"]
saved = st.session_state["meta"]
if saved != (situation, layout, transfer_space, duration, seed, tuple(algos)):
    st.warning("Settings changed. Press **Run simulation** to update the results below.")
result_cfg = st.session_state["result_cfg"]
df = pd.DataFrame([r.row() for r in results])
df["algorithm"] = df["algorithm"].map(display_method)

overview_tab, = st.tabs(["Overview"])


def seconds(value):
    return "—" if value is None else f"{value:.1f} s"


with overview_tab:
    selected = "EleWaiter AI" if any(r.algorithm == "EleWaiter AI" for r in results) else results[0].algorithm
    if len(results) > 1:
        selected = st.selectbox("Show results for", [r.algorithm for r in results],
                                index=[r.algorithm for r in results].index(selected),
                                format_func=display_method)
    res = next(r for r in results if r.algorithm == selected)
    st.caption(f"{saved[0]} · {result_cfg.num_floors} floors · {result_cfg.num_lifts} lifts · "
               f"{result_cfg.capacity} spaces per lift · {saved[3] / 60:g}-minute arrival period")
    st.write("**What happened in this run**")
    st.write(f"{display_method(selected)} completed **{res.n_served} of {res.n_total} lift trips**. "
             + (f"The average wait to board was **{res.avg_wait:.1f} seconds**. " if res.n_served else "")
             + f"**{res.n_emergency}** requests were emergencies.")
    ordinary_rate = BASE_RATE_PER_LIFT * SCENARIOS["emergency_surge"].rate_factor
    emergency_rate = (SCENARIOS["emergency_surge"].emergency_rate
                      if saved[0] == "With emergency" else 0.0)
    expected_trips = (ordinary_rate * result_cfg.num_lifts + emergency_rate) * saved[3]
    st.caption(f"Total trips = ordinary requests ({res.n_total - res.n_emergency}) + "
               f"emergency requests ({res.n_emergency}) = {res.n_total}. "
               f"Expected total ≈ ({ordinary_rate:.2f} ordinary requests/second/lift × "
               f"{result_cfg.num_lifts} lifts + {emergency_rate:.3f} emergency requests/second) "
               f"× {saved[3]:g} seconds = {expected_trips:.1f}. "
               "The actual count varies because arrivals are random. One patient with their team "
               "counts as one trip; completed means they reached their destination.")
    a, b, c, d = st.columns(4)
    a.metric("Trips completed", f"{res.n_served}/{res.n_total}")
    b.metric("Trips not completed", res.n_unserved,
             help="Still waiting or travelling when the simulation ended.")
    c.metric("Average wait to board", seconds(res.avg_wait if res.n_served else None))
    d.metric("95% waited within (P95)", seconds(res.p95_wait if res.n_served else None),
             help="The 95th percentile: about 95% of completed trips had a wait at or below this value.")
    if res.n_unserved:
        st.warning(f"{res.n_unserved} trips were not completed. The wait and journey times below describe completed trips only.")
    else:
        st.success("Every requested trip reached its destination.")

    st.write("**Emergency response**")
    if res.n_emergency:
        completed_em = len(res.em_wait_times)
        st.write(f"**{completed_em} of {res.n_emergency} emergency trips completed.** "
                 "All emergency requests were reported as urgent.")
        a, b, c, d = st.columns(4)
        a.metric("Emergency wait to board", seconds(res.em_avg_wait))
        b.metric("Call → destination", seconds(res.em_avg_journey))
        c.metric(f"Boarded within {EMERGENCY_WAIT_TARGET:.0f} s",
                 "—" if res.em_within_target is None else f"{res.em_within_target:.0%}")
        d.metric("Ordinary trip wait", seconds(res.routine_avg_wait))
        st.caption(f"Wait = request to boarding. Full trip = request to arrival at the destination. "
                   f"The {EMERGENCY_WAIT_TARGET:.0f}-second pickup target is a simulation setting. "
                   "Emergency times and target percentage use completed emergency trips only. "
                   "Ordinary wait uses completed non-emergency trips.")
        if completed_em < res.n_emergency:
            st.warning(f"{res.n_emergency - completed_em} emergency trips did not finish; they are not included in the averages or target percentage.")
    else:
        st.info("Normal traffic mode: no emergency trips were generated.")

    st.write("**Long waits and full lifts**")
    a, b, c = st.columns(3)
    a.metric("Longest wait", seconds(res.max_wait if res.n_served else None))
    b.metric("Waits of 45 s or more", res.starvation_count,
             help="Completed trips that waited at least 45 seconds to board.")
    c.metric("Full-lift pass-bys", res.left_behind,
             help="Times a lift could not pick someone up because it lacked space. One request may be passed more than once.")
    st.caption("Smaller waits are better. Full-lift pass-bys are events, not a count of different people.")

    if len(results) > 1:
        with st.expander("Compare all selected lift methods", expanded=True):
            fields = ["algorithm", "n_served", "n_unserved", "avg_wait", "em_avg_wait", "em_avg_journey"]
            st.dataframe(df[fields].rename(columns={
                "algorithm": "Lift method", "n_served": "Trips completed", "n_unserved": "Not completed",
                "avg_wait": "Average wait (s)", "em_avg_wait": "Emergency wait (s)",
                "em_avg_journey": "Emergency full trip (s)"}).round(1), hide_index=True, width="stretch")
            st.caption("Each method receives the same trip requests. This table compares independent simulation runs.")

    st.write("**How the lifts moved**")
    st.caption("Read left to right through time. Each dot is a lift opening its doors on a floor. "
               "Colours identify lifts; marker shapes distinguish emergency-lane stops. Larger dots mean more capacity spaces occupied.")
    st.plotly_chart(charts.lift_activity(res.timeline), width="stretch")

    with st.expander("Hospital layout used in this run"):
        st.write(f"**{result_cfg.num_lifts} lifts**, each with **{result_cfg.capacity} spaces**. "
                 f"Patient-transfer groups reserve **{saved[2]} spaces** when emergency trips are enabled.")
        st.caption("One space = one unit of lift capacity. An ordinary rider uses 1 space. "
                   "The default stretcher + staff transfer uses 3 spaces total; a larger team uses 5. "
                   "These are simulation assumptions, not physical measurements of a stretcher.")
        st.dataframe(pd.DataFrame([{"Floor": f.number, "Department": f.name}
                                  for f in result_cfg.floors.values()]), hide_index=True, width="stretch")

    with st.expander("How the AI chooses a lift — in simple terms"):
        st.markdown(r"""
1. **Check each lift:** estimate pickup time, travel time and available space.
2. **Score the choices (cost heuristic):** weigh urgency, waiting, crowding and delays to other trips.
3. **Plan several calls together (A\*):** search combinations of lift assignments within a limited search budget.
4. **Update the plan:** revise assignments as lifts move; use an emergency lane when the cost model favours direct service.
5. **Prevent long waits:** raise waiting passengers' priority and apply a starvation guard.

The movement chart shows the resulting stops. It does not display a recorded explanation of every individual assignment.
""")

    with st.expander("Technical details and waiting-time chart"):
        st.caption("P95 is the 95th percentile; it describes longer waits, not just the average. All timing statistics here use completed trips.")
        st.dataframe(pd.DataFrame([{
            "Average full trip (s)": res.avg_journey if res.n_served else None,
            "Wait variation — standard deviation (s)": res.wait_std if res.n_served else None,
            "Emergency P95 wait (s)": res.em_p95_wait,
            "Emergency P95 full trip (s)": res.em_p95_journey,
            "Lift floors travelled": res.travel_distance,
            "Starvation-guard assignments": res.forced_assignments,
        }]).round(1), hide_index=True, width="stretch")
        st.plotly_chart(charts.wait_histogram([res], labels=DISPLAY_NAMES), width="stretch")

    with st.expander("Advanced: tune the AI weights (optional)"):
        tuned = load_tuned_weights(result_cfg.name)
        st.write("Weights control how strongly the AI values waiting time, travel, space and fairness. "
                 "GA and simulated annealing can tune them before a simulation; they do not run for each lift call.")
        st.dataframe(pd.DataFrame({"Default": DEFAULT_WEIGHTS, "Used in this run": tuned or DEFAULT_WEIGHTS}).round(3), width="stretch")
        if st.button("Try a small GA (does not save weights)"):
            bar = st.progress(0.0)
            best, hist = genetic_algorithm(result_cfg, pop_size=8, generations=4,
                fit_kwargs=dict(scenarios=("emergency_surge",), seeds=(1000, 1001), intensities=(1.5,),
                                duration=200.0, transfer_space=saved[2]),
                progress=lambda g, f: bar.progress(g / 4))
            st.plotly_chart(charts.training_curve({"GA": hist}), width="stretch")
            st.dataframe(pd.DataFrame({"Default": DEFAULT_WEIGHTS, "Example tuned weights": best}).round(3), width="stretch")

    st.caption("Results describe this simulated hospital and trip pattern. Try several trip patterns before judging a method's performance.")
