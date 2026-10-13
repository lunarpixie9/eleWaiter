"""EleWaiter - Streamlit UI.   Run:  streamlit run ui/app.py"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import streamlit as st

from elewaiter import charts
from elewaiter.benchmark import ALGORITHMS, CORE_ALGORITHMS, run_once
from elewaiter.buildings import PRESETS
from elewaiter.dispatchers import DEFAULT_WEIGHTS, load_tuned_weights
from elewaiter.metrics import EMERGENCY_WAIT_TARGET
from elewaiter.models import Urgency
from elewaiter.optimizer import genetic_algorithm
from elewaiter.traffic import HOSPITAL_SCENARIOS, GENERAL_SCENARIOS, SCENARIOS

st.set_page_config(page_title="EleWaiter", page_icon="🛗", layout="wide")
st.title("🛗 EleWaiter")
st.caption("Emergency-first intelligent lift dispatch & fairness simulator - hospital focus, works for any building.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Setup")
    bname = st.selectbox("Building", list(PRESETS))
    cfg = PRESETS[bname]()
    pool = HOSPITAL_SCENARIOS + GENERAL_SCENARIOS if cfg.kind == "hospital" else GENERAL_SCENARIOS
    scenario = st.selectbox("Scenario", pool, format_func=lambda k: SCENARIOS[k].label)
    st.caption(SCENARIOS[scenario].description)
    intensity = st.slider("Routine traffic intensity", 0.5, 2.5, 1.5, 0.1)
    has_crit = bool(cfg.critical_floors())
    use_default_em = st.checkbox("Scenario's default emergency rate", True, disabled=not has_crit)
    em_rate = None
    if has_crit and not use_default_em:
        em_rate = st.slider("Emergencies per minute", 0.0, 6.0, 1.0, 0.1) / 60.0
    duration = st.slider("Simulated seconds", 200, 900, 600, 50)
    seed = st.number_input("Random seed", 0, 99999, 2000)
    algos = st.multiselect("Dispatchers", ALGORITHMS, default=CORE_ALGORITHMS)
    run = st.button("Run simulation", type="primary", width="stretch")

if run or "results" not in st.session_state:
    with st.spinner("Simulating..."):
        st.session_state["results"] = [run_once(cfg, scenario, a, int(seed), float(duration), float(intensity),
                                                None, em_rate) for a in (algos or CORE_ALGORITHMS)]
        st.session_state["meta"] = (bname, scenario)
results = st.session_state["results"]
df = pd.DataFrame([r.row() for r in results])

tabs = st.tabs(["Results", "Emergency focus", "Lift activity", "Optimizer", "Building & urgency rules", "About"])

# ------------------------------------------------------------------ results
with tabs[0]:
    st.caption(f"{st.session_state['meta'][0]} | {SCENARIOS[st.session_state['meta'][1]].label} | one seed - use run_benchmark.py for multi-seed statistics")
    cols = ["algorithm", "avg_wait", "p95_wait", "max_wait", "wait_std", "starvation_count",
            "left_behind", "travel_distance", "avg_journey"]
    st.dataframe(df[cols].round(2), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.comparison_bar(df, "avg_wait", "Average wait"), width="stretch")
    c2.plotly_chart(charts.comparison_bar(df, "p95_wait", "95th-percentile wait (fairness tail)"), width="stretch")
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.comparison_bar(df, "starvation_count", "Passengers waiting 45 s or more", "passengers"), width="stretch")
    c4.plotly_chart(charts.comparison_bar(df, "left_behind", "Left behind by a full lift", "events"), width="stretch")
    st.plotly_chart(charts.wait_histogram(results), width="stretch")

# ------------------------------------------------------------------ emergency
with tabs[1]:
    if df["n_emergency"].max() == 0:
        st.info("This building/scenario has no emergencies (no critical floors, or the rate is zero).")
    else:
        st.write(f"**{int(df['n_emergency'].max())} emergency trips** in this run. Emergency status is ground truth that dispatchers "
                 f"never see - they must infer it from floors and time. Target: board within {EMERGENCY_WAIT_TARGET:.0f} s.")
        ecols = ["algorithm", "em_avg_wait", "em_p95_wait", "em_max_wait", "em_within_target", "routine_avg_wait"]
        st.dataframe(df[ecols].round(2), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.comparison_bar(df, "em_avg_wait", "Emergency average wait"), width="stretch")
        c2.plotly_chart(charts.comparison_bar(df, "routine_avg_wait", "Routine passengers' average wait (the cost of priority)"), width="stretch")
        st.plotly_chart(charts.wait_histogram(results, em=True), width="stretch")
        p, r = df["infer_precision"].dropna(), df["infer_recall"].dropna()
        if len(p) and len(r):
            st.write(f"**Context inference quality (this run):** precision {p.iloc[0]:.0%} (inferred-critical trips that were real "
                     f"emergencies), recall {r.iloc[0]:.0%} (real emergencies the floor rules recognised). Code responses on "
                     f"ordinary wards cannot be recognised from floors alone - a known limitation.")

# ------------------------------------------------------------------ activity
with tabs[2]:
    pick = st.selectbox("Dispatcher", [r.algorithm for r in results])
    res = next(r for r in results if r.algorithm == pick)
    st.plotly_chart(charts.lift_activity(res.timeline), width="stretch")
    st.caption("Each dot is a door opening; size = passengers aboard; diamonds/other symbols mark emergency-lane trips.")

# ------------------------------------------------------------------ optimizer
with tabs[3]:
    tuned = load_tuned_weights()
    st.write("The AI dispatcher's cost-function weights are tuned with a Genetic Algorithm (or Simulated Annealing via "
             "`train_optimizer.py`), using the simulator itself as the fitness function.")
    wdf = pd.DataFrame({"default": DEFAULT_WEIGHTS, "tuned (shipped)": tuned or DEFAULT_WEIGHTS}).round(3)
    st.dataframe(wdf, width="stretch")
    log = os.path.join(ROOT, "elewaiter", "data", "training_log.json")
    if os.path.exists(log):
        L = json.load(open(log))
        st.plotly_chart(charts.training_curve({"GA": L["ga_history"], "SA": L["sa_history"]}), width="stretch")
        st.caption(f"Validation fitness (different seeds): { {k: round(v, 1) for k, v in L['validation'].items()} } - winner: {L['winner']}")
    with st.expander("Run a quick GA here (small; the shipped weights used a bigger run)"):
        if st.button("Run mini-GA"):
            bar = st.progress(0.0)
            best, hist = genetic_algorithm(cfg, pop_size=8, generations=4,
                                           fit_kwargs=dict(scenarios=("emergency_surge",), seeds=(1000, 1001), intensities=(1.5,), duration=200.0),
                                           progress=lambda g, f: bar.progress(g / 4))
            st.plotly_chart(charts.training_curve({"GA": hist}), width="stretch")
            st.dataframe(pd.DataFrame({"default": DEFAULT_WEIGHTS, "mini-GA best": best}).round(3), width="stretch")

# ------------------------------------------------------------------ building
with tabs[4]:
    st.subheader(cfg.name)
    st.dataframe(pd.DataFrame([{"floor": f.number, "name": f.name, "zone": f.zone.name, "role": f.role,
                                "time boost": ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in f.boost_windows) or "-"}
                               for f in cfg.floors.values()]), hide_index=True, width="stretch")
    st.write("**Urgency rules (no buttons, cards or extra hardware):** a trip from a CRITICAL zone (ER/ICU/OT) to a HIGH or "
             "CRITICAL zone is *critical*; any other trip touching a HIGH/CRITICAL zone is *elevated*; the rest are routine. "
             "Zones can be upgraded during time windows (e.g. lecture floors just before class). "
             "On top of that, urgency rises automatically with waiting time.")
    a, b, t = st.columns(3)
    o = a.selectbox("From", sorted(cfg.floors), format_func=lambda n: f"{n} - {cfg.floors[n].name}")
    d = b.selectbox("To", sorted(cfg.floors), index=min(1, len(cfg.floors) - 1), format_func=lambda n: f"{n} - {cfg.floors[n].name}")
    tt = t.number_input("At time (s)", 0, 3600, 0)
    st.metric("Inferred urgency", Urgency(cfg.infer_urgency(o, d, float(tt))).name if o != d else "choose different floors")

# ------------------------------------------------------------------ about
with tabs[5]:
    st.markdown("""
**What it is.** A simulator plus an AI dispatcher that decides which lift serves which passenger, built for hospitals
where an emergency transfer (ER to ICU/OT) must not wait behind routine traffic, while visitors and staff are not starved.

**AI components**
- Multi-objective cost: own wait/ride (scaled by urgency), delay imposed on others, crowding, predicted overflow.
- A* search over joint assignments of pending calls; dynamic re-optimisation of not-yet-boarded passengers.
- Emergency lane: critical trips get a lift committed to them (batching several critical passengers), chosen by cost
  versus normal priority service so a burst cannot stall the whole building.
- Anti-starvation: priority grows with waiting time; left-behind passengers get a boost; a hard guard force-assigns.
- GA / simulated annealing tune the weights.

**Honest limits.** Results hold for the simulated traffic, not real buildings. Urgency is inferred from context, so some
real emergencies are missed and some routine trips are boosted (measured above).
""")
