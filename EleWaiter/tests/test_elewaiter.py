"""Tests for EleWaiter. Run:  pytest -q"""
import itertools
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elewaiter.benchmark import ALGORITHMS, make_dispatcher, run_once, run_benchmark, win_rate, summarize
from elewaiter.buildings import PRESETS, hospital_10, college_8, office_12
from elewaiter.dispatchers import (AIDispatcher, Assignment, Collective, DEFAULT_WEIGHTS, Dispatcher,
                                   FCFS, Nearest, PriorityNearest, WEIGHT_BOUNDS)
from elewaiter.models import (BuildingConfig, Floor, LiftCar, LiftState, Passenger, Urgency, Zone)
from elewaiter.optimizer import composite, fitness, genetic_algorithm, simulated_annealing
from elewaiter.planner import (eta_table, look_order, model_from_lift, normal_cost, express_cost,
                               predicted_load)
from elewaiter.simulator import ElevatorSimulator
from elewaiter.traffic import SCENARIOS, TrafficGenerator


@pytest.fixture
def cfg():
    return hospital_10()


def make_lifts(cfg, n=None):
    return [LiftCar(i, floor=1, capacity=cfg.capacity, speed=cfg.speed, door_time=cfg.door_time)
            for i in range(n or cfg.num_lifts)]


# ------------------------------------------------------------------ zones & urgency
class TestUrgencyInference:
    def test_er_to_icu_is_critical(self, cfg):
        assert cfg.infer_urgency(2, 9) == Urgency.CRITICAL

    def test_er_to_imaging_is_critical(self, cfg):
        assert cfg.infer_urgency(2, 3) == Urgency.CRITICAL

    def test_lobby_to_ward_is_routine(self, cfg):
        assert cfg.infer_urgency(1, 6) == Urgency.ROUTINE

    def test_staff_to_icu_is_elevated(self, cfg):
        assert cfg.infer_urgency(6, 9) == Urgency.ELEVATED

    def test_time_window_upgrades_zone(self):
        c = college_8()
        assert c.zone_at(3, 50.0) == Zone.NORMAL
        assert c.zone_at(3, 120.0) == Zone.HIGH
        assert c.infer_urgency(1, 3, 120.0) == Urgency.ELEVATED
        assert c.infer_urgency(1, 3, 50.0) == Urgency.ROUTINE

    def test_office_has_no_critical_floors(self):
        assert office_12().critical_floors() == []


# ------------------------------------------------------------------ traffic
class TestTraffic:
    def test_reproducible(self, cfg):
        a = TrafficGenerator(cfg, 5).generate("emergency_surge", 300)
        b = TrafficGenerator(cfg, 5).generate("emergency_surge", 300)
        assert [(p.origin, p.destination, p.arrival_time) for p in a] == \
               [(p.origin, p.destination, p.arrival_time) for p in b]

    def test_sorted_and_valid_floors(self, cfg):
        for sc in SCENARIOS:
            pax = TrafficGenerator(cfg, 1).generate(sc, 300)
            assert [p.arrival_time for p in pax] == sorted(p.arrival_time for p in pax)
            assert all(1 <= p.origin <= cfg.num_floors and 1 <= p.destination <= cfg.num_floors
                       and p.origin != p.destination for p in pax)

    def test_hospital_scenarios_have_ground_truth_emergencies(self, cfg):
        pax = TrafficGenerator(cfg, 3).generate("emergency_surge", 600)
        assert sum(p.is_emergency for p in pax) > 0

    def test_mass_casualty_has_a_burst(self, cfg):
        pax = TrafficGenerator(cfg, 3).generate("mass_casualty", 300)
        in_burst = [p for p in pax if p.is_emergency and 60 <= p.arrival_time <= 120]
        assert len(in_burst) >= 10

    def test_no_emergencies_in_office(self):
        pax = TrafficGenerator(office_12(), 3).generate("up_peak", 300)
        assert not any(p.is_emergency for p in pax)

    def test_emergency_rate_independent_of_intensity(self, cfg):
        lo = sum(p.is_emergency for p in TrafficGenerator(cfg, 4).generate("emergency_surge", 1200, 1.0))
        hi = sum(p.is_emergency for p in TrafficGenerator(cfg, 4).generate("emergency_surge", 1200, 3.0))
        assert abs(lo - hi) <= 0.5 * max(lo, 1)

    def test_dispatcher_cannot_see_ground_truth_urgency_before_arrival(self, cfg):
        pax = TrafficGenerator(cfg, 3).generate("emergency_surge", 100)
        assert all(p.urgency == Urgency.ROUTINE for p in pax)   # set by the simulator on arrival


# ------------------------------------------------------------------ simulator invariants
def run_sim(cfg, algo, scenario="emergency_surge", seed=3, duration=300, intensity=1.0):
    spec = SCENARIOS[scenario]
    pax = TrafficGenerator(cfg, seed).generate(scenario, duration, intensity)
    sim = ElevatorSimulator(cfg, pax, make_dispatcher(algo), scenario, spec.disabled_lifts)
    res = sim.run()
    return sim, res


@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize("intensity", [1.0, 2.5])
def test_capacity_never_exceeded(cfg, algo, intensity):
    """Regression: the original version let lifts carry up to 19 people with capacity 10."""
    sim, res = run_sim(cfg, algo, "visiting_hours", intensity=intensity)
    assert max(t["load"] for t in res.timeline) <= cfg.capacity


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_lifts_never_teleport(cfg, algo):
    """Regression: the original version moved lifts to a different floor than scheduled."""
    sim, res = run_sim(cfg, algo, "mass_casualty", intensity=1.5)
    last = {}
    for t in res.timeline:
        if t["lift"] in last:
            dt = t["time"] - last[t["lift"]][0]
            df = abs(t["floor"] - last[t["lift"]][1])
            assert df / cfg.speed <= dt + 0.11
        last[t["lift"]] = (t["time"], t["floor"])


@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_everyone_served_and_times_ordered(cfg, algo, scenario):
    sim, res = run_sim(cfg, algo, scenario, duration=240, intensity=1.5)
    assert res.n_served == res.n_total
    for p in sim.completed:
        assert p.arrival_time <= p.board_time <= p.exit_time


@pytest.mark.parametrize("building", list(PRESETS))
def test_works_on_every_building(building):
    cfg = PRESETS[building]()
    for sc in ("up_peak", "down_peak", "mixed"):
        r = run_once(cfg, sc, "EleWaiter AI", seed=2, duration=200)
        assert r.n_served == r.n_total


def test_disabled_lift_never_moves(cfg):
    sim, res = run_sim(cfg, "EleWaiter AI", "lift_failure")
    assert sim.lifts[0].travelled == 0 and sim.lifts[0].state == LiftState.OUT_OF_SERVICE


def test_deterministic(cfg):
    a = run_once(cfg, "emergency_surge", "EleWaiter AI", 11, 300)
    b = run_once(cfg, "emergency_surge", "EleWaiter AI", 11, 300)
    assert a.avg_wait == b.avg_wait and a.em_avg_wait == b.em_avg_wait


def test_lone_passenger_wait_matches_physics(cfg):
    """Lift waits at the lobby; a passenger there boards immediately; ride time = floors/speed + doors."""
    pax = [Passenger(0, 1, 6, 10.0)]
    sim = ElevatorSimulator(cfg, pax, make_dispatcher("Nearest"), "t")
    r = sim.run()
    p = sim.completed[0]
    assert p.wait_time == pytest.approx(0.0, abs=1e-6)
    assert p.exit_time - p.board_time == pytest.approx(cfg.door_time + cfg.board_time + 5 / cfg.speed, abs=1.0)


# ------------------------------------------------------------------ starvation guard
class LazyDispatcher(Dispatcher):
    """Never assigns anyone, so only the starvation guard can serve passengers."""
    name = "lazy"

    def assign(self, unassigned, reassignable, lifts, now, config):
        return Assignment()


def test_starvation_guard_actually_fires(cfg):
    """Regression: in the original version the guard was dead code."""
    pax = [Passenger(0, 5, 8, 0.0), Passenger(1, 3, 1, 5.0)]
    sim = ElevatorSimulator(cfg, pax, LazyDispatcher(), "t", starvation_limit=30.0)
    r = sim.run()
    assert r.n_served == 2 and r.forced_assignments >= 2
    assert all(p.wait_time < 30.0 + 25.0 for p in sim.completed)


# ------------------------------------------------------------------ express mode
def _lane_sim(cfg, dispatcher, pax, disabled=()):
    sim = ElevatorSimulator(cfg, pax, dispatcher, "t", disabled_lifts=disabled)
    return sim, sim.run()


@pytest.mark.parametrize("make", [lambda: PriorityNearest(), lambda: AIDispatcher(DEFAULT_WEIGHTS)])
def test_lane_lift_never_boards_routine_passengers(cfg, make):
    """While a lift runs an emergency lane it takes only express (critical) passengers.
    (Default weights are pinned: the tuned weights deliberately use lanes sparingly.)"""
    pax = TrafficGenerator(cfg, 8).generate("mass_casualty", 300, 1.5)
    sim, res = _lane_sim(cfg, make(), pax)
    assert any(t["lane"] for t in res.timeline), "scenario should trigger emergency lanes"
    assert sum(t["routine_boarded_in_lane"] for t in res.timeline) == 0


def test_ai_batches_critical_passengers_when_lifts_are_scarce(cfg):
    """With one lift in service, three ER->ICU emergencies must share a lane trip."""
    pax = [Passenger(i, 2, 9, 0.5 * i, is_emergency=True) for i in range(3)]
    sim, res = _lane_sim(cfg, AIDispatcher(DEFAULT_WEIGHTS), pax, disabled=(1, 2))
    assert res.n_served == 3
    assert max(t["load"] for t in res.timeline if t["lane"]) >= 2


def test_ai_uses_parallel_lanes_when_lifts_are_free(cfg):
    pax = [Passenger(i, 2, 9, 0.5 * i, is_emergency=True) for i in range(3)]
    sim, res = _lane_sim(cfg, AIDispatcher(DEFAULT_WEIGHTS), pax)
    assert res.n_served == 3 and res.em_max_wait < 10.0


def test_tuned_weights_still_serve_emergencies_fast(cfg):
    pax = [Passenger(i, 2, 9, 0.5 * i, is_emergency=True) for i in range(3)]
    sim, res = _lane_sim(cfg, AIDispatcher(), pax)          # shipped (tuned) weights
    assert res.n_served == 3 and res.em_max_wait < 10.0


def test_express_serves_critical_faster_than_nearest_under_load(cfg):
    df = run_benchmark(cfg, ["emergency_surge"], ["Nearest", "Priority-Nearest"], range(6), 400, 2.0)
    s = summarize(df).set_index("algorithm")
    assert s.loc["Priority-Nearest", "em_avg_wait"] < s.loc["Nearest", "em_avg_wait"]


# ------------------------------------------------------------------ planner
class TestPlanner:
    def test_look_order_up_then_down(self):
        assert look_order(5, 1, {2, 7, 9, 4}) == [7, 9, 4, 2]
        assert look_order(5, -1, {2, 7, 9, 4}) == [4, 2, 7, 9]

    def test_look_order_idle_picks_nearer_side(self):
        assert look_order(5, 0, {6, 1}) == [6, 1]
        assert look_order(5, 0, {8, 4}) == [4, 8]

    def test_eta_table_travel_and_doors(self, cfg):
        lift = make_lifts(cfg)[0]
        m = model_from_lift(lift, 0.0)
        _, tab, tend = eta_table(m, {4, 6})
        assert tab[4] == pytest.approx(3.0)                    # 3 floors at 1 floor/s
        assert tab[6] == pytest.approx(3.0 + cfg.door_time + 2.0)

    def test_nearer_lift_has_lower_cost(self, cfg):
        lifts = make_lifts(cfg)
        lifts[0].floor, lifts[1].floor = 2, 9
        ms = [model_from_lift(l, 0.0) for l in lifts]
        c = [normal_cost(m, 4, 8, 1.0, DEFAULT_WEIGHTS) for m in ms]
        assert c[0] < c[1]

    def test_predicted_overflow_penalised(self, cfg):
        lifts = make_lifts(cfg)
        lifts[0].floor = 3
        lifts[0].passengers = [Passenger(100 + i, 1, 9, 0.0) for i in range(cfg.capacity)]
        m_full = model_from_lift(lifts[0], 0.0)
        m_free = model_from_lift(lifts[1], 0.0)
        assert normal_cost(m_full, 4, 8, 1.0, DEFAULT_WEIGHTS) > normal_cost(m_free, 4, 8, 1.0, DEFAULT_WEIGHTS)

    def test_predicted_load_counts_dropoffs_before_pickup(self, cfg):
        lift = make_lifts(cfg)[0]
        lift.floor = 1
        lift.passengers = [Passenger(i, 1, 3, 0.0) for i in range(cfg.capacity)]   # all leave at 3
        m = model_from_lift(lift, 0.0)
        _, tab, _ = eta_table(m, m.stop_set | {5, 8})
        assert predicted_load(m, tab, tab[5]) == 0      # everyone exits at floor 3 before floor 5

    def test_delay_imposed_on_others_is_charged(self, cfg):
        lifts = make_lifts(cfg)
        for i in range(6):                              # lift 0 carries 6 people to floor 9
            lifts[0].passengers.append(Passenger(i, 1, 9, 0.0))
        lifts[0].floor, lifts[0].direction, lifts[0].state = 2, 1, LiftState.MOVING
        lifts[0].next_arrival_at = 1.0
        lifts[1].floor = 1
        ms = [model_from_lift(l, 0.0) for l in lifts]
        w = dict(DEFAULT_WEIGHTS, w_delay=1.0)
        # a detour stop at 5 delays 6 riders on lift 0 but nobody on lift 1
        assert normal_cost(ms[0], 5, 6, 1.0, w) > normal_cost(ms[1], 5, 6, 1.0, w) - 3.0

    def test_urgency_changes_the_gap_between_lifts(self, cfg):
        """Regression: in the original cost function the priority/starvation terms were
        identical for every lift (a constant added to every option), so tuning them
        could never change a decision. Here urgency must change the COST GAP between
        lifts, i.e. it can flip which lift wins."""
        lifts = make_lifts(cfg)
        lifts[0].floor = 2                        # close, but crowded and heading elsewhere
        lifts[0].passengers = [Passenger(i, 1, 10, 0.0) for i in range(9)]
        lifts[0].state, lifts[0].direction, lifts[0].next_arrival_at = LiftState.MOVING, 1, 1.0
        lifts[1].floor = 8                        # farther away but empty
        m = [model_from_lift(l, 0.0) for l in lifts[:2]]
        w = dict(DEFAULT_WEIGHTS, w_delay=1.2)
        gap = lambda W: normal_cost(m[0], 1, 6, W, w) - normal_cost(m[1], 1, 6, W, w)
        assert gap(1.0) != pytest.approx(gap(5.0))

    def test_urgent_passenger_accepts_more_imposed_delay(self, cfg):
        """A routine caller avoids a crowded lift (it would delay 9 riders); an urgent
        caller takes it because it gets there sooner."""
        lifts = make_lifts(cfg, 2)
        lifts[0].floor = 2                       # crowded, already heading up past floor 4
        lifts[0].passengers = [Passenger(i, 1, 10, 0.0) for i in range(9)]
        lifts[0].state, lifts[0].direction, lifts[0].next_arrival_at = LiftState.MOVING, 1, 1.0
        lifts[1].floor = 10                      # empty but far above the call
        d = AIDispatcher(dict(DEFAULT_WEIGHTS, w_delay=0.3, w_load=0.0, a_urg=4.0))
        calm, urgent = Passenger(50, 4, 8, 0.0), Passenger(51, 4, 8, 0.0)
        urgent.urgency = Urgency.ELEVATED
        assert d.assign([calm], [], lifts, 0.0, cfg).lift_for[50] == 1
        assert d.assign([urgent], [], lifts, 0.0, cfg).lift_for[51] == 0


# ------------------------------------------------------------------ A* search
class TestAStar:
    def _setup(self, cfg):
        lifts = make_lifts(cfg)
        lifts[0].floor, lifts[1].floor, lifts[2].floor = 1, 5, 9
        pax = [Passenger(i, o, d, 0.0) for i, (o, d) in enumerate([(2, 8), (6, 1), (9, 3), (4, 10)])]
        return lifts, pax

    def test_single_passenger_equals_argmin(self, cfg):
        lifts, pax = self._setup(cfg)
        d = AIDispatcher(DEFAULT_WEIGHTS)
        got = d.assign(pax[:1], [], lifts, 0.0, cfg).lift_for[0]
        ms = [model_from_lift(l, 0.0) for l in lifts]
        costs = [normal_cost(m, pax[0].origin, pax[0].destination, d.pweight(pax[0], 0.0), d.w) for m in ms]
        assert got == costs.index(min(costs))

    def test_joint_search_no_worse_than_greedy(self, cfg):
        lifts, pax = self._setup(cfg)
        d = AIDispatcher(DEFAULT_WEIGHTS, batch=4, max_nodes=2000)
        models = tuple(model_from_lift(l, 0.0, {p.id for p in pax}) for l in lifts)
        Ws = [d.pweight(p, 0.0) for p in pax]

        def total(picks):
            ms, tot = list(models), 0.0
            for pid, lid, ex in picks:
                p = pax[pid]
                c, nm, _ = d._option(p, Ws[pid], ms[lid])
                tot += c
                ms[lid] = nm
            return tot

        a_star, _ = d._search(pax, Ws, models)
        greedy, _ = d._greedy_finish(pax, Ws, 0, models, ())
        assert len(a_star) == 4 and total(a_star) <= total(greedy) + 1e-6

    def test_all_pending_get_assigned(self, cfg):
        lifts, pax = self._setup(cfg)
        out = AIDispatcher(DEFAULT_WEIGHTS).assign(pax, [], lifts, 0.0, cfg)
        assert set(out.lift_for) == {p.id for p in pax}

    def test_out_of_service_lift_never_chosen(self, cfg):
        lifts, pax = self._setup(cfg)
        lifts[0].state = LiftState.OUT_OF_SERVICE
        out = AIDispatcher(DEFAULT_WEIGHTS).assign(pax, [], lifts, 0.0, cfg)
        assert 0 not in out.lift_for.values()


# ------------------------------------------------------------------ emergency behaviour
class TestEmergencyLogic:
    def test_critical_trip_goes_express_on_idle_lifts(self, cfg):
        lifts = make_lifts(cfg)
        lifts[0].floor = 2
        p = Passenger(1, 2, 9, 0.0)
        p.urgency = Urgency.CRITICAL
        out = AIDispatcher(DEFAULT_WEIGHTS).assign([p], [], lifts, 0.0, cfg)
        assert out.lift_for[1] == 0 and 1 in out.express

    def test_no_emergency_mode_never_expresses(self, cfg):
        lifts = make_lifts(cfg)
        p = Passenger(1, 2, 9, 0.0)
        p.urgency = Urgency.CRITICAL
        out = AIDispatcher(DEFAULT_WEIGHTS, emergency=False).assign([p], [], lifts, 0.0, cfg)
        assert not out.express

    def test_priority_boarding_when_space_short(self, cfg):
        pax = [Passenger(i, 1, 5, 0.0) for i in range(cfg.capacity)] + \
              [Passenger(100, 1, 9, 5.0, is_emergency=True)]
        sim = ElevatorSimulator(cfg, pax, make_dispatcher("Priority-Nearest"), "t")
        sim.run()
        em = sim.pax[100]
        assert em.board_time is not None and em.wait_time < 20.0

    def test_emergency_logic_beats_conventional_in_emergency_surge(self, cfg):
        df = run_benchmark(cfg, ["emergency_surge"], ["Collective (LOOK-ETA)", "EleWaiter AI"],
                           range(500, 508), 400, 1.5)
        s = summarize(df).set_index("algorithm")
        assert s.loc["EleWaiter AI", "em_avg_wait"] < s.loc["Collective (LOOK-ETA)", "em_avg_wait"]
        assert s.loc["EleWaiter AI", "em_within_target"] >= s.loc["Collective (LOOK-ETA)", "em_within_target"]

    def test_mass_casualty_burst_does_not_collapse_the_building(self, cfg):
        """Regression: with one-lift-per-emergency express mode the burst more than
        doubled everyone's wait. The lane + backlog-aware cost must keep it bounded."""
        df = run_benchmark(cfg, ["mass_casualty"], ["Collective (LOOK-ETA)", "Priority-Nearest", "EleWaiter AI"],
                           range(500, 508), 400, 1.5)
        s = summarize(df).set_index("algorithm")
        assert s.loc["EleWaiter AI", "avg_wait"] < 1.6 * s.loc["Collective (LOOK-ETA)", "avg_wait"]
        assert s.loc["EleWaiter AI", "avg_wait"] < s.loc["Priority-Nearest", "avg_wait"]

    def test_emergency_logic_costs_routine_waits_only_modestly(self, cfg):
        df = run_benchmark(cfg, ["emergency_surge"], ["EleWaiter AI", "AI (no emergency logic)"],
                           range(500, 508), 400, 1.5)
        s = summarize(df).set_index("algorithm")
        assert s.loc["EleWaiter AI", "routine_avg_wait"] < 1.4 * s.loc["AI (no emergency logic)", "routine_avg_wait"]


# ------------------------------------------------------------------ baselines
class TestBaselines:
    def test_nearest_picks_closest(self, cfg):
        lifts = make_lifts(cfg)
        lifts[0].floor, lifts[1].floor, lifts[2].floor = 2, 8, 5
        p = Passenger(0, 3, 7, 0.0)
        assert Nearest().assign([p], [], lifts, 0.0, cfg).lift_for[0] == 0

    @pytest.mark.parametrize("cls", [FCFS, Nearest, Collective, PriorityNearest])
    def test_skip_full_and_out_of_service(self, cfg, cls):
        lifts = make_lifts(cfg)
        lifts[0].state = LiftState.OUT_OF_SERVICE
        lifts[1].passengers = [Passenger(900 + i, 1, 9, 0.0) for i in range(cfg.capacity)]
        out = cls().assign([Passenger(0, 3, 7, 0.0)], [], lifts, 0.0, cfg)
        assert out.lift_for[0] == 2

    @pytest.mark.parametrize("cls", [FCFS, Nearest, Collective, PriorityNearest])
    def test_no_assignment_when_nothing_available(self, cfg, cls):
        lifts = make_lifts(cfg)
        for l in lifts:
            l.state = LiftState.OUT_OF_SERVICE
        assert cls().assign([Passenger(0, 3, 7, 0.0)], [], lifts, 0.0, cfg).lift_for == {}


# ------------------------------------------------------------------ metrics
class TestMetrics:
    def test_percentile_ordering(self, cfg):
        r = run_once(cfg, "emergency_surge", "EleWaiter AI", 4, 300)
        assert r.avg_wait <= r.p95_wait <= r.max_wait

    def test_emergency_metrics_use_ground_truth(self, cfg):
        r = run_once(cfg, "emergency_surge", "Nearest", 4, 600)
        truth = len([1 for p in TrafficGenerator(cfg, 4).generate("emergency_surge", 600) if p.is_emergency])
        assert r.n_emergency == truth and r.em_avg_wait is not None
        assert 0.0 <= r.em_within_target <= 1.0

    def test_inference_precision_recall_are_reported_and_imperfect(self, cfg):
        r = run_once(cfg, "emergency_surge", "Nearest", 4, 900)
        assert 0 < r.infer_recall < 1.0       # code responses on wards are missed by context rules
        assert 0 < r.infer_precision <= 1.0

    def test_no_emergency_metrics_in_office(self):
        r = run_once(office_12(), "mixed", "Nearest", 2, 200)
        assert r.n_emergency == 0 and r.em_avg_wait is None


# ------------------------------------------------------------------ optimizer
class TestOptimizer:
    FK = dict(scenarios=("emergency_surge",), seeds=(1,), intensities=(1.0,), duration=150.0)

    def test_fitness_deterministic(self, cfg):
        assert fitness(DEFAULT_WEIGHTS, cfg, **self.FK) == fitness(DEFAULT_WEIGHTS, cfg, **self.FK)

    def test_ga_returns_bounded_weights_and_monotone_history(self, cfg):
        best, hist = genetic_algorithm(cfg, pop_size=6, generations=3, fit_kwargs=self.FK)
        for k, (lo, hi) in WEIGHT_BOUNDS.items():
            assert lo - 1e-9 <= best[k] <= hi + 1e-9
        assert all(hist[i + 1] <= hist[i] + 1e-9 for i in range(len(hist) - 1))   # elitism

    def test_ga_never_worse_than_default(self, cfg):
        best, _ = genetic_algorithm(cfg, pop_size=6, generations=3, fit_kwargs=self.FK)
        assert fitness(best, cfg, **self.FK) <= fitness(DEFAULT_WEIGHTS, cfg, **self.FK) + 1e-9

    def test_sa_runs_and_improves_or_matches(self, cfg):
        best, hist = simulated_annealing(cfg, iterations=8, fit_kwargs=self.FK)
        assert hist[-1] <= hist[0] + 1e-9
