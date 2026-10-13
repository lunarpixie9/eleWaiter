# 🛗 EleWaiter: emergency-first intelligent lift dispatch

A Python simulator and AI dispatcher that decides **which lift serves which passenger**, built for hospitals where an emergency transfer (ER to ICU/OT) must not wait behind routine traffic, while visitors and staff are not starved. It also works for any building (office and college presets are included).

No hardware, no extra buttons or cards: urgency is **inferred from context** (which floors a trip touches, and when).

## Quick start

```bash
pip install -r requirements.txt
streamlit run ui/app.py                 # interactive UI
python run_benchmark.py --ablation      # multi-seed benchmark, hospital scenarios
python run_benchmark.py --building "12-floor Office"
python train_optimizer.py               # re-tune the AI weights (about 5 min)
pytest -q                               # 129 tests
```

## The problem

For each call, choose a lift so that average wait is low, **nobody is starved or repeatedly left behind**, and **emergencies get through fast**, without wrecking everyone else's service. Rules like "nearest lift" or first-come-first-served ignore direction, crowding and urgency.

## How it works

**Urgency from context (no new hardware).** Every floor has a zone (LOW / NORMAL / HIGH / CRITICAL; ER, ICU and OT are CRITICAL). A trip leaving a CRITICAL zone for a HIGH/CRITICAL zone (ER to ICU/OT/imaging) is inferred **critical**; other trips touching a HIGH/CRITICAL zone are **elevated**. Zones can be upgraded in time windows (e.g. lecture floors before class). Waiting time raises priority automatically.

**Dispatchers** (`elewaiter/dispatchers/`)

| Dispatcher | Idea |
|---|---|
| FCFS | calls in arrival order, to the lift with fewest stops |
| Nearest | closest lift by floors |
| Collective (LOOK-ETA) | lift that will reach the caller soonest given its direction and stops (strong conventional baseline) |
| Priority-Nearest | nearest lift + hand-written emergency rules (one lift rushed per emergency, priority boarding) |
| **EleWaiter AI** | the system below |
| AI (no emergency logic) | ablation: same AI with urgency, lanes and priority boarding switched off |

**EleWaiter AI**
1. **Multi-objective cost** (`planner.py`): the passenger's own wait and ride time, scaled by an urgency weight, plus the delay imposed on people already planned, crowding, and **predicted lift load at pickup** (who will have exited by then), with a large penalty if the lift would be full (left-behind risk).
2. **Urgency weight** `W = 1 + a_urg*urgency + a_age*log(1+waited/15) + a_lb*times_left_behind`. It multiplies only the passenger's *own* terms, so urgent or long-waiting passengers accept more delay to others. (A constant per-passenger bonus would never change which lift wins.)
3. **A\* search** over joint assignments of pending calls, with the cheapest-single-option sum as heuristic and a node cap with greedy fallback. Not-yet-boarded passengers are re-optimised, with a switching penalty against thrashing.
4. **Emergency lane**: a critical trip can get a lift committed to it (straight to the caller, straight to the destination, no routine pickups). Several critical passengers can share one lane. Whether to start a lane is decided **by cost**, including a backlog term, against normal priority service; this is what stops a burst of emergencies from stalling the building.
5. **Fairness**: aging priority, a boost for passengers a full lift already bypassed, priority boarding when space is short, and a hard **starvation guard** that force-assigns anyone waiting 45 s or more.
6. **GA / simulated annealing** tune the 13 weights using the simulator as the fitness function.

**Simulator** (`simulator.py`): discrete-event, lifts move one floor at a time (so they can stop en route and never teleport), capacity is enforced per passenger, full lifts bypass calls, LOOK/collective movement.

## Scenarios and buildings

Hospital: **emergency surge**, **mass-casualty burst** (14 emergencies in about a minute), **visiting hours**, **shift change**, **lift failure**. General (any building): up-peak, down-peak, mixed. Buildings: 10- and 15-floor hospitals, 12-floor office, 8-floor college.

Emergencies in the traffic are **ground truth that dispatchers never see**; they include patient transfers (ER to ICU/OT/imaging, OT to ICU) and code-response trips to ordinary wards that floor rules cannot recognise. Routine traffic scales with `intensity`; emergencies are an independent process.

## Results (held-out seeds)

Protocol: weights trained on seeds 1000-1002, winner picked on validation seeds 3000-3003, **all numbers below use unseen seeds 2000-2009**; 10-floor hospital, 600 s, routine intensity 1.5. Every dispatcher sees the identical passenger stream per seed. Overall = mean of the five hospital scenarios.

| Dispatcher | Avg wait (s) | p95 wait (s) | Starved (>=45 s) | Emergency avg wait (s) | Emergency p95 (s) | Emergencies <= 20 s | Routine avg wait (s) |
|---|---|---|---|---|---|---|---|
| FCFS | 17.7 | 43.1 | 15.8 | 17.5 | 32.2 | 64% | 17.7 |
| Nearest | 15.8 | 54.4 | 31.4 | 22.5 | 41.4 | 58% | 15.7 |
| Collective (LOOK-ETA) | 9.7 | 30.8 | 7.1 | 11.7 | 23.1 | 80% | 9.6 |
| Priority-Nearest | 21.3 | 71.3 | 46.9 | 12.8 | 28.9 | 82% | 21.4 |
| **EleWaiter AI** | 8.2 | 24.1 | 2.9 | 6.2 | 12.2 | 96% | 8.2 |
| AI (no emergency logic) | 7.9 | 22.8 | 1.2 | 9.0 | 17.0 | 86% | 7.9 |

Per scenario (mean of 10 seeds):

| Scenario | Metric | Collective | Priority-Nearest | **EleWaiter AI** |
|---|---|---|---|---|
| Emergency surge | avg wait (s) | 6.7 | 13.8 | **6.0** |
| | emergency wait (s) | 6.5 | 7.0 | **5.1** |
| | emergencies <= 20 s | 97% | 94% | **98%** |
| Mass casualty | avg wait (s) | 6.6 | 18.1 | **5.4** |
| | emergency wait (s) | 7.1 | 18.5 | **4.3** |
| | emergencies <= 20 s | 90% | 71% | **99%** |
| Visiting hours | avg wait (s) | 6.9 | 14.4 | **5.8** |
| | emergency wait (s) | 9.7 | 7.2 | **4.4** |
| | emergencies <= 20 s | 82% | 95% | **100%** |
| Shift change | avg wait (s) | 10.0 | 23.5 | **8.5** |
| | emergency wait (s) | 13.3 | 12.5 | **7.3** |
| | emergencies <= 20 s | 79% | 86% | **98%** |
| Lift failure | avg wait (s) | 17.9 | 36.7 | **15.3** |
| | emergency wait (s) | 21.8 | 19.0 | **9.9** |
| | emergencies <= 20 s | 53% | 66% | **87%** |

**What these numbers show**
- The AI has the lowest emergency wait in every hospital scenario, and the highest share of emergencies boarding within 20 s, while keeping routine waits close to the best conventional method.
- Naive "always rush emergencies" (Priority-Nearest) buys emergency speed with a large routine-wait penalty (21 s vs 8 s) and mass starvation.
- Emergency logic costs the routine passengers little: AI vs its own ablation is 8.2 s vs 7.9 s average wait (about 4%), for emergency waits of 6.2 s vs 9.0 s.

**Load sweep** (emergency surge, 8 seeds): the emergency advantage grows with load.

| Intensity | Collective: emergency wait / within 20 s | Priority-Nearest | EleWaiter AI |
|---|---|---|---|
| 1.0 | 4.8 s / 98% | 4.5 s / 99% | 3.8 s / 100% |
| 1.5 | 6.0 s / 100% | 6.9 s / 94% | 5.5 s / 97% |
| 2.0 | 12.9 s / 78% | 9.9 s / 89% | 7.8 s / 90% |
| 2.5 | 20.0 s / 66% | 15.5 s / 77% | 13.9 s / 77% |

At light load a good conventional method is already close; do not expect large gains there. Average wait at 2.5x: AI 14.6 s vs Collective 22.8 s vs Priority-Nearest 62.6 s.

**Other buildings** (office, college; no emergencies): the AI had the lowest average wait, p95 and starvation in every general scenario (e.g. office up-peak 5.2 s vs 5.9 s for Collective and 15.3 s for Nearest).

**What the optimizer bought** (default vs tuned weights, 8 held-out seeds, hospital scenarios; on validation seeds the GA improved the composite fitness by about 10%, 75.7 to 67.9, versus 25% on its own training seeds, so expect modest transfer): average wait -24%, p95 -23%, starvation -63%, but emergency wait unchanged (6.3 s to 6.4 s). Tuning improved fairness and routine service. It also taught the AI to start emergency lanes **sparingly** (0.6-2.3% of door openings vs 4.9-12.1% with hand-set weights): most of the emergency benefit comes from urgency-weighted costs and priority boarding, with lanes used when cheap.

## Honest limitations

- **Simulated traffic only.** Results show smarter dispatch helps under the traffic models here; they are not proof for a real building. Real hospital lift traffic, emergency rates and layouts would need calibration.
- **Context-based urgency is imprecise.** Across the held-out runs, trips inferred as critical were real emergencies only about **23%** of the time (many routine staff moves between critical floors look the same), and the rules recognised about **77%** of real emergencies (code responses to wards are missed). The AI still performs well because decisions are cost-based and selective, but better signals (e.g. a code-alert feed) would help.
- Collective control is hard to beat on average wait at light load: the AI's win over it there is small (it wins on average wait in 6-10 of 10 seeds depending on scenario, and on emergency wait in 5-8 of 10).
- Assumptions: destinations are known at call time (destination-dispatch style); passengers board a lift regardless of its direction of travel; one fixed starvation threshold (45 s); the tuned weights were trained on the 10-floor hospital only and reused unchanged for the other buildings.
- No animated lift view (the activity chart shows door openings and emergency-lane trips).

## What changed from v1

The first version ran but had defects that made its benchmark untrustworthy. All are fixed and covered by regression tests: lifts could carry up to 19 people with capacity 10 (capacity was checked once per stop, not per passenger); lifts teleported to nearer stops on 8-16% of arrivals; the starvation guard could never fire; the "priority" and "starvation" cost terms were identical for every lift, so tuning them could not change any decision; the optimizer's convergence curve re-seeded traffic every generation; and "priority delay" was a confounded group difference. The repository also contained stray folders with literal brace names from a failed shell expansion.

## Layout

```
elewaiter/
  models.py        zones, urgency inference, passenger, lift
  buildings.py     hospital / office / college presets
  traffic.py       Poisson traffic + scenarios + ground-truth emergencies
  simulator.py     discrete-event engine (floor-by-floor, lanes, starvation guard)
  planner.py       ETA / predicted-load model and cost functions
  dispatchers/     baselines.py, ai.py (A* search), base.py
  optimizer.py     genetic algorithm + simulated annealing
  metrics.py       wait, fairness, emergency and inference-quality metrics
  benchmark.py     multi-seed benchmark helpers
  charts.py        Plotly charts
  data/            tuned_weights.json, training_log.json
ui/app.py          Streamlit UI
tests/             129 tests (invariants, regressions, planner, A*, emergency logic)
run_benchmark.py   train_optimizer.py
```
