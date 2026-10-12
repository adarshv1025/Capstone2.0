# Change log

Every change and fix made to this project since it was cloned on 27 Sep 2026,
newest first. Each entry says what changed, why, which files, and how it was
checked. Keep adding to the top of this file with every change.

Where to look for more:

- `README.md` — what the project is, setup, how to run it.
- `CLAUDE.md` — the full design log. Sections 1-4 are the original build
  (GNN and the six agents); sections 5-9 are the work recorded here, with the
  measurements behind each decision. Section 8 is the agreed plan and the
  standing data limitation.
- `git log` — the commits listed at the bottom of this file.

Branch: `aayanvikramsingh` on `adarshv1025/Capstone2.0`. `main` is unchanged
since the original two commits.

---

## 12 Oct 2026 — Phase 2 finished: EV range fix, evaluation re-run

### EV range is planned on expected energy
- **What:** the Coordinator now checks an EV's battery against the route's
  *expected* energy use (typical delays included) instead of the delay-free
  figure. The solver returns `expected_ev_energy_pct`; `RouteResult` carries
  it; the live app shows expected EV energy and bases its battery message on
  it.
- **Why:** the simulated drive uses about 35% more energy at rush hour than
  the delay-free plan. A route planned at 75-100% of its battery passed the
  Coordinator and then failed the Execution agent's hard battery rule.
- **Files:** `gnn/routing_solver.py`, `agents/schemas.py`,
  `agents/route_agent.py`, `agents/coordinator_agent.py`, `live_app.py`.
- **Checked by:** Coordinator smoke test (EV case recalibrated to a 5.9 kWh
  battery: 6.07 kWh expected on the default route, 5.80 kWh after
  negotiation, so it resolves). Solver regression tests still pass.
- **Not done, on purpose:** the solver's `ev` objective still minimises
  delay-free energy. Minimising expected energy instead would save about 1%
  more on average (median 0%) over 200 random trips, so it was not worth the
  change. The check uses the mean with no reserve: a route planned exactly at
  the battery limit can still run out about half the time.

### Evaluation and dashboard re-run on the Phase 2 code
- **What:** all 5 snapshots re-run (5 scenarios x 3 systems), 75 of 75 runs
  completed; `dashboard.html` regenerated; results tables updated in
  `CLAUDE.md` 9.6 and `README.md`.
- **Why:** four of the Phase 2 fixes change which routes are chosen, so the
  earlier results no longer described the code.
- **Files:** `agents/evaluation_results/*.csv`, `agents/dashboard.html`,
  `CLAUDE.md`, `README.md`.
- **Results (expected values, typical delays included):**
  - Simulated drives land within about 3-6% of the promised time (28-48%
    before Phase 2).
  - EV feasibility on the tight-battery scenario: multi-agent 0.20 -> 0.60,
    level with the lowest-carbon baseline.
  - Multi-agent vs fastest baseline: 3.0% less CO2, 6.6% longer. Vs
    lowest-carbon baseline: 2.1% more CO2, 6.5% faster.
  - The lowest-carbon baseline beats the multi-agent system on both time and
    carbon in 1 of 5 scenarios, down from 3 of 5.
- **Cost:** 255 LLM calls, about 232,000 tokens (46,000 per snapshot), split
  across two Groq accounts.
- **Also checked:** 54 of 54 live-app checks; LLM-on results identical to the
  previous day's run, which confirms temperature 0 gives repeatable answers.

### This change log
- **What:** `CHANGELOG.md` created; linked from `README.md` and `CLAUDE.md`.
- **Why:** one place to see every change and fix, for handing the project
  over.

---

## 11 Oct 2026 — Phase 2: routing and repeatability fixes

Details and measurements: `CLAUDE.md` section 9.

### Routing graph is cached
- **What:** `RoutingSolver` builds its graph once and applies weights through
  a cost function, instead of rebuilding a weighted graph on every call.
- **Why:** rebuilding took about 380 ms of a 590 ms routing call; the
  shortest-path search itself takes about 6 ms.
- **Files:** `gnn/routing_solver.py`.
- **Checked by:** median `find_route()` 162 ms -> 3.7 ms. With the new delay
  features switched off, 150 test routes have identical cost to the old
  solver.

### Parallel roads: keep the faster one, everywhere
- **What:** new helper `canonical_edges()` keeps the faster road for each
  pair of junctions joined by two roads. The solver, the Execution agent and
  the live app all use it.
- **Why:** 83 junction pairs have two roads (for example a 9 m link and a
  1.2 km loop). The old code kept whichever came last in the file, which was
  the slower road in 53 of 83 pairs, by up to 105x.
- **Files:** `gnn/routing_solver.py`, `agents/execution_agent.py`,
  `live_app.py`.
- **Checked by:** solver test (pair 31 -> 5134 now uses the 1.7 s road, not
  the 175 s one).

### Delay cost is probability x time
- **What:** the `delay` objective is now `delay_probability * travel_time_s`
  per segment, not the bare probability.
- **Why:** summing a bare probability charged every segment about the same,
  so a high delay weight rewarded routes with fewer segments rather than less
  delay. Koramangala -> Hebbal with a delay-heavy weighting came out at
  57.5 min expected, against 38.3 min for the plain fastest route.
- **Files:** `gnn/routing_solver.py`.
- **Checked by:** the same delay-heavy weighting now gives 38.3 min.

### Plans promise the expected time, not the delay-free time
- **What:** the `time` objective is expected time (GNN time plus the expected
  cost of delays). Routes carry `expected_travel_time_s` and
  `expected_carbon_kg`. The Execution agent measures deviation against the
  expected time. The Carbon Optimizer compares expected times. The ESG
  Reporter reports expected totals and is told to judge actual against
  expected.
- **Why:** the simulation's own average delay showed up as a 30-45%
  "deviation" on almost every rush-hour route, so nearly everything was
  flagged for re-planning, and the ESG summary described normal delay
  emissions as a problem.
- **Files:** `gnn/routing_solver.py`, `agents/schemas.py`,
  `agents/route_agent.py`, `agents/execution_agent.py`,
  `agents/carbon_optimizer_agent.py`, `agents/esg_reporter_agent.py`,
  `agents/pipeline.py`.
- **Checked by:** 12-delivery batch over 6 seeds: simulated time is 98.8% of
  expected, simulated carbon 99.0%, mean per-delivery deviation 4.5%, no
  re-plans.
- **Side effect to know:** deviations are now small, so the Execution agent's
  LLM review rarely runs. It will be exercised again when Phase 4 adds
  mid-trip disruptions.

### LLM calls are repeatable and logged
- **What:** every LLM call is sent at temperature 0 by default
  (`LLM_TEMPERATURE`), and logged to `logs/llm_calls.jsonl` with prompt,
  reply, tokens and latency (`LLM_LOG_PATH`; git-ignored).
- **Why:** one baseline's on-time rate moved from 0.07 to 0.27 between two
  runs with identical routes, only because the LLM judged the same delays
  differently. The log also gives the token and cost numbers the paper needs.
- **Files:** `agents/llm_client.py`, `.gitignore`.
- **Checked by:** 41 calls logged at temperature 0 during the agent tests.

### Dashboard
- **What:** label "Total planned distance-time" corrected to "Total planned
  time (no delays)"; expected-time and expected-carbon cards added; the
  per-route chart compares expected against actual.
- **Files:** `agents/dashboard.py`.

### Live app
- **What:** uses the shared parallel-road rule; shows expected time against
  simulated time; the six demo scenarios were re-measured and rewritten.
- **Why:** the old "agents repair a bad plan" scenario (Whitefield -> Peenya,
  86.5 min) only existed because of the delay-cost bug.
- **Files:** `live_app.py`.
- **Checked by:** 54 of 54 app checks (six scenarios, LLM off and on).

### Plan and standing limitation written down
- **What:** `CLAUDE.md` section 8: the ten-phase plan, the real-data coverage
  limitation with three ways to address it, and the optional TomTom step
  (deferred).
- **Files:** `CLAUDE.md`.

---

## 11 Oct 2026 — Phase 1: measured audit, README, corrected claims

Details: `CLAUDE.md` section 7.

### Full audit (no code changed)
- **What was run:** the GNN on all 381 test snapshots; every agent's own
  test; the full evaluation; the live app on six scenarios; routing timings.
- **Result:** 9 architecture stages 39/65 = 60% complete; with cross-cutting
  promises 46/83 = 55%. GNN accuracy reproduced exactly (R^2 0.9488 / 0.9687
  / 0.9431 / 0.9445).
- **Findings that change what can be claimed:**
  - Real Kaggle traffic values cover 481 of 15,083 road segments (3.2%), not
    the "about 60%" stated before. The rest are interpolated.
  - The Kaggle data is daily; the 8 time-of-day snapshots per day come from a
    hand-set multiplier table.
  - A linear model on the raw road features beats the GNN on 3 of 4 targets;
    the GNN is ahead only on delay probability.
  - The lowest-carbon baseline was faster and greener than the multi-agent
    system in 3 of 5 scenarios.
  - The promised Tsinghua shortest-path algorithm is not implemented; the
    search is about 1% of a routing call.

### Corrected claim
- **What:** `CLAUDE.md` section 1 now states 3.2% real-data coverage and that
  the hourly pattern is modelled.
- **Files:** `CLAUDE.md`.

### README added
- **What:** `README.md`: what the project is, the six agents, setup, which
  Drive folders to copy where, run commands, results, known limitations.

### Evaluation re-run on the fixed agents
- **What:** all 5 snapshots, 75 of 75 runs completed; dashboard regenerated.
- **Files:** `agents/evaluation_results/*.csv`, `agents/dashboard.html`.

---

## 28 Sep 2026 — Demo preparation and agent reliability fixes

Details: `CLAUDE.md` section 6. Committed on 11 Oct.

### Agents silently lost LLM answers
- **What:** prompts now use a `<request_id>` placeholder, and a new
  `align_ids()` helper recovers a reply keyed by the wrong id.
- **Why:** the LLM copied the example id from the prompt (`req-0`) instead of
  the real one in about 2 of 6 replies. The agent then could not find the
  decision and fell back to "reject" or "no re-plan", once rejecting a swap
  that saved 0.5 kg for 1 second.
- **Files:** `agents/llm_client.py`, `agents/carbon_optimizer_agent.py`,
  `agents/execution_agent.py`, `agents/coordinator_agent.py`.

### Empty or invalid LLM replies are retried
- **What:** `chat_json()` asks again up to twice before giving up.
- **Files:** `agents/llm_client.py`.

### ESG summary reported 43% as "0.43%"
- **What:** the prompt now states that the deviation and rate metrics are
  fractions.
- **Files:** `agents/esg_reporter_agent.py`.

### Carbon Optimizer keeps an audit trail
- **What:** each decision records the original weights and reasoning and the
  candidate weights, so the Route Agent's own choice is still visible after
  an accepted swap.
- **Files:** `agents/carbon_optimizer_agent.py`.

### Live app: demo scenarios and delay figures
- **What:** six one-click demo scenarios with talking points; "P(any delay)"
  (always about 100%) replaced by "Expected with delays" and "90% arrive
  within".
- **Files:** `live_app.py`.

---

## 27 Sep 2026 — Live GNN and the interactive app

Details: `CLAUDE.md` section 5.

### GNN inference could not be imported
- **What:** restored the first 53 lines (docstring and imports) of
  `gnn/gnn_inference.py` from the Drive copy; the rest of the file was
  identical.
- **Files:** `gnn/gnn_inference.py`.

### Checkpoints would not load without a GPU
- **What:** `load_trained_model()` and `FeatureNormalizer.load()` pass
  `map_location="cpu"`. They were saved on a Colab GPU.
- **Files:** `gnn/gnn_trainer_v3.py`.
- **Checked by:** live predictions match the exported `sample_costs` CSVs to
  within 1.2e-4.

### Interactive planner
- **What:** `live_app.py` (Streamlit): pick source and destination on a map,
  choose traffic snapshot, priority and vehicle; runs live GNN inference and
  the six agents; shows the route against two baselines and each agent's
  reasoning. A switch runs it without the LLM.
- **Files:** `live_app.py`, `requirements-live.txt`, `.gitignore`.
- **Found along the way:** `bengaluru_graph/nodes.geojson` on the Drive has
  coordinates for every junction, in the same order as the GNN's node
  indices. The original notes said no coordinates existed.

### Evaluation re-run (first)
- **What:** all 5 snapshots re-run; dashboard regenerated.
- **Files:** `agents/evaluation_results/*.csv`, `agents/dashboard.html`.

---

## What is not in git

| Item | Where it comes from |
|---|---|
| `.env` (LLM model and API key) | Create it yourself; see `README.md`. Keys are never committed. |
| `gnn_checkpoints_v3/`, `dataset_kaggle/test/`, `bengaluru_graph/` (about 710 MB) | Team Google Drive, `Capstone/`; see `README.md` for the exact folders. |
| `logs/llm_calls.jsonl` | Written automatically when the agents run. |
| `.venv/` | `pip install -r requirements.txt` (and `requirements-live.txt` for the app). |

## Decisions on record

- **7 Oct 2026:** the team continues only with this repo's six agents
  (Dispatcher, Route Agent, Coordinator, Carbon Optimizer, Execution, ESG
  Reporter), not the Fleet/Trust/EV/ESG agents shown in the Review 1 deck.
- **11 Oct 2026:** real traffic from the TomTom API is optional and deferred
  (`CLAUDE.md` 8.3).
- **11 Oct 2026:** the Tsinghua shortest-path algorithm was promised to the
  review panel and is Phase 3 of the plan.

## What comes next

Phase 3 onward of the plan in `CLAUDE.md` section 8.2: the Tsinghua solver,
the feedback loop with mid-trip disruptions, the remaining promises, the
Review 3 package, then the paper work.

## Commits

| Commit | Date | Contents |
|---|---|---|
| `4fadb77`, `9fa2b48` | before 27 Sep | Original project (Adarsh): GNN and the six agents. On `main`. |
| `1e85246` | 27 Sep | Live planner, GNN inference fixes, first evaluation re-run. |
| `e43855f` | 11 Oct | Agent reliability fixes, demo scenarios, delay figures. |
| `0a83696` | 11 Oct | README, measured audit, evaluation re-run on fixed agents. |
| `00b0cf3` | 12 Oct | Phase 2 code: routing fixes, expected-value planning, EV range fix, repeatable and logged LLM calls. |
| `fdb8497` | 12 Oct | Evaluation re-run, regenerated dashboard, `CHANGELOG.md`, updated `README.md` and `CLAUDE.md`. |
