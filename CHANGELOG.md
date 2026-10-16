# Change log

Every change and fix made to this project since it was cloned on 27 Sep 2026,
newest first. Each entry says what changed, why, which files, and how it was
checked. Keep adding to the top of this file with every change.

Where to look for more:

- `README.md` — what the project is, setup, how to run it.
- `CLAUDE.md` — the full design log. Sections 1-4 are the original build
  (GNN and the six agents); sections 5-11 are the work recorded here, with
  the measurements behind each decision. Section 8 is the agreed plan and
  the standing data limitation.
- `git log` — the commits listed at the bottom of this file.

Branch: `aayanvikramsingh` on `adarshv1025/Capstone2.0`. `main` is unchanged
since the original two commits.

---

## 16 Oct 2026 — Handover

- **What:** `HANDOVER.md`: where everything is (branch, Drive data, deck,
  exported chat), setup steps, what is done (Phases 1-6), what is next
  (Phases 7-10 with estimates and Colab advice), the working rules, how to
  continue with Claude Code, open decisions and a security note. README and
  `CLAUDE.md` point to it.
- **Chat export:** the whole working session (27 Sep - 16 Oct) exported to
  `handover/` on Aayan's machine, outside git: `chat_conversation.md`
  (readable) and `chat_transcript_full.jsonl` (complete). The three Groq
  keys pasted during development are redacted; they should be revoked.
- **Why:** Aayan is away from 16 Oct; the team continues from here.

---

## 14 Oct 2026 — Phase 6: the Review 3 package

Details: `CLAUDE.md` section 13.

### Review 3 deck
- **What:** 19 slides in the PES template, built from the team's Review 1
  file: since Review 2 at a glance, the system as built, Review 1 agents ->
  the six, what we measured and what it corrected, completion audit, routing
  fixes and re-evaluation, Tsinghua benchmark, the closed loop (how and
  results), vehicle types and deadlines, multi-stop, distributed runtime,
  live demo, limitations, scope, roadmap. Native (editable) charts and
  speaker notes on every slide.
- **Where:** `ppt&docs/Review3/166_Capstone_Phase3_Review3.pptx` and `.pdf`
  in the team folder, with the builder `build_review3.py`. Not in git: it
  has the team's photos.
- **Why these slides:** the Review 2 deck (read in full) marked OR-Tools,
  LangGraph and an ST-GNN as done and reported results from the old solver;
  Review 3 corrects those openly and shows what has been built and measured
  since.
- **Checked by:** every slide rendered through PowerPoint and inspected
  (overlaps, overflow, labels fixed); the file passes the PowerPoint skill's
  validator against the template.

### Demo script
- **What:** `docs/review3_demo_script.md`: preparation, four demo steps with
  numbers measured with the LLM on and off, and likely panel questions with
  honest answers.
- **Checked by:** each step run through the app's automated check (LLM off)
  and once in the browser (LLM on).

### Audit re-scored
- Same checklist as 11 Oct, only changed items moved: nine stages 60% ->
  72%, all promises 55% -> 68% (`CLAUDE.md` 13.3).

### App: wording when the planner is not told
- With "Tell the planner" off, the incident message said the incident "is
  reported". It now says nobody reports it. `app_pages/single_delivery.py`.

---

## 14 Oct 2026 — The two app pages named clearly

- **What:** `streamlit run live_app.py` now opens a sidebar with two clearly
  named pages: **Single delivery: source → destination** (the original page,
  with "On the road" disruptions and the algorithm comparison; opens by
  default) and **Multi-stop fleet planner**. Each page links to the other.
- **Why:** adding the second page made Streamlit list the original one only
  as "live app" (its file name), so it looked as if the source/destination
  planner and the disruption controls had disappeared.
- **Files:** `live_app.py` is now only the entry point that lists the pages
  (`st.navigation`); the page code moved to `app_pages/single_delivery.py`
  (was `live_app.py`) and `app_pages/fleet_rounds.py` (was
  `pages/1_Multi-stop_fleet_planner.py`). Same command to run the app.
- **Checked by:** automated checks through the entry point: 15 of 15 for the
  Phase 5 features on both pages and 150 of 150 for "On the road"; the
  sidebar viewed in the browser.

---

## 14 Oct 2026 — Phase 5: vehicle types, deadlines, multi-stop, distributed runtime

Details, sources and full tables: `CLAUDE.md` section 12.

- **Vehicle-specific carbon** (`agents/vehicles.py`, `Vehicle.vehicle_type`): 6 types. Combustion
  factors from the India GHG Program (2015): diesel LDV 0.307, MDV 0.593, diesel 3W 0.132, CNG 3W
  0.108, petrol scooter 0.033 kg CO2/km, against the GNN's 0.27 reference. EV CO2 = kWh x 0.727
  (CEA v20.0, FY 2023-24). `FleetCosts` routes each request on its vehicle's own costs: combustion
  routes never change (300 of 300 identical), EV routes change in 7-47% of trips. Untyped vehicles
  behave exactly as before. Dispatcher prefers a cleaner vehicle (600 s per kg CO2) only when the
  fleet has types.
- **Deadlines** (`deadline_s` = seconds after leaving the origin): Coordinator moves weights toward
  time-only in 25% steps until met; Carbon Optimizer hard rule (seen live: the LLM accepted a greener
  swap, the rule rejected it for the deadline); Execution `deadline_met`; ESG `deadline_met_rate`;
  closed-loop `deadline` rule.
- **Multi-stop with OR-Tools** (`agents/multi_stop.py`, `DispatcherAgent.plan_tours`): payload, time
  windows, battery; baselines nearest-neighbour and one trip per delivery. 20 rounds x 10 stops
  (`multi_stop_experiment.py`): mixed fleet, greenest: 18.1 kg / 260 min vs nearest-neighbour 31.5
  kg / 271 min (59 of 100 deadlines) vs separate trips 57.4 kg / 583 min; all-diesel fleet 37.5 kg,
  so the mixed fleet halves CO2. Deadlines 100/100 expected, 97-99/100 in simulated drives.
- **Distributed runtime** (`agents/cluster.py`, Dask): worker processes over TCP, local or on other
  machines. `cluster_benchmark.py`, same answers every time: GNN on 381 snapshots 43.8 s serial,
  x1.20 on 2 workers, x1.54 on 4 (R2 0.9488/0.9687/0.9431/0.9445 reproduced); re-plan experiment
  120 trips 304 s serial, 145 s on 4 workers (x2.09). `replan_experiment.py --workers N`.
  Bug fixed: `client.close()` left local workers running (now `release()`). Not tested on 2 machines.
- **Label check / correction:** on test snapshots the time label rule holds on 97% of segments,
  carbon 95%, EV 100%. Segments with `event_disruption` > 0 are labelled ~25% FASTER, so the GNN's
  incident response follows the data (corrected in CLAUDE.md 11.3 and both disruption modules).
- **App:** vehicle selector and deadline input on the main page; new page
  `pages/1_Multi-stop_fleet_planner.py`; shared loaders moved to `app_common.py`.
- **Tests:** `tests/test_fleet.py` 11/11, `tests/test_cluster.py` 2/2 (separate scheduler and
  workers over TCP), `test_feedback_loop.py` 16/16, `test_bmssp.py` 6/6; app checks pass on both
  pages; one full LLM pipeline run with vehicle types and a deadline.
- **Requirements:** `ortools`, `dask[distributed]`.
- **Future work, written up for the guide to confirm** (`CLAUDE.md` 12.6): Neo4j, LangGraph,
  30-second live rebuild (computation takes ~0.12 s per snapshot; the missing piece is live
  data), GNN retraining from telemetry, edge-cloud deployment.
- **Docs:** `CLAUDE.md` section 12 (and plan item 5 marked done), `README.md`.

---

## 14 Oct 2026 — Disruption controls easier to find

### Live app: pointers to "On the road"
- **What:** the sidebar has a new item "5 · Disruption on the road" that
  says where the disruption controls are, with a link; after a route is
  planned, a note under the headline figures links to the same place; the
  section heading has the anchor `#on-the-road`.
- **Why:** the controls for introducing a disruption mid-trip are in the
  main page, below the baseline comparison, and only appear after "Plan
  route". Looking in the sidebar, they could not be found.
- **Files:** `live_app.py`.
- **Checked by:** in the browser: the sidebar item and the note appear, and
  the link jumps to the section.

---

## 13 Oct 2026 — Phase 4: the closed feedback loop

Details, measurements and the full results tables: `CLAUDE.md` section 11.

### The loop from Execution back to routing: `agents/feedback_loop.py`
- **What:** a vehicle drives its planned route, conditions change part-way,
  and the Execution agent decides whether to re-plan from the junction the
  vehicle is at. Both futures (finish the plan, or switch) are simulated
  with the same luck on every road they share, so the reported saving is the
  effect of the decision. Fixed rules decide first (route unchanged, EV
  battery, no faster and no cleaner, gain under 5%); the LLM is asked only
  when none of them settles it, and a fixed policy answers when the LLM is
  off or unreachable.
- **Why:** the architecture has always shown an arrow from Execution back to
  routing. In the code the Execution agent simulated the whole trip and then
  set a "re-plan" flag that nothing read. Phase 4 of the plan.
- **Files:** `agents/feedback_loop.py` (new: `drive`, `ClosedLoopExecutor`,
  `FeedbackStore`, `fleet_learning`).
- **Checked by:** `python tests/test_feedback_loop.py` (16 of 16) and
  `python agents/feedback_loop.py` (smoke test, no key needed).
- **Not a seventh agent:** it is the Execution/Monitoring agent's job
  extended. The six-agent pipeline and the evaluation harness are unchanged.

### What can change on the road: `agents/disruptions.py`, `gnn/snapshot_disruptions.py`
- **What:** two kinds of change. GNN-level: edit the snapshot and score it
  again with the GNN (rain across the city, a congestion surge on chosen
  roads, or the next snapshot of the day). World-level: a road incident as an
  assumed slowdown (default 4x) on the segments around a point.
- **Why two kinds:** measured, the GNN cannot model an incident. Raising its
  `event_disruption` feature made it predict the affected roads 10-24%
  faster. So an incident is stated as an assumption, never as a prediction.
- **Also measured:** rain gives time x1.18-1.22 and a clear rise in delay
  probability; a congestion surge does nothing at rush hour (congestion is
  already 0.98-0.99) and x1.14 / x1.28 at 13:00 / 21:00; the hour-to-hour
  snapshot change is close to one city-wide factor.
- **Caveat:** rain is rare in training and absent from the test split, so
  the rain response cannot be checked against labelled rain data.

### Solver: `path_totals()` and `reference_costs`
- **What:** `RoutingSolver.path_totals(path)` prices any path (it was
  private to `find_route`). `RoutingSolver(costs_after,
  reference_costs=costs_before)` prices changed roads in the original plan's
  units: unchanged roads cost exactly what they did, a road now r times
  worse costs r times more.
- **Why:** the solver scales costs between percentiles and caps them at 1.0,
  so a road that was already long and became 4x slower looked no worse than
  before, and the planner would not avoid it.
- **Files:** `gnn/routing_solver.py`. Default behaviour is unchanged.
- **Checked by:** tests for both; `tests/test_bmssp.py` still 6 of 6; 200
  of 200 random routes identical to the previous commit's solver, so every
  earlier result still holds.

### Execution agent: noise model shared
- **What:** the per-segment noise (jitter, delay, carbon and energy
  following time) is now the function `simulate_edge(row, rng)`, used by both
  the Execution agent and the loop. The module note about re-planning being
  "out of scope" now points to the loop.
- **Files:** `agents/execution_agent.py`.
- **Checked by:** 117 of 117 seeded runs give the same output as before.

### Learning from telemetry: `FeedbackStore`
- **What:** every drive records expected and actual time per segment. A
  segment that keeps taking far longer than predicted gets a correction
  factor, and the next vehicle is planned on corrected costs. The threshold
  comes from the noise model: one unlucky but normal report can never
  create a correction; a 4x incident does on its first report.
- **Why:** this is the loop for the case where nobody reports the problem.
- **Limits:** per segment only, not saved by the app, not fed into GNN
  retraining. A correction also never expires: once a road is flagged,
  vehicles avoid it, so no new reports arrive and it stays flagged after the
  road clears. That needs an expiry or a probe vehicle before real use.

### Experiment: `agents/replan_experiment.py`
- **What:** 120 random trips, each under five kinds of change, re-plan
  against no re-plan with paired luck; plus the unreported-incident case with
  five vehicles in a row. No LLM, so it is free and repeatable (about 6.5
  minutes). Results in `agents/evaluation_results/replan_experiment.csv`,
  `replan_summary.csv`, `fleet_learning_experiment.csv`.
- **Results:** nothing changed: 0 of 120 re-planned. Incident: 93%
  re-planned, 9.0 min and 1.65 kg CO2 saved per re-plan, trips 15%+ late
  down from 75% to 18%. Rain: none re-planned (77% late either way).
  Congestion surge: 44% re-planned, 5.1 min saved per re-plan. Next
  snapshot: none re-planned. Unreported incident: the second vehicle already
  takes the fully informed route in 68% of trips, and vehicles 2-5 recover
  94% of the gap.
- **How to read it:** re-planning helps for local changes and not for
  city-wide ones, because when every road slows by the same factor the best
  road is still the best. The incident is put on the vehicle's route on
  purpose and its 4x is assumed, so these are "what the loop does when
  something hits the route", not how often that happens.

### Live app: "On the road"
- **What:** every planned route gets a section where the trip is driven
  through a change: pick what happens, when, and whether the planner is
  told. The page shows the decision and who made it, the promised time
  against finishing the plan and against the re-plan, CO2, a map with the
  driven part, the route taken, the route not taken, the incident and the
  vehicle, the telemetry, and (when the planner is not told) what the next
  five vehicles do. It says which numbers are assumed and which are GNN
  predictions.
- **Files:** `live_app.py`.
- **Checked by:** automated app check, 3 scenarios x 5 kinds x told / not
  told, LLM off: 150 of 150. One browser run with the LLM on: Peenya to
  Rajajinagar, incident at 30% of the trip, 33.3 min finishing the plan
  against 25.0 min re-planned (promised 22.3), decided by the LLM.

### Two findings to carry forward
- **The LLM agreed with the fixed rule every time it was asked** (16 of 16
  on demo trips). All were faster and cleaner at once. Its contribution to
  the re-plan decision is the explanation, so far; Phase 8 should measure
  that rather than assume it.
- **A late trip with no new information is correctly not re-planned.** The
  rest of a best route is still the best route from any point on it.

### Fixed along the way
- `FeedbackStore.corrected()` raised on the GNN's float32 columns. Now goes
  through float64; covered by a test.

### Tests
- `tests/test_feedback_loop.py`: 16 tests, no LLM, no torch.

---

## 13 Oct 2026 — Algorithm comparison shown in the app

### Live app: "Dijkstra vs Tsinghua BMSSP" panel
- **What:** every planned route now has a section comparing the two
  shortest-path algorithms on that trip: search time, road segments
  examined, how each one searches, the work behind it, its theoretical
  running time, and a plain statement that both found the same route. Below
  it, a chart from the benchmark shows how many times slower the Tsinghua
  algorithm is as graphs grow, with the numbers available as a table.
- **Why:** both algorithms return the same route, so selecting one or the
  other changed nothing visible on the page except one small grey line. The
  real differences (how they search, how much work, how they scale) were not
  shown anywhere.
- **Files:** `live_app.py`, `gnn/routing_solver.py` (new
  `compare_algorithms()`), `gnn/bmssp.py` (new `dijkstra_reference()`, a
  counting Dijkstra used for the like-for-like comparison).
- **Checked by:** automated app check on the six scenarios with both
  algorithms (12 of 12 pass, same route every time); viewed in the browser in
  the dark theme. Chart colours validated for colour-blind separation in
  light and dark themes.
- **Example (HSR Layout to Whitefield, balanced):** same 53-segment route;
  Dijkstra examined 14,957 road segments in about 17 ms, the Tsinghua
  algorithm 54,207 in about 130 ms.
- **Replaces:** the earlier one-line caption comparing the two times, which
  only appeared when the Tsinghua algorithm was selected.
- **Note for whoever runs the app:** after pulling new code, restart
  `streamlit run live_app.py`. A page refresh alone reloads `live_app.py` but
  not the modules it imports, and gives an AttributeError.

---

## 12 Oct 2026 — Phase 3: the Tsinghua shortest-path algorithm

Details and the full benchmark table: `CLAUDE.md` section 10.

### New solver: `gnn/bmssp.py`
- **What:** our own implementation of the algorithm from Duan, Mao, Mao, Shu
  and Yin, "Breaking the Sorting Barrier for Directed Single-Source Shortest
  Paths" (arXiv:2504.17033), written from the paper's Algorithms 1-3 and
  Lemma 3.3. Selectable next to Dijkstra with
  `RoutingSolver(costs_df, algorithm="bmssp")` or the `ROUTING_ALGORITHM`
  environment variable. Dijkstra stays the default.
- **Why:** the team told the review panel this algorithm would be added.
- **Files:** `gnn/bmssp.py` (new), `gnn/routing_solver.py`.
- **Checked by:** `tests/test_bmssp.py` (6 tests, all pass) and a stress run
  of 400 random graphs with 0 mismatches against Dijkstra. On 174 real
  routes and the six demo scenarios it returns the identical route.
- **Three differences from the paper, on record:** no constant-degree
  rewrite of the graph (ours already has degree at most 4); the queue uses
  sorting and a sorted list where the paper uses linear-time selection and a
  balanced tree, so the code does not strictly reach the paper's time bound;
  and one added safeguard that deletes a stale queue entry the pseudocode
  leaves behind (it fires about 220 times per search on our graph; without
  it results are still correct but some junctions are processed twice).

### Benchmark: `gnn/benchmark_sssp.py`
- **What:** Dijkstra against the new algorithm on the Bengaluru graph and on
  test graphs from 1,000 to 262,144 junctions. Results in
  `gnn/benchmark_results/sssp_benchmark.csv`.
- **Result:** the new algorithm is slower at every size tested: 14-22x on
  the Bengaluru graph (about 465-586 ms against 26-36 ms per full search).
  On grid graphs the gap narrows from 11.4x at 1,024 junctions to 7.0x at
  262,144. It is asymptotically faster, not faster in practice at city
  scale.

### Tests folder started
- **What:** `tests/test_bmssp.py`, runnable with pytest or as a script, no
  LLM or API key needed.

### Live app: algorithm selector
- **What:** "Shortest-path algorithm" choice in the sidebar. With the
  Tsinghua algorithm selected, the page runs the same query with both
  algorithms and shows both times and whether the route is identical.
- **Files:** `live_app.py`.
- **Checked by:** automated app check with both algorithms on the six
  scenarios (12 of 12 pass, identical figures), and one full LLM run with
  the new algorithm selected.

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

### Live app: headline figures in two rows
- **What:** the six headline figures (distance, planned time, expected time,
  90% arrival, CO2, EV energy) are laid out in two rows of three instead of
  one row of six.
- **Why:** in a laptop-width or split-screen window the values were cut off
  ("16.6 ...").
- **Files:** `live_app.py`.
- **Checked by:** viewed in the browser at 800 px width; six scenarios pass
  the automated app check with the LLM off.

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
- **13 Oct 2026:** a road incident is modelled as an assumed slowdown, not
  through the GNN, because the GNN predicts incident roads to be faster
  (`CLAUDE.md` 11.3). Always present it as an assumption.
- **14 Oct 2026:** Phase 5 built in full (the user asked for all of it).
  Neo4j, LangGraph, the 30-second rebuild, retraining from telemetry and
  edge-cloud deployment are written up as future work; the guide still has
  to confirm that scope.

## What comes next

Phases 1 to 6 of the plan in `CLAUDE.md` section 8.2 are done. Next is the
paper work: Phase 7 (strengthen the GNN: baselines, forecasting version,
real vs interpolated accuracy), Phase 8 (paper-grade evaluation), Phase 9
(pytest and CI), Phase 10 (the paper). Rehearse the Review 3 demo with
`docs/review3_demo_script.md`.

## Commits

| Commit | Date | Contents |
|---|---|---|
| `4fadb77`, `9fa2b48` | before 27 Sep | Original project (Adarsh): GNN and the six agents. On `main`. |
| `1e85246` | 27 Sep | Live planner, GNN inference fixes, first evaluation re-run. |
| `e43855f` | 11 Oct | Agent reliability fixes, demo scenarios, delay figures. |
| `0a83696` | 11 Oct | README, measured audit, evaluation re-run on fixed agents. |
| `00b0cf3` | 12 Oct | Phase 2 code: routing fixes, expected-value planning, EV range fix, repeatable and logged LLM calls. |
| `fdb8497` | 12 Oct | Evaluation re-run, regenerated dashboard, `CHANGELOG.md`, updated `README.md` and `CLAUDE.md`. |
| `8885af5` | 13 Oct | Phase 3: Tsinghua BMSSP solver, solver option, tests, benchmark and results. |
| `8b14c12` | 13 Oct | Live app: algorithm selector, Dijkstra vs BMSSP comparison panel, two-row headline figures. |
| `2fc4eff` | 13 Oct | Change log, `CLAUDE.md` section 10 and `README.md` for Phase 3 and the comparison panel. |
| `22013bf` | 14 Oct | Phase 4 code: closed feedback loop, disruption models, solver additions, re-plan experiment and results, app "On the road" section, tests. |
| `dfa4f6b` | 14 Oct | Change log, `CLAUDE.md` section 11 and `README.md` for Phase 4. |
| `91e5f33` | 14 Oct | Phase 5 code: vehicle types, deadlines, OR-Tools multi-stop, Dask runtime, experiments and results, app page and controls, tests. |
| `774570b` | 14 Oct | Change log, `CLAUDE.md` section 12 (and the 11.3 correction), `README.md` for Phase 5. |
| `27b4c6c` | 14 Oct | App pages named clearly (`live_app.py` entry point, `app_pages/`); commit table updated in the next commit. |
| `6746fca` | 14 Oct | Change log commit table. |
| `5855059` | 16 Oct | Phase 6 (demo script, app wording fix, `CLAUDE.md` 13) and `HANDOVER.md`. |
