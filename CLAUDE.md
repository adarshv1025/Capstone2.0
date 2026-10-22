# Bengaluru Logistics Framework — Project Context & Build Plan

> **Continuing this project (handover of 16 Oct 2026):** read `HANDOVER.md` first. Sections 7-13 below record
> everything done from 27 Sep to 16 Oct; `CHANGELOG.md` lists each change.

This file is project memory for Claude Code. Read it fully before making changes.

**Every change and fix is also listed, newest first, in `CHANGELOG.md` — add an
entry there with each change.** Section 8 below is the current plan and the
standing data limitation.

## 1. Project Overview

Capstone project: **"Intelligent Decentralized Logistics Framework"** for Bengaluru,
built as a 4-person team. 9-layer pipeline:

```
real-world data -> ingestion -> feature engineering -> graph construction ->
GNN inference -> multi-agent decisions -> carbon-aware optimizer ->
execution -> ESG dashboard
```

**My scope:** originally owned the GNN inference layer individually. Now building
the **entire remaining pipeline** (multi-agent decisions through ESG dashboard)
independently, in parallel with my teammates' own implementation of the same
layers. Whichever version performs better on the evaluation harness (Section 3.5,
phase 8) is what gets kept for the final submission — so build for correctness
and measurability, not just "something that runs."

**Graph:** Bengaluru OSM road network filtered to motorway/trunk/primary/
secondary/tertiary roads — 7,095 nodes, 15,083 edges. Dynamic (traffic-dependent)
features come from a Kaggle Bengaluru traffic dataset covering 16 named roads.
**Corrected 2026-10-11 (was "~60% geo-matched / ~40% interpolated", which is
wrong — see 7.2):** only 481 of the 15,083 edges (3.2%; 13 of the 16 Kaggle
roads) are name-matched to real Kaggle values; the other ~96.8% are filled by
spatial interpolation from those matched edges. The Kaggle data is daily, so
the 8 time-of-day snapshots per day come from a hand-set hourly multiplier
table, not measured hourly traffic.

---

## 2. GNN Layer — STATUS: COMPLETE

### 2.1 What it predicts

Per road-segment (edge):

| Output | Range | Meaning |
|---|---|---|
| `effective_travel_time` | seconds, >= 0 | expected traversal time |
| `delay_probability` | 0-1 | likelihood of significant delay |
| `actual_carbon_kg` | kg, >= 0 | CO2 emitted traversing the segment |
| `ev_energy_pct` | %, >= 0 | % of a 50kWh EV battery consumed |

### 2.2 Architecture — `LogisticsGNN` (386,692 params)

```
Input: x [7095,13] node features, edge_attr [15083,16] edge features
  -> Node encoder: Linear(13->128) -> LayerNorm -> ReLU
  -> Edge encoder: Linear(16->128) -> LayerNorm -> ReLU
  -> 3x TransformerConv (4 attention heads, edge-conditioned, edge_dim=128)
       -> LayerNorm -> Dropout(0.2) -> residual
  -> Edge fusion: concat(src_embed, dst_embed, raw edge_attr)
       -> 2-layer MLP -> 128-dim shared edge representation
  -> 4x independent 2-layer MLP heads (128->64->1), one per output above
```

Training: Adam (lr=3e-4, weight_decay=1e-5), ReduceLROnPlateau, batch size 8,
weighted multi-task MSE loss (weights [TT 0.25, Delay 0.40, Carbon 0.20, EV
0.15]), early stopping patience 8. Dataset split: train 6,092 / val 1,143 /
test 381 snapshots. A `BaselineGNN` (GraphSAGE, no edge features) also exists
as a reference lower bound.

### 2.3 The bug that was found and fixed (important context — don't reintroduce)

The `FeatureNormalizer` z-score normalizes **all 4 target labels** (including
`delay_probability`) before loss is computed, so training targets are centered
at 0 and can be negative/unbounded. The output heads originally ended in
**ReLU** (travel_time/carbon/ev — can't output negative values) or **Sigmoid**
(delay_prob — can't output outside (0,1)). This meant the network was being
asked to match training targets it was mathematically incapable of producing
for a large fraction of samples, capping R^2 architecturally regardless of
training quality — worst for the doubly-bounded Sigmoid head.

**Diagnosed** by comparing the GNN against a non-graph baseline (Ridge +
gradient-boosted trees trained on raw `edge_attr` alone, no graph structure).
The baseline hit 0.90-0.99 R^2 across all 4 labels, proving it was not a data
or label-design problem, but a bug in the GNN's own architecture/training
pipeline.

**Fix:** removed the final activation from all 4 heads (now plain `Linear`
output, matching the unconstrained normalized targets). Domain constraints
(travel_time >= 0, delay_prob in [0,1], carbon >= 0, ev_energy_pct >= 0) are
applied via `clamp_predictions()` **only after** `normalizer.inverse_y()`
converts predictions back to real units — never inside the loss or during
training. Do not add back a final activation to these heads.

### 2.4 Results after the fix

| Label | R^2 before fix | R^2 after fix |
|---|---|---|
| Travel Time | 0.7721 | 0.9489 |
| Delay Probability | 0.3007 | **0.9696** |
| Carbon | 0.7881 | 0.9432 |
| EV Energy | 0.7899 | 0.9447 |

All four are now "Excellent" (>0.85). Delay probability now exceeds the
gradient-boosted baseline (0.90) that was used to diagnose the bug.

### 2.5 Canonical files — naming (read this before touching any gnn_* file)

- **`gnn_model_v3.py` / `gnn_trainer_v3.py` = canonical, latest, fixed files.**
  The "v3" name reflects that the bug fix was applied here, not a further
  architecture change beyond what's described in 2.2/2.3.
- `gnn_model_v2.py` / `gnn_trainer_v2.py` = superseded, contain the bug. Do not
  use; safe to delete once confirmed no longer referenced anywhere.
- Checkpoints: `gnn_checkpoints_v3/best_model.pt`, `gnn_checkpoints_v3/normalizer.pt`
  (best epoch 11/19, early-stopped).
- `gnn_inference.py` — clean wrapper: `GNNCostPredictor(checkpoint_dir="./gnn_checkpoints_v3")`,
  method `.predict(snapshot) -> pd.DataFrame` with columns
  `[src, dst, travel_time_s, delay_probability, carbon_kg, ev_energy_pct]`,
  already real-unit and clamped. Imports from `gnn_model_v3` / `gnn_trainer_v3`.
- `routing_solver.py` — deterministic, multi-objective weighted-shortest-path
  solver (`RoutingSolver` class, uses `networkx`). Takes the costs DataFrame +
  a weights dict `{time, delay, carbon, ev}`, returns a route + real-unit
  totals. Route-level `delay_probability` is computed correctly as
  `1 - product(1 - p_i)` over the route's edges (P of at least one delay),
  **not** a naive sum. Tested and confirmed working against a synthetic graph
  (fastest-weighted and carbon-weighted routing correctly diverge onto
  different paths).

  **Normalization fix (found and fixed during Phase 2/3 agent testing —
  important context, don't reintroduce):** the original per-column
  normalization used true min/max. `travel_time_s`, `carbon_kg`, and
  `ev_energy_pct` are all heavily right-skewed on the real Bengaluru data
  (a handful of outlier edges up to ~660s / ~2.2kg dwarf the typical edge,
  whose median is ~21s / ~0.05kg), so scaling by the true max squeezed
  every typical edge's normalized value near 0. `delay_probability` has no
  such long tail (roughly bell-shaped, bounded ~0-0.7), so its normalized
  values stayed meaningfully spread out. Net effect: any nonzero `delay`
  weight dominated route selection almost regardless of magnitude relative
  to the other three — confirmed by testing: mixed weight vectors like
  `{time:0.7, delay:0.15, carbon:0.1, ev:0.05}` and
  `{time:0.1, delay:0.1, carbon:0.5, ev:0.3}` produced the **identical**
  66-edge route, even though pure single-objective weights (`{time:1,...}`
  vs `{carbon:1,...}`) correctly diverged. Since Route Agent's LLM realistically
  always returns mixed (non-pure) weights, this meant "fastest" vs
  "greenest" vs "balanced" priorities were collapsing onto the same route in
  practice — a real problem for Phase 4's negotiation (nothing to negotiate
  if weight tweaks don't change routes) and Phase 8's evaluation (priority
  differentiation is a key measured behavior).

  **Fix:** `_mins`/`_maxs` now come from the 1st/99th percentile of each
  column (`costs_df[...].quantile(0.01)` / `.quantile(0.99)`) instead of the
  true min/max, and `_normalized()` clips the raw value into `[lo, hi]`
  before scaling, so results stay in `[0,1]`. This is a robust-scaling
  approach: it tames outlier influence without distorting the bulk of each
  distribution. Verified fixed: the same two mixed weight vectors above now
  produce distinct routes (2181s/7.04kg vs 2058s/6.56kg), and pure-objective
  weights still diverge correctly.

  **Extension (Phase 4, additive/backward-compatible):** `build_graph()`
  and `find_route()` now also accept `penalized_edges: set[(src,dst)]` and
  `penalty_factor: float = 5.0` -- edges in that set get their scalar cost
  multiplied by `penalty_factor` before the shortest-path search, which the
  Coordinator (agents/coordinator_agent.py) uses to steer a re-route away
  from a congested edge. Deliberately a multiplicative penalty rather than
  hard exclusion, so a route degrades gracefully (a costly detour) instead
  of risking `NetworkXNoPath` when there's no real alternative. Both new
  params default to `None`/`5.0` and existing callers are unaffected.
- `export_sample_costs.py` — one-time export script, run wherever the GNN
  checkpoints + full dataset live (e.g. Colab). Dumps a handful of snapshots'
  predicted costs to `sample_costs/*.csv` so agent development needs no
  torch/torch_geometric/GPU/full dataset locally.

### 2.6 Local environment

Working locally in VS Code + Claude Code for the agents phase (GNN training
itself still happens on Colab GPU when needed). Agent-layer development is
**lightweight on purpose** — no `torch` install needed locally; everything
operates on the exported `sample_costs/*.csv` files. Folder layout:

```
capstone/
├── .env                       (API keys, gitignored — LLM_MODEL + provider key)
├── requirements.txt           (pandas, networkx, pydantic>=2, litellm, python-dotenv)
├── gnn/
│   ├── gnn_inference.py
│   ├── routing_solver.py
│   ├── gnn_model_v3.py
│   └── gnn_trainer_v3.py
├── sample_costs/
│   └── snapshot_XXXX_costs.csv
└── agents/                    <- all 8 phases complete, see Section 3
    ├── schemas.py             (DeliveryRequest, Vehicle, RouteResult)
    ├── llm_client.py          (LiteLLM wrapper: chat, chat_json)
    ├── route_agent.py         (RouteAgent — Phase 2)
    ├── dispatcher_agent.py    (DispatcherAgent — Phase 3)
    ├── pipeline.py            (run_batch ... run_full_pipeline — Phase 3-7)
    ├── coordinator_agent.py   (CoordinatorAgent — Phase 4)
    ├── carbon_optimizer_agent.py  (CarbonOptimizerAgent — Phase 5)
    ├── execution_agent.py     (ExecutionAgent — Phase 6)
    ├── esg_reporter_agent.py  (ESGReporterAgent — Phase 7)
    ├── golden_scenarios.py    (GOLDEN_SCENARIOS — Phase 8)
    └── evaluation_harness.py  (run_evaluation, run_baseline — Phase 8)
```

---

## 3. Agents Layer — STATUS: ALL 8 PHASES COMPLETE (full pipeline + eval harness built)

### 3.1 Design decision

Build the full remaining pipeline (not just "the agents") independently,
competing against teammates' own implementation of the same layers.

### 3.2 Design principle

**Deterministic tools do the grounding work; LLM agents do the judgment.**
`routing_solver.py` (a plain graph-search tool, no LLM) is already built and
tested. LLM-driven agents sit on top of it for the parts that need judgment:
interpreting fuzzy priorities into solver weights, negotiating trade-offs
between competing vehicles, and deciding when a carbon/time trade-off is worth
making. This mirrors the proven pattern from the code-review-agent project
(deterministic tools like Bandit/Ruff/coverage.py do grounding, LLM workers
reason on top) — keep that consistency.

### 3.3 Provider-agnostic requirement

Not locked to any single provider's API. Use **LiteLLM** so any provider's key
works through one interface:

```python
from litellm import completion
response = completion(
    model="anthropic/claude-sonnet-5",   # or "openai/gpt-4o", etc.
    messages=[{"role": "user", "content": "..."}],
)
```

LiteLLM reads whichever API key is set as an environment variable
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) based on the model prefix.
Switching providers should be a `.env` change, never a code change.

**Current `.env`:** `LLM_MODEL=groq/openai/gpt-oss-120b`, `GROQ_API_KEY` set.
Verified working end-to-end — `route_agent.py` and `pipeline.py` both
confirmed making live, successful LLM calls through this provider.

Provider history, for context if switching again: an OpenAI key was tried
first (`openai/gpt-4o`) and authenticated fine but the account had zero
billing credits (`RateLimitError: insufficient_quota`). An xAI key was tried
next but turned out to actually be a **Groq** key (`gsk_...` prefix, from
groq.com — fast open-model inference) mistaken for an **xAI Grok** key
(`xai-...` prefix, from console.x.ai) — easy mix-up given the names. Settled
on Groq since the key was already in hand and it has a generous free tier;
`groq/openai/gpt-oss-120b` (OpenAI's open-weight 120B model, hosted on Groq)
was picked as a strong general-purpose default for the structured-JSON
weight-translation task Route Agent needs. Switching providers again is
still just a `.env` change (model string + matching key env var) — no code
changes required, confirmed by this switch itself.

**Rate-limit retry (added in Phase 5, benefits every agent):** a full
pipeline run chains many LLM calls back to back (5 Route Agents, then the
Coordinator's negotiation rounds, then the Carbon Optimizer's
propose/reflect calls) — enough to exceed Groq's free-tier 8000
tokens/minute cap even though no single call is oversized, something
testing hit repeatedly once the pipeline had enough stages chained
together. `llm_client.chat()` now retries on `litellm.exceptions.
RateLimitError` specifically, with exponential backoff (3s/6s/12s, 3
retries by default) before giving up; other error types propagate
immediately since retrying wouldn't fix them. This lives in the shared
client, not per-agent, so every current and future agent gets it for free.

### 3.4 Agent roster (6 roles)

| # | Agent | Role |
|---|---|---|
| 1 | **Dispatcher/Planner** | assigns delivery requests to vehicles before routing starts |
| 2 | **Route Agent** (parallel, N instances) | turns a request's priority into solver weights, calls `RoutingSolver` |
| 3 | **Coordinator/Negotiation** | resolves conflicts across Route Agents' outputs (shared-edge overload, EV range violations) |
| 4 | **Carbon-Aware Optimizer** (critic, self-reflection loop) | reviews the finalized plan, pushes greener swaps where the time cost is acceptable |
| 5 | **Execution/Monitoring** | simulates the run against GNN predictions (+ optional noise), flags deviations/re-plan triggers |
| 6 | **ESG Reporter** | aggregates + narrates sustainability metrics for the dashboard |

Six roles maps cleanly onto the four remaining framework layers (multi-agent
decisions <- #1-3, carbon-aware optimizer <- #4, execution <- #5, ESG
dashboard <- #6) and matches the complexity of the code-review-agent project's
six roles (planner + 4 workers + critic) — don't add more roles than this.

### 3.5 Build order (8 phases)

1. **DONE** — Grounding tool: `routing_solver.py` (deterministic weighted
   shortest-path solver, unit test standalone).
2. **DONE** — Wire up one Route Agent end-to-end: single delivery request in,
   LLM interprets priority into solver weights, calls the tool, returns a
   structured route. Validates the full pattern before scaling out.
3. **DONE** — Scale to parallel Route Agents across a batch of delivery
   requests.
4. **DONE** — Add the Coordinator/Negotiation Agent — the genuinely
   "decentralized" piece: detect conflicts across parallel routes, re-run
   specific Route Agents with adjusted constraints until conflicts clear.
5. **DONE** — Add the Carbon-Aware Optimizer as a critic with a
   self-reflection loop.
6. **DONE** — Build the execution/monitoring simulation (replay plan against GNN
   predictions, optionally with injected noise, log deviations).
7. **DONE** — Build ESG Reporting (aggregate carbon/EV/on-time metrics,
   generate a narrative summary).
8. **DONE** — Build the evaluation harness: a golden set of delivery scenarios, scored
   against baselines (plain Dijkstra time-only, naive greedy-carbon-only, and
   eventually the teammates' system) through the **same input/output
   contract**, so the comparison is apples-to-apples. Metrics: total time,
   total carbon, on-time rate, EV feasibility rate, negotiation rounds needed.

Estimated total effort: ~35-55 hours across all 8 phases; roughly 2-3 weeks
part-time.

### 3.6 Phase 2 — Route Agent (done)

Built in `agents/`:

- `schemas.py` — pydantic models: `DeliveryRequest`, `Vehicle`, `RouteResult`.
- `llm_client.py` — thin LiteLLM wrapper: `chat(messages, model=None)` returns
  raw text; `chat_json(...)` additionally strips markdown code fences and
  parses the result as JSON, raising `ValueError` with the raw content on
  failure so callers can decide whether to retry. Model defaults to the
  `LLM_MODEL` env var (see 3.3).
- `route_agent.py` — `RouteAgent(costs_df, model=None)`. `.route(request)`
  sends the request's `priority` string to the LLM with a system prompt
  asking for a `{time, delay, carbon, ev, reasoning}` JSON object, feeds the
  4 weights into `RoutingSolver.find_route()`, and returns a `RouteResult`
  (success or failure, with the weights and the LLM's one-line reasoning
  attached either way).
- Verified: `RoutingSolver` itself confirmed working against a real
  `sample_costs` snapshot (fastest vs. carbon-weighted queries correctly
  diverge onto different paths). `route_agent.py`'s LLM path fully verified
  live end-to-end via Groq (see 3.3 for the provider history) — for
  "fastest"/"greenest"/"balanced" priorities the LLM returns sensible
  weights with a one-line rationale, and (after the normalization fix
  documented in 2.5) each priority now produces a genuinely different route.
- Tested against `sample_costs/*.csv` directly (`pd.read_csv(...)`) — no
  torch/GNN checkpoint needed for this phase, per 2.6.

### 3.7 Phase 3 — Parallel Route Agents (done)

Built in `agents/`:

- `dispatcher_agent.py` — `DispatcherAgent(costs_df)`. `.assign(requests,
  vehicles)` matches each `DeliveryRequest` to the nearest capable, available
  `Vehicle`. **Deliberately kept LLM-free**: vehicle assignment is a
  nearest-neighbor matching problem with a clear objective, not a fuzzy
  judgment call, so it doesn't need the LLM budget that Route Agent /
  Coordinator / Carbon Optimizer spend on genuine interpretation and
  negotiation (see 3.2). Implementation: builds one time-only weighted graph
  from `RoutingSolver.build_graph()`, runs one `nx.single_source_dijkstra_path_length`
  per vehicle (not per vehicle-request pair), filters out capacity-infeasible
  pairs (`package_weight_kg > vehicle.max_payload_kg`), then greedily assigns
  all feasible (vehicle, request) pairs sorted by distance ascending — first
  come, first served, so no vehicle or request is double-booked. This is a
  greedy approximation, not a globally optimal assignment (no Hungarian
  algorithm) — acceptable at this project's fleet scale; revisit only if the
  evaluation harness (Phase 8) shows dispatch quality actually bottlenecks
  results.
- `pipeline.py` — `run_batch(requests, vehicles, costs_df, model=None,
  max_workers=8)`: runs the Dispatcher, then calls every assigned request's
  `RouteAgent.route()` concurrently via `ThreadPoolExecutor` (Route Agent
  calls are I/O-bound LLM round-trips, so threads are enough — no
  multiprocessing needed). Returns `(route_results, unassigned_requests)`.
  **Fixed in Phase 4** (was an open known limitation here): a single Route
  Agent call raising (e.g. malformed LLM JSON) no longer aborts the whole
  batch — `_route_or_failure()` wraps `RouteAgent.route()` per-request
  inside the thread pool and converts an exception into a failed
  `RouteResult` instead of letting it propagate.
- Verified: `dispatcher_agent.py` smoke-tested end-to-end (deterministic, no
  LLM needed) — 3 vehicles vs 4 requests correctly assigned 3 and left 1
  unassigned, no double-booking. `pipeline.py` fully verified live: 5
  requests / 5 vehicles, all assigned and routed concurrently through Groq
  with no errors.

### 3.8 Phase 4 — Coordinator/Negotiation Agent (done)

Built `agents/coordinator_agent.py`. Conflict **detection** is deterministic
grounding (edge-usage counts, EV battery-budget arithmetic — pure facts, no
judgment); conflict **resolution** is where the LLM's judgment comes in, per
3.2's explicit call-out of "negotiating trade-offs between competing
vehicles" as an LLM task. Two conflict types:

- **edge_overload** — more than `edge_capacity` (default 2) requests routed
  over the same directed edge. The LLM decides which request_ids give up
  the segment; the Coordinator penalizes that edge
  (`RoutingSolver.find_route`'s new `penalized_edges`, see 2.5) and
  re-solves only those requests, keeping their *original* `weights_used` —
  the objective priority doesn't change on an edge conflict, only the path.
- **ev_range** — an `is_ev` vehicle's route consumes more energy than its
  actual `battery_capacity_kwh`. `ev_energy_pct` is defined (2.1) as % of a
  reference 50kWh battery, so it's rescaled against the vehicle's real
  capacity to check feasibility. The LLM decides a new `{time, delay,
  carbon, ev}` weights dict biased toward efficiency — a genuine trade-off
  judgment (how much time/carbon to give up), unlike edge_overload's fixed
  penalize-and-reroute mechanism.

`CoordinatorAgent.resolve(results, requests_by_id)` loops up to `max_rounds`
(default 3), returning `(final_results, rounds_used, fully_resolved)` —
`rounds_used` feeds the Phase 8 "negotiation rounds needed" metric,
`fully_resolved` is discussed below. `pipeline.run_batch_with_negotiation()`
chains `run_batch()` straight into this.

**Two real problems found and fixed while testing (important context —
don't reintroduce):**

1. **Per-conflict LLM calls don't scale.** The first implementation called
   `_negotiate()` once per overloaded *edge*. A test with 3 requests fully
   overlapping on a 21-edge path produced 21 rapid-fire LLM calls and blew
   through Groq's free-tier rate limit (8000 TPM) immediately. Worse, a
   5-request batch with realistic diverse routes and `edge_capacity=1`
   produced a *single* conflict payload large enough (10,437 tokens) to
   exceed the same 8000 TPM cap outright, because every overloaded edge got
   its own near-duplicate JSON entry even when many edges were overloaded
   by the exact same set of requests. **Fix:** batch every conflict of a
   given type into ONE negotiation call per round (not one per conflict),
   and within that, group overloaded edges by their exact `frozenset` of
   competing request_ids (one shared 21-edge corridor between the same 3
   requests is one decision, not 21) — capped at the 30 largest groups sent
   to the LLM, though penalization and the no-response fallback still cover
   every overloaded edge regardless of what fits in the prompt. This is
   also just better design independent of rate limits: one coherent
   decision beats fragmented per-edge ones for a request that appears in
   several conflicts.
2. **`fully_resolved` can legitimately be False — this is not a bug to
   chase away.** Tracing a 3-way edge_overload conflict step by step
   (`edge_capacity=2`) showed the Coordinator correctly whittling 21
   overloaded edges down to a single edge, `(7086, 7087)`, that stayed
   overloaded for the rest of the run no matter how many rounds or how high
   the penalty factor. That edge is a genuine bottleneck in this graph — the
   only road into the destination node — so no reroute can avoid it. The
   Coordinator now checks conflicts one final time after the round loop
   ends and returns `fully_resolved=False` honestly instead of silently
   returning a route that still violates a constraint. **Relatedly:**
   testing an EV-range fix also found that `ev_energy_pct` correlates
   strongly with route distance/time in this dataset (unlike
   `delay_probability`, which is segment-specific and largely independent
   of route length) — so for a battery too small for a trip's *shortest
   possible* path, reweighting toward "ev" has limited room to help, and
   some EV violations are genuinely infeasible, not just a tuning miss.
   Callers (especially the Phase 8 eval harness's EV feasibility rate
   metric) must check `fully_resolved` rather than assume a `RouteResult`
   with `success=True` satisfies every constraint.

**Also fixed:** a `UnicodeEncodeError` crash on Windows when LLM-generated
reasoning text contains Unicode punctuation (e.g. a non-breaking hyphen)
that the default `cp1252` console codepage can't print. All of
`route_agent.py` / `coordinator_agent.py` / `pipeline.py`'s `__main__`
blocks now call `sys.stdout.reconfigure(encoding="utf-8")` before printing
any LLM output.

Verified live end-to-end via Groq: `coordinator_agent.py`'s own smoke test
(one scenario forcing an edge_overload among 3 requests, a separate one
forcing a resolvable ev_range violation) and the full
`pipeline.run_batch_with_negotiation()` flow (5 requests, `edge_capacity=2`)
both run without errors, with coherent LLM negotiation reasoning (e.g. "Both
balanced requests cause the largest overload on nine edges and appear in
every conflict, so rerouting them clears all groups while preserving the
fastest and greenest priorities").

### 3.9 Phase 5 — Carbon-Aware Optimizer (done)

Built `agents/carbon_optimizer_agent.py`. A critic with a genuine
self-reflection loop, structured as three steps rather than one LLM call
with a hardcoded acceptance formula (per 3.2 — the "is this trade worth it"
call has to stay real LLM judgment, not code moving that judgment into a
fixed threshold):

1. **PROPOSE (LLM)** — for every successful route in the batch, propose a
   candidate `{time, delay, carbon, ev}` weights dict shifted toward
   `carbon`, sized to the request's stated priority (a modest shift for
   "fastest", a bigger one for "greenest"/"balanced"). Batched into ONE call
   for the whole set of routes, same lesson as the Coordinator (3.8) —
   proposing one-route-at-a-time would multiply LLM calls across the batch
   for no benefit.
2. **GROUND (deterministic)** — re-solve each candidate with
   `RoutingSolver.find_route()` and compute the real `time_delta_s` /
   `carbon_delta_kg` vs. the original (`candidate - original`). This step is
   why propose-then-reflect beats accepting the LLM's proposal outright: a
   candidate's weights can look reasonable and still not actually reduce
   carbon once solved against the real graph (confirmed by testing — see
   below).
3. **REFLECT (LLM)** — given the real deltas and the request's priority,
   decide accept/reject per request, again batched into one call across the
   batch. `CarbonOptimizerAgent.optimize(results, requests_by_id)` returns
   `(final_results, decisions_log)`; `decisions_log` has one entry per
   candidate evaluated (accepted or not, with the deltas and reasoning) for
   the Phase 8 eval harness / auditing.

Threads through the Coordinator's `penalized_edges` (exposed as
`coordinator.penalized_edges` after `resolve()` runs, per §3.8) so a
carbon-motivated re-route can't silently undo an already-resolved conflict.
`pipeline.run_batch_with_optimization()` chains
`run_batch_with_negotiation()` straight into this.

**A real bug found and fixed while testing (important context — don't
reintroduce):** the first REFLECT prompt didn't state the sign convention
for `time_delta_s`/`carbon_delta_kg`. Testing surfaced a case where the LLM
returned `accept: true` with reasoning claiming *"yields a ~5.5% carbon
reduction"* for a candidate whose actual computed delta was
`carbon_delta_kg: +0.39` — an **increase**, the opposite of what the LLM's
own reasoning claimed. The deterministic safety check already in code
(`accept = llm_accept and carbon_delta_kg < 0`) caught this and correctly
overrode it to reject — which is exactly why that invariant belongs in code
rather than being left to the LLM alone — but the prompt itself was still
worth fixing so the LLM's *reasoning* is right, not just silently
overruled. **Fix:** the REFLECT prompt now explicitly spells out that
deltas are `(candidate - original)`, that positive means worse for both
metrics, and that a candidate must always be rejected if
`carbon_delta_kg >= 0` regardless of `time_delta_s`. Verified fixed:
re-running the same case produced the reasoning *"The candidate is slower
(+69s) and emits more carbon (+0.39kg), so it fails the fastest priority"*
— now consistent with the data instead of contradicting it.

Verified live end-to-end via Groq: a hand-crafted reflect-only test
confirmed the accept/reject logic itself is sound (accepted a modest
time-increase/real-carbon-reduction trade for a "balanced" request,
rejected a large-time/tiny-reduction trade for a "fastest" one, with
correct reasoning in both cases). The full
`pipeline.run_batch_with_optimization()` flow (5 requests) ran end-to-end
without errors and produced a genuine accept: one candidate came back both
faster *and* lower-carbon than the original (a clean win), correctly
accepted, while four others (no improvement, or worse on both metrics) were
correctly rejected.

### 3.10 Phase 6 — Execution/Monitoring Agent (done)

Built `agents/execution_agent.py`. Simulates actually driving the finalized
plan against the GNN's per-edge predictions (the only "truth" available
locally — there's no live telemetry), with a documented noise model
standing in for real-world deviation, then flags routes whose simulated
outcome diverges enough to warrant a re-plan.

**Noise model** (per edge, applied on top of `costs_df`'s predicted values
— see the module docstring for the full rationale):
1. Baseline prediction error: `actual_time = predicted_time * (1 +
   U(-10%, +10%))` — a stand-in for the GNN's own residual error (its
   travel_time R^2 is 0.9489, per 2.4, not perfect).
2. Delay events: a Bernoulli draw using the GNN's *own* predicted
   `delay_probability` for that edge (principled — the simulation's "did a
   delay happen" is grounded in the model's own uncertainty estimate, not
   an arbitrary constant). If it fires, that edge's time is further
   multiplied by `U(1.3, 2.0)`.
3. Carbon and EV energy scale with the realized time ratio for that edge
   rather than getting independent noise — more time on a segment
   (congestion, idling) means proportionally more emissions, so tying them
   to the same jitter is more defensible than a third independent
   parameter.

**Deterministic-grounding/LLM-judgment split (3.2), same shape as every
other agent here, with one addition — a hard deterministic override:**
- **Hard rule (no LLM, no exceptions):** if an `is_ev` vehicle's simulated
  actual EV energy use exceeds its `battery_capacity_kwh`, that's always a
  `replan_triggered=True` — a stranded EV is a hard failure, not a
  trade-off to weigh. Mirrors the Carbon Optimizer's `carbon_delta_kg >= 0`
  hard rule from Phase 5 — same lesson, a fact the LLM shouldn't get to
  override.
- **Soft judgment (LLM, batched):** for routes whose `time_deviation_pct`
  magnitude is >= 15% (`TIME_DEVIATION_FLOOR_PCT` — below that, treated as
  normal variance, not even sent to the LLM) and that don't already have a
  hard EV violation, the LLM decides per-route whether the deviation is bad
  enough to interrupt and re-plan, weighing the request's stated priority
  (e.g. observed live: a "greenest" request with a +37% deviation was
  judged tolerable — *"Greenest priority tolerates larger delays ... within
  its acceptable range"* — while "fastest"/"balanced" requests with
  smaller deviations still triggered re-plans). All routes needing
  reflection in a round go into ONE batched LLM call, same lesson as
  Phases 4/5.

`ExecutionAgent.run(results, vehicles_by_id)` returns one report dict per
successful route (simulated totals, `time_deviation_pct`,
`replan_triggered`, `reasoning`). Actually acting on `replan_triggered`
(re-invoking the Route Agent) is explicitly out of scope for this agent —
its job is to simulate and flag; a later phase or the eval harness consumes
the flag. `pipeline.run_batch_with_execution()` chains
`run_batch_with_optimization()` straight into this, and accepts a `seed`
passthrough for reproducible noise draws in tests.

Verified live end-to-end via Groq: the standalone smoke test (seeded,
reproducible) showed a small deviation (+8.4%, under the 15% floor) skip
LLM reflection entirely, while larger ones (+30%, +38%) correctly triggered
it with priority-aware reasoning. The full
`pipeline.run_batch_with_execution()` flow (5 requests, unseeded) ran
end-to-end without errors through all six chained agents (Dispatcher, Route
Agents, Coordinator, Carbon Optimizer, Execution).

### 3.11 Phase 7 — ESG Reporter (done)

Built `agents/esg_reporter_agent.py`. The one agent in the roster that
aggregates across the *whole* run rather than deciding per-route, so
there's no per-item batching concern like Phases 4-6 hit — this is
naturally one LLM call per run regardless of batch size.

`ESGReporterAgent.aggregate(results, unassigned, vehicles_by_id,
decisions_log, execution_reports, negotiation_rounds_used)` is pure
deterministic grounding — every number (total planned vs. simulated-actual
time/carbon, on-time rate from `replan_triggered`, EV feasibility rate from
`hard_ev_violation`, Carbon Optimizer swaps accepted + total kg saved,
negotiation rounds used, unassigned/routing-failed counts) is computed from
the pipeline's own prior outputs, nothing estimated. `.narrate(metrics)` is
the one LLM call: turns that metrics dict into a 3-5 sentence dashboard
summary. `.report(...)` runs both and returns `{"metrics": {...}, "summary":
str}`. `pipeline.run_full_pipeline()` chains the complete 6-agent flow
(Dispatcher through ESG Reporter, CLAUDE.md roster #1-6) and returns
everything as a **dict**, not a further-growing positional tuple like the
earlier `run_batch_with_*` helpers — this is the final integration point,
and later work (the Phase 8 eval harness especially) should reference
fields by name.

**Two narrative-quality issues found and fixed while testing (both
cosmetic — the underlying numbers were always correct, never fabricated —
but worth fixing for a dashboard-facing summary):**
1. The first `NARRATE_SYSTEM_PROMPT` said *"do not ... round any number of
   your own that isn't already in the JSON"* — intended to stop the LLM
   inventing facts, but it also suppressed rounding for *display*, so the
   summary dumped raw floats like `"15671.69233300773 s"`. **Fix:** the
   prompt now explicitly distinguishes inventing a number (forbidden) from
   rounding one for readability (expected) — e.g. "15,672s (~4.4 hours)".
2. The LLM once phrased `negotiation_rounds_used` as *"3 rounds per
   request"* when it's actually a total for the whole batch. **Fix:** the
   prompt now states that explicitly. Verified fixed: re-running produced
   *"the batch needed just three rounds for the entire batch"* and cleanly
   rounded figures throughout.

Verified live end-to-end via Groq: `esg_reporter_agent.py`'s own smoke test
and the full `pipeline.run_full_pipeline()` flow (5 requests, all 6 agents
chained, 12+ LLM calls in one run) both completed with no errors; every
number quoted in the generated narrative was cross-checked against the
`metrics` dict and matched exactly.

### 3.12 Phase 8 — Evaluation Harness (done, final phase)

Built `agents/golden_scenarios.py` and `agents/evaluation_harness.py`. This
is the artifact that serves the project's stated goal (section 1):
whichever system — this one or a teammate's independent implementation —
scores better here is what ships in the final submission. More than any
other file in `agents/`, this one has to be fair and trustworthy, not
flattering to the system built here.

**`golden_scenarios.py`** — `GOLDEN_SCENARIOS`, a fixed list of `(name,
builder)` pairs, each `builder(costs_df) -> (requests, vehicles)`. Node IDs
are derived from `costs_df` (e.g. `costs_df.iloc[0]["src"]`,
`pd.unique(...)[100]`) rather than hand-typed integers, but this is still a
"fixed" scenario set in the meaningful sense: **verified** that every
`sample_costs/*.csv` snapshot has identical `(src, dst)` columns in
identical order (only the traffic-dependent predicted values differ per
snapshot), so a builder produces the same origin/destination pair regardless
of which snapshot it's run against. Three scenarios, each chosen from a
specific behavior found during Phases 4-6 testing rather than an arbitrary
happy path:
- `edge_overload` — 3 requests sharing one origin/destination, the exact
  pair used in the Coordinator's own smoke test and in 3.8's
  irreducible-bottleneck-edge trace.
- `ev_infeasible` — one EV request with a battery (1.0kWh) far below what
  any weighting could achieve on that trip (**verified**: even the most
  energy-conscious route found needs >=4.7kWh) — a genuine infeasibility
  case per 3.8's ev-correlates-with-distance finding, not a tuning gap.
- `mixed_priority` — the 5-request, rotating-priority batch used throughout
  Phases 3-7's own smoke tests, formalized here instead of staying ad hoc.

**`evaluation_harness.py`** — `run_evaluation(costs_df, model=None,
seed=None, scenarios=GOLDEN_SCENARIOS)` runs every system in `SYSTEMS`
against every golden scenario and returns one `pandas.DataFrame` row per
(scenario, system) pair with every metric `ESGReporterAgent.aggregate()`
computes.

`SYSTEMS` has three entries: `multi_agent` (`pipeline.run_full_pipeline()`,
the system built in Phases 2-7) and two baselines,
`run_baseline(requests, vehicles, costs_df, weights, ...)` called with
`{"time":1,...}` (plain Dijkstra time-only) and `{"carbon":1,...}` (naive
greedy-carbon-only) — `RoutingSolver.find_route()` called directly with
fixed weights, no Route Agent, no LLM anywhere in the routing decision.

**Apples-to-apples design decision (important, don't reintroduce a
mismatch):** baselines still run through the SAME `DispatcherAgent`,
`ExecutionAgent` (including its own LLM-judged soft-deviation reflection),
and `ESGReporterAgent.aggregate()` as `run_full_pipeline()` — only the
routing/negotiation/optimization decisions differ (no Coordinator, no
Carbon Optimizer for baselines: `rounds_used=0`, `decisions_log=[]`,
`fully_resolved=None` meaning "not applicable", not "resolved"). This
matters because `on_time_rate` and the other metrics need to mean the same
thing across every system being compared — if baselines used a different
(or no) execution simulation, the comparison wouldn't be measuring the same
thing, contrary to 3.5 phase 8's explicit "same input/output contract"
requirement. The harness also builds fresh `requests`/`vehicles` from each
scenario's builder for every system run, not shared/reused objects, since
`DispatcherAgent.assign()` mutates `request.vehicle_id` in place.

**First results** (all 3 golden scenarios x all 3 systems, one
`sample_costs` snapshot, `seed=0` for reproducible execution noise) —
worth recording here as a baseline for comparison once teammates' systems
are added, and because a couple of the numbers are genuinely informative,
not just a sanity check:

| scenario | system | planned_time_s | planned_carbon_kg | on_time_rate | rounds_used |
|---|---|---|---|---|---|
| edge_overload | multi_agent | 6532 | 19.79 | 0.00 | 3 |
| edge_overload | baseline_time_only | 6141 | 21.06 | **0.33** | 0 |
| edge_overload | baseline_carbon_only | 6426 | **19.48** | 0.00 | 0 |
| ev_infeasible | multi_agent | 2047 | 4.23 | 0.00 | 3 |
| ev_infeasible | baseline_time_only | 1766 | 4.87 | 0.00 | 0 |
| ev_infeasible | baseline_carbon_only | 2047 | 4.23 | 0.00 | 0 |
| mixed_priority | multi_agent | 11759 | 27.52 | 0.20 | 3 |
| mixed_priority | baseline_time_only | 10229 | 28.998 | 0.20 | 0 |
| mixed_priority | baseline_carbon_only | 11067 | 24.23 | 0.20 | 0 |

(`ev_feasibility_rate` was 0.0 for every system on `ev_infeasible` — no
routing strategy can make a 1.0kWh battery cover a >=4.7kWh trip, confirming
the scenario is a genuine infeasibility test, not a routing-quality test.)

Two things worth being honest about rather than glossing over, since this
harness's credibility depends on not being rigged in the home system's
favor:
1. **On `edge_overload`, `baseline_time_only` beat `multi_agent` on both
   speed AND on-time rate.** The Coordinator's negotiated detour around the
   irreducible bottleneck edge (3.8) made the route longer, which meant more
   edges exposed to the execution simulation's delay draws. Negotiating the
   conflict succeeded at its actual job (clearing a shared-edge overload),
   but that's a different thing from "the plan performed better" — a real,
   not-flattering result the evaluation harness is supposed to surface.
2. **`on_time_rate` landed on the identical value (0.20) for all three
   systems on `mixed_priority`**, despite clearly different routes (planned
   carbon differs across all three: 27.5 / 29.0 / 24.2 kg). Investigated,
   not a bug — each system gets a freshly seeded `ExecutionAgent`, not a
   shared/stale RNG. More likely explanation: this dataset's
   `delay_probability` averages ~0.54 across edges with a tight
   distribution (2.5's normalization-fix writeup), so the simulation's
   Bernoulli delay draws fire on roughly half of *any* route's edges
   regardless of which specific route was chosen -- and with only 5
   requests, `on_time_rate` can only take 6 discrete values (0, .2, .4, .6,
   .8, 1), so 3 systems landing on the same coarse bucket isn't as
   surprising as it first looks. Worth listing as a known limitation for
   whoever extends this harness: a larger golden set, or measuring average
   deviation magnitude instead of a hard-threshold rate, would give more
   statistical resolution between systems.

**Left for whoever picks this up next (not done here, out of scope for
"build the harness"):** wiring in a teammate's independently-built system as
a fourth `SYSTEMS` entry once one exists, and running the full evaluation
across all 5 `sample_costs` snapshots (only 1 used above) to see how
robust each system's ranking is across different traffic conditions.

## 4. Completion Phase — STATUS: IN PROGRESS (items 1-4 done, items 5-6 not started)

### 4.1 Scope decision (supersedes 3.12's "left for whoever picks this up
next" framing)

Complete this project as a fully standalone, independently-built system.
**Do not wait on or depend on teammates' implementation.** Section 3.12's
"wire in a teammate's system as a 4th SYSTEMS entry" is now optional/later,
not a blocking requirement — this build is the final deliverable regardless
of what teammates produce. Everything below is scoped to make this system
complete and defensible on its own.

### 4.2 Remaining work, in priority order

1. **DONE, all 5 snapshots** (see 4.5 for a rate-limit issue hit and fixed
   along the way) — Multi-snapshot evaluation. Run
   `evaluation_harness.run_evaluation()`
   across all 5 `sample_costs/*.csv` snapshots (Phase 8 only used 1). Report
   mean +/- std per metric per (scenario, system), not a single run — this
   checks whether Phase 8's findings (e.g. `baseline_time_only` beating
   `multi_agent` on `edge_overload`) hold across different traffic
   conditions or were specific to that one snapshot. If results vary a lot
   between snapshots, that's itself worth recording as a finding.

2. **DONE** — Fix the `on_time_rate` resolution problem (flagged as a known
   limitation in 3.12). Replace the hard-threshold `on_time_rate` with a
   continuous metric: mean `abs(time_deviation_pct)` across all routes in
   the batch, sourced from `ExecutionAgent`'s reports. Keep `on_time_rate`
   as a secondary/reported stat if useful, but make the continuous metric
   the primary comparison metric in the harness output and the dashboard
   (item 4) — it won't collapse to identical values across systems the way
   a 6-bucket discrete rate does with a 5-request batch.

3. **DONE** — Expand golden scenarios beyond the original 3. Added:
   - a larger batch (10+ requests) to stress-test the Dispatcher/Coordinator
     at more realistic fleet scale than 3-5 requests
   - a genuinely *feasible* EV scenario (battery large enough for the trip)
     to test EV routing quality, not just infeasibility detection — the
     original `ev_infeasible` scenario only exercised the hard-rejection
     path, never the case where EV-aware routing actually helps

4. **DONE** — Build the ESG Dashboard, the literal 9th layer of the
   original framework, not yet visually built until now. See 4.3 for the
   spec, 4.5 for what got built and verified.

5. **Add a pytest + CI suite.** Convert the existing per-phase smoke tests
   (currently ad hoc `__main__` blocks per 3.6-3.12) into a `tests/`
   directory using a fake/recorded LLM client for deterministic, API-key-free
   CI runs — mirror the code-review-agent project's pattern of
   dependency-injected clients for testability. Add a GitHub Actions workflow
   that runs the suite on push.

6. **Write the final project report** once 1-5 are done — a comprehensive
   report covering the full 9-layer system for the capstone guide, in the
   same spirit as the earlier GNN debugging report but end-to-end. This one
   doesn't need to be built by Claude Code — bring the finished 1-5 results
   back and it'll be written directly.

### 4.3 ESG Dashboard spec (item 4)

Build `agents/dashboard.py`: a script that takes `ESGReporterAgent.aggregate()`'s
metrics dict (and optionally the evaluation harness's results for a
multi-system comparison view) and renders **one self-contained HTML file**
(Chart.js via CDN, no build step, no server) containing:

- Summary cards: total distance, total carbon, the new continuous deviation
  metric (item 2), EV feasibility rate, negotiation rounds used
- Bar chart: planned vs. simulated-actual time and carbon, per route (from
  `ExecutionAgent` reports)
- Bar chart: multi-agent vs. `baseline_time_only` vs. `baseline_carbon_only`
  per scenario (same shape as the two charts already generated this chat —
  reuse that structure)
- The narrative text from `.narrate(metrics)` displayed prominently
- Optional/nice-to-have: a simple map of one representative route using
  Leaflet.js, if node lat/lon is available from the graph construction step

Static HTML, not a live server — easiest to screenshot for the report and
hand to the guide, and Claude Code can build it without new heavy
dependencies. If a live interactive demo is wanted later for a
presentation/viva, this can be upgraded to a small Streamlit app reusing the
same aggregation logic — start with the static version first.

### 4.4 What to check back with

- Item 1: the aggregated multi-snapshot results table (mean +/- std)
- Item 2/3: confirmation the harness runs cleanly with the new metric/
  scenarios + the updated results table
- Item 4: **a screenshot of the dashboard** — the one item in this whole
  project where an image is actually the right way to check it
- Item 5: the CI run output / pytest summary
- Item 6: nothing needed from Claude Code for this one

### 4.5 Items 1-4 — results, findings, and a rate-limit bug found and fixed

Built `agents/golden_scenarios.py` (2 new scenarios), `agents/dashboard.py`,
and `agents/run_full_evaluation.py` (a dedicated multi-snapshot CLI, see
below). Updated `agents/esg_reporter_agent.py` (new metric),
`agents/evaluation_harness.py` (new `reliable` column + `summarize_
across_snapshots()` helper), `agents/llm_client.py` (retry jitter), and
`agents/pipeline.py` (`max_workers` default 8 -> 4). All 5 snapshots now
complete with 100% reliable data; `dashboard.html` generated and its
embedded data verified consistent end to end.

**Item 2 — continuous on-time metric.** Added `mean_abs_time_deviation_pct`
to `ESGReporterAgent.aggregate()`: the mean, across a batch's routes, of
`abs(time_deviation_pct)` from `ExecutionAgent`'s reports. `on_time_rate`
kept as a secondary stat; the narrate prompt now explains both explicitly
(which is "primary", and that `on_time_rate` is coarse on a small batch).

**Item 3 — two new golden scenarios**, in `golden_scenarios.py`:
- `ev_feasible` — one EV request on a pair with real EV route diversity
  (found by scanning several candidates; most pairs had under 1% spread
  between best/worst-case EV cost, confirming 3.8's ev-correlates-with-
  distance finding generalizes). **A real calibration surprise, worth not
  reintroducing:** the battery threshold has to be picked against
  SIMULATED actual EV cost, not the planned figure — `ExecutionAgent`'s
  noise model scales carbon/EV energy with the realized time ratio per
  edge (3.10), so a route with several delayed edges can use 30-50%+ more
  energy than planned. A battery set between the two *planned* extremes
  (2.5kWh) tested at `ev_feasibility_rate=0.0` for every system, including
  EV-aware ones — noise inflation swallowed the whole planned-route margin.
  Recalibrated to 3.3kWh against sampled *actual* kWh across 5 seeds. Even
  so, the two extremes' actual-kWh ranges overlap somewhat (e.g. one seed:
  worst=3.517 vs best=3.374), so this scenario's outcome is genuinely
  seed-/snapshot-sensitive by design, unlike `edge_overload`/`ev_infeasible`
  — EV-aware routing is more LIKELY to succeed here, not guaranteed every
  run. That's honest (real EV range anxiety depends on delay exposure, not
  just route choice), not a scenario flaw.
- `large_batch` — 12 requests across 12 connectivity-verified pairs (one
  planned pair, node-index 1650, turned out to have no outgoing path in
  this direction and was swapped for 1700 during scenario design).

**Item 4 — dashboard.** `agents/dashboard.py` builds one self-contained
HTML file (Chart.js via CDN): stat-tile cards, a planned-vs-actual bar
chart per route (from a fresh `mixed_priority` pipeline run), and a
multi-agent-vs-baselines bar chart per scenario (from
`evaluation_results/summary_results.csv`). Followed the dataviz skill's
method: the validated default categorical palette in fixed slot order
(blue=multi_agent, orange=baseline_time_only, aqua=baseline_carbon_only,
never reassigned), status colors (green/red) reserved for the
EV-feasibility card only, a legend on every multi-series chart, `<=24px`
bars with rounded caps and a gap between them. No lat/lon data exists
anywhere in this project (checked `sample_costs/*.csv` and every `gnn/`
file), so the spec's optional Leaflet.js route map was skipped — that data
would have to come from a teammate's graph-construction layer.

Generated `agents/dashboard.html` and validated it structurally (well-formed
HTML, no leftover template placeholder, the embedded JSON parses and every
number the LLM's narrative cites matches the stat cards exactly — e.g.
"100% feasibility" <-> the EV feasibility card, "33% longer" <-> the 33.8%
deviation card, "25.5 kg to 33.9 kg" <-> the two carbon cards verbatim).
**Could not render/screenshot it in this environment** — no `chromium-cli`
and no existing browser-automation setup here, and installing
Playwright/Node from scratch for one screenshot wasn't worth it as a
one-off. Opened the file locally instead; a visual check against the
dataviz skill's own final step ("render it and look at it — the validator
checks color, not layout") is still worth doing by eye once opened.

**Item 1 — multi-snapshot evaluation, a rate-limit bug found, root-caused,
and fixed.** Built `agents/run_full_evaluation.py` as a dedicated CLI
rather than a single call to `run_multi_snapshot_evaluation()`,
specifically so it could survive interruption: `python
run_full_evaluation.py <N>` runs ONE snapshot (by index or path) and merges
its rows into `evaluation_results/raw_results.csv` (replacing only that
snapshot's prior rows, safe to re-run), and `--summarize` recomputes
`evaluation_results/summary_results.csv` from whatever's there. This
mattered in practice — the run hit Groq's *daily* token cap (200,000/day)
partway through snapshot 5 of 5, and **4 of 5 snapshots' results survived
untouched** because of the per-snapshot save design; only the in-progress
5th snapshot's unsaved work was lost. Finished the 5th snapshot after the
key was refreshed with a new daily quota.

**A real bug found, then its root cause corrected once more evidence came
in — worth reading both stages, not just the ending:**

1. **First symptom:** `large_batch`/`multi_agent` failed to route 4-5 of 12
   requests in every one of the first 4 snapshots (`reliable_mean=0.0`,
   `num_routing_failed_mean=4.25`), while every other one of the 15
   (scenario, system) combinations was perfectly reliable. Initial
   hypothesis: `large_batch` runs last in `GOLDEN_SCENARIOS`, so its 12
   Route Agent calls were landing after 4 earlier scenarios had already
   spent a chunk of the day's 200k-token budget.
2. **That hypothesis was wrong, and the evidence that killed it:** re-running
   `large_batch` standalone against a *freshly reset daily quota* (a brand
   new API key, full 200k budget, zero prior calls that day) still failed
   4/12 requests identically. The failure had nothing to do with cumulative
   daily spend.
3. **Real root cause:** `pipeline.run_batch`'s `ThreadPoolExecutor` fires up
   to `max_workers=8` Route Agent calls at the same instant. For a 12-request
   batch, that's 8 concurrent LLM calls landing in the same fraction of a
   minute — enough alone to blow through Groq's *per-minute* (8000 TPM)
   limit in one burst, independent of the daily total. Worse, `llm_client
   .chat()`'s retry backoff was a fixed schedule (3s, 6s, 12s) with no
   jitter: several threads hitting the limit in the same instant all retry
   at the same fixed delays, collide on the per-minute limit again together,
   and some exhaust `max_retries` while still colliding in lockstep — a
   classic thundering herd, invisible in every OTHER scenario here because
   none of them fire more than 5 concurrent Route Agent calls.
4. **Fix, verified:** (a) `llm_client.chat()`'s backoff now has +-30% random
   jitter per attempt, and `max_retries` raised 3 -> 5, so colliding threads
   stop retrying in lockstep; (b) `pipeline.run_batch`'s (and every
   `run_batch_with_*`'s) `max_workers` default lowered 8 -> 4, shrinking the
   burst itself. Re-ran `large_batch` against the SAME snapshot that had
   just failed: **12/12 routed successfully.** Re-ran the full scenario
   across all 5 snapshots: **100% reliable in every one** (`reliable_mean
   =1.0` everywhere in the final summary). Raise `max_workers` back up for
   a generous/paid API tier where burst size isn't a concern; the jitter fix
   is a pure win regardless of tier.
5. **A methodological trap this also caught, independent of the rate-limit
   bug itself:** the first pass at this comparison summed `total_planned_
   time_s` / `_carbon_kg` across only the *successfully-routed* subset per
   system, so `multi_agent`'s smaller (7-8 of 12) successful set made it
   look ~40% faster and greener than both baselines' full 12/12 — a false
   "win" from doing less work, not better work. **Fix, kept even though the
   underlying failures are now gone:** `run_evaluation()` adds a `reliable`
   column (`True` iff `num_routing_failed == 0`) so a future partial-failure
   case can't silently produce a misleading comparison again;
   `summarize_across_snapshots()` was also fixed to actually aggregate it —
   pandas' `select_dtypes(include="number")` silently excludes `bool`
   columns in the version installed here, so `reliable` was disappearing
   from the summary before this fix (cast bool columns to `int` first).

**Final results, all 5 snapshots, 100% reliable** (raw:
`agents/evaluation_results/raw_results.csv`, summary:
`.../summary_results.csv`; mean values shown, see the CSV for std):

| scenario | system | time_s | carbon_kg | mean_abs_dev | on_time_rate | ev_feasibility |
|---|---|---|---|---|---|---|
| edge_overload | multi_agent | 6654 | 19.93 | **0.279** | 0.067 | n/a |
| edge_overload | baseline_time_only | 6228 | 21.07 | 0.290 | **0.133** | n/a |
| edge_overload | baseline_carbon_only | 6519 | **19.66** | 0.308 | 0.000 | n/a |
| ev_feasible | multi_agent | 1294 | 2.45 | 0.418 | 0.000 | 0.2 |
| ev_feasible | baseline_time_only | 1214 | 2.80 | 0.406 | 0.000 | 0.2 |
| ev_feasible | baseline_carbon_only | 1260 | 2.35 | 0.483 | 0.000 | **0.6** |
| ev_infeasible | multi_agent | 2013 | 4.27 | 0.360 | 0.000 | 0.0 |
| ev_infeasible | baseline_time_only | 1798 | 4.89 | 0.329 | 0.000 | 0.0 |
| ev_infeasible | baseline_carbon_only | 2013 | 4.27 | 0.360 | 0.000 | 0.0 |
| large_batch | multi_agent | 22461 | 54.64 | 0.347 | 0.367 | 1.0 |
| large_batch | baseline_time_only | 20442 | 55.26 | 0.352 | 0.383 | 1.0 |
| large_batch | baseline_carbon_only | 23685 | 53.87 | 0.357 | 0.350 | 1.0 |
| mixed_priority | multi_agent | 11802 | 26.78 | 0.355 | **0.400** | 1.0 |
| mixed_priority | baseline_time_only | 10444 | 28.70 | 0.339 | 0.120 | 1.0 |
| mixed_priority | baseline_carbon_only | 11274 | 24.45 | 0.349 | 0.280 | 1.0 |

**Reading these, honestly:**
- **`edge_overload`: the Phase 8 finding holds up across all 5 snapshots,
  not a one-off.** `baseline_time_only` is fastest and has the best
  `on_time_rate`; interestingly `multi_agent` has the LOWEST
  `mean_abs_time_deviation_pct` despite the worst `on_time_rate` — exactly
  the item-2 scenario this metric was added for (a lower *average*
  deviation doesn't guarantee fewer individual routes cross the LLM's
  priority-weighted replan threshold).
- **`ev_feasible`: `baseline_carbon_only` (pure carbon=1.0) is 3x more
  likely to hit the tight EV budget than `multi_agent`'s "greenest"-priority
  routing (0.6 vs 0.2).** This is a real, actionable finding, not scenario
  noise (see 4.5 above): the Route Agent's LLM-driven "greenest" weight
  translation is consistently less aggressive than a naive single-objective
  baseline, which costs it in a tightly EV-constrained trip. Worth revisiting
  the Route Agent's system prompt (route_agent.py, 3.6) to push harder
  toward the extremes for a "greenest" priority specifically, if EV
  feasibility under tight batteries matters for the final submission.
- **`large_batch`, now clean: `multi_agent` sits sensibly between the two
  baselines on both time and carbon** (22,461s / 54.64kg, between
  time-only's 20,442s/55.26kg and carbon-only's 23,685s/53.87kg) — the
  expected middle-ground result for a batch mixing "fastest"/"greenest"
  /"balanced" priorities, once the rate-limit bug stopped distorting it.
  Negotiation rounds used average 1.4 here (vs. exactly 3 or 0 everywhere
  else) — plausible, since 12 requests scattered across different
  origin/destination pairs create less persistent overlapping congestion
  than `edge_overload`'s 3 identical-route requests, so conflicts often
  clear before `max_rounds`.
- **`mixed_priority`: `multi_agent` actually has the BEST `on_time_rate`
  here (0.400 vs 0.120/0.280)**, the one scenario where it outright wins on
  that metric — though `mean_abs_time_deviation_pct` is roughly comparable
  across all three (0.355/0.339/0.349), a closer picture than `on_time_rate`
  alone suggests.
## 5. Live interactive planner — `live_app.py`

(Since 2026-10-14 `live_app.py` only lists the pages; this page's code is
`app_pages/single_delivery.py`, and the multi-stop page is
`app_pages/fleet_rounds.py` -- see 12.7. References to `live_app.py` below
and in sections 6-11 mean the single-delivery page.)

Streamlit upgrade of the static dashboard (4.3's "upgrade to a small
Streamlit app" follow-up). User picks source/destination on a Leaflet map
of Bengaluru (or one of the 13 named depots / BESCOM EV stations), a
traffic snapshot (date/hour/weather/incident), a priority and a vehicle;
"Plan route" runs **live GNN v3 inference** (~0.2-0.4s on CPU for all
15,083 edges) and then `pipeline.run_full_pipeline()` for that one
request (~8s on Groq), and shows the route vs. both baselines on the map,
headline metrics, and every agent's decision/reasoning. A toggle turns off
the LLM agents (solver-only, instant, no API tokens).

Run: `pip install -r requirements-live.txt`, then `streamlit run live_app.py`.

**Data it needs, not in git (from the team Drive, gitignored):**
`gnn_checkpoints_v3/{best_model.pt,normalizer.pt}` (Drive `Capstone/GNN/gnn_checkpoints_v3/`),
`dataset_kaggle/test/snapshot_*.pt` (381 files, ~690 MB, Drive `Capstone/GNN/dataset_kaggle/test/`),
`bengaluru_graph/{nodes,edges}.geojson` (Drive `Capstone/bengaluru_graph/`).

**Node coordinates — resolves 4.5's "no lat/lon anywhere" note:**
`bengaluru_graph/nodes.geojson` feature *i* == GNN node index *i*, and
`edges.geojson` feature order == `edge_index` / `sample_costs` row order
(verified exactly), so edge geometry maps straight onto routes.

**Two fixes made to get live inference running locally (don't reintroduce):**
1. `gnn/gnn_inference.py` in git had lost its first 53 lines (module
   docstring + all imports) and could not be imported. Restored from the
   Drive copy; the rest of the file was byte-identical.
2. `gnn_checkpoints_v3/*.pt` were saved on a Colab GPU, so `torch.load`
   failed on CPU-only machines. `load_trained_model()` and
   `FeatureNormalizer.load()` in `gnn_trainer_v3.py` now pass
   `map_location="cpu"`; `GNNCostPredictor` still moves everything to CUDA
   when available. Verified: live predictions match `sample_costs/*.csv`
   to <=1.2e-4, R^2 ~0.95 on snapshot 07235.

**Evaluation re-run (all 5 snapshots, 75/75 reliable):** results in
`agents/evaluation_results/`. Mostly consistent with 4.5, one change worth
noting: on `mixed_priority`, `multi_agent`'s `on_time_rate` came out 0.24
(vs 0.40 in 4.5) and `baseline_carbon_only` led at 0.36 — the 4.5 claim
that `multi_agent` wins `on_time_rate` there did not reproduce.

## 6. Demo scenarios + four agent-layer fixes found while preparing them

`live_app.py` has a "🎬 Demo scenario" picker (6 one-click cases, see
`DEMO_SCENARIOS`), picked by scanning all 156 named-place pairs on real GNN
predictions for trips where fastest/balanced/greenest genuinely diverge.
Only ~3 pairs per snapshot give three distinct routes with a real
time-vs-carbon trade-off (GNN carbon correlates strongly with time), so the
scenarios reuse the best ones: Peenya WH -> Rajajinagar Hub (−26% CO2 for
~20s at 09:00), HSR EV -> Rajajinagar, Whitefield -> Peenya.
Time of day matters: 09:00/17:00 snapshots are ~27% slower network-wide
than 21:00, with mean delay_probability 0.66 vs 0.33.

The app no longer shows route-level `P(any delay)` (always ~100%: 1-prod(1-p)
over 30-90 segments at p~0.5). It shows expected time with delays (exact:
sum t*(1 + 0.65p), using the Execution agent's noise model) and the 90th
percentile arrival time (2,000-sample Monte Carlo).

Fixes (don't reintroduce):
1. **Echoed example ids.** Prompts showed example keys like "req-0"; the
   LLM copied them back instead of the real id ~2/6 times, the lookup
   missed, and agents silently fell back (Carbon Optimizer: "no reflection
   response -- reject", even for a −0.5kg/+1s candidate). Prompts now use a
   `<request_id>` placeholder + "use the exact request_id"; code-side
   `llm_client.align_ids()` remaps when exactly one id is missing and one
   unknown key came back (Carbon Optimizer propose/reflect, Execution
   reflect, Coordinator ev_range); Coordinator treats a non-empty
   reroute list with no valid ids as malformed and uses its fallback.
2. **Empty/invalid JSON replies.** `chat_json()` re-asks up to 2 times
   before raising ValueError (gpt-oss-120b intermittently returns empty
   content).
3. **ESG narrative units.** `mean_abs_time_deviation_pct` is a fraction
   (0.43 = 43%); the narrate prompt didn't say so and the LLM wrote
   "0.43%". Prompt now states units for it, `on_time_rate`,
   `ev_feasibility_rate`. Stored metric values unchanged.
4. **Swap audit trail.** An accepted Carbon Optimizer swap replaces the
   RouteResult (reasoning None). `decisions_log` entries now also carry
   `original_weights`, `original_reasoning`, `candidate_weights` (additive;
   ESG aggregate only reads `accepted`/`carbon_delta_kg`).

Fixes 1-2 can change evaluation outcomes (fewer silent default rejects).
The evaluation was re-run on the fixed code on 2026-10-11 — see 7.3;
`evaluation_results/` and `dashboard.html` now reflect that run.

## 7. Measured audit — 2026-10-11

Every runnable component was executed and each item of the proposed
architecture (Review 1 high-level design, objectives and roadmap) scored
working (1) / partial (0.5) / absent (0). **9 architecture stages: 39/65 =
60%. Including cross-cutting promises: 46/83 = 55%.** Scope decision
(Aayan, 2026-10-07): the team continues only with this repo's six agents,
not the Fleet/Trust/EV/ESG agents shown in the Review 1 deck.

### 7.1 What was run

| Check | Result |
|---|---|
| GNN v3 on all 381 test snapshots (5,746,623 edge rows) | R^2 0.9488 / 0.9687 / 0.9431 / 0.9445 (TT / delay / carbon / EV) — matches the training report to 4 decimals; 156 ms median per snapshot on CPU |
| Each agent's `__main__` smoke test + `pipeline.py` | 7/7 exit 0, no tracebacks, no silent fallbacks |
| Full evaluation, 5 scenarios x 3 systems x 5 snapshots | 75/75 reliable, 0 routing failures |
| `live_app.py`, 6 demo scenarios, LLM off and on (Streamlit AppTest) | 54/54 checks pass |
| `dashboard.py` | regenerated from the new summary |

Not re-run: GNN training (needs the 6,092 train snapshots + GPU; verified
via `training_log.json`, best epoch 11 of 19, and the reproduced test
metrics) and the Drive data-building scripts (code read, outputs verified).

### 7.2 Data-layer facts (don't restate the old claims)

- **Real-data coverage is 3.2%, not ~60%.** `match_report.csv` sums to 519
  name matches = 481 unique edges (3.82% of road length); CMH Road, Sony
  World Junction and Yeshwanthpur Circle match nothing. Reproduced locally
  from `edges.geojson` road names with the script's own keyword map.
- **Hourly variation is synthetic.** `kaggle_to_snapshots.py` scales each
  day's Kaggle value by `HOUR_MULTIPLIERS` (e.g. 09:00 x1.45, 21:00 x0.45)
  plus N(0, 0.03) noise. 952 dates x 8 hours = 7,616 snapshots.
- **Node "logistics" features are simulated** with `np.random` (20 random
  vehicles per snapshot, pending orders, charger availability) and are
  non-zero on <=0.3% of nodes.
- **Little weather/incident signal:** 7,336 of 7,616 snapshots are "Clear";
  `weather_impact` takes 2 values in the test split (0 / 0.15);
  `has_incident` is True for 7,608 of 7,616.
- **Truly dynamic edge features:** `congestion_index`, `weather_impact`,
  `event_disruption` (non-zero on 1.5% of edges), `road_condition_score`.
  The other 12 are static (graph topology is identical in every snapshot).
- **Labels are near-deterministic functions of the edge's own features:**
  formula-vs-label R^2 = 1.0000 (EV energy), 0.998 (carbon), 0.994 (travel
  time), 0.79 (delay).
- **Non-graph baselines** (trained on 60 test snapshots, scored on the other
  321): Ridge 0.981 / 0.800 / 0.993 / 0.994, GBM 0.993 / 0.900 / 0.993 /
  0.995, vs GNN 0.949 / 0.970 / 0.943 / 0.945. The GNN wins only on
  `delay_probability`.
- Test split = 48 dates, 2024-06-23 .. 2024-08-09. The 5 `sample_costs`
  snapshots used by the evaluation are all from 2024-06-23 (13:00-21:00).

### 7.3 Evaluation on the fixed code (mean of 5 snapshots) — SUPERSEDED by 9.6 (this table is the pre-Phase-2 solver)

| scenario | system | time_s | carbon_kg | mean_abs_dev | on_time | ev_feas |
|---|---|---|---|---|---|---|
| edge_overload | multi_agent | 6935 | 20.28 | 0.279 | 0.00 | n/a |
| edge_overload | baseline_time_only | 6228 | 21.07 | 0.290 | 0.27 | n/a |
| edge_overload | baseline_carbon_only | 6519 | 19.66 | 0.308 | 0.07 | n/a |
| ev_feasible | multi_agent | 1273 | 2.43 | 0.401 | 0.00 | 0.20 |
| ev_feasible | baseline_time_only | 1214 | 2.80 | 0.406 | 0.00 | 0.20 |
| ev_feasible | baseline_carbon_only | 1260 | 2.35 | 0.483 | 0.00 | 0.60 |
| ev_infeasible | multi_agent | 1887 | 4.43 | 0.307 | 0.00 | 0.00 |
| ev_infeasible | baseline_time_only | 1798 | 4.89 | 0.329 | 0.00 | 0.00 |
| ev_infeasible | baseline_carbon_only | 2013 | 4.27 | 0.360 | 0.00 | 0.00 |
| large_batch | multi_agent | 22271 | 54.48 | 0.341 | 0.35 | 1.00 |
| large_batch | baseline_time_only | 20442 | 55.26 | 0.352 | 0.38 | 1.00 |
| large_batch | baseline_carbon_only | 23685 | 53.87 | 0.357 | 0.42 | 1.00 |
| mixed_priority | multi_agent | 11822 | 26.67 | 0.370 | 0.16 | 1.00 |
| mixed_priority | baseline_time_only | 10444 | 28.70 | 0.339 | 0.20 | 1.00 |
| mixed_priority | baseline_carbon_only | 11274 | 24.45 | 0.349 | 0.36 | 1.00 |

Reading these honestly:
- Totals over the 5 scenarios: `multi_agent` is -3.9% carbon / +10.1% time
  vs `baseline_time_only`, and +3.5% carbon / -1.3% time vs
  `baseline_carbon_only`.
- `baseline_carbon_only` is both faster AND greener than `multi_agent` in 3
  of 5 scenarios (edge_overload, ev_feasible, mixed_priority).
- `on_time_rate` is noisy: `baseline_time_only` on edge_overload moved 0.07
  -> 0.27 between two runs with identical routes and seed, because the
  Execution agent's LLM judges lateness differently run to run. Don't draw
  conclusions from small on-time differences until the LLM is pinned
  (temperature 0).
- One snapshot evaluation costs ~42k Groq tokens; 5 snapshots ~ one free
  account's daily cap (run was split across two accounts).

### 7.4 Routing timings and the promised "Tsinghua SSSP" solver

The team told the review panel a newer shortest-path algorithm (the
"Tsinghua SSSP" algorithm, asymptotically faster than Dijkstra on sparse
graphs) would replace Dijkstra. **Status when this audit was written: not
implemented. Implemented on 2026-10-12 — see section 10.**

Measured on an idle machine, 60 random origin-destination pairs:
point-to-point Dijkstra 5.9 ms; single-source to all nodes 22.5 ms;
`RoutingSolver.build_graph()` 378 ms; `find_route()` end to end 587 ms; one
LLM plan 8-36 s. The search itself is ~1% of a routing call, so a faster
SSSP cannot visibly speed anything up at this size (7,095 nodes); caching
the graph can. Plan: implement it as a second, selectable solver, verify
identical distances to Dijkstra, benchmark on this graph and on larger
ones, and report the result as measured.

### 7.5 Known defects found (the first four and the dashboard label were fixed in Phase 2 — see 9)

- `build_graph()` uses `nx.DiGraph`, so the 83 parallel edges (166 rows
  sharing a (src, dst)) collapse to whichever row comes last: 15,000 of
  15,083 edges are routable.
- The graph is rebuilt on every `find_route()` call (see 7.4).
- The `delay` objective sums per-edge probabilities, which rewards fewer
  segments rather than less delay time; a high delay weight gave slower
  routes that were more likely to run late.
- Planning uses delay-free `travel_time_s`, so at rush hour nearly every
  route runs 30-45% over plan in the Execution simulation.
- `deadline_s` is never read; `is_ev` only affects the battery check, not
  carbon; nothing acts on `replan_triggered`.
- Absent from the code: distributed runtime, Neo4j, OR-Tools, LangGraph,
  live data APIs, a retraining/feedback loop, a pytest suite.
- Dashboard card label "Total planned distance-time" shows seconds only.

### 7.6 Per-stage scores

Data sources 6/9 · Ingestion 1/5 · Feature engineering 4.5/5 · Graph
construction 3.5/5 · GNN 4/6 · Agent decisions 6/10 · Carbon optimisation
4/7 · Execution and feedback 3/7 · ESG dashboard 7/11 · Distributed
computing 0.5/3 · Evaluation 4/7 · Tests and CI 1.5/4 · Documentation 1/4.

## 8. Plan agreed 2026-10-11 — read 8.1 before relying on any GNN accuracy number

### 8.1 STANDING LIMITATION: real traffic data covers 3.2% of the network

Everything in this repo runs on the Kaggle-derived dataset described in 7.2.
Until that changes, treat these as true of every result here:

- Only 481 of 15,083 edges carry real Kaggle values; ~97% are interpolated.
- Time-of-day variation is a hand-set multiplier, not measured traffic.
- Labels are near-deterministic formulas of each edge's own features.
- So the GNN's R^2 (0.94-0.97) measures how well it reproduces those
  estimates, **not** how well it predicts real Bengaluru traffic. Agent,
  routing and app behaviour are unaffected; any claim about traffic-
  prediction accuracy is.
- The Kaggle dataset has only 16 roads/junctions, so better matching alone
  cannot fix this (estimated ceiling ~4-5% of edges).
- Say it this way in reviews and the paper: "16 real monitoring locations;
  other roads are estimated by spatial interpolation."

Ways to address it (none is done yet):

| Option | What | Cost | Raises real coverage? |
|---|---|---|---|
| A | Fix the matching (3 unmatched roads, geometry-based) | ~0.5 day | Slightly |
| B | Collect real speeds from a traffic API (see 8.3) | weeks of calendar time + retrain | Yes — the only one that does |
| C | Sparse-sensor validation: hide one real road at a time, predict it, compare GNN vs plain interpolation | 2-3 days, no new data | No, but tests the GNN on real values |

A and C are planned in Phase 7. B is optional (8.3). Any of the three
changes the dataset or training setup, so it requires a GNN retrain;
otherwise the v3 checkpoint needs no retraining (its test metrics were
reproduced exactly, 7.1).

### 8.2 Phases

Before Review 3:
1. **Secure and correct — DONE 2026-10-11.** Committed and pushed; corrected
   the coverage claim; README added.
2. **Fix known defects (7.5) — DONE 2026-10-12, see 9.** Six fixes plus EV
   range on expected energy (9.5); evaluation re-run and dashboard
   regenerated (9.6).
3. **Tsinghua SSSP solver (promised to the panel) — DONE 2026-10-12, see
   10.** Implemented from the paper, selectable next to Dijkstra, verified
   against Dijkstra, benchmarked. It is 14-22x slower than Dijkstra on our
   graph, as expected at this size.
4. **Close the feedback loop — DONE 2026-10-13, see 11.** Mid-trip change
   of conditions (GNN re-scoring for rain, a congestion surge or the next
   snapshot; an assumed slowdown for an incident, which the GNN cannot
   model), a real re-route from the vehicle's junction, old and new route
   and the saving shown in the app, per-segment telemetry feeding a
   correction store for later plans. Measured on 120 trips: it helps for
   local changes (incident: 9.0 min saved per re-plan, 75% -> 18% of trips
   15%+ late) and correctly does nothing for city-wide ones (rain, next
   snapshot). Justification for having no live feed: trace-driven
   simulation on recorded snapshots, with a swappable snapshot source.
5. **Settle remaining promises — DONE 2026-10-14, see 12.** Built: a
   distributed runtime (Dask worker processes over TCP), a small OR-Tools
   multi-stop planner, vehicle-specific carbon, delivery deadlines. Written
   up as future work (12.6), for the guide to confirm: Neo4j, LangGraph,
   30-second live rebuild, GNN retraining from feedback, edge-cloud
   deployment.
6. **Review 3 package — DONE 2026-10-14, see 13.** Deck in the PES
   template (19 slides, speaker notes, native charts) in the team folder,
   demo script with measured numbers in `docs/review3_demo_script.md`,
   audit re-scored on the same checklist (60% -> 72% of the nine stages).

For the paper:
7. **Strengthen the GNN:** baseline table (linear, GBM, MLP, GraphSAGE);
   forecasting version (predict the next snapshot); loss-rebalance
   experiment; accuracy split by real vs interpolated edges; options A and C
   from 8.1.
8. **Paper-grade evaluation:** solve the token limit (paid tier / local
   model / fewer calls); snapshots across all 48 test days, 50+ random
   trips, several seeds, confidence intervals; ablations (LLM vs fixed
   presets, no Coordinator, no Carbon Optimizer, 2-3 LLMs); explain why
   `baseline_carbon_only` beats `multi_agent` in 3 of 5 scenarios; re-plan
   vs no re-plan; Tsinghua vs Dijkstra; tokens, latency and cost per
   delivery.
9. **Tests and CI:** pytest with a fake LLM; GitHub Actions.
10. **Write the paper.**

### 8.3 OPTIONAL, not scheduled: real traffic from the TomTom API (option B)

Decision (Aayan, 2026-10-11): defer; may be done after everything else. The
snapshot source is swappable, so it can be added at any point without
changing the agents, routing or app design.

How it would work: a collector pulls current speed and free-flow speed for
Bengaluru at the 8 times of day; TomTom road shapes are matched to our edges
by geometry; a 1-2 day pilot measures real coverage before committing.

What it gives:
- **Live mode in the app** as soon as the collector and matching exist: plan
  on the traffic right now instead of a recorded snapshot.
- **A real historical dataset** only by collecting day by day — the API has
  no history. 2 weeks = 112 snapshots, 4 weeks = 224. Cannot be backfilled,
  so if it is wanted for the paper, start collecting before Phase 10;
  starting after the paper means retraining, re-evaluating and rewriting
  results.

Free tier, from TomTom's pricing page on 2026-10-11 (re-check before use):
Traffic Flow vector tiles 200,000 requests/month; Flow Segment Data
20,000/month; no credit card. Our network fits in 146 tiles at zoom 14, so
8 pulls a day is ~35,000 requests/month. That allows a city-wide refresh
about every 30 minutes, not the "every 30 seconds" in the Review 1
objectives.

Not confirmed: whether TomTom's terms allow storing the data, training on it
and publishing results (check first — biggest risk); how much of the
tertiary network (49% of our edges) TomTom covers; what happens past the
free limit; HERE's free tier. Only speed/congestion would become real:
weather needs a separate API, and carbon and EV energy would still be
computed from the speeds. The data would be from late 2026, so it is a new
dataset alongside the 2022-2024 Kaggle one, not a patch to it.

## 9. Phase 2 — routing and repeatability fixes (2026-10-11)

Six fixes. Four of them change which routes are chosen, so every number in
7.3 and in `agents/evaluation_results/` predates this section (9.4).

### 9.1 `gnn/routing_solver.py`

- **Graph cached.** The weighted DiGraph used to be rebuilt row by row on
  every `find_route()`. It is now built once in `__init__` with normalized
  per-edge objective values, and weights are applied by a cost function
  passed to `nx.shortest_path`. Median `find_route()`: 162 ms -> 3.7 ms
  (one-time setup ~35-110 ms per solver). `build_graph()` still returns a
  graph with a `"weight"` attribute (the Dispatcher uses it).
- **Parallel edges: `canonical_edges(costs_df)`.** 83 junction pairs are
  joined by two different roads. `nx.DiGraph` kept whichever row came last,
  which was the SLOWER road in 53 of 83 pairs (by up to 105x: a 9m link vs a
  1.2km loop). `canonical_edges()` keeps the faster row per (src, dst) and is
  the single rule every (src, dst)-keyed consumer must use: the solver, the
  Execution agent's `edge_costs`, and `live_app.py`. Don't build a
  `{(src, dst): row}` dict straight from `costs_df` again. (In 7 of the 83
  pairs the slower road has lower carbon; that alternative is dropped.)
- **`delay` objective is now `p * travel_time_s`** (delay exposure), not
  `p`. Summing a bare probability charged every segment about the same, so a
  high delay weight rewarded fewer segments, not less delay: Koramangala ->
  Hebbal with `{time .3, delay .7}` was 57.5 min expected (old) vs 38.3 min
  with time only; it is now 38.3.
- **`time` objective is now EXPECTED time** `t * (1 + DELAY_SLOWDOWN * p)`,
  and `find_route()` also returns `expected_travel_time_s` and
  `expected_carbon_kg` next to the unchanged delay-free
  `total_travel_time_s` / `total_carbon_kg`. `DELAY_SLOWDOWN` (0.65) is
  derived from `DELAY_EXTRA_TIME_RANGE`, which now lives in this module and
  is imported by the Execution agent, so planner and simulator share one
  definition. The `carbon` objective itself is unchanged (GNN carbon).
- Regression-tested against the previous solver: with the delay features
  switched off (`delay_slowdown=0`, delay weight 0, no parallel edges in the
  input) 150 routes have identical cost.

### 9.2 Agents

- `RouteResult` gained `expected_travel_time_s` and `expected_carbon_kg`.
- **Execution** measures `time_deviation_pct` against the EXPECTED time
  (was the delay-free time), and reports `expected_time_s`. Before, the
  simulation's own average delay showed up as a +30-45% "deviation" on
  nearly every rush-hour route and almost everything was flagged for
  re-planning. Now: on `large_batch` over 6 seeds, simulated/expected is
  0.988 for time and 0.990 for carbon, mean per-route deviation 4.5%, no
  re-plans. Consequence to know: with deviations this small the 15% floor
  is rarely crossed, so the Execution agent's LLM reflection almost never
  runs until Phase 4 injects real disruptions. The hard EV rule is
  unaffected.
- **Carbon Optimizer** `time_delta_s` / `original_time_s` are expected times.
- **ESG Reporter** adds `total_expected_time_s` and
  `total_expected_carbon_kg`; the narrate prompt tells the LLM to judge
  actual against expected (it had started describing actual carbon
  exceeding the delay-free plan as a problem).
- **Dashboard:** label "Total planned distance-time" -> "Total planned time
  (no delays)"; added expected-time and expected-carbon cards; the per-route
  chart compares expected vs actual.

### 9.3 `agents/llm_client.py`

- **`LLM_TEMPERATURE` (default 0)** applied to every call unless the caller
  passes its own; JSON-parse retries step it up (+0.3 per retry) because
  re-asking at 0 can return the same unusable reply. Empty string disables
  it for a model that rejects the parameter.
- **Call log:** `logs/llm_calls.jsonl` (git-ignored; `LLM_LOG_PATH`
  overrides, empty string disables). One line per successful call with
  messages, reply, model, temperature, latency, token counts and rate-limit
  retries. Use it for the token/latency/cost numbers Phase 8 needs.

### 9.4 Verified, and what is still pending

Verified after the changes: 7/7 agent smoke tests; `live_app.py` 54/54
checks on the six demo scenarios (LLM off and on; LLM-off plans now take
0.3-1.1 s); solver regression tests. The demo scenarios were re-measured and
rewritten: the old "agents repair a bad plan" case (Whitefield -> Peenya,
balanced, 86.5 min) no longer exists because that route was a product of
the delay-objective bug (now 66.5 min delay-free).

The evaluation and dashboard were re-run on 2026-10-12 — see 9.6.

### 9.5 EV range planned on expected energy (2026-10-12)

The open question left in Phase 2 was settled (Aayan: include it). The
Coordinator's EV-range check now compares `expected_ev_energy_pct` (delays
included) with the battery, not the delay-free `total_ev_energy_pct`. The
Execution agent's hard rule judges the simulated drive, which uses ~35% more
at rush hour, so a route planned at 75-100% of its battery on the delay-free
figure passed planning and then failed on the road. The solver returns
`expected_ev_energy_pct`; `RouteResult` carries it; the negotiation payload
gives the LLM `route_ev_energy_kwh`; `live_app.py` shows expected EV energy.

- It is the mean with no reserve: a route planned exactly at the battery
  limit can still run out about half the time. A reserve margin would be a
  further refinement.
- The solver's `ev` objective was left on delay-free energy on purpose:
  measured over 200 random trips, minimising expected energy instead saves
  1.1% more on average (median 0%, p90 3.5%). Same for `carbon` (0.7%).
- The Coordinator's own smoke test was recalibrated: its EV case used a
  4.8 kWh battery chosen against delay-free energy (4.92 kWh default route).
  With expected energy the default route is 6.07 kWh and the efficient one
  5.76 kWh, so the test battery is now 5.9 kWh to stay "just resolvable".
- `golden_scenarios.py` was NOT recalibrated: `ev_feasible`'s 3.3 kWh
  battery was already chosen against simulated actual energy (4.5).

### 9.6 Evaluation on the Phase 2 code (2026-10-12, mean of 5 snapshots)

75/75 runs reliable, 0 routing failures. 255 LLM calls, 231,930 tokens
(~46k per snapshot; run split across two Groq accounts), all at temperature
0. `dashboard.html` regenerated from this summary. Times and carbon below
are EXPECTED values (typical delays included), the quantity the plan
promises; the CSVs also hold the delay-free and simulated-actual columns.

| scenario | system | expected_time_s | expected_carbon_kg | mean_abs_dev | on_time | ev_feas |
|---|---|---|---|---|---|---|
| edge_overload | multi_agent | 8559 | 26.02 | 0.039 | 1.00 | n/a |
| edge_overload | baseline_time_only | 7950 | 26.19 | 0.039 | 1.00 | n/a |
| edge_overload | baseline_carbon_only | 8557 | 24.52 | 0.029 | 1.00 | n/a |
| ev_feasible | multi_agent | 1761 | 3.27 | 0.063 | 0.60 | 0.60 |
| ev_feasible | baseline_time_only | 1660 | 3.87 | 0.037 | 0.00 | 0.00 |
| ev_feasible | baseline_carbon_only | 1761 | 3.27 | 0.063 | 0.60 | 0.60 |
| ev_infeasible | multi_agent | 2729 | 5.92 | 0.026 | 0.00 | 0.00 |
| ev_infeasible | baseline_time_only | 2385 | 6.13 | 0.037 | 0.00 | 0.00 |
| ev_infeasible | baseline_carbon_only | 2795 | 5.91 | 0.037 | 0.00 | 0.00 |
| large_batch | multi_agent | 28868 | 70.74 | 0.047 | 0.97 | 1.00 |
| large_batch | baseline_time_only | 27378 | 72.20 | 0.052 | 0.98 | 1.00 |
| large_batch | baseline_carbon_only | 32142 | 71.35 | 0.049 | 0.97 | 1.00 |
| mixed_priority | multi_agent | 14848 | 35.38 | 0.039 | 1.00 | 1.00 |
| mixed_priority | baseline_time_only | 13884 | 37.38 | 0.031 | 1.00 | 1.00 |
| mixed_priority | baseline_carbon_only | 15449 | 33.32 | 0.044 | 1.00 | 1.00 |

Reading these honestly:
- **The plan's promise is now accurate.** Simulated-actual totals are within
  about 1-7% of expected in every row, and `mean_abs_time_deviation_pct` is
  2.6-6.3% (it was 28-48% before Phase 2).
- **`on_time_rate` is ~1.0 wherever no EV runs out of battery.** In the two
  EV scenarios it equals the EV feasibility rate, because the only re-plans
  left are the hard battery rule.
- **EV feasibility on `ev_feasible`: `multi_agent` 0.20 -> 0.60**, now level
  with `baseline_carbon_only` (0.60) and ahead of `baseline_time_only`
  (0.00). This is the effect of planning EV range on expected energy (9.5).
- **Totals over the 5 scenarios (expected values):** `multi_agent` is -3.0%
  carbon / +6.6% time vs `baseline_time_only`, and +2.1% carbon / -6.5% time
  vs `baseline_carbon_only`.
- **`baseline_carbon_only` now beats `multi_agent` on both time and carbon
  in 1 of 5 scenarios (edge_overload), down from 3 of 5.** `multi_agent`
  lies between the two baselines on both measures in ev_feasible,
  ev_infeasible and mixed_priority, and on `large_batch` it has the lowest
  expected carbon of the three (70.74 vs 71.35 / 72.20) while being 10%
  faster than `baseline_carbon_only`.
- edge_overload remains the honest loss: the negotiated detour around the
  bottleneck costs time and the result is no greener than the carbon
  baseline.
- Still 5 snapshots from a single day (2024-06-23, 13:00-21:00) and one LLM;
  see Phase 8 before drawing strong conclusions.

## 10. Phase 3 — the "Tsinghua SSSP" algorithm (2026-10-12)

### 10.1 What was implemented

`gnn/bmssp.py` implements the algorithm of R. Duan, J. Mao, X. Mao, X. Shu,
L. Yin, "Breaking the Sorting Barrier for Directed Single-Source Shortest
Paths", arXiv:2504.17033 (v2, 30 Jul 2025) — O(m log^(2/3) n) against
Dijkstra's O(m + n log n). Written from the paper's Algorithm 1
(FindPivots), Algorithm 2 (BaseCase), Algorithm 3 (BMSSP) and the Lemma 3.3
partial-sorting structure (`BlockQueue`). No third-party code. Community
implementations exist (Java, Rust) and were not used.

Parameters on the Bengaluru graph (n = 7,095, m = 15,000): k = 2, t = 5,
3 recursion levels, queue block sizes 1024 / 32 / 1. Maximum in- and
out-degree is 4.

Ties are handled as the paper's Assumption 2.1 prescribes: every label is
the triple (distance, hops, vertex id) and every bound is such a triple.
This matters here because robust scaling gives some edges cost exactly 0
(24 with balanced weights).

### 10.2 Where it differs from the paper — say these when reporting it

1. **No constant-degree transformation.** Needed for the running-time
   proof, not correctness; our graph's degree is already at most 4.
2. **`BlockQueue` simplifications.** The paper's layout is kept (two block
   sequences, blocks of at most M pairs, an upper bound per inserted block),
   but block bounds live in a sorted list searched by bisection instead of a
   balanced search tree, and medians / "the M smallest" are found by sorting
   a block instead of linear-time selection. Correctness is unaffected; the
   code therefore does **not** strictly attain the O(m log^(2/3) n) bound.
3. **One safeguard the pseudocode leaves implicit.** When a recursive call
   completes a vertex that is still waiting in the current level's queue
   with an older, larger value, that stale pair is deleted. The paper's
   pseudocode never removes it, although its Lemma 3.10 and Remark 3.8
   assume it is not there. Measured: the safeguard fires about 220 times per
   source on the Bengaluru graph. Without it distances are still correct
   (0 mismatches on 200 random graphs) but vertices are completed more than
   once (1,871 duplicate completions on those graphs); with it, none are.

### 10.3 Integration

- `RoutingSolver(costs_df, algorithm="dijkstra" | "bmssp")`, or the
  `ROUTING_ALGORITHM` environment variable (agents build their own solver
  and read it). Default stays `dijkstra`, so nothing changes unless asked.
- `find_route()` records `last_search_s`; `route_cost(path, weights)` prices
  a path under given weights.
- BMSSP is single-source: it settles every reachable vertex and the route is
  read off the predecessor tree, so it is not comparable with networkx's
  early-exit point-to-point Dijkstra call for call.
- Both return a minimum-cost route. Where several routes tie on cost they
  may return different ones, so keep the algorithm fixed within an
  evaluation. On 174 real routes (3 weightings) and the six demo scenarios
  the routes were identical.
- `live_app.py`: "Shortest-path algorithm" selector, and (added 2026-10-13)
  a "Dijkstra vs Tsinghua BMSSP" section on every plan. Both algorithms
  return the same route, so the selector alone changes nothing visible; the
  section shows what does differ. It calls
  `RoutingSolver.compare_algorithms(source, target, weights)`, which runs
  the same single-source search with `bmssp.dijkstra_reference()` (a
  counting pure-Python Dijkstra) and with BMSSP on identical adjacency lists
  and reports time, relaxations, BMSSP's internal counts and whether the
  routes agree. Below it, an Altair chart of `time_ratio` against graph size
  from `gnn/benchmark_results/sssp_benchmark.csv`, with a table view. Costs
  about 0.2-0.6 s per plan.
- After changing anything under `gnn/` or `agents/`, restart Streamlit: a
  page refresh re-reads only `live_app.py`, not imported modules.

### 10.4 Verification

`tests/test_bmssp.py` (pytest-compatible, also runs as a script, no LLM):
tiny graphs, zero-weight cycles, unreachable vertices, a long chain with
expensive shortcut edges, random graphs with five weight styles (floats,
many ties, zeros and ones, all equal, all zero), a grid, the queue against a
sorted-list model, and the Bengaluru graph (4 weightings x 5 sources). All
pass. A separate stress run of 400 random graphs (n from 3 to 6,000) had 0
mismatches against Dijkstra. The predecessor tree is checked too: every
vertex's distance equals its parent's plus the edge weight.

### 10.5 Benchmark (`gnn/benchmark_sssp.py`, results in `gnn/benchmark_results/`)

Both algorithms as single-source searches to every vertex, same adjacency
lists, pure Python, median wall time. "Theory" is the ratio of the two
bounds' leading terms, m log^(2/3) n / (m + n log n), constants ignored.

| graph | n | m | Dijkstra | BMSSP | BMSSP slower by | relaxations | theory |
|---|---|---|---|---|---|---|---|
| Bengaluru, time only | 7,095 | 15,000 | 26 ms | 576 ms | 22.0x | 3.6x | 0.78 |
| Bengaluru, carbon only | 7,095 | 15,000 | 32 ms | 465 ms | 14.4x | 3.7x | 0.78 |
| Bengaluru, balanced | 7,095 | 15,000 | 34 ms | 586 ms | 17.4x | 3.6x | 0.78 |
| Bengaluru, greenest | 7,095 | 15,000 | 36 ms | 564 ms | 15.9x | 3.7x | 0.78 |
| grid 32x32 | 1,024 | 3,968 | 3 ms | 34 ms | 11.4x | 3.3x | 1.30 |
| grid 64x64 | 4,096 | 16,128 | 12 ms | 106 ms | 9.0x | 3.2x | 1.29 |
| grid 128x128 | 16,384 | 65,024 | 111 ms | 842 ms | 7.6x | 3.2x | 1.28 |
| grid 256x256 | 65,536 | 261,120 | 432 ms | 3,182 ms | 7.4x | 3.1x | 1.27 |
| grid 512x512 | 262,144 | 1,046,528 | 2,107 ms | 14,698 ms | 7.0x | 3.2x | 1.25 |
| sparse random | 1,000 | 2,098 | 4 ms | 42 ms | 12.1x | 3.4x | 0.81 |
| sparse random | 4,000 | 8,397 | 26 ms | 166 ms | 6.3x | 3.4x | 0.78 |
| sparse random | 16,000 | 33,598 | 133 ms | 629 ms | 4.7x | 2.4x | 0.76 |
| sparse random | 64,000 | 134,400 | 688 ms | 2,972 ms | 4.3x | 2.4x | 0.74 |
| sparse random | 256,000 | 537,600 | 4,081 ms | 28,440 ms | 7.0x | 2.1x | 0.72 |

Reading these honestly:
- **BMSSP is slower than Dijkstra at every size tested**, by 4-22x, and
  14-22x on our graph. It performs 2-4x as many edge relaxations and has far
  more bookkeeping per step.
- **The gap narrows as graphs grow** on the grid (11.4x at 1k vertices to
  7.0x at 262k), which is the direction the theory predicts, but there is no
  crossover anywhere near these sizes. The largest random graph is a single
  source and is noisier (7.0x after 4.3x).
- Even the theory's own ratio is only 0.72-0.78 on graphs this sparse and
  above 1 on the grid: the advantage is a slowly growing log factor.
- In the application the search is about 1% of a routing call and far less
  of an LLM plan, so the choice of algorithm does not change what a user
  sees.

**What to tell the panel:** the algorithm is implemented from the paper,
returns the same shortest paths as Dijkstra on every test, and is selectable
in the app; it is asymptotically faster but, measured, slower at city scale,
so Dijkstra remains the default. "Better than Dijkstra" should be stated as
"asymptotically faster", not "faster in practice here".

## 11. Phase 4 — the closed feedback loop (2026-10-13)

### 11.1 What was open, and what closes it

The architecture has always shown an arrow from Execution back to routing.
In the code, `ExecutionAgent.run()` simulated a whole trip and only then set
`replan_triggered`: a flag raised after the trip was over, which nothing
read. Phase 4 adds the two loops that arrow stands for, in
`agents/feedback_loop.py`:

- **Short loop, within one trip** (`ClosedLoopExecutor`): the vehicle drives
  its plan, conditions change part-way, the planner is told, and it decides
  whether to re-route from the junction the vehicle is at.
- **Long loop, across trips** (`FeedbackStore`, `fleet_learning()`): nobody
  tells the planner anything. Vehicles report how long each segment took;
  segments that take far longer than predicted get a correction factor, and
  the next vehicle is planned on corrected costs.

It is the Execution/Monitoring agent's job extended, not a seventh agent.
`pipeline.run_full_pipeline` and the evaluation harness are unchanged, so
9.6's numbers still stand; the loop is a separate stage used by the app and
by `agents/replan_experiment.py`.

**A late trip with no new information is not re-planned, and should not
be.** If the roads have not changed, the rest of a lowest-cost route is
still the lowest-cost route from any junction on it. Time already lost to
ordinary delays cannot be bought back by re-routing; only new information
about the roads ahead can. That is why the old flag could never have led
anywhere, and why the loop needs a model of "conditions changed".

### 11.2 Files

| File | What it holds |
|---|---|
| `agents/feedback_loop.py` | `drive()`, `ClosedLoopExecutor`, `FeedbackStore`, `fleet_learning()`, a no-LLM smoke test |
| `agents/disruptions.py` | world-level incident: `edge_at_fraction()`, `edges_near()`, `apply_incident()` (no torch) |
| `gnn/snapshot_disruptions.py` | GNN-level changes: `with_rain()`, `with_congestion_surge()` |
| `gnn/routing_solver.py` | new `path_totals(path)` and `reference_costs=` (11.5) |
| `agents/execution_agent.py` | noise model extracted as `simulate_edge(row, rng)`; outputs unchanged (117 of 117 seeded runs identical) |
| `agents/replan_experiment.py` | re-plan against no re-plan over many trips; writes three CSVs to `agents/evaluation_results/` |
| `tests/test_feedback_loop.py` | 16 tests, no LLM, no torch |
| `live_app.py` | section "On the road" (11.9) |

### 11.3 What a "change" is — two kinds, and what each can claim

There is no live traffic (8.1), so a change is a second set of per-edge
costs: the world the plan was made on and the world after.

**GNN-level**: edit the snapshot's features, run the GNN again, and the
model says how the roads respond. Measured on the 2024-06-24 snapshots:

| Change | How | What the GNN predicts |
|---|---|---|
| Rain, whole city | `weather_impact` = 0.6, `road_condition_score` - 0.06 | time x1.18 / x1.21 / x1.22 at 09:00 / 13:00 / 21:00; mean delay probability 0.66 -> 0.88, 0.49 -> 0.67, 0.33 -> 0.49; carbon and EV +1% to +5% |
| Congestion surge on chosen edges | `congestion_index` -> 1.0 | on the surged edges, time x1.01 at 09:00 and 17:00 (already 0.98-0.99), x1.14 at 13:00, x1.28 at 21:00; other edges unchanged |
| Next snapshot of the day | load it | 09:00 -> 11:00 x0.85, 15:00 -> 17:00 x1.24, 19:00 -> 21:00 x0.62 on expected time; half of all roads within a few percent of that factor |

Caveats: rain is rare in training (`weather_impact` mean 0.013, s.d. 0.073)
and absent from the test split (373 Clear, 8 Overcast), so the rain response
cannot be checked against labelled rain data here. (Added 2026-10-14, 12.1: it
does match the rule that generated the travel-time labels, x1.19.) A surge
has no effect at rush hour because congestion is already at the top of its
range.

**World-level**: `apply_incident()` multiplies time, carbon and EV energy on
the affected edges by a stated factor (default 4) and raises their delay
probability to at least 0.9. This is an **assumption, not a prediction**. It
exists because **the GNN cannot model an incident**: raising
`event_disruption` to 0.6 or 1.0 on a route's edges made it predict those
edges 10-24% *faster* and less likely to be delayed (time x0.76-0.90).
**Corrected 2026-10-14 (12.1):** this is not a training failure. In the
labels themselves, segments with `event_disruption` > 0 take about 25% *less*
time than the rule that sets the other 97% of segments, so the GNN learned
what the data says. In this dataset an incident flag does not mean a slower
road. Do not route incidents through the GNN; say "we assume a 4x slowdown".

### 11.4 The re-plan decision

At the change the loop prices two options on the new conditions: finish the
planned route, or take the best route from the vehicle's junction (same
weights the route was planned with). Deterministic rules first, in order;
the LLM is asked only if none settles it:

1. `same_route` — the best route from here is the rest of the plan.
2. `ev_range` — hard rule. If finishing the plan is expected to need more
   energy than the battery has left and the alternative fits, switch. Never
   switch to a route that does not fit when the plan does.
3. `dominated` — never switch to a route expected to be both slower and
   more polluting.
4. `small_gain` — under 5% better on the route's own weighted cost
   (`REPLAN_MIN_GAIN_PCT`): not worth turning a vehicle around for.
5. `llm` — weigh the time and CO2 changes against the request's priority.
   With the LLM off, or unreachable, the answer is `policy` /
   `policy_fallback`: switch, since rule 4 was passed.

Both futures are then simulated, with **the same random numbers on every
segment they share** (each segment's noise is seeded by the run seed and the
segment). The difference between the two outcomes is the effect of the
decision, not of luck. The report carries the decision, its rule and
reasoning, both outcomes, the saving, and per-segment telemetry.

One change per trip, applied when the vehicle reaches a junction; there is
no continuous monitoring.

### 11.5 Pricing a changed world: `reference_costs`

`RoutingSolver` scales each objective to [0, 1] between its 1st and 99th
percentile and clips. For re-planning that has two faults: the scale moves
when the costs move, and a road already at the cap looks no worse after it
becomes 4x slower. `RoutingSolver(costs_after, reference_costs=costs_before)`
prices every edge in the original plan's units instead: an unchanged edge
costs exactly what it did, and an edge whose prediction is now r times
higher costs r times more, uncapped. With unchanged costs it reproduces the
plain solver exactly (tested). Default behaviour is untouched.

### 11.6 The long loop: `FeedbackStore`

Per segment it keeps the ratios actual time / expected time that vehicles
report (expected = the GNN's own, delays included). The correction factor is
a shrunk mean, (2 + sum of ratios) / (2 + reports), and a factor within the
deadband of 1.0 counts as 1.0. The deadband is derived from the noise model:
the worst single report a healthy road can give is +10% jitter with a delay
at its 2.0x maximum, a ratio of 2.2, which gives (2 + 2.2) / 3 = 1.4 — so
0.4. One unlucky but normal report can never create a correction; a 4x
incident reports at least 2.6 and clears it at once. `corrected(costs_df)`
multiplies time, carbon and EV energy by the factor; `save()` / `load()`
write JSON.

Limits: corrections are per segment (nothing spreads to neighbouring
roads); they are not persisted by the app; they are not fed into GNN
retraining (out of scope, 8.2 item 5). **They also never expire, and that
needs fixing before any real use:** once a segment is flagged, vehicles are
routed around it, so no new reports arrive and the flag stays even after
the road has cleared. An expiry time, or an occasional vehicle sent through
to check, would be needed.

### 11.7 Verification

- `python tests/test_feedback_loop.py` — 16 of 16 pass (hand-built ladder
  network plus one check on `sample_costs/`): `path_totals` equals
  `find_route`; `reference_costs` changes nothing when costs are unchanged
  and does not cap a slowed road; incident and ball construction; no change
  -> no re-plan; an incident ahead is avoided and saves time on 40 of 40
  seeds; an incident already passed changes nothing; the uninformed run's
  outcome equals the informed run's counterfactual; EV hard rule both ways;
  small gain not taken; paired and reproducible noise; the store ignores the
  worst healthy report and learns a 4x incident; fleet learning finds the
  informed route and learns nothing in an unchanged world.
- `tests/test_bmssp.py` still 6 of 6 after the solver change, and 200 of
  200 random routes (2 snapshots, 4 weightings) are identical to the
  previous commit's solver, so 9.6 and every earlier result still hold.
- App: automated check of "On the road", 3 scenarios x 5 kinds of change x
  told / not told, LLM off — 150 of 150 checks pass. One run in the browser
  with the LLM on (11.9).
- Found and fixed along the way: `FeedbackStore.corrected()` raised on the
  GNN's float32 columns (pandas refuses the float64 product); now covered by
  a test.

### 11.8 Does it help? (`agents/replan_experiment.py`, 120 trips)

120 random trips of at least 10 minutes on 2024-06-24 (40 each at 09:00,
13:00, 21:00; priorities rotated; mean 39 minutes, 67 segments). The change
is reported to the planner 30% of the way into each trip. Decision by the
fixed policy, so the run uses no tokens and is repeatable. Results in
`agents/evaluation_results/replan_experiment.csv` and `replan_summary.csv`.

| | nothing | incident (assumed 4x) | rain | surge | next snapshot |
|---|---|---|---|---|---|
| trips | 120 | 120 | 120 | 120 | 80 |
| mean slowdown of the road ahead | 1.00 | 1.52 | 1.31 | 1.24 | 0.95 |
| re-planned | 0% | 93% | 0% | 44% | 0% |
| 15%+ late, finishing the plan | 0% | 75% | 77% | 46% | 1% |
| 15%+ late, with the loop | 0% | 18% | 77% | 25% | 1% |
| mean trip, finishing the plan (min) | 39.4 | 50.8 | 47.7 | 44.6 | 43.4 |
| mean trip, with the loop (min) | 39.4 | 42.4 | 47.7 | 42.4 | 43.4 |
| mean saved when re-planned (min) | - | 9.0 | - | 5.1 | - |
| mean CO2 saved when re-planned (kg) | - | 1.65 | - | 0.61 | - |
| re-plans that ended slower | - | 2.7% | - | 5.7% | - |

(The last column has 80 trips because 21:00 has no later snapshot, so its
means are not comparable with the others.)

Why the loop did or did not re-plan: nothing 120 `same_route`; incident 112
`policy`, 5 `small_gain`, 3 `same_route`; rain 85 `same_route`, 32
`small_gain`, 3 `dominated`; surge 53 `policy`, 34 `same_route`, 32
`small_gain`, 1 `dominated`; next snapshot 74 `same_route`, 5 `small_gain`,
1 `dominated`. Surge by hour: 0% re-planned at 09:00, 42% at 13:00, 90% at
21:00.

Long loop (the same incident, never reported, five vehicles in a row, 120
trips; `fleet_learning_experiment.csv`):

| vehicle | mean trip (min) | same drive, no feedback | same drive, planner fully informed | on the informed route |
|---|---|---|---|---|
| 1 | 50.6 | 50.6 | 41.3 | 1% |
| 2 | 43.3 | 50.7 | 41.6 | 68% |
| 3 | 41.5 | 50.7 | 41.2 | 82% |
| 4 | 41.6 | 51.1 | 41.5 | 90% |
| 5 | 41.4 | 50.9 | 41.4 | 90% |

Vehicles 2-5 recover 94% of the gap between never learning and being fully
informed. The first vehicle on the informed route was the 2nd in 82 trips,
the 3rd in 17, the 4th in 8, the 5th in 2. In 10 trips none of the five
reached it: 8 settled on a different route that beat the original plan in
all 8, and 2 stayed on the original plan, where the informed route was no
faster anyway. 4.6% of vehicle 2-5 drives were slower than the same drive on
the original plan.

**Reading these honestly**
- The control passes: with nothing changed, 0 of 120 trips were re-planned.
- **Re-planning helps for local changes, not city-wide ones.** Rain slows
  the road ahead by 31% and makes 77% of trips late, and the loop re-plans
  none of them: when every road slows by about the same factor, the best
  road is still the best. The same holds for the next snapshot, because the
  hour is a city-wide multiplier in this dataset (7.2). This is a property
  of the data as much as of the loop.
- **The incident is placed on the vehicle's route by construction**, and its
  4x is assumed. 93% is how the loop behaves when a change does hit the
  route, not how often that happens. The saving scales with the assumed
  slowdown.
- The surge is the only local change that is a GNN prediction, and it only
  bites off-peak.
- A re-plan is a bet on expected values: 3-6% of them ended slower in the
  paired simulation.
- **The LLM added an explanation, not a different decision.** On 3 demo
  trips x 3 priorities x incident and surge at 13:00, 16 cases reached the
  LLM and it chose the same as the fixed policy in 16 of 16. All 16 were
  faster *and* cleaner: in this data a slow road is also a dirty one, so a
  real time-against-carbon trade-off at re-plan time is rare (in the
  120-trip run, 156 of 165 re-plans were better on both, 8 traded one for
  the other). Phase 8's ablation should report this, not assume the LLM
  matters here.

**What to tell the panel:** the loop is closed in simulation. A change on
the road triggers a real re-route from the vehicle's position, with the old
route, the new route and the saving shown, and vehicle telemetry feeds a
correction store that later plans use. There is no live feed: a change is
either the GNN re-scoring changed conditions (rain, congestion, next
snapshot) or an assumed incident slowdown, because the GNN cannot model
incidents. Not done: continuous monitoring, more than one change per trip,
retraining the GNN from telemetry.

### 11.9 Live app — "On the road"

A section on every plan, between the baseline comparison and the algorithm
comparison. A form chooses what happens (incident / rain / traffic build-up
/ next traffic update / nothing), when (share of the trip driven), whether
the planner is told, and for an incident where, how large and how slow.
"Drive the trip" runs `run_loop()` -> `ClosedLoopExecutor` with the plan's
route, weights, priority, vehicle and seed; the LLM is used if the plan was
made with it. Shown: whether it re-planned, what decided it and why;
promised time, the time finishing the plan, the time with the re-plan and
CO2; which numbers are assumed and which are GNN predictions; a map (driven
part, route taken, route not taken, incident, vehicle position); the
telemetry table. With "Tell the planner" off, it also runs
`fleet_learning()` for five following vehicles and tabulates them.

Browser run, scenario 1 (Peenya -> Rajajinagar, 09:00, fastest), LLM on,
incident 4x at the defaults: change at 6.7 min after 7 of 43 segments;
promised 22.3 min; finishing the plan 33.3 min; with the re-plan 25.0 min
(-8.3 min, -1.31 kg CO2); decided by the LLM ("Switch saves about 9
minutes, aligning with fastest priority").

Finding it (added 2026-10-14): the controls are in the main page and only
appear after "Plan route", so the sidebar has an item "5 · Disruption on the
road" and the page a note under the headline figures, both linking to the
section's anchor `#on-the-road`.

A result is tied to the plan it was run on (`plan["token"]`) and disappears
when a new route is planned. GNN re-scoring for rain and surge is cached
(`predict_changed`). As before, restart Streamlit after changing anything
under `gnn/` or `agents/`.

## 12. Phase 5 — vehicle types, deadlines, multi-stop rounds, distributed runtime (2026-10-14)

The four items from the Review 1 deck and title still open after Phase 4:
"vehicle type (EV vs diesel) in the routing objective", "OR-Tools CVRP with
carbon + EV constraints", delivery deadlines, and "over Distributed Computing
Systems". The six agents are unchanged in number; each item extends one.

### 12.1 What the training labels are made of (checked on the test snapshots)

The rules quoted in the teammate's shared chat were checked against the
labels on 10 test snapshots (every 40th):

| Label | Rule | Holds exactly on |
|---|---|---|
| travel time | base_time x (1 + congestion) x (1 + 0.3 weather) x (1 + 0.2 (1 - road condition)) | 97% of segments |
| carbon | base_carbon x (1 + 0.4 congestion) x (1 + 0.05 weather) | 95% |
| EV energy | 0.25 kWh/km x (1 + 0.5 congestion), as % of 50 kWh | 100% |

base_carbon is 0.27 kg CO2 per km on every segment (1st-99th percentile
0.267-0.273), so the GNN's carbon is for one reference vehicle emitting
0.27 kg/km in free flow, and its EV energy is for one reference EV at
0.25 kWh/km. This is what vehicle-specific carbon (12.2) scales.

The exceptions are mostly segments with `event_disruption` > 0. Their travel
time labels are about 25% BELOW the rule (median ratio 0.71-0.94 across the
10 snapshots; carbon is mixed, 0.73-1.34). Two corrections follow:
- 11.3 said the GNN's "incident = faster" response came from a feature that
  was never learned. Wrong: the GNN learned what the labels say. In this
  dataset an incident flag does not mean a slower road. 11.3 and both
  disruption modules are corrected.
- The rain response (x1.18-1.22) matches the label rule (x1.19), although
  there is still no labelled rain snapshot in the test split.

For the paper this quantifies 7.2's point: the targets are near-deterministic
functions of the edge features on 95-100% of segments.

### 12.2 Vehicle-specific carbon (`agents/vehicles.py`)

| Type | kg CO2/km | x GNN carbon | Typical payload |
|---|---|---|---|
| Diesel light goods vehicle (<3.5 t) | 0.3070 | 1.137 | 750 kg |
| Diesel medium truck (<12 t) | 0.5928 | 2.196 | 5,000 kg |
| Diesel three-wheeler | 0.1322 | 0.490 | 500 kg |
| CNG three-wheeler | 0.10768 | 0.399 | 500 kg |
| Petrol scooter (<110 cc) | 0.0334 | 0.124 | 20 kg |
| Electric light goods vehicle | 0 tailpipe; 0.25 kWh/km x 0.727 kg/kWh | — | 750 kg |

Sources: India GHG Program (WRI India, CII, TERI), "India Specific Road
Transport Emission Factors", Version 1.0, 2015 (freight LDV/MDV, three-
and two-wheeler tables); Central Electricity Authority, "CO2 Baseline
Database for the Indian Power Sector", User Guide Version 20.0 (Dec 2024),
Table S, FY 2023-24 weighted average 0.727 t CO2/MWh. Payloads are typical
values, not from those sources. The EV energy is the dataset's own figure.

How it is used:
- `Vehicle.vehicle_type` (schemas). When set, `is_ev` follows the type and a
  missing `max_payload_kg` takes the typical payload. None keeps the
  pre-Phase-5 behaviour exactly.
- `costs_for_vehicle(costs, type)`: combustion -> carbon x factor; electric
  -> carbon = its energy x grid factor. Time and delay unchanged.
- `FleetCosts`: per-type cost frames and solvers, shared by the Route Agent,
  Coordinator and Carbon Optimizer; the Execution agent simulates each route
  on its vehicle's costs; the pipeline labels results with `vehicle_type`;
  ESG reports `vehicle_types`. Every carbon figure for a typed vehicle is
  then its own CO2.
- Measured on 100 trips x 3 weightings: a combustion type never changes the
  route (a uniform scale, removed by the solver's percentile scaling);
  an EV's route is identical in 93 / 86 / 53 of 100 trips for balanced /
  greenest / carbon-only weights, because its CO2 follows energy use.
- Dispatcher: when any vehicle has a type, each (vehicle, delivery) pair is
  scored as seconds to reach the pickup + 600 s x trip CO2 (kg) -- a vehicle
  up to 10 minutes further away wins if it emits 1 kg less. 600 is a policy
  setting. Without types the score is the distance alone, as before.

Caveats to state: travel time is the same for every vehicle type (the GNN
has one time per road); combustion figures are tailpipe only, electric are
electricity at the power station without transmission or charging losses;
the published factors come from rated fuel efficiency.

### 12.3 Delivery deadlines

`DeliveryRequest.deadline_s` = latest acceptable arrival, in seconds after
the vehicle leaves the origin (it used to say "from dispatch" and was unused).
- Coordinator: new conflict type `deadline`, deterministic (a hard
  constraint, not a trade-off). The request's weights move toward time-only
  in 25% steps; the first route that meets the deadline is taken; if even
  the fastest misses, the plan is not fully resolved. `meet_deadline()` is
  public so the app's solver-only mode uses the same rule.
- Carbon Optimizer: second hard rule -- a greener swap that would miss the
  deadline is rejected whatever the LLM says.
- Execution: `deadline_met` per route (simulated arrival vs deadline).
- ESG: `num_with_deadline`, `deadline_met_rate`.
- Closed loop (11.4): rule `deadline` between `ev_range` and `dominated`.
- The evaluation's golden scenarios have no deadlines or vehicle types, so
  9.6 still stands.

Seen in a full LLM pipeline run (sample costs, Peenya -> Rajajinagar,
greenest, EV, deadline 19.5 min): the Route Agent's route was expected to
take 20.3 min; the Coordinator moved the weights 100% to time-only (18.6
min); the Carbon Optimizer's LLM then ACCEPTED a greener swap (-0.585 kg,
+106 s) and the deadline rule rejected it. Simulated arrival 17.8 min,
deadline met.

### 12.4 Multi-stop rounds with OR-Tools (`agents/multi_stop.py`)

A capacitated vehicle routing problem with time windows and battery limits
for a few vehicles leaving one depot: `plan_tours()` (also
`DispatcherAgent.plan_tours()`).
- Legs: for each vehicle type and each pair of points, the route the
  single-delivery planner would take (that type's costs, the delivery's
  weights) with its expected time, CO2 and energy. One one-to-all search
  per point and type (predecessors only; building all paths was most of
  the time).
- Objective per vehicle: (time + delay weight) x expected time / mean leg
  time + (carbon + ev weight) x CO2 / mean leg CO2 -- real units, so a
  mixed fleet is steered to its cleaner vehicles when carbon matters.
- Constraints: payload, deadlines as time windows (2 minutes hand-over per
  stop), EV battery. A stop that cannot be served is dropped and reported.
- Baselines on the same legs: nearest-neighbour (each vehicle in turn takes
  the closest stop that fits) and every delivery as its own round trip.
- `simulate_tours()` drives each tour through the Execution noise model and
  checks deadlines against simulated arrivals.
- It minimises time and CO2 summed over vehicles, NOT the time the last
  parcel arrives: without deadlines it may give everything to one vehicle.

`agents/multi_stop_experiment.py`: 20 random rounds (random depot, 10
drop-offs 10-40 minutes away, 10-80 kg parcels, 300 kg per vehicle, a
deadline on every other stop = 1.5 x direct time + 20 min), on
sample_costs/. Means per round; deadlines summed over 100:

| Fleet | Priority | Method | Time (min) | CO2 (kg) | Deadlines met | Simulated |
|---|---|---|---|---|---|---|
| diesel x3 | greenest | OR-Tools | 256.7 | 37.47 | 100 | 97 |
| diesel x3 | greenest | nearest-neighbour | 270.1 | 40.75 | 59 | |
| diesel x3 | greenest | separate trips | 582.7 | 91.73 | (100) | |
| mixed | greenest | OR-Tools | 260.1 | 18.05 | 100 | 98 |
| mixed | greenest | nearest-neighbour | 270.8 | 31.52 | 59 | |
| mixed | greenest | separate trips | 582.7 | 57.36 | (100) | |
| diesel x3 | fastest | OR-Tools | 243.7 | 39.20 | 100 | 99 |
| mixed | fastest | OR-Tools | 244.1 | 19.46 | 100 | 98 |

(mixed = electric LGV + diesel LGV + CNG three-wheeler. "Separate trips"
deadlines are counted as if each trip started fresh, the most generous
reading.)

Reading it: OR-Tools served every stop and met every deadline in plan; the
nearest-neighbour heuristic met 56-59 of 100. Bundling stops halves the
driving against one trip per delivery. The same plan with the mixed fleet
emits 52% less CO2 than with three diesel vehicles for about 1% more time --
that is the value of vehicle type in the objective. 1-3 of 100 deadlines
were missed in the simulated drives (the plan uses expected times).

### 12.5 Distributed runtime (`agents/cluster.py`)

Dask (`dask[distributed]`): a scheduler and worker processes that talk over
TCP. `connect(workers=N)` starts N local worker processes;
`connect(scheduler="tcp://host:8786")` joins workers started by hand,
possibly on other machines (commands in the module docstring; each worker
needs its own copy of the repo and data). `release(client)` disconnects and
stops workers it started -- `client.close()` alone left a local cluster's
workers running, which is how a 4-worker run first ran out of memory.

Distributed jobs: `score_snapshots()` (GNN inference, combined into exact
pooled R^2), `route_batch()`, and the re-plan experiment
(`replan_experiment.py --workers N`, one trip per task; the experiment was
split into `sample_trips()` / `run_trip()` / `run_all()` for this, with the
same trips and rows either way). Each worker loads the GNN once. LLM calls
are not distributed (they wait on the network; the pipeline already overlaps
them with threads).

`agents/cluster_benchmark.py`, this laptop (i5-13500H, 12 cores, 16 GB):

| Job | Serial | 1 worker | 2 workers | 4 workers | Same answer |
|---|---|---|---|---|---|
| GNN on all 381 test snapshots | 43.8 s | 44.6 s (x0.98) | 36.5 s (x1.20) | 28.4 s (x1.54) | yes |
| Re-plan experiment, 24 trips | 44.6 s | 62.6 s (x0.71) | 42.3 s (x1.06) | 32.7 s (x1.36) | yes |
| Re-plan experiment, 120 trips | 303.9 s | | | 145.3 s (x2.09) | yes |

The distributed GNN run reproduces the training report's test R^2 (0.9488 /
0.9687 / 0.9431 / 0.9445) to 9 decimal places; different thread counts
change float summation order in the 11th. Results:
`agents/evaluation_results/cluster_benchmark.csv` and
`cluster_benchmark_replan120.csv`.

Reading it honestly: speed-ups are modest. The serial GNN run already uses
all cores through PyTorch; each re-plan worker first rebuilds its own GNN
and road worlds (~0.9 GB and several seconds each), which only pays off on
bigger jobs (x1.36 at 24 trips, x2.09 at 120); four such workers is the
memory limit of this laptop. Tested as separate processes on one machine
(`tests/test_cluster.py` starts a scheduler and two workers from the
command line and connects over TCP); NOT tested across two machines.

**What to tell the panel:** the system runs its heavy jobs on a distributed
task runtime -- worker processes coordinated by a scheduler over the
network, which can sit on separate machines -- with identical results to a
single process and measured speed-ups up to x2.1 on four local workers. The
edge-cloud deployment in the Review 1 abstract is future work (12.6).

### 12.6 Scope: written up as future work, not built (for the guide to confirm)

| Review 1 item | Status | Why, with the measured facts |
|---|---|---|
| Neo4j graph database | Future work | The graph (7,095 junctions, 15,083 segments) fits in memory and is searched in milliseconds; a database adds operations, not results. The Drive's "Neo4j-style" explorer is a visualisation only. |
| LangGraph orchestration | Future work | The six agents are orchestrated in plain Python (`agents/pipeline.py`) with the same structure a LangGraph graph would have; switching changes the framework, not the behaviour. |
| Rebuild the graph every 30 seconds | Partly answered | The computation is not the obstacle: GNN re-scoring takes about 0.12 s per snapshot (43.8 s for 381). The missing piece is a live data feed (8.1, option B in 8.3). |
| GNN retraining from telemetry | Future work | Telemetry already corrects later plans (11.6), but retraining needs real telemetry; simulated telemetry would only teach the GNN its own noise model. |
| Edge-cloud deployment | Future work | The runtime supports workers on other machines (12.5); vehicles as edge nodes and a cloud planner are not built or tested. |
| Fleet / Warehouse / EV / ESG agents | Replaced | Team decision 2026-10-07: the six agents in this repo. Mapping for the deck in Phase 6. |

### 12.7 Files and verification

New: `agents/vehicles.py`, `agents/multi_stop.py`,
`agents/multi_stop_experiment.py`, `agents/cluster.py`,
`agents/cluster_benchmark.py`, `app_common.py`, the multi-stop page,
`tests/test_fleet.py`, `tests/test_cluster.py`.

App layout (final, 2026-10-14): `live_app.py` is the entry point and only
lists two pages with `st.navigation`: "Single delivery: source ->
destination" (`app_pages/single_delivery.py`, the former `live_app.py`,
default) and "Multi-stop fleet planner" (`app_pages/fleet_rounds.py`). The
first attempt put the second page in Streamlit's `pages/` folder, which (a)
listed the original page only as "live app", so it looked missing, and (b)
as `pages/multi_stop.py` shadowed `agents/multi_stop.py` on the import path.
Test the app through `live_app.py` (AppTest `switch_page` for the second
page): `st.page_link` between the pages needs the page list.
Changed: schemas, all six agents' modules except the ESG narrative, the
pipeline, the feedback loop, the re-plan experiment, `live_app.py`,
`requirements.txt` (+ ortools, dask[distributed]).

- `tests/test_fleet.py` 11/11, `tests/test_cluster.py` 2/2,
  `tests/test_feedback_loop.py` 16/16, `tests/test_bmssp.py` 6/6.
- Automated app checks: every vehicle type plans (time identical across
  combustion types), deadlines met and missed are reported, the loop's
  deadline rule fires, the multi-stop page plans for two priorities; viewed
  in the browser.
- One full LLM pipeline run with two vehicle types and a deadline (12.3).
- Untyped fleets: dispatcher smoke test output unchanged; default routing
  unchanged (11.7).

App: the main page has a "Vehicle" selector (reference vehicle or a type;
the EV toggle only for the reference vehicle) and a deadline input; demo
scenarios reset both so their talking points still hold. The CO2 metric is
labelled tailpipe or grid electricity. The second page, "Multi-stop fleet
planner", takes a depot, drop-offs, a priority, deadlines and a fleet of up
to three vehicles (three, because only the palette's first three colours
are validated as distinguishable everywhere on a map), and shows the
comparison, a map of each vehicle's round and per-stop expected and
simulated arrivals.

## 13. Phase 6 — the Review 3 package (2026-10-14)

### 13.1 What was made

- **Deck:** `ppt&docs/Review3/166_Capstone_Phase3_Review3.pptx` (and a PDF
  export) in the team folder `E:/CollegeWork/CAPSTONE/SEM7/ppt&docs/`, NOT in
  git (it carries the team's photos). 19 slides, 20 x 11.25 in, built from
  the team's own Review 1 file so it keeps the PES template: the title/team
  slide, problem statement and thank-you slide are the template's; every
  other slide copies the template's orange arcs and PES logo. New text uses
  Cambria (titles, amber F59E0B as in the template) and Calibri: the
  template's Canva fonts (Open Sans Bold Italics, Arvo Italics, Canva Sans)
  are not installed on Office machines and made the old slides overflow.
  Charts are native PowerPoint charts (editable). Every slide has speaker
  notes. The builder is `ppt&docs/Review3/build_review3.py` (python-pptx;
  needs the Review 1 .pptx as input); rendered for checking through
  PowerPoint COM (`Slide.Export`), since LibreOffice is not installed here.
- **Slides:** title · problem statement · since Review 2 (six stat cards) ·
  the system as built (diagram; future work dashed) · Review 1 agents ->
  the six · what we measured and what it corrected · completion audit chart
  · routing fixes + re-evaluation chart · Tsinghua (log-scale benchmark) ·
  the closed loop (how) · the closed loop (results chart) · vehicle types
  and deadlines · multi-stop chart · distributed runtime · live demo ·
  limitations · scope (built / future work) · roadmap to the paper · thank
  you.
- **Demo script:** `docs/review3_demo_script.md` (in git): preparation, four
  demo steps with the numbers measured with the LLM on and off, and the
  questions the panel is likely to ask with honest answers.
- **App:** with "Tell the planner" off, the incident message no longer says
  the incident "is reported" (found while measuring the demo).

### 13.2 What the Review 2 deck said that Review 3 corrects

The Review 2 deck (28 Sep, `ppt&docs/Review2_28sept/`) was read in full.
Slide "what we measured, and what it corrected" addresses:
- "Spatio-Temporal GNN on real Bengaluru traffic" -> TransformerConv on
  single snapshots; real values on 3.2% of segments.
- Completion slide "GNN: ST-GCN / TGN — Done", "Carbon-Aware Optimizer:
  OR-Tools CVRP — Done", "Agentic AI: Fleet, Warehouse, EV, ESG agents —
  LangGraph — Done": none of these was in the code at Review 2. OR-Tools is
  built now (12.4); LangGraph is future work; the agents are the six (team
  decision 2026-10-07).
- Results slide "-5.4 to -12.7% carbon, +6.6 to +13% time vs the speed
  baseline" came from the pre-Phase-2 solver (four routing bugs). Current,
  on expected values: -3.0% CO2, +6.6% time (9.6).
- Its HLD still showed Truck/Fleet/Warehouse/EV/ESG agents and Neo4j; the
  new "system as built" diagram replaces it.

### 13.3 Audit re-scored (same checklist as 7, 2026-10-14)

Only items whose status changed, each backed by a test or run:

| Stage | 11 Oct | 14 Oct | Change |
|---|---|---|---|
| Ingestion | 1/5 | 1.5/5 | feedback intake partial (telemetry corrects later plans, not the data layer) |
| Agent decisions | 6/10 | 9/10 | Tsinghua algorithm, multi-stop routes, deadlines |
| Carbon optimisation | 4/7 | 6/7 | vehicle-specific carbon, OR-Tools CVRP (Pareto selection still absent) |
| Execution and feedback | 3/7 | 5/7 | acting on re-plans, feedback loop (retraining, real telemetry absent) |
| ESG dashboard | 7/11 | 7.5/11 | what-if disruption demo complete |
| Distributed computing | 0.5/3 | 1.5/3 | worker processes over TCP on one machine (1); multi-node partial (0.5) |
| Tests and CI | 1.5/4 | 2/4 | pytest-compatible suite partial (35 tests, run as scripts; no pytest/CI) |
| Documentation | 1/4 | 2/4 | README (CHANGELOG too) |

Unchanged: data sources 6/9, feature engineering 4.5/5, graph construction
3.5/5, GNN 4/6, evaluation 4/7 (the new experiments do not cover the
missing ablations or stronger routing baseline). **Nine stages: 39 -> 47 of
65 (60% -> 72%). With cross-cutting promises: 46 -> 56.5 of 83 (55% ->
68%).**

### 13.4 Not done in Phase 6

The deck was not rehearsed with the team or checked against a department
template for Review 3 (none was available; it follows the Review 1/2
format). Slide timing: about 20-25 minutes for 19 slides; slides 2, 10 and
16 can be skipped for a 15-minute slot.

---

## 15. UI design system (branch `better`, 2026-10-22)

The app's look is defined in two files and the pages only call them. **No agent,
solver or page logic changed**; every widget key and label that code or tests
look up is unchanged.

- `.streamlit/config.toml`: native theme for light and dark (the app follows the
  viewer's setting), semantic colours, `toolbarMode = "minimal"` (hides Deploy).
- `ui_theme.py`: tokens, global CSS (`inject()`), and components: `header`,
  `chip`, `step`, `note`, `badge`, `legend`, `ends`, `weights_meter`,
  `comparison_table`, `delta_chips`, `route_bar_chart`, `style_map`, `pin`.
  Each page calls `ui.inject()` right after `st.set_page_config`.

### 15.1 Rules (so a future page looks like the rest)

1. **Colour.** Neutrals and the three series colours are the validated dataviz
   palette (both modes). The accent is its blue: it means "our system" (the
   multi-agent route, primary buttons, the active page). Orange and aqua are the
   two baselines, on the map, in the table and in the charts. **Red is for
   critical states only** (a battery that runs out, a closed road, a bad delta).
2. **Icons.** Material Symbols only: `icon=":material/name:"` on buttons,
   expanders, alerts and page links; `ui.icon()` inside HTML components. No emoji.
3. **Text wears text tokens;** colour appears on marks (a dot, a line, a bar), and
   a status is always icon plus label (`ui.badge`), never colour alone.
4. **Native widgets stay native** (`st.metric`, `st.dataframe`, `st.expander`), only
   styled, so `AppTest` and screen readers still see them. Every custom table has
   a "Table view" expander with the plain dataframe.
5. **Words.** No "live" for anything that is not live traffic: the page title is
   "Route planner"; the badge says "Scored 15,083 roads in N ms".
6. **Type.** The system sans stack (`"Segoe UI Variable Text", "Segoe UI",
   system-ui, ...`), no web font. JetBrains Mono was tried for all text (bundled
   font files, `theme.fontFaces`, `server.enableStaticServing`) and reverted on
   request: it is about 20% wider and the app has many explanatory sentences, so
   it needed shrunken tiles and shortened labels. If a monospace face is wanted
   again, use it for numbers and tables only, not body text.
7. **Map.** The standard light OpenStreetMap in both themes (no tile filter); the
   pins, zoom buttons and reference dots are styled for a light map.

### 15.2 Layout

Planner: header and snapshot chips, then **map (left, 60%)
beside the headline tiles (right, 40%)**, then the comparison, "On the road",
the Dijkstra vs Tsinghua panel and the agent trace. Fleet: header, KPI row, the
deadline banner, map, comparison, per-vehicle rounds. The basemap is the same
OpenStreetMap tiles, always light (an earlier version of this redesign used a muted grey map in
light and an inverted map in dark; both were dropped on request); pins are lettered
(A source, B destination, D depot, V vehicle, ! incident, numbers for stops).

### 15.3 Things to know

- **Restart Streamlit after editing `ui_theme.py`** (a refresh reloads only the
  page script, not imported modules).
- `st.metric` values scale with the window (`clamp()`), because a narrow column
  clipped them ("22.2 ...") at 1100 px.
- `scenario_label` (the quick-start display name) is idempotent: Streamlit's
  `AppTest` applies a `format_func` twice. In tests select a scenario with
  `selectbox.select_index(i)`.
- The scenario keys keep their circled digits (the docs and the deck refer to
  them); only the display changed.
- The scenarios' talking points (`DEMO_SCENARIOS[...]["talk"]`) are not drawn on
  the page any more: they are the hover tooltip (?) of the *Quick start scenario*
  field (`scenario_help()`), so the screen stays clean and the speaker can still
  read them. Keep any new scenario's `talk` text there, not in a note box.
- To look at the result, drive the app with a real browser (Playwright on the
  installed Chrome works: no browser download needed); `AppTest` cannot see CSS.

### 15.4 Checked and not done

Checked in Chrome, dark and light, at 1440 and 1100 px, LLM off and on, empty and
after a drive; all numbers identical to before (CHANGELOG 22 Oct). Not done: the
fleet editor in the sidebar shows only its first column in the narrow sidebar
(it scrolls); no mobile layout (the app is for a laptop or projector).
