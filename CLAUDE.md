# Bengaluru Logistics Framework — Project Context & Build Plan

This file is project memory for Claude Code. Read it fully before making changes.

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
features come from a Kaggle Bengaluru traffic dataset covering 16 named roads
(~60% of edges geo-matched directly; remaining ~40% filled by spatial
interpolation from nearby matched edges).

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
