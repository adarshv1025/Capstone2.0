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
├── .env                       (API keys, gitignored)
├── requirements.txt
├── gnn/
│   ├── gnn_inference.py
│   ├── routing_solver.py
│   ├── gnn_model_v3.py
│   └── gnn_trainer_v3.py
├── sample_costs/
│   └── snapshot_XXXX_costs.csv
└── agents/                    <- being built now, see Section 3
```

---

## 3. Agents Layer — STATUS: IN PROGRESS (Phase 1 of 8 complete)

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

1. **NEXT** — Grounding tool: `routing_solver.py` (deterministic weighted
   shortest-path solver, unit test standalone).
2. **NEXT** — Wire up one Route Agent end-to-end: single delivery request in,
   LLM interprets priority into solver weights, calls the tool, returns a
   structured route. Validates the full pattern before scaling out.
3. Scale to parallel Route Agents across a batch of delivery requests.
4. Add the Coordinator/Negotiation Agent — the genuinely "decentralized"
   piece: detect conflicts across parallel routes, re-run specific Route
   Agents with adjusted constraints until conflicts clear.
5. Add the Carbon-Aware Optimizer as a critic with a self-reflection loop.
6. Build the execution/monitoring simulation (replay plan against GNN
   predictions, optionally with injected noise, log deviations).
7. Build ESG Reporting (aggregate carbon/EV/on-time metrics, generate a
   narrative summary).
8. Build the evaluation harness: a golden set of delivery scenarios, scored
   against baselines (plain Dijkstra time-only, naive greedy-carbon-only, and
   eventually the teammates' system) through the **same input/output
   contract**, so the comparison is apples-to-apples. Metrics: total time,
   total carbon, on-time rate, EV feasibility rate, negotiation rounds needed.

Estimated total effort: ~35-55 hours across all 8 phases; roughly 2-3 weeks
part-time.

### 3.6 Immediate next step (Phase 2 detail)

Build, in `agents/`:

- `schemas.py` — pydantic models: `DeliveryRequest`, `Vehicle`, `RouteResult`
- `llm_client.py` — thin LiteLLM wrapper (see 3.3)
- `route_agent.py` — single Route Agent: takes a `DeliveryRequest` + stated
  priority (e.g. "fastest", "greenest", "balanced"), has the LLM translate
  that into a `{time, delay, carbon, ev}` weights dict, calls
  `RoutingSolver.find_route()` from `routing_solver.py`, returns a structured
  `RouteResult`.
- Test against `sample_costs/*.csv` directly (`pd.read_csv(...)` instead of
  `predictor.predict(...)`) — no torch/GNN checkpoint needed for this phase.
