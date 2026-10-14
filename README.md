# Agentic AI and GNNs for Sustainable Logistics — Bengaluru

Capstone project (PES University, Team 166). A Graph Neural Network predicts
the cost of every road segment in Bengaluru (travel time, delay probability,
CO2, EV energy), and six agents use those predictions to assign vehicles,
plan routes, resolve conflicts, cut carbon, simulate the drive, re-plan
when conditions change on the road, and report ESG metrics.

Design rule used throughout: **plain code computes every number; the LLM only
makes judgment calls**, and code-enforced rules stop the LLM from getting
facts wrong.

## How it works

```
traffic snapshot -> GNN (per-road costs) -> Dispatcher -> Route Agents
   -> Coordinator -> Carbon Optimizer -> Execution (simulated drive) -> ESG Reporter
                          ^                          |
                          +--- re-plan mid-trip; telemetry corrects later plans
```

| Agent | File | Uses the LLM? | Job |
|---|---|---|---|
| Dispatcher | `agents/dispatcher_agent.py` | No | Assigns each delivery to the nearest capable vehicle |
| Route Agent | `agents/route_agent.py` | Yes | Turns a priority ("fastest", "greenest", free text) into solver weights |
| Coordinator | `agents/coordinator_agent.py` | On conflicts | Resolves overloaded roads and EV range violations |
| Carbon Optimizer | `agents/carbon_optimizer_agent.py` | Yes | Proposes greener routes; code verifies the saving before accepting |
| Execution | `agents/execution_agent.py`, `agents/feedback_loop.py` | When rules do not settle it | Simulates the drive with delays; re-plans mid-trip when conditions change |
| ESG Reporter | `agents/esg_reporter_agent.py` | Yes (summary) | Aggregates metrics and writes a short report |

The road graph has 7,095 junctions and 15,083 road segments. Routing is a
weighted shortest-path search (`gnn/routing_solver.py`) over the GNN's
predicted costs. Two search algorithms are available: Dijkstra (default) and
the Duan et al. 2025 "Tsinghua" algorithm (`gnn/bmssp.py`, see below).

## Repository layout

```
gnn/                 GNN model, trainer, inference wrapper, routing solver, BMSSP solver and its benchmark
tests/               tests that need no LLM (the BMSSP solver, the feedback loop)
agents/              the six agents, pipeline, feedback loop, evaluation harness, static dashboard
agents/evaluation_results/   raw and summary evaluation CSVs, re-plan experiment results
sample_costs/        GNN predictions for 5 snapshots (CSV), for running agents without PyTorch
live_app.py          interactive Streamlit planner (map, live GNN, agent trace)
CLAUDE.md            full design log, findings and the measured audit (section 7)
CHANGELOG.md         every change and fix since 27 Sep 2026, newest first
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

Create a `.env` file in the repo root with an LLM model and its key. Any
provider supported by LiteLLM works; the project was developed on Groq:

```
LLM_MODEL=groq/openai/gpt-oss-120b
GROQ_API_KEY=<your key>
```

`.env` is git-ignored. Never commit keys.

## Data that is not in git

The agent layer runs on `sample_costs/` alone. The live app also needs these
three folders in the repo root (git-ignored, about 710 MB). They live on the
team's Google Drive under `Capstone/`; ask a team member for access.

| Local folder | Drive location | Contents |
|---|---|---|
| `gnn_checkpoints_v3/` | `Capstone/GNN/gnn_checkpoints_v3/` | `best_model.pt`, `normalizer.pt` |
| `dataset_kaggle/test/` | `Capstone/GNN/dataset_kaggle/test/` | 381 test snapshots (`snapshot_*.pt`) |
| `bengaluru_graph/` | `Capstone/bengaluru_graph/` | `nodes.geojson`, `edges.geojson` (coordinates and road shapes) |

## Running

No API key needed:

```bash
python agents/dispatcher_agent.py        # vehicle assignment smoke test
python tests/test_bmssp.py               # Tsinghua solver vs Dijkstra (or: pytest tests/)
python tests/test_feedback_loop.py       # the closed loop: re-planning and learning from telemetry
python agents/feedback_loop.py           # one trip through an incident, reported and not reported
python gnn/benchmark_sssp.py --quick     # Dijkstra vs Tsinghua solver timings (drop --quick for the full run)
python agents/replan_experiment.py       # re-plan vs no re-plan on 120 trips (needs the Drive data, ~7 min)
```

Needs the LLM key:

```bash
python agents/pipeline.py                # full six-agent run on 5 deliveries
python agents/run_full_evaluation.py 0   # evaluate one snapshot (index 0-4), ~42k tokens
python agents/run_full_evaluation.py --summarize
python agents/dashboard.py               # writes agents/dashboard.html
```

Interactive app (needs the Drive data above):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-live.txt
streamlit run live_app.py
```

Pick a source and destination on the map (or a demo scenario from the
sidebar), choose the traffic snapshot, priority and vehicle, and press
**Plan route**. Switching off "Use multi-agent LLM pipeline" plans instantly
with fixed weights and uses no tokens.

Under every planned route, **On the road** drives the trip through a change
of conditions (a road incident, rain, a traffic build-up, the next traffic
update, or nothing) and shows whether the system re-plans, who decided, and
what it saved. After pulling new code, restart Streamlit; a page refresh
does not reload the modules the app imports.

## Results (measured 11 Oct 2026)

- **GNN v3**, all 381 test snapshots: R^2 0.949 (travel time), 0.969 (delay
  probability), 0.943 (carbon), 0.945 (EV energy); about 0.16 s per snapshot
  on CPU.
- **Evaluation** (re-run 12 Oct 2026 after the routing fixes), 5 scenarios x
  3 systems x 5 snapshots: 75 of 75 runs completed. On expected values
  (typical delays included) the multi-agent system emits 3.0% less CO2 than
  a fastest-route baseline and takes 6.6% longer; against a lowest-carbon
  baseline it emits 2.1% more and is 6.5% faster. Simulated drives land
  within about 3-6% of the time the plan promises.

## The closed loop: re-planning on the road

`agents/feedback_loop.py` closes the loop from Execution back to routing.
Part-way through a trip the road conditions change; the system prices
"finish the planned route" against "the best route from here" on the new
conditions and decides, by fixed rules first and the LLM only when they do
not settle it. Each drive also records expected and actual time per road
segment, and roads that keep running far slower than predicted are
corrected for the next vehicle.

There is no live traffic feed, so a change is one of two things: the GNN
re-scoring the roads on changed conditions (rain, congestion, the next
snapshot), or, for a road incident, an **assumed** slowdown. The GNN cannot
model incidents, so incident results are never GNN predictions.

Measured on 120 random trips (13 Oct 2026, change reported 30% into the
trip, fixed decision policy, both outcomes simulated with the same luck):

| Change | Re-planned | Trips 15%+ late: plan -> loop | Saved per re-plan |
|---|---|---|---|
| Nothing (control) | 0% | 0% -> 0% | - |
| Incident on the route (assumed 4x) | 93% | 75% -> 18% | 9.0 min, 1.65 kg CO2 |
| Rain across the city | 0% | 77% -> 77% | - |
| Congestion surge on the route | 44% | 46% -> 25% | 5.1 min, 0.61 kg CO2 |
| Next traffic snapshot | 0% | 1% -> 1% | - |

Re-planning helps when the change is local and does nothing when every road
slows together, which is the right behaviour. The incident is placed on the
vehicle's route on purpose. When an incident is never reported, the second
vehicle on the same trip already takes the route a fully informed planner
would in 68% of cases. `CLAUDE.md` section 11 has the details.

## The Tsinghua shortest-path algorithm

`gnn/bmssp.py` is our implementation of R. Duan, J. Mao, X. Mao, X. Shu and
L. Yin, "Breaking the Sorting Barrier for Directed Single-Source Shortest
Paths", arXiv:2504.17033 (2025), which runs in O(m log^(2/3) n) against
Dijkstra's O(m + n log n). Use it with
`RoutingSolver(costs_df, algorithm="bmssp")`, by setting
`ROUTING_ALGORITHM=bmssp`, or from the app's sidebar.

It returns the same shortest paths as Dijkstra on every test. Measured, it
is 14-22 times slower than Dijkstra on the Bengaluru graph (7,095 junctions)
and still 7 times slower at 262,144 junctions: its advantage is asymptotic
and does not show at city scale, so Dijkstra stays the default. `CLAUDE.md`
section 10 lists the three places the code differs from the paper.

## Known limitations

- Real Kaggle traffic values cover 481 of 15,083 road segments (3.2%); the
  rest are interpolated. Time-of-day variation is modelled, not measured.
- There is no live traffic feed; the system runs on recorded snapshots, and
  a mid-trip change is simulated (GNN re-scoring, or an assumed slowdown for
  incidents).
- The GNN cannot model a road incident: its incident feature was never
  learned and pushes predictions the wrong way.
- A linear model on the raw road features matches or beats the GNN on three
  of the four targets; the GNN is ahead only on delay probability.
- The lowest-carbon baseline is both faster and greener than the multi-agent
  system in 1 of 5 evaluation scenarios (the shared-bottleneck case).
- The evaluation uses 5 snapshots from a single day and one LLM.
- The loop handles one change per trip, decided at a junction; there is no
  continuous monitoring, and telemetry corrections are not used to retrain
  the GNN.
- In the re-plan decision the LLM has so far agreed with the fixed rule
  every time (16 of 16).
- Not implemented: multi-stop routing, delivery deadlines, vehicle-specific
  carbon, distributed execution, automated tests beyond `tests/`.

`CLAUDE.md` section 7 has the full audit, per-stage completion scores and the
list of known defects.

## Team

Aayan Vikram Singh, Adarsh V, Saksham Gupta, Rohan Varghese.
Guide: Dr. Arti Arya, Department of CSE (AI & ML), PES University.
