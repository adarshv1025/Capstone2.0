# Agentic AI and GNNs for Sustainable Logistics — Bengaluru

Capstone project (PES University, Team 166). A Graph Neural Network predicts
the cost of every road segment in Bengaluru (travel time, delay probability,
CO2, EV energy), and six agents use those predictions to assign vehicles,
plan routes, resolve conflicts, cut carbon, simulate the drive and report
ESG metrics.

Design rule used throughout: **plain code computes every number; the LLM only
makes judgment calls**, and code-enforced rules stop the LLM from getting
facts wrong.

## How it works

```
traffic snapshot -> GNN (per-road costs) -> Dispatcher -> Route Agents
   -> Coordinator -> Carbon Optimizer -> Execution (simulated drive) -> ESG Reporter
```

| Agent | File | Uses the LLM? | Job |
|---|---|---|---|
| Dispatcher | `agents/dispatcher_agent.py` | No | Assigns each delivery to the nearest capable vehicle |
| Route Agent | `agents/route_agent.py` | Yes | Turns a priority ("fastest", "greenest", free text) into solver weights |
| Coordinator | `agents/coordinator_agent.py` | On conflicts | Resolves overloaded roads and EV range violations |
| Carbon Optimizer | `agents/carbon_optimizer_agent.py` | Yes | Proposes greener routes; code verifies the saving before accepting |
| Execution | `agents/execution_agent.py` | Large deviations only | Simulates the drive with delays, flags re-plans |
| ESG Reporter | `agents/esg_reporter_agent.py` | Yes (summary) | Aggregates metrics and writes a short report |

The road graph has 7,095 junctions and 15,083 road segments. Routing is a
weighted shortest-path search (`gnn/routing_solver.py`, Dijkstra via
NetworkX) over the GNN's predicted costs.

## Repository layout

```
gnn/                 GNN model, trainer, inference wrapper, routing solver
agents/              the six agents, pipeline, evaluation harness, static dashboard
agents/evaluation_results/   raw and summary evaluation CSVs
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

## Known limitations

- Real Kaggle traffic values cover 481 of 15,083 road segments (3.2%); the
  rest are interpolated. Time-of-day variation is modelled, not measured.
- There is no live traffic feed; the system runs on recorded snapshots.
- A linear model on the raw road features matches or beats the GNN on three
  of the four targets; the GNN is ahead only on delay probability.
- The lowest-carbon baseline is both faster and greener than the multi-agent
  system in 1 of 5 evaluation scenarios (the shared-bottleneck case).
- The evaluation uses 5 snapshots from a single day and one LLM.
- The Execution agent only flags re-plans; nothing re-routes yet.
- Not implemented: the Tsinghua SSSP solver, multi-stop routing, delivery
  deadlines, vehicle-specific carbon, distributed execution, automated tests.

`CLAUDE.md` section 7 has the full audit, per-stage completion scores and the
list of known defects.

## Team

Aayan Vikram Singh, Adarsh V, Saksham Gupta, Rohan Varghese.
Guide: Dr. Arti Arya, Department of CSE (AI & ML), PES University.
