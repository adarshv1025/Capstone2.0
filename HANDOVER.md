# Handover — Capstone 2.0, Team 166 (16 Oct 2026)

Aayan is away from 16 Oct. This file is the starting point for whoever
continues. Read it first, then `CLAUDE.md` (the full design log, with every
measurement) and `CHANGELOG.md` (every change since 27 Sep, newest first).

## 1. Where everything is

| What | Where |
|---|---|
| Code | GitHub `adarshv1025/Capstone2.0`, branch **`aayanvikramsingh`**. `main` still has only the original two commits; merging is a team decision. |
| Data (not in git, ~13 GB with the training split) | Team Google Drive, `Capstone/` — folders listed in `README.md` ("Data that is not in git"). Aayan's Drive: ask him to share it before he leaves. |
| Review 3 deck | `ppt&docs/Review3/` on Aayan's machine: `166_Capstone_Phase3_Review3.pptx`, a PDF, and `build_review3.py` that rebuilds it. Not in git (team photos). Upload the folder to the team Drive. |
| Demo script | `docs/review3_demo_script.md` (in git): the four demo steps with measured numbers and likely panel questions. |
| The working chat with Claude | `handover/` on Aayan's machine: `chat_conversation.md` (readable, ~200 KB) and `chat_transcript_full.jsonl` (complete, 40 MB). API keys are redacted. Upload to the team Drive. |

## 2. Setting up (one teammate, about an hour)

```bash
git clone https://github.com/adarshv1025/Capstone2.0.git
cd Capstone2.0
git checkout aayanvikramsingh
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-live.txt
```

Copy from the Drive into the repo root: `gnn_checkpoints_v3/`,
`dataset_kaggle/test/` (381 snapshots, ~660 MB) and `bengaluru_graph/`.
Create `.env` with **your own** key (never commit it):

```
LLM_MODEL=groq/openai/gpt-oss-120b
GROQ_API_KEY=<your key>
```

Check that everything works (no key needed for the tests):

```bash
python tests/test_bmssp.py
python tests/test_feedback_loop.py
python tests/test_fleet.py
python tests/test_cluster.py
streamlit run live_app.py
```

All 35 tests passed on 16 Oct. The app's first load after starting can take
about a minute; if the page stays blank, refresh.

## 3. What is done

The plan agreed on 11 Oct is `CLAUDE.md` 8.2. Phases 1-6 are done:

| Phase | What | Details |
|---|---|---|
| 1 | Measured audit, corrected claims, README | `CLAUDE.md` 7 |
| 2 | Routing fixes (graph cache, parallel roads, delay cost, expected-time plans, EV range); evaluation re-run | 9 |
| 3 | The Tsinghua shortest-path algorithm (BMSSP), verified and benchmarked | 10 |
| 4 | Closed feedback loop: mid-trip re-planning, telemetry learning, 120-trip experiment | 11 |
| 5 | Vehicle types (sourced CO2), deadlines, OR-Tools multi-stop, Dask distributed runtime | 12 |
| 6 | Review 3 deck, demo script, audit re-scored (60% -> 72% of the nine stages) | 13 |

The live app has two pages: **Single delivery** (map, GNN, six agents,
disruptions on the road, algorithm comparison) and **Multi-stop fleet
planner**.

## 4. What is next

| Phase | What | Estimate |
|---|---|---|
| 7 | Strengthen the GNN for the paper: baselines table (linear, GBM, MLP, GraphSAGE), accuracy on real vs interpolated roads, loss-rebalance retrain, forecasting version, options A and C from `CLAUDE.md` 8.1 | 3-5 working days, 6-10 h of Colab GPU |
| 8 | Paper-grade evaluation: all 48 test days, 50+ trips, seeds, confidence intervals, ablations (no LLM, no Coordinator, ...), cost | 2-4 days; needs ~2.2M LLM tokens (paid Groq tier, a local model, or a smaller sample) |
| 9 | pytest + GitHub Actions (the tests already exist) | half a day |
| 10 | Write the paper | 1-2 weeks |

Start Phase 7 with the parts that need no GPU: the accuracy split between the
481 real-data roads and the rest, and the linear / boosted-tree baselines.

**Colab (free tier is enough):** the trainer streams snapshots from disk,
the model is small (386,692 parameters) and a full run took 83.5 min on a
Colab GPU. Copy the data from Drive to the Colab machine's disk at the start
of each session (reading 6,092 files from mounted Drive is slow), save a
checkpoint to Drive every epoch, and use the trainer's resume option after a
disconnect. Keep the tab active. GPU access is not guaranteed on the free
tier. Option C as first planned (one retrain per held-out road) is too heavy;
hide roads at prediction time instead, or drop it.

Optional, any time: real traffic from the TomTom API (`CLAUDE.md` 8.3). It
cannot be backfilled, so start collecting before the paper if you want it.

## 5. Working rules (please keep them)

1. **Record every change in `CHANGELOG.md`** (newest first: what, why,
   files, how it was checked) and keep `CLAUDE.md` in step. The project has
   been handed over twice now; this is what makes that possible.
2. **Only the six agents in this repo** (Dispatcher, Route Agent,
   Coordinator, Carbon Optimizer, Execution, ESG Reporter) — team decision
   of 7 Oct, not the Fleet/Warehouse/EV/ESG agents of the Review 1 deck.
3. **Measure before claiming.** Every number in the deck and docs comes from
   a run that can be repeated. State limitations openly (`README.md`,
   "Known limitations"; `CLAUDE.md` 8.1 for the 3.2% real-data coverage).
4. **Never commit API keys.** `.env` is git-ignored. Before each commit:
   `git diff --cached | grep -E "gsk_[A-Za-z0-9]{20,}"` must print nothing.
5. **Commit to the branch, push when the team agrees;** end commit messages
   with a short body saying what changed and why.
6. After changing anything under `gnn/` or `agents/`, restart Streamlit; a
   page refresh does not reload imported modules.

## 6. Continuing with Claude Code

Open Claude Code in the repo folder. `CLAUDE.md` is loaded automatically
and gives Claude the whole design and every measurement, so it does not need
the old chat. A good first message:

> Read HANDOVER.md and CHANGELOG.md, then start Phase 7 with the parts that
> need no GPU. Keep CHANGELOG.md and CLAUDE.md updated with every change.

The old chat (`handover/chat_conversation.md`) is there for people: it
explains why decisions were made. Claude can read it too if a question
needs the history.

## 7. Things to know

- **Groq free tier:** about 200k tokens per day per account. One app plan
  with the LLM on uses ~3-4k; one evaluation snapshot ~42-46k (so a full
  5-snapshot evaluation is a day's quota). `LLM_TEMPERATURE` defaults to 0 so
  runs repeat; every call is logged to `logs/llm_calls.jsonl`.
- **The GNN cannot model incidents** (its data labels incident roads
  faster); the app models them as an assumed slowdown and says so.
- **Windows + Dask:** scripts that start worker processes must be real
  files with `if __name__ == "__main__":`, not piped to Python on stdin.
- **pytest is not installed** in the project's environment yet; the test
  files run as plain scripts (Phase 9 adds pytest and CI).
- **Deck fonts:** the PES template's Canva fonts are missing on most Office
  machines; the new slides use Cambria and Calibri.

## 8. Open decisions for the team and the guide

- The scope statement in `CLAUDE.md` 12.6 (Neo4j, LangGraph, 30-second
  rebuild, retraining from telemetry, edge-cloud deployment as future work)
  still needs the guide's agreement.
- Whether to present the deck's "What we measured, and what it corrected"
  slide as it is.
- When to merge `aayanvikramsingh` into `main`.
- Whether to start TomTom data collection.

## 9. Security note

Three Groq keys were pasted into the working chat during development. They
are redacted in the exported chat, but they should be **revoked in the Groq
console** (Aayan) and not reused. Each teammate creates their own key.
