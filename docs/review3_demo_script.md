# Review 3 — demo script and likely questions

The deck is `ppt&docs/Review3/166_Capstone_Phase3_Review3.pptx` in the team
folder (not in git: it has team photos). Its slide 15 is this demo in short;
every slide has speaker notes. Numbers below were measured on 14 Oct 2026.

## Before the review (10 minutes)

1. Start the app from the repo root and leave it running:

   ```bash
   streamlit run live_app.py
   ```

   It opens on **Single delivery: source → destination**. The sidebar also
   lists **Multi-stop fleet planner**.
2. Plan one route with the LLM on, to warm up the GNN and check the Groq key
   has quota. One plan uses about 3-4k tokens; the demo below about 10k.
3. If Groq is rate-limited or the network is down, switch **Use multi-agent
   LLM pipeline** off. Everything still runs, and the numbers are then fixed
   (the "LLM off" column below).
4. Close other heavy programs: the app holds the GNN in memory (~1-2 GB).

## The demo (about 6 minutes)

### 1. Plan a delivery

Sidebar → **Demo scenario** → ① *Greener vs faster — Peenya → Rajajinagar,
9 AM* → **Plan route**.

| | LLM on (varies slightly) | LLM off (fixed) |
|---|---|---|
| Distance | 9.3 km | 9.7 km |
| Expected with delays | 22.3 min | 22.2 min |
| CO₂ | 2.84 kg | 2.95 kg |

Say: the GNN scored all 15,083 roads for 9 AM; the Route Agent turned
"fastest" into weights; the Carbon Optimizer weighed a greener swap and
explained its decision (scroll to "What each agent did"). The route itself
always comes from the shortest-path solver, never from the LLM.

### 2. Disrupt it

Scroll to **On the road** (or the 🚧 link). Keep the defaults: *A road
incident ahead on the route*, 30%, *Tell the planner* on → **Drive the trip**.

| | LLM on | LLM off |
|---|---|---|
| When | 6.7 min in, after 7 of 43 segments | 7.0 min in |
| Finishing the planned route | 33.3 min | 30.8 min |
| With the re-plan | 25.0 min | 22.5 min |
| Decided by | the LLM ("switch saves ~9 minutes…") | the fixed rule (gain above 5%) |

Say: the planner compared finishing the plan with the best route from the
vehicle's junction, both on the new conditions; both futures were simulated
with the same random delays, so the saving is the effect of the decision.
Point at the map: grey = driven, blue = new route, dashed = abandoned, red =
incident. **Be clear the incident's 4× slowdown is an assumption** (the GNN
cannot model incidents: its data labels incident roads faster).

Optional: choose *Heavy rain starts across the city* → it keeps the route,
because rain slows every road about equally.

### 3. Nobody tells the planner

Switch **Tell the planner** off → **Drive the trip**.

The vehicle drives into it (LLM off: 32.9 min). The table below shows the
next five vehicles on the same trip: from vehicle 2 they take the route a
fully informed planner would (22.6-24.2 min against 27.9-31.0 without
feedback), learned only from the first vehicle's telemetry.

### 4. A delivery round

Sidebar → **Multi-stop fleet planner** → Depot *Koramangala Depot*, 8
drop-offs, priority *greenest*, deadlines on, default fleet (EV, diesel van,
CNG three-wheeler) → **Plan the round**.

| Method | Time (min) | CO₂ (kg) | Deadlines (expected) |
|---|---|---|---|
| OR-Tools | 284 | 16.6 | 4 of 4 |
| Nearest-neighbour | 306 | 31.7 | 3 of 4 |
| Every delivery its own trip | 465 | 45.9 | 4 of 4 |

The green box confirms the deadlines held in a simulated drive. OR-Tools
gave the work to the EV and the CNG three-wheeler and left the diesel van
unused. *fastest* instead: 268 min, 19.0 kg, EV and three-wheeler 4 stops
each.

## Questions to expect, and the honest answers

**Is the traffic data real?** Partly. Real Kaggle values are on 481 of
15,083 road segments (3.2%); the rest is interpolated, and hourly change is
a fixed multiplier. We say this openly; live collection (TomTom) is the
option to fix it.

**Your GNN R² is 0.94-0.97 — is that good?** Reproduced exactly on all 381
test snapshots. But the labels follow fixed formulas on 95-100% of segments,
and simple per-road models match it on 3 of 4 targets. The paper will report
those baselines next to it.

**You promised the Tsinghua algorithm "is better than Dijkstra".** It is
asymptotically faster. We implemented it from the paper and it returns the
same paths on every test, but it is 14-22× slower on our 7,095-junction
graph (the gap narrows on larger graphs: 11.4× → 7.0× on grids). So it is
selectable and Dijkstra stays the default.

**What does the LLM actually add?** It turns free-text priorities into
weights, negotiates conflicts and judges greener swaps. Hard facts are rules
in code (carbon must drop, EV range, deadlines). In re-plan decisions it
agreed with the fixed rule 16 of 16 times; Phase 8 will measure it with an
ablation.

**Why doesn't rain trigger a re-route?** Because it slows every road by
about the same factor, so the best road stays the best. The loop re-routes
for local changes (incident: 93% of trips, 9 min saved per re-plan).

**What is "distributed" here?** Heavy jobs run on a Dask cluster: a
scheduler and worker processes over TCP that can be on separate machines.
Same answers as one process; 2.1× faster on 4 workers for the 120-trip
experiment. Tested on one machine, not yet two.

**Why not LangGraph / Neo4j?** LangGraph would change the framework, not the
behaviour — our pipeline has the same structure in plain Python. The graph
fits in memory and is searched in milliseconds, so a graph database adds
operations, not results. Both are listed as future work.

**The 30-second rebuild?** Not blocked by computation (the GNN re-scores the
city in ~0.12 s); blocked by having no live data feed.

**What changed since Review 2?** Slide 3 of the deck; the full record is
`CHANGELOG.md`, and `CLAUDE.md` sections 9-13 hold every measurement.
