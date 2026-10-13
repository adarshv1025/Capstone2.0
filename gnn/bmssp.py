"""
bmssp.py

Single-source shortest paths by the algorithm of

    R. Duan, J. Mao, X. Mao, X. Shu, L. Yin,
    "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths",
    arXiv:2504.17033 (STOC 2025)

-- the "Tsinghua SSSP" algorithm promised to the review panel (CLAUDE.md 7.4,
10). It runs in O(m log^(2/3) n) in the comparison-addition model, against
Dijkstra's O(m + n log n). This is our own implementation written from the
paper's Algorithms 1-3 and Lemma 3.3; no third-party code is used.

HOW IT WORKS (paper section 3)
------------------------------
Dijkstra keeps the whole frontier sorted. This algorithm avoids that by
working on a bounded sub-problem, BMSSP(l, B, S): given frontier vertices S
and an upper bound B, complete every vertex whose distance is below B (or
stop early with a smaller bound B' once enough work has been done).

  bmssp(l, B, S)                                        [Algorithm 3]
    1. find_pivots: k rounds of Bellman-Ford-style relaxation from S. The
       vertices reached form W. Only the "pivots" P -- vertices of S that
       root a shortest-path subtree of at least k vertices -- need to be
       carried further, which shrinks the frontier by a factor of k.
                                                        [Algorithm 1]
    2. Put the pivots in a BlockQueue D, a partial-sorting structure that
       can hand back "the M smallest" without fully sorting. [Lemma 3.3]
    3. Repeatedly pull the M smallest from D, recurse on them one level
       down with a tighter bound, then relax the edges leaving the newly
       completed vertices and feed the results back into D.
    4. At level 0, base_case runs a tiny Dijkstra that stops after k + 1
       vertices.                                        [Algorithm 2]

  with k = floor(log2(n)^(1/3)), t = floor(log2(n)^(2/3)), and the top-level
  call bmssp(ceil(log2(n) / t), infinity, {source}).

TIES (paper Assumption 2.1)
---------------------------
The analysis assumes no two paths have the same length. As the paper
prescribes, paths are compared lexicographically by (length, number of
hops, then vertex ids from the end), so every label below is the triple
(distance, hops, vertex) and every bound B is such a triple. This matters
here: after robust scaling ~1% of our edges have cost exactly 0.

WHERE THIS DIFFERS FROM THE PAPER (be precise about these when reporting)
-------------------------------------------------------------------------
1. No constant-degree transformation. The paper first rewrites the graph so
   every vertex has in/out-degree <= 2; that is needed for the running-time
   proof, not for correctness. Road graphs already have small degree (the
   Bengaluru graph's maximum out-degree is reported by the benchmark).
2. BlockQueue keeps the paper's structure (two block sequences D0/D1, blocks
   of at most M pairs, an upper bound per D1 block), but:
     - the D1 bounds are kept in a sorted Python list searched by bisection,
       not a balanced search tree (same search cost, O(#blocks) updates);
     - medians and "the M smallest" are found by sorting a block, not by
       linear-time selection (an extra log M factor inside a block).
   These keep correctness and the algorithm's shape, but mean this code does
   not strictly attain the O(m log^(2/3) n) bound.
3. One safeguard the pseudocode leaves implicit: when a recursive call
   completes a vertex that is still waiting in this level's D with an older,
   larger value, that stale pair is deleted. The paper's own Lemma 3.10 and
   Remark 3.8 assume no such pair exists; `stats["stale_deleted"]` counts
   how often the safeguard actually fires.

Verified against Dijkstra in tests/test_bmssp.py; benchmarked by
gnn/benchmark_sssp.py.
"""

from __future__ import annotations

import heapq
import math
from bisect import bisect_left

INF = (math.inf, math.inf, math.inf)  # a bound above every label


# ══════════════════════════════════════════════════════════════════════════
# Lemma 3.3 -- the partial-sorting structure
# ══════════════════════════════════════════════════════════════════════════

class _Block:
    __slots__ = ("items", "upper", "in_d1")

    def __init__(self, items: dict, upper, in_d1: bool):
        self.items = items    # key -> value; a dict gives the O(1) delete the paper's linked list does
        self.upper = upper    # D1 only: every value here is <= upper <= every value in the next block
        self.in_d1 = in_d1


class BlockQueue:
    """
    Holds (key, value) pairs with values below B and supports
      insert(key, value)      keep the smaller value if the key exists
      batch_prepend(pairs)    every value must be below everything stored
      pull() -> (x, keys)     up to M keys with the smallest values, and a
                              bound x with max(pulled) < x <= min(remaining)
                              (x = B if nothing remains)
    """

    def __init__(self, M: int, B):
        self.M = max(1, M)
        self.B = B
        self.value: dict = {}          # key -> value currently stored
        self._where: dict = {}         # key -> block holding it
        self._d0: list[_Block] = []    # batch-prepended blocks; the FRONT of the sequence is the END of this list
        self._d1: list[_Block] = [_Block({}, B, True)]  # inserted blocks, ordered by upper bound
        self._d1_bounds: list = [B]

    def __len__(self) -> int:
        return len(self.value)

    def __contains__(self, key) -> bool:
        return key in self.value

    def delete(self, key) -> None:
        block = self._where.pop(key)
        del block.items[key]
        del self.value[key]
        # An emptied D1 block gives up its bound, except the last block: it
        # owns the bound B and must stay to receive later inserts.
        if block.in_d1 and not block.items and block.upper != self.B:
            i = bisect_left(self._d1_bounds, block.upper)
            while self._d1[i] is not block:
                i += 1
            del self._d1[i]
            del self._d1_bounds[i]

    def insert(self, key, value) -> None:
        old = self.value.get(key)
        if old is not None:
            if value >= old:
                return
            self.delete(key)
        # The block whose upper bound is the smallest one that is >= value.
        i = min(bisect_left(self._d1_bounds, value), len(self._d1) - 1)
        block = self._d1[i]
        block.items[key] = value
        self.value[key] = value
        self._where[key] = block
        if len(block.items) > self.M:
            self._split(i)

    def _split(self, i: int) -> None:
        block = self._d1[i]
        ordered = sorted(block.items.items(), key=lambda kv: kv[1])
        half = len(ordered) // 2
        low = _Block(dict(ordered[:half]), ordered[half - 1][1], True)
        block.items = dict(ordered[half:])  # keeps its old upper bound
        for key in low.items:
            self._where[key] = low
        self._d1.insert(i, low)
        self._d1_bounds.insert(i, low.upper)

    def batch_prepend(self, pairs) -> None:
        best: dict = {}
        for key, value in pairs:
            if key not in best or value < best[key]:
                best[key] = value
        fresh = []
        for key, value in best.items():
            old = self.value.get(key)
            if old is not None:
                if value >= old:
                    continue
                self.delete(key)
            fresh.append((key, value))
        if not fresh:
            return
        if len(fresh) <= self.M:
            chunks = [fresh]
        else:
            fresh.sort(key=lambda kv: kv[1])
            size = (self.M + 1) // 2  # ceil(M / 2)
            chunks = [fresh[j:j + size] for j in range(0, len(fresh), size)]
        for chunk in reversed(chunks):  # smallest chunk ends up at the front
            block = _Block(dict(chunk), None, False)
            self._d0.append(block)
            for key, value in chunk:
                self.value[key] = value
                self._where[key] = block

    def pull(self):
        M = self.M
        if not self.value:
            return self.B, []

        # A prefix of each sequence: stop once it is exhausted or holds M.
        collected = []
        i0, n0 = len(self._d0) - 1, 0
        while i0 >= 0 and n0 < M:
            collected.extend(self._d0[i0].items.items())
            n0 += len(self._d0[i0].items)
            i0 -= 1
        i1, n1 = 0, 0
        while i1 < len(self._d1) and n1 < M:
            collected.extend(self._d1[i1].items.items())
            n1 += len(self._d1[i1].items)
            i1 += 1

        if len(collected) == len(self.value) and len(collected) <= M:
            keys = [key for key, _ in collected]
            self.__init__(M, self.B)
            return self.B, keys

        # Either prefix that stopped early holds >= M pairs, all no larger
        # than the rest of its sequence, so the M smallest overall are here.
        collected.sort(key=lambda kv: kv[1])
        taken, left = collected[:M], collected[M:]

        # x = the smallest value that remains: among what was collected but
        # not taken, and the first non-empty block after each prefix.
        candidates = [left[0][1]] if left else []
        while i0 >= 0:
            if self._d0[i0].items:
                candidates.append(min(self._d0[i0].items.values()))
                break
            i0 -= 1
        while i1 < len(self._d1):
            if self._d1[i1].items:
                candidates.append(min(self._d1[i1].items.values()))
                break
            i1 += 1
        x = min(candidates) if candidates else self.B

        for key, _ in taken:
            self.delete(key)
        while self._d0 and not self._d0[-1].items:
            self._d0.pop()
        return x, [key for key, _ in taken]


# ══════════════════════════════════════════════════════════════════════════
# Algorithms 1-3
# ══════════════════════════════════════════════════════════════════════════

class BMSSP:
    """
    adj: {u: [(v, weight), ...]} with non-negative weights. Vertices that
    only appear as targets need no entry. Vertex ids must be mutually
    comparable (ints here) -- they are the final tie-breaker.

        dist, pred = BMSSP(adj).run(source)

    dist holds every reachable vertex; pred[v] is v's parent in the
    shortest-path tree (None for the source).
    """

    def __init__(self, adj: dict, n_vertices: int | None = None):
        self.adj = adj
        if n_vertices is None:
            nodes = set(adj)
            for edges in adj.values():
                nodes.update(v for v, _ in edges)
            n_vertices = len(nodes)
        self.n = max(1, n_vertices)
        log_n = max(1.0, math.log2(self.n))
        self.k = max(1, math.floor(log_n ** (1 / 3)))
        self.t = max(1, math.floor(log_n ** (2 / 3)))
        self.top_level = max(1, math.ceil(log_n / self.t))
        self.stats: dict = {}

    def run(self, source):
        self.d = {source: 0.0}      # d-hat: distance labels, only ever decreased
        self.h = {source: 0}        # hops on the path that set d (tie-break)
        self.pred = {source: None}
        self._pw = {}               # weight of the edge (pred[v], v) that set d[v]
        self.stats = {"relaxations": 0, "base_cases": 0, "pulls": 0, "find_pivots": 0,
                      "stale_deleted": 0, "duplicate_completions": 0}
        self._bmssp(self.top_level, INF, [source])
        return self.d, self.pred

    # -- labels and relaxation ------------------------------------------------

    def _label(self, v):
        return (self.d[v], self.h[v], v)

    def _relax(self, u, v, w) -> bool:
        """
        The paper's "if d[u] + w <= d[v]: d[v] <- d[u] + w" under the
        path order of Assumption 2.1. Equality holds when (u, v) is the
        edge that already set d[v] -- the non-strict test is what lets a
        higher level re-use an edge a lower level relaxed first.
        """
        self.stats["relaxations"] += 1
        nd = self.d[u] + w
        dv = self.d.get(v, math.inf)
        if nd > dv:
            return False
        nh = self.h[u] + 1
        if nd == dv:
            hv = self.h[v]
            if nh > hv:
                return False
            if nh == hv:
                pv = self.pred[v]
                if pv is None or u > pv:
                    return False
        self.d[v] = nd
        self.h[v] = nh
        self.pred[v] = u
        self._pw[v] = w
        return True

    # -- Algorithm 1 ----------------------------------------------------------

    def _find_pivots(self, B, S):
        self.stats["find_pivots"] += 1
        adj, k = self.adj, self.k
        W = set(S)
        frontier = S
        for _ in range(k):
            reached = set()
            for u in frontier:
                for v, w in adj.get(u, ()):
                    if self._relax(u, v, w) and self._label(v) < B:
                        reached.add(v)
            W |= reached
            frontier = reached
            if len(W) > k * len(S):
                return list(S), W

        # F: the edges of W along which a label was set and is still tight.
        # Under the tie order each vertex has one such parent, so F is a forest.
        children: dict = {}
        has_parent = set()
        for v in W:
            p = self.pred.get(v)
            if p is not None and p in W and self.d[v] == self.d[p] + self._pw[v] and self.h[v] == self.h[p] + 1:
                children.setdefault(p, []).append(v)
                has_parent.add(v)

        pivots = []
        for u in S:
            if u in has_parent:
                continue
            size, stack = 0, [u]
            while stack and size < k:   # only need to know whether the tree reaches k vertices
                x = stack.pop()
                size += 1
                stack.extend(children.get(x, ()))
            if size >= k:
                pivots.append(u)
        return pivots, W

    # -- Algorithm 2 ----------------------------------------------------------

    def _base_case(self, B, S):
        self.stats["base_cases"] += 1
        adj, k = self.adj, self.k
        x = S[0]
        done = {x}
        order = [x]
        heap = [(self._label(x), x)]
        while heap and len(order) < k + 1:
            label, u = heapq.heappop(heap)
            if label != self._label(u) or (u in done and u != x):
                continue  # an outdated heap entry (this heap has no decrease-key)
            if u not in done:
                done.add(u)
                order.append(u)
            for v, w in adj.get(u, ()):
                nd, nh = self.d[u] + w, self.h[u] + 1
                if (nd, nh, v) < B and self._relax(u, v, w) and v not in done:
                    heapq.heappush(heap, (self._label(v), v))
        if len(order) <= k:
            return B, order
        B_prime = max(self._label(v) for v in order)
        return B_prime, [v for v in order if self._label(v) < B_prime]

    # -- Algorithm 3 ----------------------------------------------------------

    def _bmssp(self, level, B, S):
        if level == 0:
            return self._base_case(B, S)

        adj, k, t = self.adj, self.k, self.t
        pivots, W = self._find_pivots(B, S)

        D = BlockQueue(2 ** ((level - 1) * t), B)
        for x in pivots:
            D.insert(x, self._label(x))

        B_prime = min((self._label(x) for x in pivots), default=B)  # B'_0
        U: list = []
        in_U: set = set()
        limit = k * 2 ** (level * t)

        while len(U) < limit and len(D):
            self.stats["pulls"] += 1
            B_i, S_i = D.pull()
            B_prime, U_i = self._bmssp(level - 1, B_i, S_i)

            for u in U_i:
                if u in in_U:
                    self.stats["duplicate_completions"] += 1
                else:
                    in_U.add(u)
                    U.append(u)
                if u in D:  # safeguard 3 in the module docstring
                    D.delete(u)
                    self.stats["stale_deleted"] += 1

            K = []
            for u in U_i:
                for v, w in adj.get(u, ()):
                    if self._relax(u, v, w):
                        label = self._label(v)
                        if B_i <= label < B:
                            D.insert(v, label)
                        elif B_prime <= label < B_i:
                            K.append((v, label))
            for x in S_i:
                label = self._label(x)
                if B_prime <= label < B_i:
                    K.append((x, label))
            D.batch_prepend(K)

        # The paper's "B' <- min{B'_i, B}": B'_i <= B_i <= B throughout, and
        # with no pivots B'_0 is B itself.
        for x in W:
            if x not in in_U and self._label(x) < B_prime:
                in_U.add(x)
                U.append(x)
        return B_prime, U


def dijkstra_reference(adj: dict, source):
    """
    Plain binary-heap Dijkstra over the same adjacency lists, counting its
    work, for like-for-like comparison with BMSSP (same language, same data
    structure for the graph). Returns (dist, pred, stats).
    """
    dist = {source: 0.0}
    pred = {source: None}
    heap = [(0.0, source)]
    done = set()
    relaxations = pushes = 0
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v, w in adj.get(u, ()):
            relaxations += 1
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))
                pushes += 1
    return dist, pred, {"relaxations": relaxations, "heap_pushes": pushes}


def bmssp_shortest_paths(adj: dict, source, n_vertices: int | None = None):
    """Convenience wrapper: returns (dist, pred, stats)."""
    solver = BMSSP(adj, n_vertices)
    dist, pred = solver.run(source)
    return dist, pred, solver.stats
