"""
gnn_model_v3.py
===============
Updated GNN model where ALL 4 output heads are regression.
ev_energy_pct replaces binary ev_feasible — now a continuous target.

v3 = v2 architecture + the head-activation fix below.
Nothing else changed: forward(), edge_mlp, build_model() and every
layer dimension are identical to gnn_model_v2.py.

Changes from gnn_model.py:
  - ev head: removed Sigmoid activation (was for binary), now plain ReLU
  - Loss: pure MSE for all 4 heads (no more BCEWithLogitsLoss)
  - Added: build_model() correctly filters kwargs per model type
            (fixes the heads kwarg bug that crashed BaselineGNN)

THE v3 FIX — head-activation / normalization mismatch:
  gnn_trainer_v3.py's FeatureNormalizer z-score normalizes data.y for ALL
  4 labels before loss is computed, so training targets are centered on 0
  and can be negative (delay_prob's normalized target can swing well
  outside [0,1]). The old heads finished with ReLU / Sigmoid, which
  structurally cannot output negative values (ReLU) or anything outside
  (0,1) (Sigmoid) -- so the model could never match roughly half of its
  own training targets, capping R^2 architecturally regardless of how
  well training went (worst for the doubly-bounded Sigmoid delay head).

  Fix: heads now end in a plain Linear layer (no final activation), so
  they can freely match the unconstrained normalized targets. Domain
  constraints (travel_time >= 0, delay_prob in [0,1], carbon >= 0,
  ev_energy_pct >= 0) are enforced with clamp_predictions() AFTER
  normalizer.inverse_y() converts predictions back to real units --
  i.e. only at inference/reporting time, never inside the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TransformerConv, BatchNorm


# ══════════════════════════════════════════════════════════════════════════
# BASELINE GNN — GraphSAGE, node features only
# ══════════════════════════════════════════════════════════════════════════

class BaselineGNN(nn.Module):
    """
    3-layer GraphSAGE. No edge features. Fast training baseline.
    Predicts all 4 regression labels per edge via src+dst node embeddings.
    """

    def __init__(self, node_feat=13, hidden=128, layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.node_enc = nn.Sequential(
            nn.Linear(node_feat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(BatchNorm(hidden))

        edge_in = hidden * 2
        # All 4 heads regress in NORMALIZED (z-scored) space -- no final
        # activation. Domain constraints are applied post-hoc via
        # clamp_predictions() on the real-unit (inverse-normalized) output.
        self.heads = nn.ModuleList([
            self._make_head(edge_in),   # travel_time
            self._make_head(edge_in),   # delay_prob
            self._make_head(edge_in),   # carbon_kg
            self._make_head(edge_in),   # ev_energy_pct
        ])
        self._init_weights()

    def _make_head(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_dim // 2, 1),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.node_enc(x)
        for conv, bn in zip(self.convs, self.bns):
            h_new = bn(F.relu(conv(h, edge_index)))
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h     = h + h_new

        src, dst = edge_index[0], edge_index[1]
        e = torch.cat([h[src], h[dst]], dim=-1)
        return torch.cat([head(e) for head in self.heads], dim=-1)


# ══════════════════════════════════════════════════════════════════════════
# LOGISTICS GNN — TransformerConv, uses node + edge features
# ══════════════════════════════════════════════════════════════════════════

class LogisticsGNN(nn.Module):
    """
    Full model using TransformerConv with edge features.
    4 regression heads: travel_time, delay_prob, carbon, ev_energy_pct.
    """

    def __init__(self, node_feat=13, edge_feat=16,
                 hidden=128, layers=3, heads=4, dropout=0.2):
        super().__init__()
        self.hidden  = hidden
        self.dropout = dropout

        assert hidden % heads == 0
        head_dim = hidden // heads

        self.node_enc = nn.Sequential(
            nn.Linear(node_feat, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_feat, hidden), nn.LayerNorm(hidden), nn.ReLU())

        self.convs    = nn.ModuleList()
        self.bns      = nn.ModuleList()
        self.res_proj = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(TransformerConv(
                in_channels=hidden, out_channels=head_dim,
                heads=heads, edge_dim=hidden,
                dropout=dropout, concat=True))
            self.bns.append(nn.LayerNorm(hidden))
            self.res_proj.append(nn.Linear(hidden, hidden, bias=False))

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_feat, hidden),
            nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU())

        # All 4 heads regress in NORMALIZED (z-scored) space -- no final
        # activation. Domain constraints are applied post-hoc via
        # clamp_predictions() on the real-unit (inverse-normalized) output.
        # See module docstring for why the old ReLU/Sigmoid heads were a bug.
        self.head_travel = self._make_head(hidden)   # travel_time
        self.head_delay  = self._make_head(hidden)   # delay_prob
        self.head_carbon = self._make_head(hidden)   # carbon_kg
        self.head_ev     = self._make_head(hidden)   # ev_energy_pct

        self._init_weights()

    def _make_head(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(in_dim // 2, 1),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr):
        h  = self.node_enc(x)
        ea = self.edge_enc(edge_attr)

        for conv, bn, res in zip(self.convs, self.bns, self.res_proj):
            h_new = conv(h, edge_index, ea)
            h_new = bn(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h     = F.relu(res(h) + h_new)

        src, dst = edge_index[0], edge_index[1]
        e = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        e = self.edge_mlp(e)

        return torch.cat([
            self.head_travel(e),
            self.head_delay(e),
            self.head_carbon(e),
            self.head_ev(e),
        ], dim=-1)


# ══════════════════════════════════════════════════════════════════════════
# INFERENCE-TIME DOMAIN CONSTRAINTS
# ══════════════════════════════════════════════════════════════════════════

def clamp_predictions(pred_real: torch.Tensor) -> torch.Tensor:
    """
    Apply real-world domain constraints to REAL-UNIT predictions, i.e.
    AFTER normalizer.inverse_y(pred_norm). Never apply this in normalized
    space or inside the loss -- the heads must stay unconstrained there
    so they can match the (possibly negative) z-scored training targets.

    pred_real columns: [travel_time, delay_prob, carbon_kg, ev_energy_pct]
    """
    out = pred_real.clone()
    out[:, 0] = out[:, 0].clamp(min=0)          # travel_time   >= 0
    out[:, 1] = out[:, 1].clamp(min=0, max=1)   # delay_prob    in [0, 1]
    out[:, 2] = out[:, 2].clamp(min=0)          # carbon_kg     >= 0
    out[:, 3] = out[:, 3].clamp(min=0)          # ev_energy_pct >= 0
    return out


# ══════════════════════════════════════════════════════════════════════════
# MODEL FACTORY — correctly filters kwargs per model type
# ══════════════════════════════════════════════════════════════════════════

def build_model(model_type="logistics", **kwargs):
    """
    Build model, filtering kwargs to only what each model accepts.
    This fixes the heads kwarg crash in BaselineGNN.
    """
    if model_type == "baseline":
        valid = {"node_feat", "hidden", "layers", "dropout"}
        cfg   = {k: v for k, v in kwargs.items() if k in valid}
        defaults = dict(node_feat=13, hidden=128, layers=3, dropout=0.2)
        model = BaselineGNN(**{**defaults, **cfg})

    elif model_type == "logistics":
        valid = {"node_feat", "edge_feat", "hidden", "layers", "heads", "dropout"}
        cfg   = {k: v for k, v in kwargs.items() if k in valid}
        defaults = dict(node_feat=13, edge_feat=16,
                        hidden=128, layers=3, heads=4, dropout=0.2)
        model = LogisticsGNN(**{**defaults, **cfg})

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'baseline' or 'logistics'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, n_params


# ══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  gnn_model_v3 — Architecture Test")
    print("=" * 50)

    N, E = 100, 200
    x  = torch.randn(N, 13)
    ei = torch.randint(0, N, (2, E))
    ea = torch.randn(E, 16)

    for mtype in ["baseline", "logistics"]:
        model, n = build_model(
            mtype, hidden=64, layers=2, heads=4,  # heads passed to both, filtered inside
            node_feat=13, edge_feat=16, dropout=0.1
        )
        if mtype == "logistics":
            out = model(x, ei, ea)
        else:
            out = model(x, ei)

        assert out.shape == (E, 4), f"Got {out.shape}"
        # NOTE: outputs are now UNCONSTRAINED (normalized-space regression),
        # so no range asserts here -- that's the whole point of the fix.
        # clamp_predictions() is applied to REAL-unit predictions instead,
        # after normalizer.inverse_y() in the trainer / inference helper.
        clamped_example = clamp_predictions(out)
        assert (clamped_example[:, 0] >= 0).all()
        assert (clamped_example[:, 1] >= 0).all() and (clamped_example[:, 1] <= 1).all()
        assert (clamped_example[:, 2] >= 0).all()
        assert (clamped_example[:, 3] >= 0).all()
        print(f"\n  {mtype}: raw output {out.shape}  params {n:,}")
        print(f"    raw travel_time range: [{out[:,0].min():.3f}, {out[:,0].max():.3f}]  (unconstrained, expected)")
        print(f"    raw delay_prob  range: [{out[:,1].min():.3f}, {out[:,1].max():.3f}]  (unconstrained, expected)")

    print("\n  All tests passed.")