"""
gnn_inference.py

Single entry point for downstream consumers (the multi-agent routing
layer) to get per-edge cost predictions from the trained LogisticsGNN,
without needing to know anything about normalization or clamping.

INPUT CONTRACT
--------------
A torch_geometric.data.Data snapshot with:
    data.x          [num_nodes, 13] node features (RAW, unnormalized)
    data.edge_index [2, num_edges]
    data.edge_attr  [num_edges, 16] edge features (RAW, unnormalized)

(data.y is not required/used for inference)

OUTPUT CONTRACT
---------------
A pandas.DataFrame, one row per edge (same order as edge_index), columns:

    src, dst
        node indices for that edge

    travel_time_s
        seconds, >= 0

    delay_probability
        0-1

    carbon_kg
        kg CO2, >= 0

    ev_energy_pct
        % of a 50kWh battery, >= 0

All 4 values are already real-unit and domain-clamped -- safe to use
directly as edge costs, no further transformation needed.

USAGE
-----
    from gnn_inference import GNNCostPredictor

    predictor = GNNCostPredictor(checkpoint_dir="./gnn_checkpoints_v3")
    costs_df = predictor.predict(snapshot)
"""

import torch
import pandas as pd

from gnn_trainer_v3 import load_trained_model
from gnn_model_v3 import clamp_predictions


class GNNCostPredictor:
    """
    Loads the trained model + normalizer once.

    The model and all tensor attributes inside the normalizer are placed
    on the same device (CPU or CUDA), so inference works correctly
    regardless of where the checkpoint was originally saved.
    """

    def __init__(self, checkpoint_dir="./gnn_checkpoints_v3", device=None):

        # Automatically use CUDA when available, otherwise CPU.
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"Using inference device: {self.device}")

        # Load trained model, normalizer, and configuration.
        self.model, self.normalizer, self.cfg = load_trained_model(
            checkpoint_dir
        )

        # Move the trained model to the selected device.
        self.model = self.model.to(self.device)

        # IMPORTANT:
        # load_trained_model() loads the normalizer separately from the
        # model. Its tensors (x_mean, x_std, y_mean, y_std, etc.) may
        # remain on CPU. Move every tensor stored inside the normalizer
        # to the same device as the model.
        self._move_normalizer_to_device()

        # Put model into inference mode.
        self.model.eval()

    def _move_normalizer_to_device(self):
        """
        Move all torch.Tensor attributes stored directly inside the
        normalizer to self.device.

        This avoids hard-coding names such as x_mean, x_std, y_mean,
        y_std, edge_mean, edge_std, etc.

        The trained GNN itself is NOT modified.
        """

        for name, value in vars(self.normalizer).items():

            if isinstance(value, torch.Tensor):

                setattr(
                    self.normalizer,
                    name,
                    value.to(self.device)
                )

    @torch.no_grad()
    def predict(self, snapshot) -> pd.DataFrame:
        """
        Run inference on one PyTorch Geometric snapshot.

        Parameters
        ----------
        snapshot:
            torch_geometric.data.Data containing RAW, unnormalized:
                x
                edge_index
                edge_attr

        Returns
        -------
        pandas.DataFrame
            One row per edge containing:

                src
                dst
                travel_time_s
                delay_probability
                carbon_kg
                ev_energy_pct
        """

        # ---------------------------------------------------------------
        # 1. Move the snapshot to the same device as the model.
        # ---------------------------------------------------------------

        data = snapshot.clone().to(self.device)

        # ---------------------------------------------------------------
        # 2. Normalize node and edge features.
        #
        #    The normalizer has already been moved to self.device in
        #    __init__, so this operation is now device-compatible.
        # ---------------------------------------------------------------

        data_norm = self.normalizer.transform(data)

        # ---------------------------------------------------------------
        # 3. Run the trained LogisticsGNN.
        # ---------------------------------------------------------------

        pred_norm = self.model(
            data_norm.x,
            data_norm.edge_index,
            data_norm.edge_attr
        )

        # ---------------------------------------------------------------
        # 4. Convert predictions from normalized space back to real
        #    physical units.
        # ---------------------------------------------------------------

        pred_real = self.normalizer.inverse_y(pred_norm)

        # ---------------------------------------------------------------
        # 5. Apply domain-specific constraints/clamping.
        # ---------------------------------------------------------------

        pred_real = clamp_predictions(pred_real)

        # ---------------------------------------------------------------
        # 6. Extract edge source/destination indices.
        # ---------------------------------------------------------------

        src = data.edge_index[0].detach().cpu().numpy()
        dst = data.edge_index[1].detach().cpu().numpy()

        # Move predictions back to CPU for pandas.
        pred_real = pred_real.detach().cpu().numpy()

        # ---------------------------------------------------------------
        # 7. Construct output DataFrame.
        # ---------------------------------------------------------------

        return pd.DataFrame({
            "src": src,
            "dst": dst,
            "travel_time_s": pred_real[:, 0],
            "delay_probability": pred_real[:, 1],
            "carbon_kg": pred_real[:, 2],
            "ev_energy_pct": pred_real[:, 3],
        })


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST / DEMO
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import glob

    # ---------------------------------------------------------------
    # Find test snapshots.
    # ---------------------------------------------------------------

    test_files = sorted(
        glob.glob("./dataset_kaggle/test/snapshot_*.pt")
    )

    if not test_files:
        raise SystemExit(
            "No test snapshots found at ./dataset_kaggle/test/ -- "
            "update the path in this smoke test if your data lives elsewhere."
        )

    # ---------------------------------------------------------------
    # Load one sample snapshot.
    # ---------------------------------------------------------------

    print(f"Loading a sample snapshot: {test_files[0]}")

    snapshot = torch.load(
        test_files[0],
        weights_only=False
    )

    # ---------------------------------------------------------------
    # Create predictor.
    # ---------------------------------------------------------------

    predictor = GNNCostPredictor(
        checkpoint_dir="./gnn_checkpoints_v3"
    )

    # ---------------------------------------------------------------
    # Run inference.
    # ---------------------------------------------------------------

    costs = predictor.predict(snapshot)

    # ---------------------------------------------------------------
    # Display results.
    # ---------------------------------------------------------------

    print(
        f"\nScored {len(costs):,} edges. "
        "First 10 rows:"
    )

    print(
        costs.head(10).to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Summary statistics.
    # ---------------------------------------------------------------

    print("\nSummary stats:")

    print(
        costs.describe()
    )