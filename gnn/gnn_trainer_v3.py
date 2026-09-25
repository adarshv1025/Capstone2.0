"""
gnn_trainer_v3.py
=================
Updated trainer with:
  1. Pure MSE loss for all 4 regression heads (no BCE)
  2. Per-label loss scaling to handle different value ranges
  3. Better metrics: R² for all 4 labels, MAE in real units
  4. Fixed heads kwarg bug (uses gnn_model_v3.py)
  5. Works with label-fixed snapshots from fix_labels.py

v3 vs v2: this trainer is functionally UNCHANGED from gnn_trainer_v2.py.
Only the module/checkpoint names were bumped, plus a predict() helper that
applies clamp_predictions() after inverse_y(). The real fix lives in
gnn_model_v3.py (heads no longer end in ReLU/Sigmoid). Metrics are computed
exactly as in v2 so the v2-vs-v3 R2 comparison stays apples-to-apples.

Label columns (after fix_labels.py):
  y[:,0] effective_travel_time  (normalized seconds)
  y[:,1] delay_probability      (0–1, ratio-based)
  y[:,2] actual_carbon_kg       (normalized kg)
  y[:,3] ev_energy_pct          (% of 50kWh battery, regression)

Run:
    python gnn_trainer_v3.py
    python gnn_trainer_v3.py --model baseline --epochs 30
    python gnn_trainer_v3.py --model logistics --epochs 50 --hidden 256
"""

import os, json, time, argparse, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from gnn_model_v3 import build_model, clamp_predictions

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
DEFAULT = dict(
    dataset_dir    = "./dataset_kaggle",
    fallback_dir   = "./dataset",
    checkpoint_dir = "./gnn_checkpoints_v3",
    model_type  = "logistics",
    hidden      = 128,
    layers      = 3,
    heads       = 4,
    dropout     = 0.20,
    epochs      = 50,
    batch_size  = 8,
    lr          = 3e-4,
    weight_decay= 1e-5,
    patience    = 8,
    # Loss weights for [travel_time, delay_prob, carbon, ev_energy_pct]
    # Higher weight on delay_prob to force the model to focus on it
    loss_weights= [0.25, 0.40, 0.20, 0.15],
    seed        = 42,
)

LABEL_NAMES = ["travel_time", "delay_prob", "carbon_kg", "ev_energy_pct"]


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════

class SnapshotDataset(Dataset):
    def __init__(self, folder, normalizer=None):
        super().__init__()
        self.files      = sorted(Path(folder).glob("snapshot_*.pt"))
        self.normalizer = normalizer

    def len(self): return len(self.files)

    def get(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        if self.normalizer:
            data = self.normalizer.transform(data)
        return data


def find_dataset(primary, fallback):
    for d in [primary, fallback]:
        if Path(d).exists() and any(Path(d,"train").glob("*.pt")):
            n = len(list(Path(d,"train").glob("*.pt")))
            print(f"  Using dataset: {d}  ({n} train snapshots)")
            return d
    raise FileNotFoundError(
        f"No dataset found at '{primary}' or '{fallback}'.\n"
        "Run fix_labels.py first if using dataset_kaggle.")


# ══════════════════════════════════════════════════════════════════════════
# FEATURE NORMALIZER
# ══════════════════════════════════════════════════════════════════════════

class FeatureNormalizer:
    def __init__(self):
        self.x_mean = self.x_std = None
        self.ea_mean = self.ea_std = None
        self.y_mean = self.y_std = None
        self.fitted = False

    def fit(self, loader, device="cpu"):
        print("  Computing normalization statistics ...")
        x_s=x_sq=x_n=0; ea_s=ea_sq=ea_n=0; y_s=y_sq=y_n=0
        for data in tqdm(loader, desc="  Fitting", leave=False):
            x=data.x.to(device); ea=data.edge_attr.to(device); y=data.y.to(device)
            x_s+=x.sum(0);  x_sq+=(x**2).sum(0);  x_n+=x.shape[0]
            ea_s+=ea.sum(0);ea_sq+=(ea**2).sum(0);ea_n+=ea.shape[0]
            y_s+=y.sum(0);  y_sq+=(y**2).sum(0);  y_n+=y.shape[0]

        def stats(s,sq,n):
            mean=s/n; std=torch.sqrt(torch.clamp(sq/n-mean**2,min=1e-8))
            std=torch.where(std<1e-6,torch.ones_like(std),std)
            return mean.cpu(), std.cpu()

        self.x_mean,  self.x_std  = stats(x_s,  x_sq,  x_n)
        self.ea_mean, self.ea_std = stats(ea_s,  ea_sq, ea_n)
        self.y_mean,  self.y_std  = stats(y_s,   y_sq,  y_n)
        self.fitted = True
        print(f"  Normalizer fitted — {x_n:,} node, {ea_n:,} edge, {y_n:,} label samples")

    def transform(self, data):
        if not self.fitted: return data
        d = data.clone()
        d.x         = (data.x        - self.x_mean)  / self.x_std
        d.edge_attr = (data.edge_attr - self.ea_mean) / self.ea_std
        d.y         = (data.y        - self.y_mean)  / self.y_std
        return d

    def inverse_y(self, y_norm):
        return y_norm * self.y_std.to(y_norm.device) + self.y_mean.to(y_norm.device)

    def save(self, path):
        torch.save(dict(x_mean=self.x_mean, x_std=self.x_std,
                        ea_mean=self.ea_mean, ea_std=self.ea_std,
                        y_mean=self.y_mean,  y_std=self.y_std), path)
        print(f"  Normalizer saved → {path}")

    @classmethod
    def load(cls, path):
        obj=cls(); ck=torch.load(path, weights_only=True)
        obj.x_mean=ck["x_mean"]; obj.x_std=ck["x_std"]
        obj.ea_mean=ck["ea_mean"]; obj.ea_std=ck["ea_std"]
        obj.y_mean=ck["y_mean"];  obj.y_std=ck["y_std"]
        obj.fitted=True; return obj


# ══════════════════════════════════════════════════════════════════════════
# MULTI-TASK LOSS — pure MSE for all 4 regression targets
# ══════════════════════════════════════════════════════════════════════════

class MultiTaskLoss(nn.Module):
    """
    Weighted MSE for all 4 regression labels.
    Higher weight on delay_prob (0.40) to push the model to focus on it.

    All labels are normalized before loss computation, so MSE values
    are comparable across different-scale targets.
    """
    def __init__(self, weights=(0.25, 0.40, 0.20, 0.15)):
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float)
        self.register_buffer("w", w / w.sum())
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        losses = {}
        total  = 0.0
        for i, name in enumerate(LABEL_NAMES):
            l = self.mse(pred[:, i], target[:, i])
            losses[name] = l.item()
            total = total + self.w[i] * l
        return total, losses


# ══════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(pred_list, target_list, normalizer=None):
    """
    Compute R², RMSE, MAE for all 4 labels.
    If normalizer provided, also report in real units.
    """
    pred   = torch.cat(pred_list,   0).cpu()   # [N, 4]
    target = torch.cat(target_list, 0).cpu()   # [N, 4]

    # Inverse transform to real units for interpretable metrics
    if normalizer:
        pred_real   = normalizer.inverse_y(pred)
        target_real = normalizer.inverse_y(target)
    else:
        pred_real   = pred
        target_real = target

    results = {}
    for i, name in enumerate(LABEL_NAMES):
        p = pred[:, i].numpy()
        t = target[:, i].numpy()
        p_r = pred_real[:, i].numpy()
        t_r = target_real[:, i].numpy()

        rmse_norm = float(np.sqrt(np.mean((p - t)**2)))
        mae_real  = float(np.mean(np.abs(p_r - t_r)))
        r2        = float(1 - np.sum((p-t)**2) / (np.sum((t-t.mean())**2)+1e-8))

        results[f"rmse_{name}"] = rmse_norm
        results[f"mae_{name}"]  = mae_real
        results[f"r2_{name}"]   = r2

    return results


# ══════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL STEPS
# ══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, tasks, nb = 0.0, {k:0.0 for k in LABEL_NAMES}, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        if hasattr(data,"edge_attr") and data.edge_attr is not None:
            pred = model(data.x, data.edge_index, data.edge_attr)
        else:
            pred = model(data.x, data.edge_index)
        loss, sub = criterion(pred, data.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); nb += 1
        for k,v in sub.items(): tasks[k] += v
    avg = total/max(nb,1)
    return avg, {k:v/max(nb,1) for k,v in tasks.items()}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, normalizer=None):
    model.eval()
    total, tasks, nb = 0.0, {k:0.0 for k in LABEL_NAMES}, 0
    preds, targets = [], []
    for data in loader:
        data = data.to(device)
        if hasattr(data,"edge_attr") and data.edge_attr is not None:
            pred = model(data.x, data.edge_index, data.edge_attr)
        else:
            pred = model(data.x, data.edge_index)
        loss, sub = criterion(pred, data.y)
        total += loss.item(); nb += 1
        for k,v in sub.items(): tasks[k] += v
        preds.append(pred.cpu()); targets.append(data.y.cpu())
    avg     = total/max(nb,1)
    metrics = compute_metrics(preds, targets, normalizer)
    return avg, {k:v/max(nb,1) for k,v in tasks.items()}, metrics


# ══════════════════════════════════════════════════════════════════════════
# MAIN TRAIN
# ══════════════════════════════════════════════════════════════════════════

def train(cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    print("\n" + "="*62)
    print("  Bengaluru Logistics GNN — Training v3")
    print("="*62)
    print(f"  Device : {device}  |  Model : {cfg['model_type']}")
    print(f"  Hidden : {cfg['hidden']}  Layers : {cfg['layers']}  Heads : {cfg['heads']}")
    print(f"  Epochs : {cfg['epochs']}  Batch : {cfg['batch_size']}  LR : {cfg['lr']}")
    print(f"  Loss weights : {cfg['loss_weights']}  [TT, DP, CO2, EV]")

    # ── Dataset ──────────────────────────────────────────────────────────
    ddir     = find_dataset(cfg["dataset_dir"], cfg["fallback_dir"])
    raw_ds   = SnapshotDataset(f"{ddir}/train")
    norm_ldr = DataLoader(raw_ds, batch_size=cfg["batch_size"],
                          shuffle=False, num_workers=0)

    # ── Normalizer ───────────────────────────────────────────────────────
    normalizer = FeatureNormalizer()
    normalizer_path = os.path.join(cfg["checkpoint_dir"], "normalizer.pt")
    if cfg.get("resume", False) and os.path.exists(normalizer_path):
        print(f"  Loading normalizer from: {normalizer_path}")
        normalizer = FeatureNormalizer.load(normalizer_path)
    else:
        normalizer.fit(norm_ldr, device=device)
        normalizer.save(normalizer_path)

    train_ds = SnapshotDataset(f"{ddir}/train", normalizer=normalizer)
    val_ds   = SnapshotDataset(f"{ddir}/val",   normalizer=normalizer)
    test_ds  = SnapshotDataset(f"{ddir}/test",  normalizer=normalizer)
    print(f"\n  Snapshots: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_ldr = DataLoader(train_ds, batch_size=cfg["batch_size"],
                           shuffle=True, num_workers=0)
    val_ldr   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                           shuffle=False, num_workers=0)
    test_ldr  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                           shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model, n_params = build_model(
        cfg["model_type"],
        hidden=cfg["hidden"], layers=cfg["layers"],
        heads=cfg["heads"],   dropout=cfg["dropout"],
    )
    model = model.to(device)
    print(f"  Parameters : {n_params:,}\n")

    optimizer = Adam(model.parameters(),
                     lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=4, min_lr=1e-6)
    criterion = MultiTaskLoss(weights=cfg["loss_weights"]).to(device)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val = float("inf")
    patience_counter = 0
    history = []
    t0_total = time.time()
    start_epoch = 1

    latest_ckpt_path = os.path.join(cfg["checkpoint_dir"], "checkpoint_latest.pt")
    if cfg.get("resume", False):
        if os.path.exists(latest_ckpt_path):
            print(f"\n  Resuming from checkpoint: {latest_ckpt_path}")
            resume_ckpt = torch.load(latest_ckpt_path, weights_only=False, map_location=device)

            model.load_state_dict(resume_ckpt["model_state"])
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])

            if "scheduler_state" in resume_ckpt and resume_ckpt["scheduler_state"] is not None:
                scheduler.load_state_dict(resume_ckpt["scheduler_state"])

            best_val = resume_ckpt.get("best_val_loss", best_val)
            patience_counter = resume_ckpt.get("patience_counter", patience_counter)
            history = resume_ckpt.get("history", history)
            start_epoch = resume_ckpt.get("epoch", 0) + 1

            if start_epoch > cfg["epochs"]:
                print(f"  Checkpoint epoch ({start_epoch - 1}) already reached/exceeded target epochs ({cfg['epochs']}).")
                print("  Skipping training loop and proceeding to final evaluation.")
            else:
                print(f"  Resume successful. Continuing at epoch {start_epoch}/{cfg['epochs']}")
        else:
            print("\n  Resume requested but checkpoint not found.")
            print("  Starting training from scratch.")

    hdr = (f"{'Ep':>4}  {'Train':>8}  {'Val':>8}  "
           f"{'R²-TT':>7}  {'R²-DP':>7}  {'R²-CO2':>8}  "
           f"{'R²-EV':>7}  {'LR':>8}  {'Time':>6}")
    print(hdr)
    print("-" * len(hdr))

    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg["epochs"]+1):
        t0 = time.time()
        tl, tt = train_epoch(model, train_ldr, optimizer, criterion, device)
        vl, vt, vm = eval_epoch(model, val_ldr, criterion, device, normalizer)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(vl)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": round(tl,5), "val_loss": round(vl,5),
            "train_tasks": {k:round(v,5) for k,v in tt.items()},
            "val_tasks":   {k:round(v,5) for k,v in vt.items()},
            "metrics":     {k:round(v,4) for k,v in vm.items()},
            "lr": lr, "time": round(elapsed,1),
        })

        print(f"{epoch:>4}  {tl:>8.4f}  {vl:>8.4f}  "
              f"{vm['r2_travel_time']:>7.4f}  {vm['r2_delay_prob']:>7.4f}  "
              f"{vm['r2_carbon_kg']:>8.4f}  {vm['r2_ev_energy_pct']:>7.4f}  "
              f"{lr:>8.2e}  {elapsed:>5.1f}s")

        if vl < best_val:
            best_val = vl; patience_counter = 0
            torch.save({"epoch":epoch, "model_state":model.state_dict(),
                        "val_loss":vl, "metrics":vm, "config":cfg},
                       f"{cfg['checkpoint_dir']}/best_model.pt")
        else:
            patience_counter += 1

        ckpt_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val,
            "patience_counter": patience_counter,
            "history": history,
            "config": cfg,
        }

        latest_tmp_path = latest_ckpt_path.replace(".pt", ".tmp")
        torch.save(ckpt_data, latest_tmp_path)
        os.replace(latest_tmp_path, latest_ckpt_path)

        if epoch % 5 == 0:
            torch.save(ckpt_data, os.path.join(cfg["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pt"))

        with open(f"{cfg['checkpoint_dir']}/training_log.json","w") as f:
            json.dump({"config":cfg, "best_val_loss":best_val, "history":history}, f, indent=2)

        if patience_counter >= cfg["patience"]:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    torch.save({"epoch":epoch,"model_state":model.state_dict(),"config":cfg},
               f"{cfg['checkpoint_dir']}/last_model.pt")

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\n  Loading best model for test evaluation ...")
    ck = torch.load(f"{cfg['checkpoint_dir']}/best_model.pt", weights_only=False)
    model.load_state_dict(ck["model_state"])
    tl_test, _, test_m = eval_epoch(model, test_ldr, criterion, device, normalizer)
    total_time = time.time() - t0_total

    # ── Save logs ─────────────────────────────────────────────────────────
    log = {"config":cfg, "best_val_loss":best_val, "test_loss":tl_test,
           "test_metrics":{k:round(v,4) for k,v in test_m.items()},
           "total_time_min":round(total_time/60,1), "history":history}
    with open(f"{cfg['checkpoint_dir']}/training_log.json","w") as f:
        json.dump(log,f,indent=2)

    # ── Report ────────────────────────────────────────────────────────────
    def grade(val, thresholds):
        # thresholds: [(value, label), ...] descending
        for v, label in thresholds:
            if val >= v: return label
        return "Needs work"

    r2_grades = [(0.85,"Excellent"),(0.70,"Good"),(0.50,"Acceptable"),(0.0,"Needs work")]

    print(f"""
Bengaluru Logistics GNN v3 — Training Report
=============================================
Model        : {cfg['model_type']}   Parameters: {n_params:,}
Total epochs : {epoch}   Time: {total_time/60:.1f} min   Device: {device}

Dataset
-------
Train: {len(train_ds)}   Val: {len(val_ds)}   Test: {len(test_ds)}

Best Val Loss : {best_val:.5f}
Test Loss     : {tl_test:.5f}

Test Metrics (normalized RMSE + real-unit MAE + R²)
-----------------------------------------------------
                       RMSE(norm)   R²       Grade
  Travel Time          {test_m['rmse_travel_time']:.4f}       {test_m['r2_travel_time']:.4f}    {grade(test_m['r2_travel_time'],r2_grades)}
  Delay Probability    {test_m['rmse_delay_prob']:.4f}       {test_m['r2_delay_prob']:.4f}    {grade(test_m['r2_delay_prob'],r2_grades)}
  Carbon kg            {test_m['rmse_carbon_kg']:.4f}       {test_m['r2_carbon_kg']:.4f}    {grade(test_m['r2_carbon_kg'],r2_grades)}
  EV Energy %          {test_m['rmse_ev_energy_pct']:.4f}       {test_m['r2_ev_energy_pct']:.4f}    {grade(test_m['r2_ev_energy_pct'],r2_grades)}

Target thresholds: R² > 0.70 = Good, R² > 0.85 = Excellent

Checkpoints saved → {cfg['checkpoint_dir']}/
  best_model.pt   last_model.pt   normalizer.pt   training_log.json
""")

    return model, normalizer, log


# ══════════════════════════════════════════════════════════════════════════
# INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════════

def load_trained_model(checkpoint_dir="./gnn_checkpoints_v3"):
    """
    Load trained model + normalizer for use by routing agents.

    Usage:
        model, norm, cfg = load_trained_model()
        data_norm = norm.transform(snapshot)
        with torch.no_grad():
            pred_norm = model(data_norm.x, data_norm.edge_index, data_norm.edge_attr)
        pred_real = norm.inverse_y(pred_norm)
        pred_real = clamp_predictions(pred_real)   # v3: enforce domain limits

        # Per edge:
        travel_time   = pred_real[:, 0]   # seconds
        delay_prob    = pred_real[:, 1]   # 0–1
        carbon_kg     = pred_real[:, 2]   # kg CO₂
        ev_energy_pct = pred_real[:, 3]   # % of 50kWh battery consumed
    """
    ck  = torch.load(f"{checkpoint_dir}/best_model.pt", weights_only=False)
    cfg = ck["config"]
    model, _ = build_model(cfg["model_type"], **{k:cfg[k] for k in
               ["hidden","layers","heads","dropout"]})
    model.load_state_dict(ck["model_state"])
    model.eval()
    norm = FeatureNormalizer.load(f"{checkpoint_dir}/normalizer.pt")
    print(f"Loaded {cfg['model_type']} from epoch {ck['epoch']} "
          f"(val_loss={ck['val_loss']:.4f})")
    return model, norm, cfg


@torch.no_grad()
def predict(model, norm, snapshot, device=None):
    """
    Full inference path for routing agents: normalize -> forward ->
    inverse-transform -> clamp to real-world domain limits.

    Returns a [E, 4] real-unit tensor:
        [:,0] effective_travel_time  (seconds,  >= 0)
        [:,1] delay_probability      (in [0, 1])
        [:,2] actual_carbon_kg       (kg CO2,   >= 0)
        [:,3] ev_energy_pct          (% of 50kWh battery, >= 0)

    The clamp lives HERE and nowhere else -- never in the loss, never in
    normalized space. That is the whole point of the v3 fix: the heads stay
    unconstrained during training so they can match z-scored targets that
    legitimately go negative.
    """
    device = device or next(model.parameters()).device
    model.eval()
    d = norm.transform(snapshot)
    pred_norm = model(d.x.to(device),
                      d.edge_index.to(device),
                      d.edge_attr.to(device))
    return clamp_predictions(norm.inverse_y(pred_norm.cpu()))


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default=DEFAULT["model_type"],
                   choices=["baseline","logistics"])
    p.add_argument("--epochs",  type=int,   default=DEFAULT["epochs"])
    p.add_argument("--batch",   type=int,   default=DEFAULT["batch_size"])
    p.add_argument("--lr",      type=float, default=DEFAULT["lr"])
    p.add_argument("--hidden",  type=int,   default=DEFAULT["hidden"])
    p.add_argument("--layers",  type=int,   default=DEFAULT["layers"])
    p.add_argument("--heads",   type=int,   default=DEFAULT["heads"])
    p.add_argument("--dropout", type=float, default=DEFAULT["dropout"])
    p.add_argument("--patience",type=int,   default=DEFAULT["patience"])
    p.add_argument("--dataset", default=DEFAULT["dataset_dir"])
    p.add_argument("--outdir",  default=DEFAULT["checkpoint_dir"])
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = {**DEFAULT,
           "model_type":cfg_val if (cfg_val:=args.model) else DEFAULT["model_type"],
           "epochs":args.epochs, "batch_size":args.batch, "lr":args.lr,
           "hidden":args.hidden, "layers":args.layers, "heads":args.heads,
           "dropout":args.dropout, "patience":args.patience,
            "dataset_dir":args.dataset, "checkpoint_dir":args.outdir,
            "resume":args.resume}
    train(cfg)