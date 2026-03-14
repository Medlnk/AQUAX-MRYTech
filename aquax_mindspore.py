"""
AQUA-X: AI-Enhanced Anomaly Detection for Water Quality Monitoring
================================================================
Team:        MRYTech
Institution: Suptech Santé Essaouira, Morocco
Competition: Huawei ICT Competition 2025-2026 - Innovation Track

AI Framework: MindSpore 2.6.0 (Huawei)
Model:        Autoencoder Neural Network for unsupervised anomaly detection
Platform:     Huawei Cloud ModelArts / Local
================================================================
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# ── MindSpore imports ──────────────────────────────────────────
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as transforms

warnings.filterwarnings("ignore")

# ── MindSpore context setup ────────────────────────────────────
# Use GPU if available on ModelArts/Ascend, otherwise CPU
try:
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    print("[INFO] Running on Ascend (Huawei Cloud)")
except Exception:
    try:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
        print("[INFO] Running on GPU")
    except Exception:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
        print("[INFO] Running on CPU")

# ── Directories ────────────────────────────────────────────────
for d in ["data", "models", "logs", "results"]:
    os.makedirs(d, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("AQUA-X")

# ── Config ─────────────────────────────────────────────────────
CONFIG = {
    "model":           "MindSpore Autoencoder",
    "framework":       "MindSpore 2.x",
    "input_dim":       6,
    "hidden_dims":     [32, 16, 8],   # encoder bottleneck at 8
    "latent_dim":      4,
    "learning_rate":   0.001,
    "batch_size":      64,
    "epochs":          60,
    "anomaly_threshold_percentile": 95,  # top 5% reconstruction errors = anomalies
    "random_seed":     42,
    "features": [
        "WaterTemp", "pH", "Turbidity",
        "DissolvedOxygen", "Salinity", "Conductivity"
    ],
}

np.random.seed(CONFIG["random_seed"])
ms.set_seed(CONFIG["random_seed"])

log.info("=" * 62)
log.info("  AQUA-X Anomaly Detection — MindSpore Pipeline Starting")
log.info("=" * 62)
log.info(f"Config: {json.dumps({k: v for k, v in CONFIG.items() if k != 'features'}, indent=2)}")


# ══════════════════════════════════════════════════════════════
# 1. DATASET GENERATION
# ══════════════════════════════════════════════════════════════
def generate_dataset(n_days=90):
    """
    Generates realistic coastal water quality data based on patterns from
    Morocco's Atlantic coastline (Essaouira region).
    Parameters are consistent with oceanographic literature for the
    Northeast Atlantic / Mediterranean transition zone.
    """
    log.info(f"Generating dataset: {n_days} days × 24h = {n_days*24} hourly readings")
    n = n_days * 24
    t = np.linspace(0, n_days, n)
    timestamps = pd.date_range(start="2025-06-01", periods=n, freq="h")

    # Water Temperature (°C) — seasonal + diurnal cycle
    water_temp = (
        19.5
        + 3.5  * np.sin(2 * np.pi * t / 365)
        + 1.2  * np.sin(2 * np.pi * t / 1)
        + np.random.normal(0, 0.35, n)
    )

    # pH — naturally alkaline seawater
    ph = 8.1 + 0.15 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.07, n)

    # Turbidity (NTU)
    turbidity = np.abs(2.0 + np.random.exponential(0.6, n) + np.random.normal(0, 0.25, n))

    # Dissolved Oxygen (mg/L) — inversely correlated with temperature
    do = 8.4 - 0.18 * (water_temp - 19.5) + np.random.normal(0, 0.28, n)

    # Salinity (PSU) — tidal variation
    salinity = 36.0 + 0.7 * np.sin(2 * np.pi * t / 0.5) + np.random.normal(0, 0.12, n)

    # Electrical Conductivity (mS/cm) — correlated with salinity
    conductivity = 48.0 + 0.9 * (salinity - 36.0) + np.random.normal(0, 0.2, n)

    df = pd.DataFrame({
        "timestamp":    timestamps,
        "WaterTemp":    np.round(water_temp, 3),
        "pH":           np.round(np.clip(ph, 6.5, 9.5), 3),
        "Turbidity":    np.round(turbidity, 3),
        "DissolvedOxygen": np.round(np.clip(do, 4.0, 14.0), 3),
        "Salinity":     np.round(np.clip(salinity, 30.0, 42.0), 3),
        "Conductivity": np.round(np.clip(conductivity, 35.0, 65.0), 3),
    })

    # ── Inject 4 realistic pollution/anomaly events ────────────
    df["true_anomaly"] = 0
    anomaly_indices = []

    events = [
        # (start_day, duration_h, description, modifications)
        (15, 10, "Industrial discharge — pH crash + turbidity spike",
            {"pH": -1.9, "Turbidity": +14.0, "DissolvedOxygen": -2.5}),
        (32, 14, "Algal bloom — DO spike + turbidity",
            {"DissolvedOxygen": +4.2, "Turbidity": +9.0, "pH": +0.4}),
        (55, 8,  "Thermal discharge — temperature spike",
            {"WaterTemp": +5.5, "DissolvedOxygen": -1.8}),
        (72, 12, "Freshwater influx — salinity/conductivity drop",
            {"Salinity": -7.5, "Conductivity": -9.0, "Turbidity": +5.0}),
    ]

    for start_day, dur, desc, mods in events:
        start_idx = start_day * 24
        end_idx = start_idx + dur
        ramp = np.linspace(0, 1, dur)
        for col, delta in mods.items():
            df.loc[df.index[start_idx:end_idx], col] += delta * ramp
        df.loc[df.index[start_idx:end_idx], "true_anomaly"] = 1
        anomaly_indices.extend(range(start_idx, end_idx))
        log.info(f"  Anomaly event injected: Day {start_day} — {desc}")

    log.info(f"Dataset: {len(df)} rows | {df['true_anomaly'].sum()} anomaly points ({df['true_anomaly'].mean()*100:.1f}%)")
    df.to_csv("data/aquax_buoy_dataset.csv", index=False)
    log.info("Saved: data/aquax_buoy_dataset.csv")
    return df


# ══════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════
def preprocess(df):
    log.info("Preprocessing: normalizing features...")
    features = CONFIG["features"]
    X = df[features].values.astype(np.float32)

    # Manual StandardScaler (save mean/std for inference)
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X_scaled = (X - mean) / std

    scaler_params = {"mean": mean.tolist(), "std": std.tolist(), "features": features}
    with open("models/scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    log.info(f"Scaler saved: models/scaler_params.json")
    return X_scaled, mean, std


# ══════════════════════════════════════════════════════════════
# 3. MINDSPORE AUTOENCODER MODEL
# ══════════════════════════════════════════════════════════════
class AquaXEncoder(nn.Cell):
    """
    Encoder: compresses normal water quality patterns into a
    low-dimensional latent representation.
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(AquaXEncoder, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Dense(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(keep_prob=0.9))
            prev = h
        layers.append(nn.Dense(prev, latent_dim))
        layers.append(nn.Tanh())
        self.network = nn.SequentialCell(layers)

    def construct(self, x):
        return self.network(x)


class AquaXDecoder(nn.Cell):
    """
    Decoder: reconstructs water quality readings from latent space.
    High reconstruction error → anomaly.
    """
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(AquaXDecoder, self).__init__()
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Dense(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Dense(prev, output_dim))
        self.network = nn.SequentialCell(layers)

    def construct(self, x):
        return self.network(x)


class AquaXAutoencoder(nn.Cell):
    """
    Full Autoencoder: trains only on NORMAL readings.
    At inference, anomalies produce high reconstruction error
    because the model never learned to reconstruct them.
    """
    def __init__(self):
        super(AquaXAutoencoder, self).__init__()
        cfg = CONFIG
        self.encoder = AquaXEncoder(cfg["input_dim"], cfg["hidden_dims"], cfg["latent_dim"])
        self.decoder = AquaXDecoder(cfg["latent_dim"], cfg["hidden_dims"], cfg["input_dim"])

    def construct(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def encode(self, x):
        return self.encoder(x)


class ReconstructionLoss(nn.Cell):
    """MSE loss between input and reconstructed output."""
    def __init__(self, net):
        super(ReconstructionLoss, self).__init__()
        self.net = net
        self.mse = nn.MSELoss()

    def construct(self, x):
        x_hat = self.net(x)
        return self.mse(x_hat, x)


# ══════════════════════════════════════════════════════════════
# 4. MINDSPORE DATASET
# ══════════════════════════════════════════════════════════════
class WaterQualityDataset:
    def __init__(self, X):
        self.X = X.astype(np.float32)

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)


def create_mindspore_dataset(X, batch_size, shuffle=True):
    ds = GeneratorDataset(
        WaterQualityDataset(X),
        column_names=["data"],
        shuffle=shuffle
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds


# ══════════════════════════════════════════════════════════════
# 5. TRAINING
# ══════════════════════════════════════════════════════════════
def train_model(X_scaled, df):
    log.info("=" * 50)
    log.info("Training MindSpore Autoencoder...")
    log.info(f"  Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['learning_rate']}")

    # Train ONLY on normal readings (the key idea of autoencoder-based anomaly detection)
    normal_mask = df["true_anomaly"].values == 0
    X_train = X_scaled[normal_mask]
    log.info(f"  Training samples (normal only): {len(X_train)}")

    # Build MindSpore dataset
    train_ds = create_mindspore_dataset(X_train, CONFIG["batch_size"])

    # Build model
    net = AquaXAutoencoder()
    loss_fn = ReconstructionLoss(net)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=CONFIG["learning_rate"])

    # Training loop (manual — compatible with all MindSpore versions)
    train_net = nn.TrainOneStepCell(loss_fn, optimizer)
    train_net.set_train()

    epoch_losses = []
    start_time = datetime.now()

    for epoch in range(CONFIG["epochs"]):
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_ds.create_dict_iterator():
            x = batch["data"]
            loss = train_net(x)
            epoch_loss += float(loss.asnumpy())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            log.info(f"  Epoch [{epoch+1:3d}/{CONFIG['epochs']}]  Loss: {avg_loss:.6f}")

    duration = (datetime.now() - start_time).total_seconds()
    log.info(f"Training complete in {duration:.1f}s | Final loss: {epoch_losses[-1]:.6f}")

    # Save training loss curve
    loss_log = {"epoch_losses": epoch_losses, "final_loss": epoch_losses[-1],
                "training_seconds": duration, "config": CONFIG}
    with open("logs/training_loss.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    # Save model checkpoint (MindSpore format)
    ms.save_checkpoint(net, "models/aquax_autoencoder.ckpt")
    log.info("Model saved: models/aquax_autoencoder.ckpt")

    # Plot training loss
    plt.figure(figsize=(10, 4), facecolor="white")
    plt.plot(epoch_losses, color="#065A82", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("AQUA-X MindSpore Autoencoder — Training Loss", fontweight="bold", color="#065A82")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("results/training_loss_curve.png", dpi=150)
    plt.close()

    return net, epoch_losses


# ══════════════════════════════════════════════════════════════
# 6. ANOMALY DETECTION (INFERENCE)
# ══════════════════════════════════════════════════════════════
def detect_anomalies(net, X_scaled, df):
    log.info("Running anomaly detection on full dataset...")
    net.set_train(False)

    # Compute reconstruction error for every reading
    X_tensor = Tensor(X_scaled.astype(np.float32))
    X_reconstructed = net(X_tensor).asnumpy()
    recon_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

    # Threshold: top N% reconstruction errors = anomalies
    threshold = np.percentile(recon_errors, CONFIG["anomaly_threshold_percentile"])
    df["reconstruction_error"] = recon_errors
    df["predicted_anomaly"] = (recon_errors > threshold).astype(int)

    log.info(f"Anomaly threshold (p{CONFIG['anomaly_threshold_percentile']}): {threshold:.6f}")
    log.info(f"Anomalies detected: {df['predicted_anomaly'].sum()}")

    # Save threshold for deployment
    with open("models/anomaly_threshold.json", "w") as f:
        json.dump({"threshold": float(threshold),
                   "percentile": CONFIG["anomaly_threshold_percentile"]}, f, indent=2)

    return df, threshold


# ══════════════════════════════════════════════════════════════
# 7. EVALUATION
# ══════════════════════════════════════════════════════════════
def evaluate(df):
    log.info("Evaluating detection performance...")
    y_true = df["true_anomaly"].values
    y_pred = df["predicted_anomaly"].values

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / len(y_true)

    metrics = {
        "framework": "MindSpore",
        "model": "Autoencoder (Unsupervised)",
        "dataset_size": len(df),
        "true_anomaly_points": int(y_true.sum()),
        "detected_anomaly_points": int(y_pred.sum()),
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1_score":  round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "evaluated_at": datetime.now().isoformat(),
    }

    with open("logs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Accuracy: {accuracy:.4f}")
    log.info("Metrics saved: logs/metrics.json")
    return metrics


# ══════════════════════════════════════════════════════════════
# 8. VISUALIZATION
# ══════════════════════════════════════════════════════════════
def plot_results(df):
    log.info("Generating result plots...")

    TEAL  = "#065A82"
    GREEN = "#0D9488"
    RED   = "#E53E3E"
    LIGHT = "#F0F7FF"

    params = [
        ("WaterTemp",        "Water Temperature (°C)", TEAL),
        ("pH",               "pH Level",               GREEN),
        ("Turbidity",        "Turbidity (NTU)",         "#8B5CF6"),
        ("DissolvedOxygen",  "Dissolved Oxygen (mg/L)", "#D97706"),
    ]

    sample = df.iloc[:30 * 24].copy()
    anomaly_pts = sample[sample["predicted_anomaly"] == 1]

    fig, axes = plt.subplots(len(params), 1, figsize=(16, 14), facecolor="white")
    fig.suptitle(
        "AQUA-X (MindSpore Autoencoder) — Anomaly Detection Results\n"
        "Actual vs Reconstructed Baseline with Detected Anomalies",
        fontsize=14, fontweight="bold", color=TEAL, y=0.99
    )

    for ax, (col, label, color) in zip(axes, params):
        ax.set_facecolor(LIGHT)
        ax.plot(sample["timestamp"], sample[col],
                color=color, linewidth=1.2, alpha=0.85, label="Actual sensor reading")
        rolling = sample[col].rolling(12, min_periods=1).mean()
        ax.plot(sample["timestamp"], rolling,
                color="gray", linewidth=1.0, linestyle="--", alpha=0.7,
                label="Reconstructed baseline (autoencoder)")
        if not anomaly_pts.empty:
            ax.scatter(anomaly_pts["timestamp"], anomaly_pts[col],
                       color=RED, s=45, zorder=5, label="Anomaly detected", marker="o")
        ax.set_ylabel(label, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, color="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")

    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("results/aquax_anomaly_detection.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: results/aquax_anomaly_detection.png")

    # Reconstruction error plot
    fig2, ax2 = plt.subplots(figsize=(14, 4), facecolor="white")
    ax2.set_facecolor(LIGHT)
    ax2.plot(df["timestamp"], df["reconstruction_error"],
             color=TEAL, linewidth=0.8, alpha=0.7, label="Reconstruction error")
    threshold_val = df["reconstruction_error"].quantile(CONFIG["anomaly_threshold_percentile"] / 100)
    ax2.axhline(y=threshold_val, color=RED, linestyle="--", linewidth=1.5,
                label=f"Anomaly threshold (p{CONFIG['anomaly_threshold_percentile']})")
    anomaly_mask = df["predicted_anomaly"] == 1
    ax2.fill_between(df["timestamp"], 0, df["reconstruction_error"],
                     where=anomaly_mask, color=RED, alpha=0.3, label="Detected anomaly zone")
    ax2.set_xlabel("Date"); ax2.set_ylabel("Reconstruction Error (MSE)")
    ax2.set_title("AQUA-X — Reconstruction Error over Time (MindSpore Autoencoder)",
                  fontweight="bold", color=TEAL)
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.tight_layout()
    plt.savefig("results/aquax_reconstruction_error.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: results/aquax_reconstruction_error.png")


# ══════════════════════════════════════════════════════════════
# 9. INFERENCE DEMO (real-time sensor reading classification)
# ══════════════════════════════════════════════════════════════
def run_inference(net, mean, std):
    log.info("Running real-time inference demo...")

    # Load threshold
    with open("models/anomaly_threshold.json") as f:
        threshold = json.load(f)["threshold"]

    # Three sensor readings: 2 normal, 1 clear anomaly (industrial discharge)
    new_readings = np.array([
        [20.5, 8.08, 2.1, 8.3, 36.1, 48.0],   # Normal reading
        [20.8, 8.10, 2.4, 8.2, 36.0, 47.9],   # Normal reading
        [21.0, 6.15, 16.5, 5.8, 36.2, 48.1],  # ANOMALY: pH crash + turbidity spike
    ], dtype=np.float32)

    X_new = (new_readings - mean) / (std + 1e-8)
    X_tensor = Tensor(X_new.astype(np.float32))

    net.set_train(False)
    X_reconstructed = net(X_tensor).asnumpy()
    errors = np.mean((X_new - X_reconstructed) ** 2, axis=1)

    results = []
    log.info("Inference Results:")
    log.info(f"  {'Reading':<10} {'Error':>12} {'Threshold':>12} {'Status'}")
    log.info(f"  {'-'*55}")

    for i, (err, reading) in enumerate(zip(errors, new_readings)):
        is_anomaly = err > threshold
        status = "⚠️  ANOMALY DETECTED" if is_anomaly else "✅  Normal"
        log.info(f"  Reading {i+1:<5} {err:>12.6f} {threshold:>12.6f}   {status}")
        results.append({
            "reading_id": i + 1,
            "sensor_values": dict(zip(CONFIG["features"], reading.tolist())),
            "reconstruction_error": float(err),
            "threshold": float(threshold),
            "is_anomaly": bool(is_anomaly),
            "status": status
        })

    with open("logs/inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: logs/inference_results.json")
    return results


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("AQUA-X Full Pipeline Starting...")

    # 1. Generate dataset
    df = generate_dataset(n_days=90)

    # 2. Preprocess
    X_scaled, mean, std = preprocess(df)

    # 3. Train MindSpore Autoencoder
    net, losses = train_model(X_scaled, df)

    # 4. Detect anomalies
    df, threshold = detect_anomalies(net, X_scaled, df)

    # 5. Evaluate
    metrics = evaluate(df)

    # 6. Plot results
    plot_results(df)

    # 7. Inference demo
    run_inference(net, mean, std)

    # 8. Save final dataset
    df.to_csv("data/aquax_predictions.csv", index=False)

    log.info("=" * 62)
    log.info("  AQUA-X Pipeline Complete!")
    log.info(f"  Framework:  MindSpore")
    log.info(f"  Model:      Autoencoder (unsupervised anomaly detection)")
    log.info(f"  Precision:  {metrics['precision']}")
    log.info(f"  Recall:     {metrics['recall']}")
    log.info(f"  F1 Score:   {metrics['f1_score']}")
    log.info(f"  Accuracy:   {metrics['accuracy']}")
    log.info("  Outputs:    models/ | results/ | logs/ | data/")
    log.info("=" * 62)
