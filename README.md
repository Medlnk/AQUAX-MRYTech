# AQUA-X 🌊 — AI Smart Buoy for Water Quality Monitoring

> **Huawei ICT Innovation Competition 2025-2026 | Regional Phase**
> Team: MRYTech | Morocco

---

## 📌 Project Overview

AQUA-X is an autonomous AI-powered smart buoy that monitors water quality in real time using a network of environmental sensors. It detects anomalies such as industrial pollution, algal blooms, thermal discharges, and freshwater influxes using a deep learning autoencoder built with **MindSpore 2.8.0** (Huawei AI framework).

---

## 🤖 AI Model — MindSpore Autoencoder

- **Framework:** MindSpore 2.8.0 (Huawei AI)
- **Architecture:** Unsupervised Autoencoder (6→32→16→8→4→8→16→32→6)
- **Task:** Anomaly detection on multivariate water quality time-series
- **Training:** 2,116 normal readings over 90 days

### Performance Metrics
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 95.74% |
| Precision | 27.78% |
| Recall    | 68.18% |
| F1 Score  | 39.47% |

---

## 📡 Monitored Parameters

| Sensor              | Unit  |
|---------------------|-------|
| Water Temperature   | °C    |
| pH Level            | —     |
| Turbidity           | NTU   |
| Dissolved Oxygen    | mg/L  |
| Salinity            | PSU   |
| Conductivity        | mS/cm |

---

## 📁 Repository Structure

```
AQUAX-MRYTech/
├── data/
│   ├── aquax_buoy_dataset.csv       # 2,160 rows, 90-day dataset
│   └── aquax_predictions.csv        # Model predictions
├── models/
│   ├── aquax_autoencoder.ckpt       # Trained MindSpore model weights
│   ├── scaler_params.json           # Normalization parameters
│   └── anomaly_threshold.json       # Detection threshold (95th percentile)
├── logs/
│   ├── training_loss.json           # Epoch-by-epoch training loss
│   ├── metrics.json                 # Evaluation metrics
│   └── inference_results.json       # Real-time inference demo results
├── results/
│   ├── aquax_anomaly_detection.png  # Anomaly detection chart
│   └── training_loss_curve.png      # Training loss curve
└── README.md
```

---

## 🚀 How to Reproduce

### Requirements
```bash
pip install mindspore==2.8.0 numpy pandas matplotlib
```

### Run the Model
```bash
python src/aquax_mindspore.py
```

The script will:
1. Generate a 90-day synthetic buoy dataset with 4 real pollution events
2. Train the MindSpore autoencoder (60 epochs)
3. Detect anomalies using reconstruction error thresholding
4. Save all results to `logs/` and `results/`

### Expected Output
```
✅ Training done
   Final loss: ~0.248
   Accuracy: 95.74%
Reading 3: ⚠️  ANOMALY  (error=67.10 | threshold=0.67)
```

---

## 🛰️ Huawei Technology Stack

| Technology     | Usage                                      |
|----------------|--------------------------------------------|
| MindSpore 2.8.0 | AI model training & inference             |
| ModelArts      | Cloud-based model training environment     |
| Huawei Cloud   | Data storage & dashboard hosting           |
| AIoT           | Device-to-cloud sensor integration         |
| 4G/5G NB-IoT   | Buoy-to-cloud connectivity                 |

---

## 🌍 Deployment Zones

- Coastal monitoring stations
- River delta pollution detection
- Aquaculture farms
- Industrial discharge zones

---

## 👥 Team

**MRYTech** — Huawei ICT Innovation Competition 2025-2026, Morocco Regional Phase
