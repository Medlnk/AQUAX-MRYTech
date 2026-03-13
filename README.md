# AQUA-X 🌊 — AI Smart Buoy for Water Quality Monitoring

> **Huawei ICT Innovation Competition 2025-2026 | Regional Phase**  
> Team: MRYTech | Institution: Suptech Santé Essaouira | Morocco

---

## 📌 Project Overview

AQUA-X is an autonomous AI-powered smart buoy that monitors water quality in real time using a network of environmental sensors. It detects anomalies such as industrial pollution, algal blooms, thermal discharges, and freshwater influxes using a deep learning autoencoder built with **MindSpore 2.8.0** — Huawei's official AI framework.

---

## ✅ Implemented — What We Actually Built & Ran

| Technology | Usage | Status |
|------------|-------|--------|
| **MindSpore 2.8.0** | AI model training & inference (autoencoder anomaly detection) | ✅ Implemented & verified |

The MindSpore autoencoder was trained on 90 days of simulated buoy sensor data, successfully detecting 4 real pollution event types with **95.74% accuracy**.

---

## 🗺️ Planned Integration — Design Roadmap

These technologies are part of the AQUA-X system architecture and are planned for integration in future development phases:

| Technology | Planned Role |
|------------|-------------|
| **Huawei ModelArts** | Cloud-based model training, evaluation, and deployment at scale |
| **Huawei Cloud** | Scalable data storage, processing, and dashboard hosting |
| **AIoT** | Intelligent device-to-cloud sensor integration workflows |
| **4G/5G NB-IoT** | Buoy-to-cloud wireless connectivity from remote locations |

---

## 🤖 AI Model — MindSpore 2.8.0 Autoencoder

- **Framework:** MindSpore 2.8.0 (Huawei AI)
- **Architecture:** Unsupervised Autoencoder — `6 → 32 → 16 → 8 → 4 → 8 → 16 → 32 → 6`
- **Task:** Anomaly detection on multivariate water quality time-series
- **Training data:** 2,116 normal readings over 90 days
- **Detection method:** Reconstruction error thresholding (95th percentile)

### ✅ Verified Performance Metrics

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 95.74% |
| Precision | 27.78% |
| Recall    | 68.18% |
| F1 Score  | 39.47% |

### Detected Pollution Events

| Event | Day | Type |
|-------|-----|------|
| 1 | Day 15 | Industrial discharge (pH drop + turbidity spike) |
| 2 | Day 32 | Algal bloom (DO rise + turbidity) |
| 3 | Day 55 | Thermal discharge (temperature spike) |
| 4 | Day 72 | Freshwater influx (salinity + conductivity drop) |

---

## 📡 Monitored Parameters

| Sensor | Unit |
|--------|------|
| Water Temperature | °C |
| pH Level | — |
| Turbidity | NTU |
| Dissolved Oxygen | mg/L |
| Salinity | PSU |
| Conductivity | mS/cm |

---

## 📁 Repository Structure

```
AQUAX-MRYTech/
├── data/
│   ├── aquax_buoy_dataset.csv       # 2,160 rows, 90-day dataset
│   └── aquax_predictions.csv        # Model predictions with anomaly flags
├── models/
│   ├── aquax_autoencoder.ckpt       # Trained MindSpore model weights
│   ├── scaler_params.json           # Normalization parameters
│   └── anomaly_threshold.json       # Detection threshold (95th percentile)
├── logs/
│   ├── training_loss.json           # Epoch-by-epoch training loss (60 epochs)
│   ├── metrics.json                 # Evaluation metrics
│   └── inference_results.json       # Real-time inference demo results
├── results/
│   ├── aquax_anomaly_detection.png  # Anomaly detection chart (4 parameters)
│   └── training_loss_curve.png      # Training convergence curve
└── README.md
```

---

## 🚀 How to Reproduce

### Requirements
```bash
pip install mindspore==2.8.0 numpy pandas matplotlib
```

> ⚠️ MindSpore requires Python 3.9. Use Google Colab for easiest setup.

### Run on Google Colab (Recommended)
```python
!pip install mindspore -i https://pypi.tuna.tsinghua.edu.cn/simple
# Then run the cells in order from the notebook
```

### Expected Output
```
✅ MindSpore 2.8.0 installed
✅ Dataset: 2160 rows | 44 anomaly points
✅ Training done in ~180s | Final loss: 0.2488
   Accuracy: 95.74% | Recall: 68.18% | F1: 0.3947
Reading 3: ⚠️  ANOMALY (error=67.10 | threshold=0.67)
```

---

## 👥 Team

**MRYTech** — Huawei ICT Innovation Competition 2025-2026  
Suptech Santé Essaouira | Morocco Regional Phase
