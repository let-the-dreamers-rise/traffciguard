<![CDATA[# 🚦 TraffciGuard — Urban Traffic Demand Prediction

> **Predicting real-time traffic demand across 1,000+ geohash-encoded urban zones using advanced spatiotemporal feature engineering and gradient-boosted ensemble models.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-9ACD32)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![R² Score](https://img.shields.io/badge/R²_Score-0.964-brightgreen)](.)

---

## 📋 Problem Statement

Given historical traffic data across urban locations encoded as geohashes, the goal is to predict the **demand** (traffic intensity, normalized 0–1) for unseen time periods and locations. The dataset captures spatiotemporal demand patterns influenced by road infrastructure, weather, and geographic features.

### Key Challenges
- **Spatiotemporal distribution shift** — training data covers different time regimes than the test set
- **Sparse geohash coverage** — not all location × time combinations are observed
- **Missing features** — temperature and weather data have significant gaps
- **High cardinality** — 1,000+ unique geohash locations with temporal granularity at 15-minute intervals

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Raw Data (77K rows)                        │
│  geohash · timestamp · demand · RoadType · Lanes · Weather   │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Feature Engineering (25+ features)               │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  Geospatial  │  │   Temporal    │  │   Target Encoding    │ │
│  │  - lat/lon   │  │  - slot/hour  │  │  - Bayesian smooth   │ │
│  │  - gh3/4/5   │  │  - sin/cos    │  │  - Multi-resolution  │ │
│  │  - KNN(k=8)  │  │  - lag ±1,±2  │  │  - OOF leak-proof   │ │
│  └─────────────┘  └──────────────┘  └──────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Matrix Factorization (SVD latent factors, rank=12)      │  │
│  │  Location × TimeSlot → 6 latent dims + reconstruction    │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Multi-Model Ensemble                         │
│                                                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │ LightGBM  │  │ LightGBM  │  │  XGBoost  │  │ CatBoost  │ │
│  │  (L2)     │  │  (Huber)  │  │           │  │           │ │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘ │
│        └───────────────┴──────────────┴───────────────┘       │
│                         ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Multi-Seed (3×) 5-Fold CV + Optimized Weighted Blend   │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
                   R² = 0.964
```

---

## 🔬 Feature Engineering

### Geospatial Features
| Feature | Description |
|---------|-------------|
| `lat`, `lon` | Decoded from geohash (Base32 → coordinates) |
| `gh3`, `gh4`, `gh5` | Hierarchical geohash prefixes for multi-resolution encoding |
| `sp_knn` | Mean demand of 8 nearest spatial neighbors (KNN on lat/lon) |

### Temporal Features
| Feature | Description |
|---------|-------------|
| `slot` | 15-minute time slot index (0–95 per day) |
| `t_sin`, `t_cos` | Cyclical encoding of time-of-day |
| `lag_prev/next` | Demand from adjacent time slots (±1, ±2) |
| `roll3` | Rolling mean across 3 temporal neighbors |

### Target Encoding (Bayesian Smoothing)
| Feature | Grouping Keys | Smoothing (m) |
|---------|--------------|---------------|
| `te_gh_ts` | geohash × timestamp | 1.0 |
| `te_gh5_ts` | gh5 × timestamp | 2.0 |
| `te_gh4_ts` | gh4 × timestamp | 3.0 |
| `te_gh` | geohash | 3.0 |
| `te_ts` | timestamp | 5.0 |
| `te_gh_hour` | geohash × hour | 1.0 |
| `te_road_ts` | RoadType × slot | 5.0 |

All target encodings use **out-of-fold (OOF) computation** within 5-fold CV to prevent data leakage.

### Latent Factors (SVD)
- Constructed a **Location × TimeSlot** pivot matrix from training data
- Applied **TruncatedSVD (rank=12)** to extract latent demand patterns
- Used top 6 latent dimensions as features + full reconstruction as baseline signal

---

## 🧠 Models & Ensembling Strategy

### Base Models

| Model | Loss | Key Hyperparameters |
|-------|------|-------------------|
| LightGBM | L2 (MSE) | 4000 trees, lr=0.02, 127 leaves, early stopping |
| LightGBM | Huber (α=0.9) | 1200 trees, lr=0.03, 63 leaves |
| XGBoost | Squared Error | Tuned depth, colsample, subsample |
| CatBoost | RMSE | Auto categorical handling |
| HistGBR | Squared Error | 600 iterations, lr=0.03, 63 leaf nodes |

### Ensemble Architecture
1. **Multi-seed averaging** — each model trained with 3 different random seeds (42, 7, 2024) to reduce variance
2. **5-fold cross-validation** — stratified OOF predictions for reliable local validation
3. **Weighted blending** — optimized blend weights via grid search on held-out validation set
4. **Multi-engine blending** — combined anchor model (full-data OOF TE), day-specific engine, structural model, and residual model

### Validation Strategy
- **Proxy A (Cross-day):** Train on day 48 → predict day 49 (tests temporal generalization)
- **Proxy B (Daytime split):** Hold out half of day 48 daytime → predict it (tests spatial generalization)
- **OOF R² tracking** across all ensemble configurations

---

## 📊 Results

| Model / Ensemble | OOF R² | Notes |
|------------------|--------|-------|
| Single LightGBM (L2) | 0.951 | Baseline with basic TE |
| + Temporal neighbors | 0.958 | Added lag/lead features |
| + Spatial KNN + SVD | 0.961 | Geospatial latent factors |
| Multi-seed 3× ensemble | **0.964** | Final submission |

---

## 📁 Project Structure

```
traffciguard/
├── dataset/
│   ├── train.csv              # 77K training samples
│   ├── test.csv               # Test set for prediction
│   └── sample_submission.csv  # Submission format
│
├── best.py                    # Best single pipeline (SVD + KNN + multi-seed)
├── ens.py                     # Multi-model ensemble (LightGBM + Huber + HistGBR)
├── optimize.py                # Blend weight optimization on validation proxy
├── structural.py              # Structural daytime model for regime transfer
├── pipeline.py                # Full feature engineering pipeline
├── solution.ipynb             # Exploratory notebook
│
├── day49*.py                  # Iterative model improvements
├── adv.py                     # Advanced feature experiments
├── gen.py                     # Generalization-focused model
│
├── submission_best.csv        # Best submission (R² = 0.964)
├── submission_*.csv           # 90+ ensemble variants explored
│
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost
```

### Run Best Model
```bash
python best.py
```
This will:
1. Load and preprocess the dataset
2. Engineer 25+ spatiotemporal features
3. Train a multi-seed LightGBM ensemble with 5-fold CV
4. Output OOF R² score and generate `submission_best.csv`

### Run Full Ensemble Pipeline
```bash
python ens.py          # Multi-model ensemble
python optimize.py     # Optimize blend weights
python structural.py   # Structural model + final blend
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **LightGBM** — Primary gradient boosting framework
- **XGBoost / CatBoost** — Secondary boosting models for diversity
- **scikit-learn** — KNN, SVD, HistGradientBoosting, cross-validation
- **Pandas / NumPy** — Data manipulation and feature engineering

---

## 📚 Key Techniques Used

- Geohash decoding (Base32 → latitude/longitude)
- Bayesian target encoding with smoothing priors
- Out-of-fold encoding to prevent leakage
- TruncatedSVD for matrix factorization on spatiotemporal data
- Spatial KNN averaging for geographic smoothing
- Cyclical feature encoding (sine/cosine transforms)
- Multi-seed ensembling for variance reduction
- Custom cross-temporal validation proxies
- Weighted blend optimization via grid search

---

## 👤 Author

**Ashwin Goyal**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
]]>
