# Airbnb Price Prediction — ML Pipeline

Reproduction and extension of [Rezazadeh Kalehbasti et al. (arXiv:1907.12665)](https://arxiv.org/abs/1907.12665) on the NYC Airbnb dataset.

**Course:** IEOR E4579 — Machine Learning in Practice, Columbia University, Spring 2026  
**Authors:** Thomas Flamand (`tf2636`)

---

## Results summary

| Model | Test R² | Test MAE | Test MSE |
|---|---|---|---|
| Linear Regression (baseline) | −287,387 | 17.709 | 134,308 |
| Ridge | 0.692 | 0.273 | 0.144 |
| K-Means + Ridge | 0.698 | 0.270 | 0.141 |
| Neural Network | 0.677 | 0.273 | 0.151 |
| SVR | 0.723 | 0.251 | 0.129 |
| **Gradient Boosting (XGBoost)** | **0.745** | **0.244** | **0.119** |
| LightGBM (Optuna) | 0.752 | 0.240 | 0.116 |
| Stacking Ensemble | 0.754 | 0.238 | 0.115 |

Target: `log(price)`. Key fix vs. original paper: MinMaxScaler is fit on the training set only (no leakage).

---

## Project structure

```
Airbnb_Price_Prediction/
├── Airbnb_Price_Pipeline_v4.ipynb   ← main notebook (all modules)
├── Models
├── .gitignore
└── README.md
```

> **Note on data & model files:** The `Data/` directory is **not included in this repository** — raw data files exceeded GitHub's file size limits and are excluded via `.gitignore`. It must be recreated locally by following the setup instructions below. All intermediate outputs are generated automatically by the notebook and saved to Google Drive.

When fully set up (and after running the code), your Google Drive will contain:

```
MyDrive/Colab Notebooks/ML_Practice/
├── Data/
│   ├── listings.csv                      ← raw listings (user-provided, see below)
│   ├── reviews_original.csv              ← raw reviews (user-provided, see below)
│   ├── reviews_sentiment.csv             ← MODULE 1 output
│   ├── data_cleaned.csv                  ← MODULE 2 output
│   ├── data_cleaned_train_X/y.csv        ← MODULE 2 output
│   ├── data_cleaned_val_X/y.csv          ← MODULE 2 output
│   ├── data_cleaned_test_X/y.csv         ← MODULE 2 output
│   ├── selected_features_lasso.npy       ← MODULE 3 output
│   ├── selected_features_pvals.npy       ← MODULE 3 output
│   ├── feature_selection_r2.json         ← MODULE 3 output
│   ├── data_trainval_lasso_X.csv         ← MODULE 4 output
│   └── data_trainval_lasso_y.csv         ← MODULE 4 output
└── Models/
    ├── scaler.pkl                        ← MinMaxScaler (MODULE 2)
    ├── results_metrics.csv               ← MODULE 4 output
    ├── best_params.json                  ← MODULE 4 output
    ├── variance_results.csv              ← MODULE 6 output
    ├── arch_results.csv                  ← MODULE 7 output
    ├── arch_best_configs.json            ← MODULE 7 output
    └── trained_models/
        ├── Linear_Baseline.joblib
        ├── Ridge.joblib
        ├── Gradient_Boosting.joblib
        ├── KMeans+Ridge.joblib
        ├── SVR.joblib
        └── Neural_Network.keras
```

---

## Setup on Google Colab

### 1 — Get the raw data

Download the NYC Airbnb dataset from [Inside Airbnb](https://insideairbnb.com/get-the-data/) (select **New York City**, version dated **16 February 2026**):

- `listings.csv.gz` → decompress → `listings.csv`
- `reviews.csv.gz` → decompress → rename to `reviews_original.csv`

Upload both files to your Google Drive at:

```
MyDrive/Colab Notebooks/ML_Practice/Data/
```

The `Models/` directory and all other `Data/` files are created automatically by the notebook. You only need to provide these two raw files.

### 2 — Open the notebook in Colab

Clone or download this repository, then open `Airbnb_Price_Pipeline_v4.ipynb` in Google Colab.

> **Recommended runtime: GPU** (T4 or A100 via Colab Pro). Modules 4 and 6 use cuML and XGBoost with CUDA and will not run on CPU.

> **Note for local execution:** This pipeline is optimised for Google Colab with GPU. Running locally requires a CUDA-capable GPU for Modules 4 and 6, and manual installation of cuML (not available via pip on all platforms).

### 3 — Configure paths (if needed)

All modules define the same two path constants at the top of each cell:

```python
DATA_DIR   = '/content/drive/MyDrive/Colab Notebooks/ML_Practice/Data'
MODELS_DIR = '/content/drive/MyDrive/Colab Notebooks/ML_Practice/Models'
```

If your Drive layout differs, update these two variables. They are repeated in each module cell so that each module can be run independently.

### 4 — Run MODULE 0 first (every session)

MODULE 0 installs cuML in the background and mounts Google Drive. Run it once per Colab session before anything else.

---

## Modules

Each module is **fully independent**: it loads its inputs from Drive and saves its outputs to Drive. You can re-run any single module without re-running upstream ones, as long as their outputs already exist.

| Module | Description | Key inputs | Key outputs |
|---|---|---|---|
| **0** | Setup & GPU install | — | cuML installed, Drive mounted |
| **1** | Sentiment analysis (TextBlob) | `reviews_original.csv` | `reviews_sentiment.csv` |
| **2** | Preprocessing, train/val/test split, MinMaxScaler | `listings.csv`, `reviews_sentiment.csv` | split CSVs, `data_cleaned.csv`, `scaler.pkl` |
| **3** | Feature selection (Lasso + p-value) | split CSVs | `selected_features_lasso.npy`, `selected_features_pvals.npy`, `feature_selection_r2.json` |
| **4** | Model training — 6 models with Optuna tuning | split CSVs + lasso features | `results_metrics.csv`, `best_params.json`, `trained_models/` |
| **4b** | Results summary + bar chart | `results_metrics.csv` | plot (inline) |
| **5** | SHAP interpretability | trained models + lasso data | SHAP plots (inline) |
| **6** | Multi-seed variance assessment | `data_cleaned.csv` + lasso features | `variance_results.csv` |
| **7** | Architectural exploration (LightGBM, stacking, tuned NN) | lasso train/val + test CSVs | `arch_results.csv`, `arch_best_configs.json` |

**Sentinel pattern:** every module checks at startup whether its outputs already exist. If they do, it skips computation and loads the cached results. To force a re-run, delete the relevant output file from Drive.

---

## Models

### Reproduced (Module 4)

- **Linear Regression** — OLS on all 764 features (unregularised baseline)
- **Ridge** — L2 regression, α tuned with Optuna (cuML GPU)
- **K-Means + Ridge** — cluster-then-regress, k and α tuned with Optuna (cuML GPU)
- **SVR** — RBF kernel, C/γ tuned with Optuna (cuML GPU)
- **Gradient Boosting** — XGBoost with `device="cuda"`, hyperparameters tuned with Optuna
- **Neural Network** — 3-layer MLP (Keras/TensorFlow), units and lr tuned with Optuna

### Extended (Module 7)

- **LightGBM** — leaf-wise boosting, 30-trial Optuna study
- **Stacking Ensemble** — Ridge + 2× LightGBM as base learners, Ridge meta-learner trained on 5-fold out-of-fold predictions

---

## Key implementation notes

**Data leakage fix:** the original codebase fits `MinMaxScaler` on the concatenation of train and test sets. This pipeline fits the scaler on the training split only and applies `.transform()` to val and test.

**Feature selection:** Lasso with λ swept over 50 log-spaced values; optimal λ selected by validation R²_adj. Retains 139 of 764 features. All subsequent models use this Lasso-selected subset.

**Train/val/test split:** 90/5/5. Models are tuned on validation, then retrained on train+val before final test evaluation.

**GPU requirements:** Modules 4 and 6 require a CUDA GPU. Ridge and SVR use cuML drop-in replacements; Gradient Boosting uses `xgb.XGBRegressor(device="cuda")`. Module 7 (LightGBM, stacking) runs on CPU.

---

## Dependencies

Installed automatically within the notebook:

```
optuna
lightgbm
shap
cuml-cu12==25.02.*     # GPU modules only (MODULE 0)
cudf-cu12==25.02.*     # GPU modules only (MODULE 0)
xgboost
tensorflow
scikit-learn >= 1.0
textblob
joblib
```

---

## Reference

Rezazadeh Kalehbasti, P., Nikolenko, L., & Rezaei, H. (2019). *Airbnb Price Prediction Using Machine Learning and Sentiment Analysis*. [arXiv:1907.12665](https://arxiv.org/abs/1907.12665).
