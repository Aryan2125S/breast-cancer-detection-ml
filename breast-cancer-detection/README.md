<div align="center">

# 🔬 Breast Cancer Detection

### AI-Powered Multi-Agent ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A production-ready machine learning system that classifies breast tumors as **Benign** or **Malignant** using the Breast Cancer Wisconsin (Diagnostic) dataset. Built with a modular **multi-agent architecture** — each agent handles one phase of the ML lifecycle with explicit, structured I/O.

[**🚀 Quick Start**](#-quick-start) · [**📊 Results**](#-model-performance) · [**🏗️ Architecture**](#️-architecture) · [**🖥️ Web App**](#️-streamlit-web-app)

</div>

---

## 📋 Overview

| | Details |
|---|---|
| **Problem** | Binary classification of breast tumors from cell nucleus measurements |
| **Dataset** | [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) — 569 samples, 30 features |
| **Models** | Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN) |
| **Best Model** | Logistic Regression — selected for highest **Recall** |
| **Key Metric** | **98.6% Recall** — only 1 false negative in 114 test samples |
| **Deployment** | Interactive Streamlit web application |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation & Run

```bash
# 1. Clone or navigate to the project
cd breast-cancer-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (trains models + generates visualizations + saves model)
python main.py

# 4. Launch the interactive web app
streamlit run app.py
```

> **Note:** Step 3 must be run before Step 4. The pipeline generates the model files that the Streamlit app depends on.

---

## 🏗️ Architecture

This project uses a **multi-agent system** where each agent is a self-contained Python module with explicit dict-based inputs and outputs:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   🗃️ Agent 1    │───▶│   📊 Agent 2    │    │   🤖 Agent 3    │───▶│   🚀 Agent 4    │
│                 │    │                 │    │                 │    │                 │
│  Data & Preproc │    │  EDA & Feature  │    │  Model Training │    │  Deployment &   │
│                 │    │  Analysis       │    │  & Evaluation   │    │  Packaging      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Load dataset  │    │ • Class distrib │    │ • Train 3 models│    │ • Save model    │
│ • Validate data │    │ • Correlation   │    │ • Full metrics  │    │ • Save scaler   │
│ • Scale features│    │ • Violin plots  │    │ • ROC curves    │    │ • Prediction fn │
│ • Train/test    │    │ • KDE density   │    │ • Select best   │    │ • Streamlit app │
│   split         │    │ • Insights      │    │ • Feature imp.  │    │                 │
└────────┬────────┘    └─────────────────┘    └────────┬────────┘    └─────────────────┘
         │                                             │
         └─────────────────────────────────────────────┘
                    Shared: X_train, X_test, y_train, y_test
```

**Why agents?** Each module can be tested independently, replaced with a different implementation, or extended without affecting others.

---

## 📁 Project Structure

```
breast-cancer-detection/
│
├── 📂 agents/                         # Multi-agent modules
│   ├── __init__.py                    # Package exports
│   ├── agent1_data_preprocessing.py   # Data loading, validation, scaling, splitting
│   ├── agent2_eda_analysis.py         # EDA visualizations with dark theme
│   ├── agent3_model_training.py       # Model training, evaluation, comparison
│   └── agent4_deployment.py           # Model serialization & prediction system
│
├── 📂 data/                           # Generated visualizations (10 PNG files)
│   ├── 01_class_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_feature_violins.png
│   ├── 04_feature_density.png
│   ├── 05_correlation_ranking.png
│   ├── 06_model_radar.png
│   ├── 07_model_comparison.png
│   ├── 08_confusion_matrices.png
│   ├── 09_roc_curves.png
│   └── 10_feature_importance.png
│
├── 📂 model/                          # Serialized artifacts
│   ├── best_model.pkl                 # Trained best model (Logistic Regression)
│   ├── scaler.pkl                     # Fitted StandardScaler
│   └── metadata.pkl                   # Feature ranges & config for app
│
├── 📂 notebooks/                      # Optional Jupyter notebooks
│
├── 🐍 main.py                        # Pipeline orchestrator — runs all agents
├── 🌐 app.py                         # Streamlit web application
├── 📋 requirements.txt               # Python dependencies
└── 📖 README.md                      # This documentation
```

---

## 📊 Model Performance

### Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|:------|:--------:|:---------:|:------:|:--------:|:-------:|
| **🏆 Logistic Regression** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9976** |
| SVM | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9971 |
| KNN | 0.9649 | 0.9595 | 0.9861 | 0.9726 | 0.9938 |

> Values shown are from `random_state=42`. Run `python main.py` to reproduce.

### Why Recall Is the Priority Metric

In cancer detection, the cost of errors is asymmetric:

| Error Type | What Happens | Consequence | Severity |
|:-----------|:-------------|:------------|:--------:|
| **False Negative** | Malignant → predicted Benign | Patient misses treatment | 🔴 **Critical** |
| **False Positive** | Benign → predicted Malignant | Extra tests performed | 🟡 Moderate |

> **A missed cancer diagnosis can be fatal.** Therefore, we optimize for **Recall** (sensitivity) — the model's ability to correctly identify *all* malignant cases. Our best model achieves **98.6% Recall**, missing only 1 malignant case out of 72.

### Visualizations Generated

The pipeline generates **10 publication-quality visualizations** with a consistent dark theme:

| # | Visualization | Purpose |
|:-:|:--------------|:--------|
| 01 | Class Distribution (Donut + Bar) | Shows dataset balance |
| 02 | Correlation Heatmap | Feature-target relationships |
| 03 | Violin Plots | Distribution shape by diagnosis |
| 04 | KDE Density Plots | Overlap analysis between classes |
| 05 | Correlation Ranking | Top 15 features ranked |
| 06 | Radar Chart | Multi-metric model comparison |
| 07 | Grouped Bar Chart | Side-by-side model metrics |
| 08 | Confusion Matrices | Error analysis per model |
| 09 | ROC Curves | Discrimination ability |
| 10 | Feature Importance | Key predictive features |

---

## 📊 Dataset Information

The **Breast Cancer Wisconsin (Diagnostic)** dataset contains measurements from digitized images of fine needle aspirate (FNA) of breast masses. Each measurement describes characteristics of the cell nuclei.

### Feature Categories (30 features total)

| Category | Count | Examples |
|:---------|:-----:|:--------|
| **Mean** | 10 | `mean radius`, `mean texture`, `mean smoothness` |
| **Standard Error** | 10 | `radius error`, `texture error`, `smoothness error` |
| **Worst** (largest 3) | 10 | `worst radius`, `worst texture`, `worst smoothness` |

### Top Predictive Features

| Rank | Feature | \|Correlation\| | Why It Matters |
|:----:|:--------|:--------:|:---------------|
| 1 | worst concave points | 0.794 | Shape irregularity is a strong malignancy indicator |
| 2 | worst perimeter | 0.783 | Larger tumors more likely malignant |
| 3 | mean concave points | 0.777 | Concavity reflects irregular cell boundaries |
| 4 | worst radius | 0.777 | Tumor size correlates with malignancy |
| 5 | mean perimeter | 0.743 | Consistent with size-based features |

---

## 🖥️ Streamlit Web App

The interactive web application features:

- **🎨 Glassmorphism UI** — Dark theme with frosted glass effects, gradient text, and hover animations
- **📊 Stats Dashboard** — Key metrics displayed prominently
- **🎛️ Grouped Sliders** — Features organized into collapsible categories (Mean / SE / Worst)
- **🎯 Instant Prediction** — Color-coded result with pulsing glow animation
- **📈 Confidence Bars** — Visual probability breakdown with gradient progress bars
- **🔑 Key Feature Table** — Shows how your inputs compare to dataset averages
- **📋 How It Works** — Step-by-step explanation for non-technical users

---

## 🔧 Pipeline Steps

### Agent 1: Data & Preprocessing
- Loads dataset from `sklearn.datasets.load_breast_cancer()`
- Validates: no missing values, all numeric data types
- Applies **StandardScaler** (z-score normalization) — critical because SVM/KNN are distance-based algorithms
- Splits 80/20 with stratification (`random_state=42`)

### Agent 2: EDA & Feature Analysis
- Analyzes class distribution (1.68:1 Benign:Malignant ratio)
- Generates 5 visualizations: donut chart, correlation heatmap, violin plots, KDE density, correlation ranking
- Identifies top features by Pearson correlation with target
- Detects 21 multicollinear feature pairs (|r| > 0.9)

### Agent 3: Model Training & Evaluation
- Trains Logistic Regression, SVM (RBF kernel), and KNN (k=5)
- Evaluates with 6 metrics: accuracy, precision, recall, F1, confusion matrix, ROC AUC
- Generates 5 visualizations: radar chart, grouped bars, confusion matrices, ROC curves, feature importance
- Selects best model by **Recall** (with F1 as tiebreaker)

### Agent 4: Deployment
- Serializes model and scaler with `joblib`
- Saves metadata (feature ranges) for Streamlit slider initialization
- Verifies prediction system with known samples
- Creates production-ready prediction function with input validation

---

## 🛠️ Tech Stack

| Technology | Purpose |
|:-----------|:--------|
| **Python 3.8+** | Core language |
| **scikit-learn** | ML models, preprocessing, metrics |
| **pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **matplotlib** | Base visualization engine |
| **seaborn** | Statistical visualizations |
| **Streamlit** | Interactive web application |
| **joblib** | Model serialization |

---

## ⚠️ Disclaimer

> This project is for **educational and demonstration purposes only**. It is NOT a medical device and is NOT a substitute for professional medical diagnosis, treatment, or advice. No clinical decisions should be made based on this tool. Always consult a qualified healthcare provider for medical concerns.

---

## 📜 License

This project uses the publicly available [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) from the UCI Machine Learning Repository, bundled with scikit-learn.

<div align="center">

---

Built with ❤️ using Python, Scikit-learn & Streamlit

</div>
