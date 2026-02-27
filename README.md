
# GAMs Rashomon Set — Breast Cancer Wisconsin (7 Features)

> Exploring the **Rashomon Set of Generalized Additive Models (GAMs)** on the Breast Cancer Wisconsin dataset using variable importance ranges, monotonicity constraints, user-preferred shape functions, and abnormal pattern detection.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Applications](#applications)
  - [Application 1: Sample Models from Rashomon Set](#application-1-sample-models-from-rashomon-set)
  - [Application 2: Variable Importance Range](#application-2-variable-importance-range)
  - [Application 3: Monotonicity Constraints](#application-3-monotonicity-constraints)
  - [Application 4: User-Preferred Shape Functions](#application-4-user-preferred-shape-functions)
  - [Application 5: Detect Abnormal Jumps](#application-5-detect-abnormal-jumps)
- [Results Summary](#results-summary)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Citation](#citation)

---

## Overview

This project applies the **GAMs Rashomon Set** framework to breast cancer malignancy prediction. Rather than selecting a single best-performing model, it explores the entire set of near-optimal GAM models (the *Rashomon Set*) to understand how predictions and feature effects can vary across equally-good models.

Key capabilities demonstrated:

- Sampling multiple GAM models with similar loss from the Rashomon Set
- Computing variable importance ranges across all near-optimal models
- Finding models that satisfy domain-specific monotonicity constraints
- Projecting onto user-preferred shape functions
- Detecting non-monotone or abnormal patterns in shape functions

The framework is based on the paper:  
**"GAMs Rashomon Set"** — [GAMsRashomonSet GitHub](https://github.com/chudizhong/GAMsRashomonSet)

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Samples | 569 |
| Original features | 30 |
| Selected features | 7 (top by Random Forest importance) |
| Target | `diagnosis` — Malignant (1) / Benign (0) |
| Binning | 5 bins per feature (quantile-based) |

**Top 7 features selected:**

| Feature | Description |
|---|---|
| `concave points_worst` | Worst concave portions of the contour |
| `perimeter_worst` | Worst perimeter measurement |
| `area_worst` | Worst area measurement |
| `concave points_mean` | Mean concave portions of the contour |
| `radius_worst` | Worst radius (mean of distances to perimeter) |
| `perimeter_mean` | Mean perimeter |
| `radius_mean` | Mean radius |

> **Why 5 bins?** The variable importance LP solver runs 2^(n_bins) solves per feature. Using 5 bins (32 solves/feature × 7 features = 224 total) is ~32× faster than 10 bins (7,168 solves) while preserving sufficient granularity.

---

## Repository Structure

```
.
├── Cancer_GAMRS_ori_7features.ipynb   # Main notebook (all 5 applications)
├── datasets/
│   └── breast_cancer_wisconsin.csv    # Pre-processed, binned dataset (auto-generated)
├── GAMsRashomonSet/                   # Cloned dependency (auto-generated)
│   └── src/
│       ├── prepare_gam.py
│       ├── rset_app.py
│       ├── rset_opt.py
│       ├── run_app.py
│       └── utils.py
├── breast_cancer_wisconsin_0.001_0.001_1.01.p   # Saved model pickle (auto-generated)
└── README.md
```

---

## Installation

**1. Clone this repository**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**2. Clone the GAMsRashomonSet dependency**
```bash
git clone https://github.com/chudizhong/GAMsRashomonSet.git
```

**3. Install Python dependencies**
```bash
pip install numpy pandas matplotlib scikit-learn torch cvxpy pickle5
```

**4. Download the dataset**

Download `Breast Cancer Wisconsin.csv` from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) and place it at:
```
/content/Breast Cancer Wisconsin.csv   # if using Colab
```
or update the `RAW_CSV` path in the notebook.

---

## Quick Start

Open and run `Cancer_GAMRS_ori_7features.ipynb` in **Google Colab** or Jupyter, executing all cells in order.

The notebook is fully self-contained and will:
1. Clone the GAMsRashomonSet repo automatically
2. Pre-process and bin the dataset
3. Fit an initial Logistic Regression as a starting point
4. Optimise the Rashomon Set ellipsoid
5. Run all 5 applications

**Initial model performance (Logistic Regression baseline):**

| Metric | Value |
|---|---|
| Accuracy | **94.73%** |
| AUC | **98.42%** |
| Log-loss | 3.7394 |
| Rashomon bound (ε) | 3.7768 |
| Rashomon multiplier | 1.01 (1% above optimal loss) |

---

## Applications

### Application 1: Sample Models from Rashomon Set

Samples 100 GAM models from within the Rashomon Set ellipsoid and plots the shape functions for each feature across all sampled models. This visualises the range of plausible feature effects that are consistent with near-optimal predictive performance.

```python
w_samples = get_models_from_rset(FILEPATH, n_samples=100, plot_shape=True)
```

**Output:** 7 shape function plots (one per feature), each showing the spread of 100 sampled models.

---

### Application 2: Variable Importance Range

Computes the **minimum and maximum variable importance** each feature can achieve across all models in the Rashomon Set, using Linear Programming (LP) optimisation.

```python
variable_importance_range(FILEPATH, mip=False, plot_shape=True, plot_vir=True)
```

**Key results:**

| Feature | VIR Min (log obj) | VIR Max (log obj) |
|---|---|---|
| `concave points_worst` | 0.2574 | 0.5559 |
| `perimeter_worst` | 0.2016 | 0.2958 |
| `area_worst` | 0.2044 | 0.3267 |
| `concave points_mean` | 0.2094 | 0.3232 |
| `radius_worst` | 0.1961 | 0.5186 |
| `perimeter_mean` | 0.2156 | 0.3268 |
| `radius_mean` | 0.2237 | 0.3634 |

> `concave points_worst` and `radius_worst` show the widest variable importance ranges, meaning their true importance is most ambiguous across near-optimal models.

---

### Application 3: Monotonicity Constraints

Finds the model within the Rashomon Set that satisfies **monotone increasing** constraints for clinically meaningful features — i.e., higher feature values should correspond to higher malignancy risk.

```python
get_monotone(FILEPATH, f="radius_worst",    direction="increase")
get_monotone(FILEPATH, f="perimeter_worst", direction="increase")
get_monotone(FILEPATH, f="area_worst",      direction="increase")
```

**Results:** All three features (`radius_worst`, `perimeter_worst`, `area_worst`) have valid monotone-increasing models within the Rashomon Set, confirming that clinical domain knowledge can be encoded without sacrificing near-optimal accuracy.

| Feature | Log-loss after monotone constraint |
|---|---|
| `radius_worst` | 0.1899 |
| `perimeter_worst` | 0.1919 |
| `area_worst` | 0.1902 |

---

### Application 4: User-Preferred Shape Functions

Projects the optimal model onto a **user-specified shape function** for a given feature, finding the closest model in the Rashomon Set that respects the user's preference.

```python
w_user = np.linspace(-0.5, 0.5, cnt)   # linear ramp: monotone increasing preference
get_projection(FILEPATH, f="concave points_worst", w_user=w_user)
```

**Result for `concave points_worst`:**
- Original weights: `[-1.952, -1.678, 0.026, 2.909, 13.703]`
- Projected weights: `[-0.500, -0.250, 0.000, 0.250, 0.500]`
- Log-loss after projection: 0.2433 (still within Rashomon bound of 3.7768)

This shows the framework can accommodate analyst preferences while remaining near-optimal.

---

### Application 5: Detect Abnormal Jumps

Tests what **proportion of models** in the Rashomon Set exhibit non-monotone (abnormal) jumps between consecutive bins of a feature's shape function, using 500 sampled models.

```python
cnt_jump = test_jump(FILEPATH, n_samples=500, i=l, j=l+1, k=l+2)
```

**Results:**

| Feature | Non-monotone jump proportion |
|---|---|
| `radius_worst` | **53.8%** — majority of models show a jump |
| `concave points_worst` | **39.4%** — significant minority show a jump |

> A high proportion indicates that the shape function behaviour at those bins is genuinely ambiguous — some near-optimal models increase smoothly while others exhibit a non-monotone pattern. This flags features that may need monotonicity constraints enforced.

---

## Results Summary

| Application | Key Finding |
|---|---|
| Model sampling | Shape functions vary considerably across 100 sampled near-optimal models |
| Variable importance range | `concave points_worst` has the widest importance range (0.26–0.56) |
| Monotonicity | Clinically monotone models exist within the Rashomon Set for all 3 tested features |
| Shape projection | User-preferred linear shape for `concave points_worst` is achievable within the bound |
| Abnormal jumps | `radius_worst` shows non-monotone jumps in 53.8% of sampled models |

---

## Configuration

Key parameters set in the notebook:

| Parameter | Value | Description |
|---|---|---|
| `N_BINS` | 5 | Number of bins per feature after discretisation |
| `LAMB0` | 0.001 | L0 regularisation |
| `LAMB2` | 0.001 | L2 regularisation |
| `MULTIPLIER` | 1.01 | Rashomon bound = optimal loss × multiplier |
| `n_iters` | 500 | Ellipsoid optimisation iterations |
| `n_samples` | 100/500 | Models sampled from Rashomon Set |

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `pandas` | Data handling |
| `scikit-learn` | Logistic Regression, preprocessing |
| `torch` | Ellipsoid optimisation (gradient-based) |
| `cvxpy` | LP/QP solving for VIR and monotonicity |
| `matplotlib` | Shape function plots |
| `pickle` | Model serialisation |
| `GAMsRashomonSet` | Core Rashomon Set framework |

---

## Citation

If you use this work, please cite the original GAMs Rashomon Set paper:

```bibtex
@misc{gams-rashomon-set,
  author = {Chudi Zhong et al.},
  title  = {GAMs Rashomon Set},
  year   = {2023},
  url    = {https://github.com/chudizhong/GAMsRashomonSet}
}
```

---

## License

This project is for research and educational purposes. See the [GAMsRashomonSet](https://github.com/chudizhong/GAMsRashomonSet) repository for the core library license.
