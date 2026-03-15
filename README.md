# 🏠 California Housing Price Prediction — End-to-End ML Project

A complete end-to-end machine learning pipeline built on the **California Housing dataset**, covering everything from raw data ingestion to model deployment. This project was developed as part of the B.Tech Computer Science curriculum and demonstrates practical, industry-aligned ML engineering practices.

---

## 📌 Project Overview

The goal is to **predict median house values** for California districts using socioeconomic and geographic features. The project follows the full ML lifecycle: data acquisition → EDA → preprocessing → model training → hyperparameter tuning → evaluation → model export.

---

## 🗂️ Dataset

- **Source:** [Aurélien Géron's dataset](https://github.com/ageron/data/raw/main/housing.tgz)
- **Size:** ~20,640 districts across California
- **Target:** `median_house_value`
- **Features:**

| Feature | Description |
|---|---|
| `longitude`, `latitude` | Geographic coordinates |
| `housing_median_age` | Median age of houses in district |
| `total_rooms`, `total_bedrooms` | Room counts per district |
| `population`, `households` | Population data |
| `median_income` | Median income (in $10k units) |
| `ocean_proximity` | Categorical — distance to ocean |

---

## 🔧 Tech Stack

- **Language:** Python 3
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn, SciPy, Joblib

---

## 🔄 Pipeline Steps

### 1. Data Loading
Fetches and extracts the `.tgz` dataset automatically if not already present locally.

### 2. Exploratory Data Analysis (EDA)
- Histogram distributions of all features
- Geographic scatter plots (color-coded by house value, sized by population)
- Correlation matrix analysis
- Scatter matrix for top correlated features

### 3. Stratified Train/Test Split
Income categories are used to perform **stratified sampling**, ensuring the test set is representative of the full income distribution.

### 4. Feature Engineering
Three derived features added:
- `rooms_per_house = total_rooms / households`
- `bedrooms_per_room = total_bedrooms / total_rooms`
- `population_per_household = population / households`

### 5. Preprocessing Pipeline
Separate pipelines for numerical and categorical features, composed with `ColumnTransformer`:

- **Numerical:** `SimpleImputer(median)` → `StandardScaler`
- **Categorical:** `SimpleImputer(most_frequent)` → `OneHotEncoder`

### 6. Model Training & Evaluation

| Model | Cross-Val RMSE |
|---|---|
| Linear Regression | ~$68,000 |
| Decision Tree | ~$71,000 (overfitting) |
| Random Forest | ~$47,000 ✅ Best |

### 7. Hyperparameter Tuning
- **Grid Search** over `n_estimators` and `max_features`
- **Randomized Search** with `scipy.stats.randint` distributions (10 iterations, 5-fold CV)

### 8. Final Evaluation
Best model evaluated on the held-out test set with a **95% confidence interval** on RMSE using `scipy.stats.t.interval`.

### 9. Model Export
Final model persisted with **Joblib**:
```python
joblib.dump(final_model, "my_california_housing_model.pkl")
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn scipy joblib
```

### Run
Open the notebook in Jupyter or Google Colab:
```bash
jupyter notebook 01_End_End_machine_learning_project.ipynb
```

The dataset is auto-downloaded on first run — no manual setup needed.

---

## 📁 Project Structure

```
.
├── 01_End_End_machine_learning_project.ipynb   # Main notebook
├── datasets/
│   └── housing/
│       └── housing.csv                          # Auto-downloaded
└── my_california_housing_model.pkl              # Exported model (after run)
```

---

## 📈 Key Results

- **Best Model:** Random Forest Regressor (tuned via Randomized Search)
- **Final Test RMSE:** ~$47,000
- **Top Predictors:** `median_income`, `rooms_per_house`, `bedrooms_per_room`, geographic coordinates

---

## 💡 Concepts Demonstrated

- Stratified train/test splitting
- Scikit-learn `Pipeline` and `ColumnTransformer` design patterns
- `OrdinalEncoder` vs `OneHotEncoder` for categorical data
- `TransformedTargetRegressor` for target scaling
- `GridSearchCV` and `RandomizedSearchCV`
- Feature importance analysis from Random Forest
- Confidence interval estimation on test RMSE

---

## 📚 Reference

Based on Chapter 2 of **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron (O'Reilly).
