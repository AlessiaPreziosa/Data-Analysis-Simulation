# Predicting Water Quality Index (WQI)

## Overview

This project aims to **predict the Water Quality Index (WQI)** using **Machine Learning** techniques based on water quality data collected across **18 Indian states**.  
Water quality has direct implications on public health, ecosystems, and socio-economic development. Predicting WQI helps policymakers and environmental agencies to detect contamination early and implement preventive measures.

---

## Objectives

1. **Understanding water quality dynamics**  
   Analyze relationships between environmental and water quality parameters to identify trends and correlations.

2. **Predictive modelling**  
   Develop machine learning models to estimate WQI based on environmental data, contributing to sustainable water management.

---

## Dataset

- **Source:** Government agency responsible for environmental monitoring in India  
- **Size:** 534 rows × 11 columns (48 KB)  
- **Features:**
  - `STATION CODE` – Unique identifier  
  - `LOCATIONS` – Monitoring station name  
  - `STATE` – Indian state  
  - `TEMP` – Water temperature (°C)  
  - `DO` – Dissolved Oxygen (mg/L)  
  - `pH` – pH level  
  - `CONDUCTIVITY` – Electrical conductivity (μS/cm)  
  - `BOD` – Biochemical Oxygen Demand (mg/L)  
  - `NITRATE_N_NITRITE_N` – Nitrate/Nitrite concentration (mg/L)  
  - `FECAL_COLIFORM` – Fecal coliform (CFU/mL)  
  - `TOTAL_COLIFORM` – Total coliform (CFU/mL)

### Label computation – Water Quality Index (WQI)

WQI = Σ(Qn × Wn), where:  
| Parameter | Weight (Wn) |
|------------|-------------|
| DO | 0.281 |
| pH | 0.165 |
| CONDUCTIVITY | 0.009 |
| BOD | 0.234 |
| NITRATE_N_NITRITE_N | 0.028 |
| FECAL_COLIFORM | 0.281 |

---

## Data Preparation

- Dataset split: **70% train / 30% test**
- **Missing values:**  
  - Most missing: `FECAL_COLIFORM` (16.58%) and `CONDUCTIVITY` (5.03%)  
  - **Imputation:** Mean substitution using Spark’s `Imputer`  
  - **Missing indicators:** Added dummy columns (1 = missing, 0 = observed)

- **Feature excluded:** `TOTAL_COLIFORM` (high correlation with `FECAL_COLIFORM`, r = 0.915)

---

## Exploratory Data Analysis

- **Skewness:** High in `CONDUCTIVITY`, `BOD`, `NITRATE_N_NITRITE_N`, and `FECAL_COLIFORM`  
- **Correlations:**
  - `TEMP` ↘️ `DO` (r = -0.19)
  - `DO` ↘️ `BOD` (r = -0.5)
  - `DO` ↗️ `WQI` (r = 0.61)
  - `BOD` ↘️ `WQI` (r = -0.56)

- **WQI Classes:**
  | Class | Range | Description |
  |--------|--------|-------------|
  | 0 | 0–59 | Poor |
  | 1 | 60–79 | Bad |
  | 2 | 80–84 | Medium |
  | 3 | 85–89 | Good |
  | 4 | 90–100 | Excellent |

---

## Modelling

### Feature Engineering
- **VectorAssembler:** Merges numeric features  
- **StandardScaler:** Standardizes features to Z-scores  
- **Pipeline:** `[assembler → scaler → final_assembler → model]`

### Algorithms
1. **Linear Regression (for continuous WQI)**
2. **Random Forest Classifier (for discrete WQI classes)**

### Hyperparameter Tuning
**Linear Regression:**
- `regParam`: [0.001, 0.01, 0.1, 0.2, 0.5]  
- `elasticNetParam`: [0.0, 0.5, 1.0]

**Random Forest:**
- `maxDepth`: [10, 15, 20]  
- `numTrees`: [15, 20, 25, 30]

**Validation:** 5-fold cross-validation using `CrossValidator`

---

## Results

| Model | Metric | Score |
|--------|---------|--------|
| **Linear Regression** | R² | 0.55 |
|  | RMSE | 9.21 |
|  | MSE | 84.80 |
|  | MAE | 6.63 |
| **Random Forest Classifier** | F1 | 0.8966 |
|  | Accuracy | 0.8970 |
|  | Weighted Precision | 0.8998 |
|  | Weighted Recall | 0.8971 |

**Random Forest performed best**, likely due to WQI discretization improving classification robustness.

---

## Key Insights

- `DO` and `BOD` are the most influential parameters for WQI.  
- `Temperature` has little to no effect on WQI.  
- Higher WQI is associated with:
  - Higher `DO`
  - Neutral `pH`
  - Lower `BOD`, `NITRATE_N_NITRITE_N`, and `FECAL_COLIFORM`.

---

## Conclusions

- Machine Learning, especially ensemble models like Random Forests, can effectively predict water quality categories.  
- The methodology provides a foundation for **data-driven environmental monitoring** and **early contamination detection**.  
- Future improvements may include:
  - Expanding dataset size  
  - Testing other ensemble or neural models  
  - Incorporating temporal or spatial dependencies.

---

## Technologies Used

- **Apache Spark (PySpark)**
- **Python**
- **Machine Learning Pipeline API**
- **Matplotlib / Pandas** (for visualization and data analysis)
