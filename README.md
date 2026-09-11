# 100 Days of Machine Learning

A structured repository documenting practical implementations, theoretical notes, and end-to-end workflows across the machine learning lifecycle. This repository covers data acquisition, exploratory data analysis (EDA), advanced feature engineering, dimensionality reduction, and foundational supervised learning algorithms.

---

## Overview

This repository serves as a comprehensive reference for core machine learning concepts implemented in Python using industry-standard libraries such as NumPy, Pandas, Scikit-Learn, Matplotlib, and Seaborn. Each module includes hands-on Jupyter Notebooks, datasets, and step-by-step mathematical breakdowns.

Key areas covered:
- Data extraction and ingestion across diverse formats (CSV, JSON, SQL, REST APIs, Web Scraping)
- Exploratory data analysis (Univariate, Bivariate, Multivariate, Automated Profiling)
- Feature engineering (Scaling, Categorical Encoding, Scikit-Learn Pipelines, ColumnTransformers)
- Missing value imputation (Univariate, Multivariate, MICE, KNN)
- Outlier detection and treatment (Z-score, IQR, Percentile/Winsorization)
- Dimensionality reduction (Principal Component Analysis from scratch and Scikit-Learn)
- Supervised learning foundations (Simple, Multiple Linear Regression, Assumptions, Evaluation Metrics)

---

## Curriculum and Repository Structure

### Phase 1: Data Ingestion and Acquisition

| Day | Module / Topic | Directory | Key Concepts & Libraries |
| :--- | :--- | :--- | :--- |
| Day 15 | Working with CSV Files | [day15 - working with csv files](day15%20-%20working%20with%20csv%20files/) | `read_csv` parameters, encodings, chunking, parsing large files |
| Day 16 | Working with JSON and SQL | [day16 - working-with-json-and-sql](day16-working-with-json-and-sql/) | JSON normalization, relational database queries with SQLite / SQL |
| Day 17 | API to DataFrame | [day17-api-to-dataframe](day17-api-to-dataframe/) | Consuming REST APIs, pagination, handling nested JSON payloads |
| Day 18 | Web Scraping to DataFrame | [day18-pandas-dataframe-using-web-scraping](day18-pandas-dataframe-using-web-scraping/) | HTML parsing, BeautifulSoup, tabular extraction |

---

### Phase 2: Exploratory Data Analysis (EDA)

| Day | Module / Topic | Directory | Key Concepts & Libraries |
| :--- | :--- | :--- | :--- |
| Day 19 | Descriptive Statistics | [day19-understanding-your-data-descriptive-stats](day19-understanding-your-data-descriptive-stats/) | Central tendency, dispersion, skewness, kurtosis, shape inspection |
| Day 20 | Univariate Analysis | [day20-univariate-analysis](day20-univariate-analysis/) | Histograms, KDE plots, box plots, bar charts, frequency distributions |
| Day 21 | Bivariate and Multivariate Analysis | [day21-bivariate-analysis](day21-bivariate-analysis/) | Scatter plots, correlation heatmaps, pairplots, cross-tabulations |
| Day 22 | Automated EDA | [day22-pandas-profiling](day22-pandas-profiling/) | Rapid data profiling, distribution reports, correlation matrices |

---

### Phase 3: Feature Engineering and Data Preprocessing

#### 1. Feature Scaling and Transformation
| Day | Module / Topic | Directory | Key Concepts & Libraries |
| :--- | :--- | :--- | :--- |
| Day 24 | Standardization | [day24-standardization](day24-standardization/) | Z-score normalization, `StandardScaler`, effect on distance-based models |
| Day 25 | Normalization | [day25-normalization](day25-normalization/) | Min-Max scaling, robust scaling, scale bounds |
| Day 30 | Function Transformer | [day30-function-transformer](day30-function-transformer/) | Log transform, reciprocal, square root transforms for normality |
| Day 31 | Power Transformer | [day31-power-transformer](day31-power-transformer/) | Box-Cox and Yeo-Johnson transformations |
| Day 32 | Discretization and Binning | [day32-binning-and-binarization](day32-binning-and-binarization/) | Equal width, equal frequency binning, `KBinsDiscretizer`, `Binarizer` |

#### 2. Categorical Encoding and Data Types
| Day | Module / Topic | Directory | Key Concepts & Libraries |
| :--- | :--- | :--- | :--- |
| Day 26 | Ordinal Encoding | [day26-ordinal-encoding](day26-ordinal-encoding/) | `OrdinalEncoder`, `LabelEncoder`, ordered categorical attributes |
| Day 27 | One-Hot Encoding | [day27-one-hot-encoding](day27-one-hot-encoding/) | Dummy variable trap, handling rare categories, `OneHotEncoder` |
| Day 33 | Handling Mixed Variables | [day33-handling mixed variables](day33-handling%20mixed%20variables/) | Splitting alphanumeric and composite feature columns |
| Day 34 | Handling Date and Time Variables | [day34-handling-date-and-time-variables](day34-handling-date-and-time-variables/) | Timestamp extraction, cyclical features, elapsed time calculations |

#### 3. Pipelines and Orchestration
| Day | Module / Topic | Directory | Key Concepts & Libraries |
| :--- | :--- | :--- | :--- |
| Day 28 | ColumnTransformer | [day28-column-transformer](day28-column-transformer/) | Heterogeneous feature preprocessing via `sklearn.compose.ColumnTransformer` |
| Day 29 | Scikit-Learn Pipelines | [day29-sklearn-pipelines](day29-sklearn-pipelines/) | End-to-end transformation and estimation pipelines, preventing data leakage |

#### 4. Missing Value Imputation
| Day | Module / Topic | Directory | Key Concepts & Libraries |
| :--- | :--- | :--- | :--- |
| Day 35 | Complete Case Analysis | [day35-complete-case-analysis](day35-complete-case-analysis/) | MCAR assumptions, listwise deletion, data loss considerations |
| Day 36 | Numerical Data Imputation | [day36-imputing-numerical-data](day36-imputing-numerical-data/) | Mean, median, arbitrary value, and random sample imputation |
| Day 37 | Categorical Data Imputation | [day37-handling-missing-categorical-data](day37-handling-missing-categorical-data/) | Mode imputation, missing category indicator creation |
| Day 38 | Missing Indicator | [day38-missing-indicator](day38-missing-indicator/) | Binary missingness flags via `MissingIndicator` |
| Day 39 | KNN Imputer | [day39-knn-imputer](day39-knn-imputer/) | Distance-weighted missing value estimation via `KNNImputer` |
| Day 40 | Iterative Imputer | [day40-iterative-imputer](day40-iterative-imputer/) | Multivariate Imputation by Chained Equations (MICE) |

#### 5. Outlier Detection and Handling
| Day | Module / Topic | Directory | Key Concepts & Libraries |
| :--- | :--- | :--- | :--- |
| Day 42 | Z-Score Outlier Removal | [day42-outlier-detection-and-removal](day42-outlier-detection-and-removal/) | Normal distribution boundary identification, trimming, and capping |
| Day 43 | IQR Outlier Removal | [day43-outlier-removal-using-IQR](day43-outlier-removal-using-IQR/) | Interquartile range thresholds for skewed feature distributions |
| Day 44 | Percentile Method | [day44-outlier-detection-using-percentile-method](day44-outlier-detection-using-percentile-method/) | Quantile trimming and Winsorization |
| Day 45 | Feature Construction and Splitting | [day45-feature-construction-and-feature-splitting](day45-feature-construction-and-feature-splitting/) | Domain-specific feature engineering, interaction terms, column parsing |

---

### Phase 4: Dimensionality Reduction

| Day | Module / Topic | Directory | Key Concepts & Libraries |
| :--- | :--- | :--- | :--- |
| Day 47 | Principal Component Analysis (PCA) | [day47-pca](day47-pca/) | Eigenvalues, eigenvectors, covariance matrix, variance explained, Scikit-Learn PCA |

---

### Phase 5: Supervised Learning - Regression

| Day | Module / Topic | Directory | Key Concepts & Libraries |
| :--- | :--- | :--- | :--- |
| Day 48 | Simple Linear Regression | [day48-simple-linear-regression](day48-simple-linear-regression/) | Closed-form OLS solution, slope and intercept derivation, baseline models |
| Day 49 | Regression Metrics | [day49-regression-metrics](day49-regression-metrics/) | MAE, MSE, RMSE, R-squared, Adjusted R-squared |
| Day 50 | Multiple Linear Regression | [day50-multiple-regression](day50-multiple-regression/) | Matrix formulation of OLS, multi-variable modeling, parameter interpretation |
| Module | Linear Regression Assumptions | [linear-regression-assumption](linear-regression-assumption/) | Linearity, homoscedasticity, multicollinearity (VIF), normality of residuals |

---

## Supplementary Resources

- **Lecture and Theoretical Notes**: [notes.pdf](notes.pdf) contains comprehensive reference notes covering mathematical derivations, workflows, and algorithmic intuition.

---

## Getting Started

### Prerequisites

Ensure Python 3.9 or higher is installed on your system.

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/RaunakSachdeva2004/ML-100-Days.git
   cd ML-100-Days
   ```

2. Create and activate a virtual environment:
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. Install the recommended scientific packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter requests beautifulsoup4
   ```

4. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```

---

## Technical Stack

- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Data Visualization**: Matplotlib, Seaborn
- **Data Ingestion**: Requests, BeautifulSoup4, SQLite
- **Environment**: Jupyter Notebook

---

## License

This project is licensed under the MIT License. See the repository license file for full details.