# 🧠 Behavioral Informatics & Risk Prediction Pipeline

> 📱 **Developer Note:** This entire project—from the initial exploratory data analysis and data engineering to the pipeline architecture and deployment—was built completely on a **mobile device**. When I hit roadblocks or got stuck with specific visual styling scripts and framework quirks, I leveraged AI as an agile peer collaborator to debug and refine the codebase.

---

## 🚀 Live Interactive Application
The final production model has been fully containerized and deployed as a live, high-vibrancy analytical dashboard. 
🔗 **Access the Live Web App here:** [Behavioral Informatics Engine Dashboard](https://behavioral-informatics-pipeline-inline.streamlit.app/)

---

## 📌 Project Overview
This repository contains an end-to-end Machine Learning lifecyle engineered to evaluate, profile, and predict psychological health indicators based on modern lifestyle, screen habit, and emotional metrics. 

Instead of relying purely on a basic classification structure, this project implements a hybrid approach:
1. **Unsupervised Clustering ($K$-Means):** Automatically derives hidden behavioral user archetypes from high-dimensional screen and sleep habits.
2. **Feature Engineering:** Injecting the unsupervised user segments directly back into the core dataset as high-value categorical variables.
3. **Supervised Tournament:** Running a leak-free execution framework between three state-of-the-art gradient boosting frameworks (**XGBoost**, **LightGBM**, and **CatBoost**) optimized to handle severe minority class imbalances.

---

## 🛠️ Pipeline Architecture & Phase Breakdown

### 📥 Phase 1: Data Ingestion & Structural Inspection
* Comprehensive schema auditing using `.info()` and `.describe()` to isolate continuous decimals from categorical string data types.
* Integrity validation using chained `.isnull().sum()` operations, confirming zero missing entries across all 1,200 observation blocks.
* Target label inspection exposing a severe class imbalance problem: **1,169 healthy samples (Class 0)** versus a rare subset of **31 positive cases (Class 1)**.

### 📉 Phase 2: Unsupervised Optimization (The Elbow Method)
* Isolated core behavioral columns (`daily_social_media_hours`, `screen_time_before_sleep`, and `sleep_hours`) and normalized them using `StandardScaler` to ensure scale-neutral geometric coordinates.
* Run an iterative loop across multiple cluster ranges, tracking the reduction in mathematical `inertia_` (Within-Cluster Sum of Squares).
* Plotted an **Elbow Curve** to locate the exact point of diminishing returns, identifying **3 clusters** as the optimal behavioral separation boundary.

### 🏷️ Phase 3: High-Dimensional Personas & Feature Engineering
* Initialized the definitive $K$-Means engine utilizing `k-means++` initialization parameters for optimized, reproducible convergence.
* Stamped group flags across rows via `.fit_predict()`, saving them as a new feature column: `User_Segment`.
* Explicitly cast these labels into categorical text strings (`.astype(str)`) to prevent tree classifiers from interpreting them as sequential ordinal values.
* Rendered an interactive **Plotly 3D Scatter Map** to visually profile the 3 distinct human lifestyle archetypes discovered by the model:
  * 🟦 **The Balanced User Persona:** Minimal social media usage, lowest pre-bed screen exposure, and highly stable sleep hours.
  * 🟥 **The Extreme Screen User Persona:** Excessive screen exposure right up until bedtime, resulting in severely restricted sleep windows.
  * 🟨 **The Standard Mixed Baseline:** Moderate lifestyle variables sitting securely between both ends of the behavioral spectrum.

### 🛡️ Phase 4: Leak-Free Supervised Framework Tournament
* Divided data into an 80/20 train-test configuration using `stratify=y` to lock exact class proportions into both splits.
* Enforced a strict anti-leakage barrier: **Fitted the scaler and clusterer strictly on the training set**, then passed test observations through matching `.transform()` and `.predict()` rules.
* Constructed a unified `ColumnTransformer` processing factory line: continuous variables were scaled, and text objects were vectorized via `OneHotEncoder(handle_unknown='ignore')`.
* Calculated the precise imbalance inversion ratio ($1,169 \div 31 \approx 37.7$) and injected it into the tree frameworks as a hard penalty multiplier (`scale_pos_weight=ratio`).
* Wrapped the preprocessor and classifiers inside scikit-learn `Pipelines`, tracking test performance using the class-specific F1-Score.

---

## 📊 Evaluation & Model Discoveries

### 🏆 The Tournament Winner: LightGBM
Following a competitive head-to-head battle, **LightGBM** emerged as the definitive champion. Evaluated using a comprehensive **Confusion Matrix Heatmap**, the pipeline delivered absolute classification precision and sensitivity boundaries ($1.0000$ F1-Score):
* **True Positives Caught:** 6 out of 6 rare class cases safely identified ($100\%$ Recall).
* **False Alarms Triggered:** 0 out of 234 healthy profiles misclassified ($100\%$ Precision).

### 🧠 Feature Importance Insights
By extracting the internal tree logic via `.feature_importances_` from the winning LightGBM pipeline, the structural weights revealed the following priority hierarchies:
1. **The Core Drivers:** `sleep_hours` and `daily_social_media_hours` emerged as the single most informative lifestyle features by a massive margin.
2. **The Clinical Flags:** Self-reported `stress_level` and `anxiety_level` provided the primary secondary splits.
3. **The Background Noise:** Demographics (`gender`), specific social app categories (`TikTok` vs `Instagram`), and academic marks were ranked right at the bottom, proving that **how long you look at a screen and how well you sleep matters vastly more than which app you open**.

---

## ⚠️ Critical Note on Synthetic Data Lifecycle

While achieving a perfect $1.0000$ F1-Score looks incredible on a dashboard scoreboard, a flawless classification profile on real human metrics typically points to a massive bug or data leakage. To stress-test this model, an aggressive column-stripping test was performed during data analysis—dropping high-importance pillars like sleep metrics and clinical scale inputs entirely. 

Despite missing its top drivers, the boosting classifiers instantly reverse-engineered alternative mathematical correlations hidden across background columns to maintain perfect scores. 

**Conclusion:** The source data is an AI-generated, synthetic dataset containing deterministic, closed-loop relationships rather than volatile, noisy real-world data patterns. Thus, while the $100\%$ metric bounds are a byproduct of artificial correlation loops, the engineering value of this project stands as a fully operational, industry-standard **MLOps architectural template** for safe data preprocessing and live deployment pipelines.

---

## 💻 Tech Stack & Dependencies
* **Core Languages & Libraries:** Python, Pandas, NumPy
* **Machine Learning Engine:** Scikit-Learn (v1.6.1), XGBoost, LightGBM, CatBoost, Joblib
* **Data Visualization:** Plotly Express, Plotly Figure Factory
* **Production Deployment:** Streamlit Web Framework, Streamlit Cloud Platform Engine



## 📞 Contact Information:-
* **Email:-**[englandengland271@gmail.com]
* **Linkedin:-**[https://www.linkedin.com/in/mohammed-nafay-ali-16519138a?utm_source=share_via&utm_content=profile&utm_medium=member_android]
* **GitHub:-**[https://github.com/M-Nafay-Ali]

