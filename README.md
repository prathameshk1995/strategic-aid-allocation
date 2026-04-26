# 🌍 Strategic Aid Allocation using Unsupervised Learning

## 📌 Overview

HELP International, a global humanitarian NGO, aims to allocate a limited fund of **$10 million** effectively to countries in dire need.

This project leverages **unsupervised machine learning** to segment countries based on socio-economic and health indicators, enabling **data-driven and impactful decision-making**.

---

## 🎯 Objective

* Identify countries that require **immediate financial aid**
* Categorize countries into **High, Medium, and Low Risk groups**
* Provide **explainable insights** to support strategic allocation

---

## 🌐 Live Application

🚀 **Streamlit App:** *(Add your deployed link here)*

Users can input country-level indicators and get:

* Risk category
* Cluster assignment
* Key reasons driving the prediction

---

## 📊 Dataset Description

The dataset contains country-level indicators:

* **Child Mortality** – Deaths under age 5 per 1000 births
* **Exports / Imports** – % of GDP per capita
* **Health** – Health expenditure (% of GDP)
* **Income** – Net income per person
* **Inflation** – Annual GDP growth rate
* **Life Expectancy** – Average lifespan
* **Total Fertility** – Births per woman
* **GDP per Capita (gdpp)**

---

## 🧠 Approach

### 1. Data Preprocessing

* Applied **log transformation** to handle skewed distributions
* Used **winsorization (capping)** for extreme outliers
* Performed **MinMax Scaling** for clustering

---

### 2. Feature Engineering

* Created meaningful indicators such as:

  * High Child Mortality
  * Low Income
  * High Inflation
  * Trade Ratio
  * Health Efficiency

---

### 3. Clustering Techniques

* **K-Means (Primary Model)**
* Hierarchical Clustering (validation)
* DBSCAN (outlier-aware clustering)

---

### 4. Model Selection

* Evaluated using **Silhouette Score**
* Optimal clusters: **k = 3**
* Balanced interpretability with performance

---

## 📌 Cluster Interpretation

### 🔴 High Risk (Needs Immediate Aid)

* High child mortality
* Low income & GDP
* Low life expectancy
* High fertility

---

### 🟡 Medium Risk

* Moderate development
* High inflation (economic instability)

---

### 🟢 Low Risk

* Strong economy
* High life expectancy
* Low mortality

---

## 🚀 Deployment Architecture

```text
User → Streamlit UI → Preprocessing → K-Means Model → Prediction + Reasons
```

* Model and preprocessing artifacts are serialized using **joblib**
* Real-time predictions are served through a **Streamlit interface**
* Explainability is provided using **data-driven thresholds**

---

## 🧩 Explainability

Since K-Means is unsupervised, interpretability is achieved through:

* **Quantile-based thresholds** derived from data
* Mapping clusters to business-friendly categories
* Rule-based reasoning for transparency

---

## 🛠️ Tech Stack

* **Python** (Pandas, NumPy, Scikit-learn)
* **Visualization** (Matplotlib, Seaborn)
* **Deployment** (Streamlit)
* **Model Serialization** (Joblib)
* **Version Control** (Git, GitHub)

---

## 📈 Key Insights

* Higher income is strongly associated with **lower child mortality**
* Increased health spending improves **life expectancy**
* High inflation signals **economic instability in mid-tier countries**

---

## 📂 Project Structure

```
Strategic Aid Allocation/
│
├── app/                # Streamlit application
├── model/              # Serialized model artifacts
├── notebooks/          # EDA and modeling
├── data/               # Dataset
├── requirements.txt
└── README.md
```

---

## 🧠 Learnings

* Importance of **feature scaling in clustering algorithms**
* Trade-off between **model performance and interpretability**
* Handling **unsupervised model explainability**
* Building an **end-to-end ML solution (EDA → Model → Deployment)**

---

## 📬 Contact

**Prathamesh Kamble**
Data Analyst | Aspiring Data Scientist
Mumbai, India

---
