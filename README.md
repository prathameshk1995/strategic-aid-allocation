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

* Handled skewness using **log transformation**
* Avoided aggressive outlier removal to preserve critical country signals
* Applied **MinMax Scaling** for distance-based clustering

---

### 2. Feature Engineering

* Created additional indicators such as:

  * High Child Mortality
  * Low Income
  * High Inflation
  * Trade Ratio
  * Health Efficiency

---

### 3. Clustering Techniques Explored

| Algorithm                   | Purpose                                 |
| --------------------------- | --------------------------------------- |
| **K-Means**                 | Primary model for deployment            |
| **Hierarchical Clustering** | Validation & structure understanding    |
| **DBSCAN**                  | Outlier-aware clustering (experimental) |

---

### 4. Model Selection

* Evaluated using **Silhouette Score**
* Optimal clusters chosen: **k = 3**
* Achieved balanced performance with strong interpretability

---

## 📌 Cluster Interpretation

### 🔴 High Risk (Needs Immediate Aid)

* High child mortality
* Low income & GDP
* Low life expectancy
* High fertility rate

👉 Represents countries in **critical condition**

---

### 🟡 Medium Risk

* Moderate development
* High inflation (economic instability)

👉 Requires **targeted economic support**

---

### 🟢 Low Risk

* High income & GDP
* Strong healthcare
* Low mortality

👉 **Lowest priority** for aid allocation

---

## 🚀 Deployment

### 🔹 Flask API

A REST API was built to serve predictions in real-time.

#### Endpoint:

```bash
POST /predict
```

#### Input:

```json
{
  "features": [child_mort, exports, health, imports, income, inflation, life_expec, total_fer, gdpp]
}
```

#### Output:

```json
{
  "cluster": 1,
  "category": "High Risk - Needs Aid",
  "reasons": [
    "High child mortality",
    "Low income",
    "Low life expectancy"
  ]
}
```

---

## 🧩 Explainability

Since K-Means is unsupervised, interpretability was achieved using:

* **Data-driven thresholds (quantiles)**
* Mapping clusters to **business-friendly categories**
* Rule-based reasoning for transparency

---

## 🛠️ Tech Stack

* **Python** (Pandas, NumPy, Scikit-learn)
* **Data Visualization** (Matplotlib, Seaborn)
* **Model Deployment** (Flask)
* **Version Control** (Git, GitHub)

---

## 📈 Key Insights

* Strong inverse relationship between **income and child mortality**
* Health expenditure positively impacts **life expectancy**
* High inflation correlates with **economic instability**

---

## 📂 Project Structure

```
Strategic Aid Allocation/
│
├── app/                # Flask API
├── model/              # Saved model, scaler, features, thresholds
├── notebooks/          # EDA & model development
├── data/               # Dataset
├── docs/               # Supporting doc
├── requirements.txt
└── README.md
```

---

## 🧠 Learnings

* Importance of **feature scaling in clustering**
* Trade-off between **model performance vs interpretability**
* Handling **unsupervised explainability challenges**
* Building **end-to-end ML pipelines (EDA → Model → API)**

---

## 🚀 Future Improvements

* Integrate **Streamlit UI** for interactive exploration
* Add **real-time data ingestion**
* Experiment with **GMM for probabilistic clustering**
* Deploy on **cloud (AWS / Render / GCP)**

---

## 🤝 Contribution

Open for suggestions and improvements. Feel free to fork or raise issues.

---

## 📬 Contact

**Prathamesh Kamble**
Data Analyst | Aspiring Data Scientist
Mumbai, India

---
