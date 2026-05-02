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

🚀 **Streamlit App:** *https://prathameshk1995-strategic-aid-allocatio-appstreamlit-app-wfangh.streamlit.app/*

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
  * Low Life Expectancy

---

### 3. Clustering Techniques

* **K-Means (Primary Model)**
* Hierarchical Clustering (validation)

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

## 📊 Tableau Dashboard

This interactive dashboard highlights countries based on socio-economic risk factors such as child mortality, inflation, fertility, life expectancy and income.

It enables stakeholders to:
- Identify countries requiring immediate financial aid
- Analyze key drivers of risk
- Compare socio-economic indicators across clusters

👉 [View Interactive Dashboard](https://public.tableau.com/app/profile/prathamesh.kamble/viz/Strategic_AID_Allocation_dashboard/StrategicAIDAllocation)

---

## 📝 Medium Blog

📖 Deep dive into the project:

👉 [Read the Full Blog on Medium](https://medium.com/@prathameshkamble99/how-i-built-a-data-driven-system-to-identify-countries-that-need-financial-aid-d603a478fbce)

> Learn how machine learning can help identify countries that need financial aid using real-world data.

---

## 📂 Project Structure

```
Strategic Aid Allocation/
│
├── app/                # Streamlit application
├── model/              # Serialized model artifacts
├── notebooks/          # EDA and modeling
├── data/               # Dataset, Tableau dashboard
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

Data Scientist
Email: prathameshkamble99@gmail.com
Mumbai, India

---
