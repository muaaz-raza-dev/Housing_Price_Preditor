# 🏠 California Housing Price Predictor

A Machine Learning project to predict median house values in California districts using census data.  
This project is based on **Chapter 2** of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Future Work](#future-work)
8. [Acknowledgements](#acknowledgements)

---

## 📌 Overview
This project aims to:
- Explore and preprocess the California housing dataset.
- Train and evaluate regression models.
- Compare model performance and discuss improvements.

---

## 📊 Dataset
- **Source**: California Housing dataset (available via `sklearn.datasets.fetch_california_housing`)
- **Features**:
  - `longitude`, `latitude`
  - `housing_median_age`
  - `total_rooms`, `total_bedrooms`
  - `population`, `households`
  - `median_income`
  - `ocean_proximity` (excluded)
- **Target**:
  - `median_house_value`

---

## ⚙️ Installation
```bash
git clone https://github.com/muaaz-raza-dev/housing-price-predictor.git
cd housing-price-predictor
pip install -r requirements.txt
```

## Acknowledgements

- A big thanks to Aurélien Géron for writing Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
- This book is an outstanding resource for learning machine learning concepts and applying them through practical, real-world projects.


