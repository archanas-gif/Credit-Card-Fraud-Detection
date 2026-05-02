Credit Card Fraud Detection System

## 📌 Overview

This project is an end-to-end Machine Learning system designed to detect fraudulent credit card transactions.
It uses a trained classification model along with an interactive Streamlit web application to predict whether a transaction is **fraudulent or legitimate** in real time.

---

## 🚨 Problem Statement

Credit card fraud is a critical issue in the financial industry.
Fraudulent transactions are **very rare compared to normal transactions**, making this an **imbalanced classification problem**.

---

## 🎯 Objective

* Detect fraudulent transactions accurately
* Minimize false negatives (missed fraud cases)
* Build a user-friendly ML-powered interface
* Demonstrate a real-world machine learning pipeline

---

## 🧠 Solution Approach

### 🔹 Data Processing

* Loaded transaction dataset
* Performed preprocessing and scaling
* Handled class imbalance using `class_weight="balanced"`

### 🔹 Model

* Random Forest Classifier
* Optimized for performance and speed

### 🔹 Features

* PCA-transformed features (V1–V28)
* Transaction Amount and Time

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib

---

## 🚀 Key Features

* Fraud prediction with confidence score
* Auto-load real transaction samples (Normal / Fraud / Random)
* Actual vs Predicted comparison
* Feature importance visualization
* Class distribution graph
* Interactive input interface
* Pre-trained model loading (no retraining required)

---

## 📂 Project Structure

Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── outputs/
│   └── confusion_matrix.png
│
├── images/
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md

---

## ▶️ How to Run

1. Clone the repository
   git clone https://github.com/archanas-gif/Credit-Card-Fraud-Detection
   cd Credit-Card-Fraud-Detection

2. Create virtual environment
   python -m venv venv

3. Activate environment

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

4. Install dependencies
   pip install -r requirements.txt

5. Train model (run once)
   python train_model.py

6. Run application
   streamlit run app.py

## 📊 Evaluation Metrics

* Precision
* Recall
* Confusion Matrix

Focus is on **high recall** to minimize missed fraud cases.

---

## 💼 Real-World Relevance

Fraud detection systems are widely used in:

* Banking
* Fintech
* Payment systems

This project simulates a **real-world fraud detection pipeline**.

---

## 🧠 Key Learnings

* Handling imbalanced datasets
* Model evaluation (precision vs recall trade-off)
* Feature importance interpretation
* Building ML-powered web applications

---

## 🔮 Future Improvements

* Real-time streaming integration
* Advanced models (XGBoost, LightGBM)
* Cloud deployment
* Dashboard analytics
