# 🏦 Bank Customer Churn Prediction with an Artificial Neural Network

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Sequential-D00000?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-preprocessing%20%26%20metrics-F7931E?logo=scikitlearn&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-85.9%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

> A feed-forward neural network, built with TensorFlow/Keras, that predicts whether a bank customer is going to **leave (churn)**, so the bank can act *before* the customer walks away.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Model Architecture](#-model-architecture)
- [Training Setup](#-training-setup)
- [Results](#-results)
- [Key Takeaways](#-key-takeaways)
- [Future Improvements](#-future-improvements)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🎯 Problem Statement

Acquiring a new customer costs far more than keeping an existing one. Given a customer's profile and banking behaviour, this project answers one binary question:

> **Will this customer exit the bank?** (`Exited = 1` → churned, `Exited = 0` → stayed)

---

## 📊 Dataset

**File:** `Churn_Modelling.csv`: 10,000 customers, 14 columns.

| Feature | Type | Description |
|---|---|---|
| `CreditScore` | numeric | Customer's credit score |
| `Geography` | categorical | France / Germany / Spain |
| `Gender` | categorical | Male / Female |
| `Age` | numeric | Age in years |
| `Tenure` | numeric | Years as a customer |
| `Balance` | numeric | Account balance |
| `NumOfProducts` | numeric | Number of bank products held |
| `HasCrCard` | binary | Owns a credit card |
| `IsActiveMember` | binary | Active member flag |
| `EstimatedSalary` | numeric | Estimated annual salary |
| **`Exited`** | **target** | **1 = churned, 0 = retained** |

`RowNumber`, `CustomerId` and `Surname` are **dropped**. They're identifiers with no predictive meaning and would only add noise.

---

## 🔄 Workflow

```
Raw CSV ─► Feature/Target split ─► Encode categoricals ─► Train/Test split ─► Scale features
                                                                                    │
        Evaluate on test set ◄─ Train with Early Stopping ◄─ Build ANN ◄────────────┘
```

### 1. Feature selection
Columns 3–12 become the independent features `X`; column 13 (`Exited`) is the target `y`.

### 2. Categorical encoding
`Geography` and `Gender` are converted with `pd.get_dummies(..., drop_first=True)`. Dropping the first level avoids the **dummy variable trap** (perfect multicollinearity), leaving `Germany`, `Spain` and `Male` columns. Final feature count: **11**.

### 3. Train / test split
80 / 20 split with `random_state=42`, giving **8,000 training** and **2,000 test** rows.

### 4. Feature scaling
`StandardScaler` is **fit on the training set only** and then applied to the test set, so no information from the test data leaks into training. Scaling matters for ANNs because gradient descent converges much faster and more stably when features share a similar range.

---

## 🧠 Model Architecture

A fully connected `Sequential` network:

| Layer | Units | Activation | Notes |
|---|---|---|---|
| Input / Dense | 11 | ReLU | One unit per input feature |
| Hidden 1 | 7 | ReLU | Followed by **Dropout(0.3)** |
| Hidden 2 | 6 | ReLU | |
| Output | 1 | Sigmoid | Outputs churn probability |

- **Sigmoid** output gives a probability in [0, 1], which suits binary classification.
- **Dropout (30%)** randomly deactivates neurons during training to fight overfitting.

---

## ⚙️ Training Setup

| Setting | Value |
|---|---|
| Optimizer | Adam (`learning_rate=0.01`) |
| Loss | Binary cross-entropy |
| Metric | Accuracy |
| Batch size | 10 |
| Max epochs | 100 |
| Validation split | 33% of the training data |
| Early stopping | monitor `val_loss`, `patience=20`, `min_delta=0.0001` |

**Early stopping** removes the guesswork of picking an epoch count. Training halted automatically at **epoch 47** once validation loss stopped improving.

> 💡 **Why 536 iterations per epoch?** 8,000 training rows × 0.67 (after the 33% validation split) = 5,360 rows ÷ batch size 10 = **536 steps**.

---

## 📈 Results

The notebook plots accuracy and loss curves for training vs. validation, to check how well the model generalises.

**Test-set performance (threshold = 0.5):**

| Metric | Value |
|---|---|
| **Accuracy** | **85.9%** |
| Precision (churn) | 77.9% |
| Recall (churn) | 39.4% |

**Confusion matrix** (2,000 test customers):

|  | Predicted: Stay | Predicted: Churn |
|---|---|---|
| **Actual: Stay** | 1563 ✅ | 44 |
| **Actual: Churn** | 238 | 155 ✅ |

Final epoch: train accuracy ≈ 86.3%, validation accuracy ≈ 85.5%. The small gap suggests the model isn't badly overfitting.

---

## 🔍 Key Takeaways

- A small ANN with just two hidden layers reaches **~86% accuracy** on unseen customers.
- When the model flags a customer as likely to churn, it's right about **78%** of the time (high precision).
- **Accuracy alone flatters the model.** The dataset is imbalanced (roughly 80% of customers stay), so always predicting "stay" would already score about 80%. The model catches only **~39% of actual churners** (recall), which is the metric that matters most for a retention team.
- Dropout, standardisation and early stopping together keep training stable and the train/validation gap small.

---

## 🚀 Future Improvements

- **Handle class imbalance**: class weights, SMOTE, or tuning the decision threshold below 0.5 to raise recall.
- **Evaluate with better metrics**: F1-score, ROC-AUC and precision-recall curves.
- **Hyperparameter tuning**: layer sizes, dropout rate and learning rate via Keras Tuner or Optuna.
- **Restore best weights**: set `restore_best_weights=True` in early stopping.
- **Benchmark** against Logistic Regression, Random Forest and XGBoost.
- **Explainability** with SHAP to see which features drive churn.
- **Deployment**: wrap the trained model and scaler in a FastAPI/Streamlit app.

---

## 🛠 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/bank-customer-churn-prediction-ann.git
cd bank-customer-churn-prediction-ann

# 2. Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib jupyter

# 3. Launch the notebook
jupyter notebook
```

Open the notebook (see below) and run all cells. It also runs on Google Colab, where TensorFlow is pre-installed. Just upload `Churn_Modelling.csv`.

---

## 📁 Project Structure

```
.
├── Churn_Modelling.csv     # Dataset (10,000 customers)
├── simple_ann.ipynb        # End-to-end notebook: preprocessing, model, evaluation
├── LICENSE
└── README.md
```

---

## 📄 License

Distributed under the terms of the [LICENSE](LICENSE) file in this repository.
