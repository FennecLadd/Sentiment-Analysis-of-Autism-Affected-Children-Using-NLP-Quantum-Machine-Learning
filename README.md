Improved Sentiment Analysis of Autism-Affected Children

Using NLP + Quantum Machine Learning

This project explores **sentiment analysis of autism-related behavioral statements** using a hybrid pipeline combining **Natural Language Processing (NLP)** and **Quantum Machine Learning (QML)**.

The system filters autism-related text, performs linguistic preprocessing, generates TF-IDF features, reduces dimensionality using PCA, and then compares performance between:

* **Classical Machine Learning model (Linear SVM)**
* **Quantum Variational Classifier (VQC) using Qiskit**

The goal is to analyze **emotional sentiment patterns in autism-related communication** and evaluate whether **quantum ML can compete with classical models** on real-world NLP tasks.

---

# Project Pipeline

```
Dataset
   ↓
Autism-related text filtering
   ↓
Text preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Dimensionality Reduction (PCA)
   ↓
Model Training
   ├── Classical Model (Linear SVM)
   └── Quantum Model (Variational Quantum Classifier)
   ↓
Performance Comparison
```

---

# Features

* Autism-specific text filtering using domain keywords
* NLP preprocessing using **NLTK**
* TF-IDF vectorization
* Dimensionality reduction using **PCA**
* Classical baseline using **Linear Support Vector Machine**
* Quantum classifier using **Qiskit Variational Quantum Classifier**
* Performance evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score

---

# Dataset

Dataset file used:

```
AutismData.csv
```

Expected column:

```
statement
```

The dataset is filtered to keep only **autism-related statements** using keywords such as:

```
autism
autistic
asd
spectrum
sensory
meltdown
stimming
communication
social
```

---

# Text Preprocessing

Steps performed:

1. Lowercasing
2. Removing special characters
3. Tokenization
4. Stopword removal
5. Removing short/noisy samples

Example:

```
Original:
"My autistic child had a sensory meltdown today"

Cleaned:
"autistic child sensory meltdown today"
```

---

# Sentiment Label Generation

Sentiment labels are automatically generated using **VADER sentiment analysis**.

```
compound score ≥ 0  → Positive
compound score < 0  → Negative
```

This converts the dataset into a **binary classification task**.

---

# Feature Engineering

### TF-IDF Vectorization

```
max_features = 600
ngram_range = (1,1)
min_df = 3
sublinear_tf = True
```

### Dimensionality Reduction

Since quantum circuits require **low-dimensional inputs**, PCA is applied.

```
PCA components = 8
```

---

# Models Used

## 1. Classical Model

Algorithm:

```
Linear Support Vector Classifier (LinearSVC)
```

Parameters:

```
class_weight = balanced
```

Advantages:

* Strong baseline for NLP tasks
* Efficient on high-dimensional TF-IDF features

---

## 2. Quantum Model

Quantum algorithm used:

```
Variational Quantum Classifier (VQC)
```

Components:

**Feature Map**

```
ZZFeatureMap
feature_dimension = 8
entanglement = linear
reps = 1
```

**Optimizer**

```
COBYLA
maxiter = 80
```

**Sampler Backend**

```
Qiskit Aer Sampler
```

---

# Results

## Classical NLP Model

Accuracy:

```
65.59%
```

Classification Report:

```
Negative
Precision: 0.76
Recall:    0.65
F1-score:  0.70

Positive
Precision: 0.54
Recall:    0.67
F1-score:  0.60
```

---

## Quantum Sentiment Model

Accuracy:

```
62.38%
```

Classification Report:

```
Negative
Precision: 0.67
Recall:    0.79
F1-score:  0.72

Positive
Precision: 0.51
Recall:    0.36
F1-score:  0.42
```

---

# Key Observations

* Classical **SVM performs slightly better overall**.
* Quantum model shows **strong recall for negative sentiment**.
* Performance gap is relatively small considering **quantum hardware limitations**.
* Demonstrates the **potential of hybrid classical-quantum NLP pipelines**.

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/autism-sentiment-qml.git
cd autism-sentiment-qml
```

Install dependencies:

```
pip install pandas numpy nltk scikit-learn qiskit qiskit-machine-learning
```

Download NLTK resources:

```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
```

---

# Running the Project

Place the dataset in the project folder:

```
AutismData.csv
```

Run the notebook or script:

```
python autism_sentiment_qml.py
```

---

# Technologies Used

* Python
* NLTK
* Scikit-learn
* Qiskit
* Qiskit Machine Learning
* NumPy
* Pandas

---

# Research Motivation

Understanding emotional patterns in autism-related communication can support:

* behavioral research
* mental health analysis
* assistive technologies
* therapeutic support systems

This project investigates whether **quantum machine learning techniques can complement classical NLP models** in such domains.

---

# Future Improvements

* Use **transformer embeddings (BERT / RoBERTa)**
* Use **quantum kernels**
* Increase **quantum circuit depth**
* Try **hybrid classical-quantum pipelines**
* Expand dataset with **clinical or behavioral corpora**

---

# License

MIT License

