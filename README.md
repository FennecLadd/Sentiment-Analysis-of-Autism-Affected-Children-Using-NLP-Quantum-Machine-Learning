# Sentiment Analysis using NLP and Quantum Machine Learning

A hybrid **Natural Language Processing (NLP) and Quantum Machine Learning (QML)** project for sentiment classification of autism and mental-health-related behavioral statements.

The project combines **weakly supervised sentiment labeling, text preprocessing, TF-IDF, Transformer-based sentence embeddings, dimensionality reduction, quantum feature encoding, and classical machine learning** to investigate sentiment classification using both classical and quantum-enhanced approaches.

---

## 📌 Project Overview

The dataset contains behavioral statements related to different mental-health conditions. While the original dataset provides condition/status labels, it does not directly provide sentiment labels.

To build a sentiment-analysis pipeline, a pretrained Transformer-based sentiment model was used to generate **weak sentiment labels**:

* **Negative**
* **Neutral**
* **Positive**

Low-confidence predictions were filtered to improve the reliability of the generated labels.

The project then follows two main approaches:

1. **Classical NLP and Machine Learning**
2. **Hybrid Quantum-Classical Machine Learning**

---

## 🎯 Objectives

* Build an automated sentiment classification pipeline for behavioral text.
* Apply NLP preprocessing techniques to clean and normalize textual data.
* Generate weak sentiment labels using a pretrained Transformer model.
* Establish classical ML baselines using TF-IDF representations.
* Generate semantic sentence embeddings using a Transformer model.
* Reduce high-dimensional embeddings using PCA for quantum processing.
* Encode classical features into an 8-qubit quantum circuit.
* Extract quantum-generated features using circuit measurements.
* Evaluate classification performance using multiple metrics.

---

## 🗂️ Dataset

The project uses a dataset containing behavioral statements and their associated mental-health-related status.

The original dataset contains approximately **53,000 records**.

After preprocessing and duplicate removal, approximately **50,204 usable statements** were obtained.

### Original Dataset Structure

| Column      | Description                                            |
| ----------- | ------------------------------------------------------ |
| `statement` | Behavioral or mental-health-related textual statement  |
| `status`    | Original category/status associated with the statement |

The original `status` column was **not used as the sentiment target** because it represents mental-health-related categories rather than sentiment.

---

## 🧠 Sentiment Label Generation

Since explicit sentiment labels were not available, a pretrained sentiment model was used for **weak supervision**.

### Model

`cardiffnlp/twitter-roberta-base-sentiment-latest`

The model generates:

* Negative
* Neutral
* Positive

along with a confidence score.

A confidence threshold of **0.60** was applied, retaining predictions with sufficient model confidence.

After confidence filtering, approximately **41,902 samples** remained.

### Important Note

The sentiment labels are **automatically generated pseudo-labels**, rather than manually annotated ground truth. Therefore, the results should be interpreted as an experimental sentiment-classification study rather than a clinically validated classification system.

---

## 🔄 Project Pipeline

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Text Preprocessing
     │
     ▼
Weak Sentiment Labeling
     │
     ▼
Confidence Filtering
     │
     ▼
Train / Test Split
     │
     ├──────────────────────────┐
     ▼                          ▼
TF-IDF                    Transformer Embeddings
     │                          │
     ▼                          ▼
Classical ML                   PCA
     │                          │
     │                       384 → 8
     │                          │
     │                          ▼
     │                   Quantum Encoding
     │                          │
     │                          ▼
     │                   Quantum Circuit
     │                          │
     │                          ▼
     │                   Quantum Features
     │                          │
     │                          ▼
     │                   Classical Classifier
     │
     ▼
Evaluation & Comparison
```

---

# 🔧 Methodology

## 1. Data Preprocessing

The textual data was cleaned using several preprocessing operations:

* Conversion to lowercase
* Removal of URLs
* Removal of unnecessary special characters
* Whitespace normalization
* Removal of very short statements
* Duplicate removal

This produces cleaner and more consistent text for subsequent NLP processing.

---

## 2. Train-Test Split

The labeled dataset was divided into:

* **80% training data**
* **20% testing data**

A stratified split was used to preserve the relative distribution of sentiment classes across the training and testing sets.

---

# 📊 Classical NLP Approach

## 3. TF-IDF Feature Extraction

The first representation used was **Term Frequency-Inverse Document Frequency (TF-IDF)**.

Configuration:

```text
Maximum features: 15,000
N-grams: Unigrams + Bigrams
Minimum document frequency: 2
Sublinear TF: Enabled
```

Using both unigrams and bigrams allows the model to capture individual words as well as short word combinations.

For example:

```text
"I feel anxious"
```

can produce features such as:

```text
feel
anxious
feel anxious
```

---

## 4. Classical Classification

Two classical classifiers were evaluated:

### Logistic Regression

Used as a linear baseline for multi-class sentiment classification.

### Linear Support Vector Machine

A Linear SVM was used because linear models are particularly effective with high-dimensional sparse text representations such as TF-IDF.

Class balancing was applied to reduce the effect of the imbalanced sentiment distribution.

---

# 🤖 Transformer-Based Representation

## 5. Sentence Embeddings

A pretrained Sentence Transformer was used:

```text
all-MiniLM-L6-v2
```

Each statement was converted into a **384-dimensional dense embedding**.

Unlike TF-IDF, which primarily represents lexical importance, Transformer embeddings provide a dense representation intended to capture semantic information from the sentence.

Example:

```text
"I feel terrible"

        ↓

384-dimensional embedding
```

The embeddings were then used with a classical Logistic Regression classifier.

---

# ⚛️ Quantum Machine Learning

## 6. Dimensionality Reduction with PCA

The Transformer model produces 384-dimensional embeddings, which are too large for the small quantum circuit used in this project.

Therefore, **Principal Component Analysis (PCA)** was applied:

```text
384 dimensions
      ↓
     PCA
      ↓
8 dimensions
```

The first eight principal components retained approximately **25.42% of the total variance**.

The resulting eight-dimensional representation was then scaled to a suitable range for quantum rotation gates.

---

## 7. Quantum Feature Encoding

An **8-qubit quantum circuit** was designed to process the reduced feature representation.

The classical features were encoded using parameterized quantum rotations.

Conceptually:

```text
Classical Features
       │
       ▼
   8 Values
       │
       ▼
RY / RZ Rotations
       │
       ▼
Quantum State
```

Each input feature is mapped to a quantum rotation angle.

---

## 8. Quantum Circuit Architecture

The quantum circuit consists of:

* 8 qubits
* Parameterized `RY` rotations
* Parameterized `RZ` rotations
* Ring-based entanglement using `CX` gates
* Measurement-based feature extraction

The ring entanglement structure connects neighboring qubits:

```text
Q0 ── Q1 ── Q2 ── Q3
│                 │
Q7 ── Q6 ── Q5 ── Q4
```

This allows interactions between the encoded features through quantum entanglement.

---

## 9. Quantum Feature Extraction

After executing the quantum circuit, measurements are collected over multiple shots.

The measurement results are converted into **Z-axis expectation values**.

Conceptually:

```text
Quantum Circuit
      │
      ▼
Measurements
      │
      ▼
Expectation Values
      │
      ▼
8-Dimensional Quantum Feature Vector
```

These quantum-generated features can then be supplied to a conventional classical classifier.

This creates a **hybrid quantum-classical architecture**.

---

# 📈 Results

The project compares multiple representations and classification approaches.

| Approach           | Representation           | Classifier          |   Accuracy |   Macro F1 |
| ------------------ | ------------------------ | ------------------- | ---------: | ---------: |
| Classical Baseline | TF-IDF                   | Logistic Regression |     84.72% |     0.7532 |
| Classical Baseline | TF-IDF                   | Linear SVM          | **87.48%** | **0.7673** |
| Transformer        | 384-D Sentence Embedding | Logistic Regression |     85.43% |     0.7611 |

The classical TF-IDF + Linear SVM pipeline achieved the strongest performance among the primary classification approaches evaluated.

---

# 📏 Evaluation Metrics

Multiple metrics were considered because the sentiment classes are not perfectly balanced.

### Accuracy

Measures the overall proportion of correctly classified samples.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many samples predicted as a particular class actually belong to that class.

### Recall

Measures how many samples belonging to a particular class were correctly identified.

### F1 Score

Combines precision and recall:

```text
F1 = 2 × Precision × Recall
          -----------------
          Precision + Recall
```

### Macro F1

F1 is calculated independently for each class and then averaged, giving equal importance to each sentiment class.

This makes Macro F1 useful when evaluating imbalanced multi-class classification problems.

---

# 🛠️ Technologies Used

### Programming

* Python

### NLP & Machine Learning

* Pandas
* NumPy
* Scikit-learn
* Transformers
* Sentence Transformers

### Quantum Machine Learning

* Qiskit
* Qiskit Aer
* Qiskit Machine Learning

### Visualization & Analysis

* Matplotlib
* Seaborn
* Jupyter Notebook

---

# 📁 Project Structure

```text
Sentiment-Analysis-NLP-QML/
│
├── rp2.ipynb
├── AutismData.xls
├── README.md
└── requirements.txt
```

> Dataset files may need to be obtained separately depending on redistribution and licensing restrictions.

---

# 🚀 How to Run

## 1. Clone the repository

```bash
git clone <repository-url>
cd Sentiment-Analysis-NLP-QML
```

## 2. Install dependencies

```bash
pip install pandas numpy scikit-learn
pip install transformers sentence-transformers
pip install qiskit qiskit-aer qiskit-machine-learning
pip install matplotlib seaborn jupyter
```

## 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
rp2.ipynb
```

and execute the notebook cells sequentially.

---

# 💡 Key Concepts Demonstrated

This project demonstrates practical experience with:

* Natural Language Processing
* Weak supervision
* Sentiment classification
* TF-IDF
* N-gram features
* Transformer sentence embeddings
* Dimensionality reduction using PCA
* Classical machine learning
* Support Vector Machines
* Logistic Regression
* Quantum feature encoding
* Quantum circuits
* Qubits and quantum gates
* Quantum entanglement
* Measurement and expectation values
* Hybrid quantum-classical machine learning
* Model evaluation and comparison

---

# ⚠️ Limitations

* Sentiment labels are generated through weak supervision rather than manual annotation.
* The source text is from a mental-health-related domain, while the sentiment model is a general pretrained model.
* PCA reduces the 384-dimensional representation to only eight dimensions, resulting in information loss.
* Quantum experiments were performed using a simulator rather than physical quantum hardware.
* Quantum feature extraction is computationally more expensive than the classical feature-extraction pipeline.

---

# 🔮 Future Improvements

Potential extensions include:

* Creating a manually annotated sentiment dataset.
* Using a domain-specific sentiment model.
* Experimenting with different embedding models.
* Evaluating different PCA dimensionalities.
* Exploring alternative quantum feature maps.
* Testing different quantum circuit architectures.
* Evaluating the approach on real quantum hardware.
* Investigating quantum noise and error mitigation.
* Comparing additional classical and quantum classifiers.

---

# 👨‍💻 Author

**Kushagra Bhagoliwal**

B.Tech — Computer Science Engineering (AI & ML)
VIT-AP University, Amaravati

**Interests:** Machine Learning • NLP • Generative AI • Quantum Machine Learning • Software Development
