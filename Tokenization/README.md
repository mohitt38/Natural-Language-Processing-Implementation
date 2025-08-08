# 🧠 Natural Language Processing (NLP) Projects

This repository contains various NLP implementations, from basic preprocessing techniques to classification models, using Python and popular libraries like **scikit-learn**, **NLTK**, **gensim**, and **NumPy**.

## 📂 Contents

1. **Tokenization**
2. **TF-IDF (Term Frequency–Inverse Document Frequency)**
3. **Bag of Words (BoW) Implementation**
4. **Spam-Ham Classification**  
   - Using **TF-IDF**
   - Using **BoW**
   - Using **Word2Vec** (average word vectors)

---

## 1️⃣ Tokenization

Tokenization is the process of breaking text into smaller units called tokens, such as words, subwords, or sentences.  
It is the first and most essential step in NLP, enabling further text analysis and processing.

---

### 🔹 Stopwords Removal
Stopwords are common words like *is*, *the*, *and*, which generally carry less meaningful information for NLP tasks.  
Removing them helps reduce noise and improves model performance.

---

### 🔹 Stemming
Stemming is the process of reducing words to their base or root form by chopping off prefixes or suffixes.  
It is a rule-based approach and may result in non-dictionary words.

**Common Types of Stemming:**
- **Porter Stemmer** – Balanced between speed and accuracy, widely used.
- **Lancaster Stemmer** – More aggressive in cutting words.
- **Snowball Stemmer** – An improvement over Porter, supports multiple languages.

---

### 🔹 Lemmatization
Lemmatization reduces words to their meaningful base form (lemma) using dictionary and morphological analysis.  
Unlike stemming, lemmatization always returns valid dictionary words and considers the part of speech for better accuracy.

---

### 🔹 Part-of-Speech (POS) Tagging
POS tagging assigns grammatical categories (noun, verb, adjective, etc.) to each token.  
It helps in understanding the syntactic structure of sentences and is useful for many downstream NLP tasks.

---

### 🔹 Named Entity Recognition (NER)
NER identifies and classifies named entities in text into predefined categories such as:
- Person names  
- Organizations  
- Locations  
- Dates & time expressions  

NER is crucial for extracting structured information from unstructured text.

---

**📌 Summary:**  
This repository covers tokenization along with:
- Stopwords Removal  
- Stemming (Porter, Lancaster, Snowball)  
- Lemmatization  
- POS Tagging  
- Named Entity Recognition  

These preprocessing steps are essential for preparing text data for advanced NLP tasks.

---

## 2️⃣ TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF measures how important a word is to a document in a collection of documents (corpus).  
It helps reduce the weight of commonly used words and increase the weight of rare but significant words.

- **Term Frequency (TF):**
\[
TF(t, d) = \frac{\text{Number of times term t appears in document d}}{\text{Total terms in document d}}
\]

- **Inverse Document Frequency (IDF):**
\[
IDF(t) = \log\frac{\text{Total documents}}{\text{Number of documents containing term t}}
\]

---

## 3️⃣ Bag of Words (BoW) Implementation

BoW represents text as a vector of word counts.  
It ignores grammar and word order but is simple and effective for many NLP tasks.

---

## 4️⃣ Spam-Ham Classification

A binary classification problem where the goal is to predict whether a message is **spam** or **ham**.

### 📌 Dataset
- Example: [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

---

### 🔹 4.1 Using TF-IDF
- Convert text into TF-IDF vectors.
- Train a classification model (e.g., Logistic Regression, Naive Bayes).
- Evaluate performance with metrics like Accuracy, Precision, Recall, F1-score.

---

### 🔹 4.2 Using Bag of Words (BoW)
- Represent text using word counts.
- Train and evaluate using standard ML classifiers.

---

### 🔹 4.3 Using Word2Vec (Average Word Vectors)
- Train or load pre-trained Word2Vec embeddings.
- Convert each sentence into a vector by averaging word embeddings.
- Train a classifier using these vector representations.

---

