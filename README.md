# MediSentinel - Drug Review Sentiment Analyzer

Live Application:  
https://medisentinel-jwu2wvb8amhbbkve6d4tpq.streamlit.app/

---

## Overview

MediSentinel is a machine learning-based web application that analyzes patient drug reviews and classifies them into sentiment categories:

- Positive  
- Negative  
- Uncertain (based on confidence threshold)

The system uses Natural Language Processing (NLP) techniques and a supervised learning model to interpret user-generated medical reviews.

---

## Features

### Single Review Analysis
- Input a patient review manually  
- Get instant sentiment prediction  
- View confidence score and probability distribution  

### Batch Analysis
- Upload a CSV file containing multiple reviews  
- Perform sentiment analysis on large datasets  
- Download prediction results  

### EDA Insights
- Visualize rating distribution  
- View dataset statistics  
- Analyze top conditions by review count  

### Advanced Controls
- Adjustable confidence threshold  
- Option to view preprocessed tokens  
- Option to display raw prediction probabilities  

---

## Machine Learning Pipeline

Input Text  
↓  
Text Preprocessing  
↓  
TF-IDF Vectorization (1–2 grams, max 30,000 features)  
↓  
Logistic Regression Model  
↓  
Probability Output  
↓  
Threshold Filtering  
↓  
Final Sentiment Classification  

---

## Model Details

Component        | Description  
-----------------|------------  
Algorithm        | Logistic Regression  
Vectorizer       | TF-IDF (Unigrams and Bigrams)  
Max Features     | 30,000  
Accuracy         | Approximately 85%  
Label Rule       | Rating ≥ 7 → Positive, ≤ 4 → Negative  
Dataset          | UCI Drug Review Dataset  

---

## Dataset

- Source: UCI Machine Learning Repository  
- Contains patient reviews, ratings, drug names, and conditions  
- Key columns used:
  - review  
  - rating  
  - drugName  
  - condition  

---

## Text Preprocessing

The following preprocessing steps are applied:

1. Convert text to lowercase  
2. Remove URLs  
3. Remove special characters  
4. Normalize whitespace  
5. Remove stopwords  

---

## Technology Stack

- Frontend: Streamlit  
- Backend: Python  
- Machine Learning: scikit-learn  
- Data Processing: Pandas, NumPy  
- Visualization: Streamlit built-in charts  

---

## Project Structure

drug-review-sentiment/  
│── app.py  
│── requirements.txt  
│── model.pkl  
│── vectorizer.pkl  
│── README.md  

---

## Run Locally

git clone https://github.com/sherinovia19/drug-review-sentiment.git  
cd drug-review-sentiment  
pip install -r requirements.txt  
streamlit run app.py  

---

## Deployment

This application is deployed using Streamlit Cloud:

https://medisentinel-jwu2wvb8amhbbkve6d4tpq.streamlit.app/

---

## Future Improvements

- Integration of deep learning models such as LSTM or BERT  
- Drug recommendation system based on sentiment  
- API deployment for external integration  
- Advanced analytics dashboard  

---

## Author

Sherin Ovia  
https://github.com/sherinovia19  

---

## Acknowledgements

- UCI Machine Learning Repository  
- Streamlit  
- scikit-learn  
