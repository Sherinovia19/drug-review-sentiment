# MediSentinel - Drug Review Sentiment Analysis

MediSentinel is a Machine Learning project that analyzes patient drug reviews and predicts sentiment (Positive/Negative) based on textual feedback.

---

## Overview

MediSentinel uses Natural Language Processing (NLP) techniques to classify drug reviews into sentiments. It helps in understanding patient experiences and can be useful for healthcare analytics and decision-making.

---

## Features

* Text preprocessing and cleaning
* TF-IDF vectorization
* Trained Machine Learning model for sentiment prediction
* Simple and reusable Python-based implementation
* Supports custom user input for real-time predictions

---

## Project Structure

```id="3l9xkd"
drug-review-sentiment/
│
├── app.py                  
├── resume_project.py       
├── model.pkl              
├── vectorizer.pkl         
├── sentiment_model.pkl    
├── sample_reviews.xlsx    
├── requirements.txt       
├── README.md              
└── .gitignore
```

---

## Dataset

The dataset used for training is sourced from Kaggle and is not included in this repository due to size constraints.

Dataset Source: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

Instructions:
1. Visit the dataset link above
2. Download the dataset files
3. Extract and place the required file (e.g., drugsComTest_raw.xlsx) in the project root directory
---

## Installation

1. Clone the repository:

```id="7e7ld7"
git clone https://github.com/sherinovia19/drug-review-sentiment.git
cd drug-review-sentiment
```

2. Install dependencies:

```id="d9m38s"
pip install -r requirements.txt
```

---

## Usage

Run the application:

```id="r2v8o9"
python app.py
```

Then input a drug review and get the predicted sentiment.

---

## Example

Input:

```id="u6e5n0"
"This medicine worked great and had no side effects."
```

Output:

```id="f9c2qm"
Positive
```

---

## Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* NLP (TF-IDF)

---

## Future Improvements

* Deploy as a web app (Streamlit/Flask)
* Add multi-class sentiment (neutral, mixed)
* Integrate deep learning models (LSTM/BERT)
* Build a healthcare dashboard

---

## Author

Sherinovia
GitHub: https://github.com/sherinovia19

---

## Contributing

Feel free to fork this repository and submit pull requests.

---

## License

This project is licensed under the MIT License.
