import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("drugsComTrain_raw.csv\\drugsComTrain_raw.csv")

# Remove missing reviews
df = df.dropna(subset=['review'])

# Create sentiment labels
def get_sentiment(rating):
    if rating >= 7:
        return "positive"
    elif rating <= 4:
        return "negative"
    else:
        return "neutral"

df['sentiment'] = df['rating'].apply(get_sentiment)

# Remove neutral (binary classification)
df = df[df['sentiment'] != 'neutral']

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_review'] = df['review'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("=== MODEL PERFORMANCE ===")
print(classification_report(y_test, y_pred))

# Sample prediction
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    return model.predict(vectorized)[0]

print("\n=== SAMPLE TEST ===")
print("Input: This drug worked very well")
print("Prediction:", predict_sentiment("This drug worked very well"))

print("Input: This made my condition worse")
print("Prediction:", predict_sentiment("This made my condition worse"))

import pickle

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved!")