
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

# Always resolve path relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'spam.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'spam_model.joblib')

def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"spam.csv not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'text'})
    df = df[['label', 'text']]
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB()),
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
    joblib.dump(pipeline, MODEL_PATH)

def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return joblib.load(MODEL_PATH)

def predict_spam(text):
    model = load_model()
    return int(model.predict([text])[0])
