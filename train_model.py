import pandas as pd
import numpy as np
import joblib
import re
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove Twitter handles
    text = re.sub(r'@\w+', '', text)
    # Remove special chars and digits (keep accented chars)
    text = re.sub(r'[^a-zA-Z\u00C0-\u00FF\s]', '', text)
    # Lowercase
    text = text.lower()
    return text.strip()

def load_and_combine_data():
    path = 'data/TrainingDataSets'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    df_list = []
    for filename in all_files:
        print(f"Loading {filename}...")
        try:
            # Read only essential columns: tweet_text, sentiment
            # Using nrows to avoid OOM if files are massive, but goal is high accuracy so we want all data.
            # Assuming machine has 8GB+ RAM, this should be fine.
            df = pd.read_csv(filename)
            
            # Normalize column names if needed (e.g. some might be review_text vs tweet_text)
            # Inspection showed 'tweet_text' and 'sentiment' for Twitter datasets.
            # If MobileReviews is included, it has 'review_text'. We'll normalize.
            if 'tweet_text' in df.columns:
                df = df.rename(columns={'tweet_text': 'text'})
            elif 'review_text' in df.columns:
                df = df.rename(columns={'review_text': 'text'})
            
            if 'sentiment' in df.columns and 'text' in df.columns:
                df_list.append(df[['text', 'sentiment']])
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    if not df_list:
        raise ValueError("No data loaded!")
        
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    return combined_df

def train_model():
    print("Loading datasets...")
    df = load_and_combine_data()
    
    print(f"Total samples: {len(df)}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Map sentiments to user-facing labels
    # Twitter datasets usually have 'Positivo', 'Negativo', 'Neutro'
    # Mobile dataset had 'Positive', 'Negative', 'Neutral'
    # We map all to: Bom, Ruim, Neutro
    
    norm_map = {
        'Positivo': 'Bom', 'Positive': 'Bom', 'Bom': 'Bom',
        'Negativo': 'Ruim', 'Negative': 'Ruim', 'Ruim': 'Ruim',
        'Neutro': 'Neutro', 'Neutral': 'Neutro'
    }
    
    df['sentiment_label'] = df['sentiment'].map(norm_map)
    
    # Drop unknown labels
    df = df.dropna(subset=['sentiment_label'])
    
    print("Preprocessing...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Remove empty text
    df = df[df['clean_text'].str.len() > 2]
    
    X = df['clean_text']
    y = df['sentiment_label']
    
    print(f"Training data size after cleaning: {len(df)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    print("Training model (TF-IDF + Logistic Regression)...")
    # Increased max_iter for convergence
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=100000)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("Saving model...")
    joblib.dump(pipeline, 'data/sentiment_model.pkl')
    print("Model saved to data/sentiment_model.pkl")

if __name__ == "__main__":
    train_model()
