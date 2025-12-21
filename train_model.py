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
    # Remover tags HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Remover URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remover handles do Twitter
    text = re.sub(r'@\w+', '', text)
    # Remover caracteres especiais e dígitos (manter acentos)
    text = re.sub(r'[^a-zA-Z\u00C0-\u00FF\s]', '', text)
    # Converter para minúsculas
    text = text.lower()
    return text.strip()

def load_and_combine_data():
    path = 'data/TrainingDataSets'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    df_list = []
    for filename in all_files:
        print(f"Loading {filename}...")
        try:
            # Ler apenas colunas essenciais: tweet_text, sentiment
            # Usando nrows para evitar OOM se os arquivos forem massivos, mas o objetivo é alta acurácia então queremos todos os dados.
            # Assumindo que a máquina tem 8GB+ RAM, isso deve ser OK.
            df = pd.read_csv(filename)
            
            # Normalizar nomes de colunas se necessário (ex: alguns podem ser review_text vs tweet_text)
            # Inspeção mostrou 'tweet_text' e 'sentiment' para datasets do Twitter.
            # Se MobileReviews for incluído, ele tem 'review_text'. Vamos normalizar.
            if 'tweet_text' in df.columns:
                df = df.rename(columns={'tweet_text': 'text'})
            elif 'review_text' in df.columns:
                df = df.rename(columns={'review_text': 'text'})
            
            if 'sentiment' in df.columns and 'text' in df.columns:
                df_list.append(df[['text', 'sentiment']])
        except Exception as e:
            print(f"Erro ao carregar {filename}: {e}")
            
    if not df_list:
        raise ValueError("No data loaded!")
        
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    return combined_df

def train_model():
    print("Carregando datasets...")
    df = load_and_combine_data()
    
    print(f"Total de amostras: {len(df)}")
    print(f"Distribuição de Sentimentos:\n{df['sentiment'].value_counts()}")
    
    # Mapear sentimentos para rótulos de usuário
    # Datasets de Twitter geralmente têm 'Positivo', 'Negativo', 'Neutro'
    # Dataset Mobile tinha 'Positive', 'Negative', 'Neutral'
    # Mapeamos tudo para: Bom, Ruim, Neutro
    
    norm_map = {
        'Positivo': 'Bom', 'Positive': 'Bom', 'Bom': 'Bom',
        'Negativo': 'Ruim', 'Negative': 'Ruim', 'Ruim': 'Ruim',
        'Neutro': 'Neutro', 'Neutral': 'Neutro'
    }
    
    df['sentiment_label'] = df['sentiment'].map(norm_map)
    
    # Remover rótulos desconhecidos
    df = df.dropna(subset=['sentiment_label'])
    
    print("Pré-processamento...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Remover texto vazio
    df = df[df['clean_text'].str.len() > 2]
    
    X = df['clean_text']
    y = df['sentiment_label']
    
    print(f"Tamanho dos dados de treino após limpeza: {len(df)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    print("Treinando modelo (TF-IDF + Regressão Logística)...")
    # Aumentado max_iter para convergência
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=100000)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("Avaliando...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("Salvando modelo...")
    joblib.dump(pipeline, 'data/sentiment_model.pkl')
    print("Modelo salvo em data/sentiment_model.pkl")

if __name__ == "__main__":
    train_model()
