from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import pickle
import os
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')
nltk.download('stopwords')

nltk.data.path.append('/home/airflow/nltk_data')

DEFAULT_ARGS = {
    'owner': 'ml_engineer',
    'start_date': datetime(2025, 6, 1),
}

MODEL_PATH = '/opt/airflow/models/logreg_spam_pipeline.pkl'
INPUT_PATH = '/opt/airflow/data/incoming_emails_20250601.csv'
OUTPUT_PATH = '/opt/airflow/data/scored_emails_20250601.csv'
BEST_THRESHOLD = 0.620

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def ingest_data(**kwargs):
    df = pd.read_csv(INPUT_PATH)
    kwargs['ti'].xcom_push(key='raw_data', value=df.to_json())

def preprocess_data(**kwargs):
    raw_json = kwargs['ti'].xcom_pull(key='raw_data')
    df = pd.read_json(raw_json)
    df['clean_text'] = df['text'].apply(preprocess_text)
    kwargs['ti'].xcom_push(key='clean_data', value=df.to_json())

def score_model(**kwargs):
    df = pd.read_json(kwargs['ti'].xcom_pull(key='clean_data'))
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    probs = model.predict_proba(df['clean_text'])[:, 1]
    df['prob_spam'] = probs
    df['prediction'] = (probs >= BEST_THRESHOLD).astype(int)
    df['label'] = df['prediction'].map({0: 'ham', 1: 'spam'})
    df.to_csv(OUTPUT_PATH, index=False)

with DAG(
    dag_id='spam_batch_scoring',
    default_args=DEFAULT_ARGS,
    schedule_interval=None,
    catchup=False
) as dag:

    ingest = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data
    )

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    score = PythonOperator(
        task_id='score_model',
        python_callable=score_model
    )

    ingest >> preprocess >> score