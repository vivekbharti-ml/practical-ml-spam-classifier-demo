from flask import Flask, request, jsonify
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

# Load model pipeline
with open('../models/logreg_spam_pipeline.pkl', 'rb') as f:
    logreg_pipeline = pickle.load(f)

# Preprocessing function (same as yours)
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text, language='english')
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# Inference threshold
BEST_THRESHOLD = 0.620

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    raw_text = data['text']
    clean_text = preprocess_text(raw_text)
    
    prob = logreg_pipeline.predict_proba([clean_text])[0][1]
    pred = 1 if prob >= BEST_THRESHOLD else 0
    
    label = 'spam' if pred == 1 else 'ham'
    
    return jsonify({
        'prediction': label,
        'probability_spam': prob
    })

if __name__ == '__main__':
    app.run(debug=True)