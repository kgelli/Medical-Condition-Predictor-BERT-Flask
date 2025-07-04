from flask import Flask, render_template, request, redirect, session, jsonify, flash
import os
import joblib
import pandas as pd
import re
import torch
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import logging
from datetime import datetime
from functools import wraps
import traceback
import time
from bert_handler import BERTHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Configuration
TFIDF_MODEL_PATH = 'model/passmodel.pkl'
TFIDF_VECTORIZER_PATH = 'model/tfidfvectorizer.pkl'
DATA_PATH = 'data/drugsComTrain_raw.tsv' 

BERT_MODEL_PATH = 'model/distilbert-drug-review-model'
BERT_TOKENIZER_PATH = 'model/distilbert-drug-review-tokenizer'
BERT_LABEL_ENCODER_PATH = 'model/label_encoder_fixed.pkl'

# User credentials
USERS = {
    "kgelli@gsu.edu": "nlp",
    "admin@medical.com": "admin123", 
    "demo@medical.com": "demo"
}

# Global variables
tfidf_model = None
tfidf_vectorizer = None
lemmatizer = None
stop_words = None
data_df = None
bert_handler = None

def initialize_nltk():
    """Initialize NLTK components"""
    global lemmatizer, stop_words
    try:
        import nltk
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        return True
    except Exception as e:
        logger.error(f"Error initializing NLTK: {str(e)}")
        return False

def load_models():
    """Load all available models"""
    global tfidf_model, tfidf_vectorizer, data_df, bert_handler
    
    models_loaded = {'tfidf': False, 'bert': False}
    
    # Load TF-IDF model (existing code)
    try:
        if (os.path.exists(TFIDF_MODEL_PATH) and 
            os.path.exists(TFIDF_VECTORIZER_PATH)):
            
            logger.info("Loading TF-IDF model...")
            tfidf_model = joblib.load(TFIDF_MODEL_PATH)
            tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            models_loaded['tfidf'] = True
            logger.info("TF-IDF model loaded successfully!")
        else:
            logger.warning("TF-IDF model files not found")
    except Exception as e:
        logger.error(f"Error loading TF-IDF model: {str(e)}")
    
    try:
        bert_handler = BERTHandler(BERT_MODEL_PATH, BERT_TOKENIZER_PATH, BERT_LABEL_ENCODER_PATH)
        if bert_handler.load_models():
            models_loaded['bert'] = True
    except Exception as e:
        logger.error(f"Error initializing BERT handler: {str(e)}")
    
    # Load data
    try:
        if os.path.exists(DATA_PATH):
            if DATA_PATH.endswith('.tsv'):
                data_df = pd.read_csv(DATA_PATH, sep='\t')
            else:
                data_df = pd.read_csv(DATA_PATH)
            logger.info(f"Data loaded: {len(data_df)} records")
        else:
            logger.warning(f"Data file not found: {DATA_PATH}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
    
    return models_loaded


def predict_with_bert(text):
    """Predict using BERT model"""
    try:
        if bert_handler is None or not bert_handler.is_available():
            return None, 0.0, "BERT model not available"
        
        return bert_handler.predict(text, clean_text)
        
    except Exception as e:
        logger.error(f"BERT prediction error: {str(e)}")
        return None, 0.0, str(e)

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect('/')
        return f(*args, **kwargs)
    return decorated_function

def clean_text(raw_text):
    """Clean text for TF-IDF model"""
    try:
        if not raw_text or pd.isna(raw_text):
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(raw_text, 'html.parser').get_text()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Convert to lowercase and split
        words = text.lower().split()
        
        # Remove stopwords and short words
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Lemmatize words
        lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
        
        return ' '.join(lemmatized_words)
        
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return raw_text

def predict_with_tfidf(text):
    """Predict using TF-IDF model"""
    try:
        if tfidf_model is None or tfidf_vectorizer is None:
            return None, 0.0, "TF-IDF model not loaded"
        
        cleaned_text = clean_text(text)
        
        if not cleaned_text.strip():
            return None, 0.0, "Text too short"
        
        # Vectorize
        text_vector = tfidf_vectorizer.transform([cleaned_text])
        
        # Get prediction
        prediction = tfidf_model.predict(text_vector)[0]
        
        # Get confidence (approximation)
        if hasattr(tfidf_model, 'decision_function'):
            scores = tfidf_model.decision_function(text_vector)[0]
            confidence = min(max(scores.max() / 10.0, 0.1), 1.0)
        else:
            confidence = 0.8  # Default confidence
        
        return prediction, confidence, None
        
    except Exception as e:
        logger.error(f"TF-IDF prediction error: {str(e)}")
        return None, 0.0, str(e)

def get_top_drugs(condition, min_rating=8.0, min_useful_count=50):
    """Get top drug recommendations with accurate review counts"""
    try:
        if data_df is None:
            return []
        
        # Filter by condition and quality metrics
        condition_drugs = data_df[
            (data_df['condition'] == condition) & 
            (data_df['rating'] >= min_rating) & 
            (data_df['usefulCount'] >= min_useful_count)
        ]
        
        if condition_drugs.empty:
            # Try with relaxed criteria
            condition_drugs = data_df[
                (data_df['condition'] == condition) & 
                (data_df['rating'] >= 7.0) & 
                (data_df['usefulCount'] >= 25)
            ]
        
        if condition_drugs.empty:
            return []
        
        # Group by drug and calculate stats
        drug_stats = condition_drugs.groupby('drugName').agg({
            'rating': 'mean',
            'usefulCount': 'sum',
            'condition': 'count'
        }).round(2)
        
        # Sort by rating and useful count
        drug_stats = drug_stats.sort_values(
            by=['rating', 'usefulCount'], 
            ascending=[False, False]
        )
        
        # Get top 3 drugs with ACTUAL review counts
        top_drugs = []
        for drug_name, stats in drug_stats.head(3).iterrows():
            # Get actual total review count for this drug + condition
            actual_review_count = len(data_df[
                (data_df['drugName'] == drug_name) & 
                (data_df['condition'] == condition)
            ])
            
            top_drugs.append({
                'name': drug_name,
                'avg_rating': stats['rating'],
                'total_useful_count': stats['usefulCount'],
                'review_count': actual_review_count  # Now shows ALL reviews
            })
        
        return top_drugs
        
    except Exception as e:
        logger.error(f"Error getting top drugs: {str(e)}")
        return []

# Routes
@app.route('/')
def login():
    """Login page"""
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect('/')

@app.route('/login_validation', methods=['POST'])
def login_validation():
    """Validate login"""
    try:
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html')
        
        if username in USERS and USERS[username] == password:
            session['user_id'] = username
            flash(f'Welcome back, {username}!', 'success')
            return redirect('/index')
        else:
            flash('Invalid username or password.', 'error')
            return render_template('login.html')
            
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        flash('An error occurred during login.', 'error')
        return render_template('login.html')

@app.route('/index')
@login_required
def index():
    """Home page"""
    models_status = {
        'bert_available': bert_handler and bert_handler.is_available(),
        'tfidf_available': tfidf_model is not None
    }
    return render_template('home.html', 
                         user_id=session.get('user_id'),
                         models_status=models_status)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Prediction page"""
    if request.method == 'POST':
        try:
            raw_text = request.form.get('rawtext', '').strip()
            
            if not raw_text:
                flash('Please enter some text for prediction.', 'error')
                return render_template('predict.html')
            
            if len(raw_text) < 10:
                flash('Please enter at least 10 characters for better prediction.', 'warning')
                return render_template('predict.html')
            
            # Record start time
            start_time = time.time()
            
            # Use BERT as primary model
            if bert_handler and bert_handler.is_available():
                condition, confidence, error = predict_with_bert(raw_text)
                model_used = 'BERT (DistilBERT)'
            elif tfidf_model is not None:
                # TF-IDF as fallback only
                condition, confidence, error = predict_with_tfidf(raw_text)
                model_used = 'TF-IDF (Fallback)'
            else:
                flash('No prediction models available.', 'error')
                return render_template('predict.html')
            
            # Record end time
            prediction_time = round(time.time() - start_time, 3)
            
            if error:
                flash(f'Prediction error: {error}', 'error')
                return render_template('predict.html')
            
            if condition is None:
                flash('Unable to make prediction. Please try different text.', 'warning')
                return render_template('predict.html')
            
            # Get drug recommendations
            top_drugs = get_top_drugs(condition)
            
            # Log prediction
            logger.info(f"Prediction by {session.get('user_id')} using {model_used}: {condition} ({confidence:.2%}) in {prediction_time}s")
            
            return render_template('predict.html',
                                 rawtext=raw_text,
                                 result=condition,
                                 confidence=round(confidence * 100, 2),
                                 top_drugs=top_drugs,
                                 show_results=True,
                                 model_used=model_used,
                                 prediction_time=prediction_time)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            flash('An error occurred during prediction.', 'error')
            return render_template('predict.html')
    
    return render_template('predict.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check system resources
        import psutil
        memory_info = psutil.virtual_memory()
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_info = None
        if gpu_available:
            gpu_info = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown'
            }
        
        # Model status
        bert_status = bert_handler and bert_handler.is_available()
        tfidf_status = tfidf_model is not None
        data_status = data_df is not None
        
        # Overall health
        overall_health = 'healthy' if (bert_status or tfidf_status) and data_status else 'degraded'
        
        return jsonify({
            'status': overall_health,
            'timestamp': datetime.now().isoformat(),
            'models': {
                'bert_available': bert_status,
                'tfidf_available': tfidf_status,
                'primary_model': 'BERT' if bert_status else 'TF-IDF' if tfidf_status else 'None'
            },
            'data': {
                'dataset_loaded': data_status,
                'record_count': len(data_df) if data_df is not None else 0
            },
            'system': {
                'memory_usage_percent': memory_info.percent,
                'memory_available_gb': round(memory_info.available / (1024**3), 2),
                'gpu_available': gpu_available,
                'gpu_info': gpu_info
            },
            'nltk_initialized': lemmatizer is not None and stop_words is not None
        })
        
    except ImportError:
        # Fallback if psutil is not available
        return jsonify({
            'status': 'healthy' if (bert_handler and bert_handler.is_available()) or tfidf_model is not None else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'models': {
                'bert_available': bert_handler and bert_handler.is_available(),
                'tfidf_available': tfidf_model is not None,
                'primary_model': 'BERT' if bert_handler and bert_handler.is_available() else 'TF-IDF' if tfidf_model is not None else 'None'
            },
            'data': {
                'dataset_loaded': data_df is not None,
                'record_count': len(data_df) if data_df is not None else 0
            },
            'system': {
                'gpu_available': torch.cuda.is_available(),
                'memory_info': 'psutil not available'
            },
            'nltk_initialized': lemmatizer is not None and stop_words is not None
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy', 
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/api/reviews', methods=['POST'])
@login_required
def get_reviews():
    """Get reviews for a specific drug"""
    try:
        data = request.get_json()
        drug_name = data.get('drug_name')
        condition = data.get('condition')
        
        if not drug_name or not condition:
            return jsonify({'error': 'Drug name and condition required'}), 400
        
        if data_df is None:
            return jsonify({'error': 'Data not available'}), 500
        
        # Get reviews for the drug and condition
        reviews = data_df[
            (data_df['drugName'] == drug_name) & 
            (data_df['condition'] == condition)
        ].sort_values('usefulCount', ascending=False).head(10)
        
        if reviews.empty:
            return jsonify({'reviews': []})
        
        review_list = []
        for _, row in reviews.iterrows():
            # Clean the review text
            review_text = str(row['review']).strip()
            if review_text.startswith('"""') and review_text.endswith('"""'):
                review_text = review_text[3:-3]
            
            # Limit review length
            if len(review_text) > 400:
                review_text = review_text[:400] + '...'
            
            review_list.append({
                'rating': float(row['rating']) if pd.notna(row['rating']) else 0,
                'usefulCount': int(row['usefulCount']) if pd.notna(row['usefulCount']) else 0,
                'review': review_text,
                'date': str(row['date']) if pd.notna(row['date']) else 'N/A'
            })
        
        return jsonify({'reviews': review_list})
        
    except Exception as e:
        logger.error(f"Error fetching reviews: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == "__main__":
    # Initialize NLTK
    if not initialize_nltk():
        logger.error("Failed to initialize NLTK")
        exit(1)
    
    # Load models
    models_loaded = load_models()
    
    if not any(models_loaded.values()):
        logger.error("No models could be loaded!")
        exit(1)
    
    # Update the status display to focus on BERT
    print("\n" + "="*60)
    print("MEDICAL CONDITION PREDICTOR LOADED!")
    print("="*60)
    print(f"Primary Model: {'BERT (DistilBERT) ✅' if models_loaded.get('bert', False) else 'TF-IDF (Fallback) ✅'}")
    print(f"Dataset: {'161K+ Medical Records ✅' if data_df is not None else 'Not Available ❌'}")
    print(f"Processing: {'GPU Accelerated' if torch.cuda.is_available() else 'CPU Processing'}")
    print("="*60)
    print("Ready for high-accuracy medical condition prediction!")
    
    # Run the application
    app.run(debug=True, host="localhost", port=8080, use_reloader=False)