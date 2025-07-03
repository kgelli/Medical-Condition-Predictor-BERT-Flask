import os
import torch
import joblib
import logging
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BERTHandler:
    def __init__(self, model_path, tokenizer_path, label_encoder_path):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_encoder_path = label_encoder_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.is_loaded = False
        
    def load_models(self):
        """Load BERT models with error handling"""
        try:
            # Check if files exist
            if not os.path.exists(self.model_path):
                logger.warning("BERT model directory not found")
                return False
            
            if not os.path.exists(self.label_encoder_path):
                logger.warning("BERT label encoder not found")
                return False
            
            # Load models
            logger.info("Loading BERT model...")
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.tokenizer_path)
            self.label_encoder = joblib.load(self.label_encoder_path)
            
            self.is_loaded = True
            logger.info("BERT model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            self.is_loaded = False
            return False
    
    def predict(self, text, clean_text_func):
        """Make prediction using BERT model"""
        try:
            if not self.is_loaded:
                return None, 0.0, "BERT model not loaded"
            
            # Clean text
            cleaned_text = clean_text_func(text)
            
            if not cleaned_text.strip():
                return None, 0.0, "Text too short"
            
            # Tokenize
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Get prediction and confidence
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Convert to condition name
            condition = self.label_encoder.inverse_transform([predicted_class])[0]
            
            return condition, confidence, None
            
        except Exception as e:
            logger.error(f"BERT prediction error: {str(e)}")
            return None, 0.0, str(e)
    
    def is_available(self):
        """Check if BERT model is available"""
        return self.is_loaded
