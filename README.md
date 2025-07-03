# Medical Condition Predictor & Drug Recommendation System

## Master's Capstone Project
**Student:** Kusal Venkata Sai Shravanth Gelli  
**Project Advisor:** Dr. Anu Bourgeous  
**Institution:** Georgia State University  

## Project Overview
An AI-powered web application that predicts medical conditions from patient-described symptoms and provides drug recommendations based on real patient reviews and ratings. The system utilizes advanced Natural Language Processing with BERT (Bidirectional Encoder Representations from Transformers) for high-accuracy medical condition classification.

## Key Features
- **BERT-Powered Predictions** using DistilBERT for sequence classification with 60-95% confidence scores
- **Fallback TF-IDF Model** ensures reliability when BERT is unavailable
- **Real Patient Reviews** with interactive modal interface showing actual patient experiences
- **Smart Drug Recommendations** based on 161K+ medical records with ratings and usefulness scores
- **Modern Responsive UI** with Bootstrap 5 and professional medical styling
- **Comprehensive Medical Disclaimers** for user safety and legal compliance

## Technical Architecture

### Machine Learning Models
- **Primary Model:** DistilBERT (distilbert-base-uncased)
  - Fine-tuned on medical drug review dataset
  - Achieves 60-95% confidence scores
  - Superior contextual understanding of medical descriptions
- **Fallback Model:** TF-IDF + Passive Aggressive Classification
  - Ensures system reliability
  - Fast processing for backup scenarios

### Technology Stack
- **Backend:** Flask (Python web framework)
- **Machine Learning:** PyTorch, Transformers, scikit-learn
- **Natural Language Processing:** NLTK, BeautifulSoup
- **Data Processing:** pandas, NumPy
- **Frontend:** Bootstrap 5, HTML5, CSS3, JavaScript
- **Database:** CSV-based patient review dataset

### Dataset
- **Source:** Drugs.com patient reviews
- **Size:** 161,297 patient reviews
- **Medical Conditions:** 4 primary conditions
  - High Blood Pressure
  - Depression
  - Diabetes Type 2
  - Birth Control
- **Data Features:** Patient reviews, drug ratings, usefulness scores, timestamps

## System Performance
- **Model Accuracy:** High accuracy with BERT's advanced NLP capabilities
- **Confidence Scores:** 60-95% for BERT predictions
- **Processing Time:** ~1-3 seconds per prediction (including drug recommendations)
- **Scalability:** Handles concurrent users with session management

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- 4GB+ RAM (for BERT model)

### Installation Steps
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure model files are present in `model/` directory
4. Run the application: `python app.py`
5. Access at `http://localhost:8080`

### Model Files Required
- `model/distilbert-drug-review-model/` - Fine-tuned BERT model
- `model/distilbert-drug-review-tokenizer/` - BERT tokenizer
- `model/label_encoder.pkl` - Label encoder for condition mapping
- `model/passmodel.pkl` - Fallback TF-IDF model
- `model/tfidfvectorizer.pkl` - TF-IDF vectorizer

## User Interface Features
- **Professional Medical Styling** with healthcare-appropriate color schemes
- **Interactive Drug Cards** with click-to-view patient reviews
- **Confidence Visualization** with progress bars and metrics
- **Mobile-Responsive Design** for accessibility across devices
- **Real-time Processing Indicators** showing prediction progress

## Academic Contributions
This Master's Capstone Project demonstrates:
- **Advanced NLP Implementation** with state-of-the-art BERT architecture
- **Healthcare AI Applications** with real-world medical data
- **Full-Stack Development** from data processing to web deployment
- **User Experience Design** for medical applications
- **Professional Software Engineering** practices and documentation

## Safety & Legal Compliance
- **Medical Disclaimers** prominently displayed throughout the application
- **Educational Purpose** clearly stated for academic research
- **Professional Consultation** recommended for all medical decisions
- **Data Privacy** considerations for patient review information

## Future Enhancements
- **Additional Medical Conditions** expansion of supported conditions
- **Enhanced Drug Information** including side effects and contraindications
- **User Accounts** for prediction history and personalized recommendations
- **API Integration** for external medical databases
- **Multi-language Support** for broader accessibility

## File Structure Note
Due to GitHub file size limitations, some large files are not included in this repository:
- Model files (256 MB BERT model)

The complete working version is available in the local development environment.
All application code and architecture are preserved in this repository.

## License
Academic project for educational purposes only. Not intended for commercial medical use.

## Acknowledgments
Special thanks to Dr. Anu Bourgeous for invaluable guidance and support throughout this Master's Capstone Project. This work represents the culmination of advanced study in Machine Learning and Natural Language Processing applied to healthcare informatics.

---

**Note:** This system is designed for educational and research purposes. Always consult with qualified healthcare professionals for medical advice and treatment decisions.
