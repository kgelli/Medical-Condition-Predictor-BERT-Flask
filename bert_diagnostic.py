# BERT DIAGNOSTIC SCRIPT
# This will help us identify exactly what's failing

import sys
import os

print("=== BERT DIAGNOSTIC REPORT ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Test 1: Basic imports
print("1. TESTING BASIC IMPORTS...")
try:
    import numpy
    print(f"   ✅ numpy {numpy.__version__}")
except Exception as e:
    print(f"   ❌ numpy failed: {e}")
    
try:
    import torch
    print(f"   ✅ torch {torch.__version__}")
except Exception as e:
    print(f"   ❌ torch failed: {e}")

try:
    import pandas
    print(f"   ✅ pandas {pandas.__version__}")
except Exception as e:
    print(f"   ❌ pandas failed: {e}")

try:
    import sklearn
    print(f"   ✅ sklearn {sklearn.__version__}")
except Exception as e:
    print(f"   ❌ sklearn failed: {e}")

print()

# Test 2: Transformers step-by-step
print("2. TESTING TRANSFORMERS STEP BY STEP...")

try:
    import transformers
    print(f"   ✅ transformers {transformers.__version__} imported")
except Exception as e:
    print(f"   ❌ transformers import failed: {e}")
    print("   🛑 STOPPING - transformers won't import")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    print("   ✅ AutoTokenizer imported")
except Exception as e:
    print(f"   ❌ AutoTokenizer failed: {e}")

try:
    from transformers import DistilBertTokenizer
    print("   ✅ DistilBertTokenizer imported")
except Exception as e:
    print(f"   ❌ DistilBertTokenizer failed: {e}")

try:
    from transformers import DistilBertForSequenceClassification
    print("   ✅ DistilBertForSequenceClassification imported")
except Exception as e:
    print(f"   ❌ DistilBertForSequenceClassification failed: {e}")

print()

# Test 3: Check model files
print("3. CHECKING MODEL FILES...")
model_dir = "model/distilbert-drug-review-model"
tokenizer_dir = "model/distilbert-drug-review-tokenizer"
label_file = "model/label_encoder.pkl"

if os.path.exists(model_dir):
    print(f"   ✅ Model directory exists: {model_dir}")
    files = os.listdir(model_dir)
    print(f"   📁 Files in model dir: {files}")
else:
    print(f"   ❌ Model directory missing: {model_dir}")

if os.path.exists(tokenizer_dir):
    print(f"   ✅ Tokenizer directory exists: {tokenizer_dir}")
    files = os.listdir(tokenizer_dir)
    print(f"   📁 Files in tokenizer dir: {files}")
else:
    print(f"   ❌ Tokenizer directory missing: {tokenizer_dir}")

if os.path.exists(label_file):
    print(f"   ✅ Label encoder exists: {label_file}")
else:
    print(f"   ❌ Label encoder missing: {label_file}")

print()

# Test 4: Try loading model components individually
print("4. TESTING MODEL COMPONENT LOADING...")

if os.path.exists(model_dir):
    try:
        from transformers import DistilBertForSequenceClassification
        print("   🔄 Attempting to load BERT model...")
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        print("   ✅ BERT model loaded successfully!")
    except Exception as e:
        print(f"   ❌ BERT model loading failed: {e}")
        print(f"   📝 Error type: {type(e).__name__}")

if os.path.exists(tokenizer_dir):
    try:
        from transformers import DistilBertTokenizer
        print("   🔄 Attempting to load tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_dir)
        print("   ✅ Tokenizer loaded successfully!")
    except Exception as e:
        print(f"   ❌ Tokenizer loading failed: {e}")

if os.path.exists(label_file):
    try:
        import joblib
        print("   🔄 Attempting to load label encoder...")
        label_encoder = joblib.load(label_file)
        print("   ✅ Label encoder loaded successfully!")
        print(f"   📋 Label classes: {label_encoder.classes_}")
    except Exception as e:
        print(f"   ❌ Label encoder loading failed: {e}")

print()

# Test 5: Try BERT handler
print("5. TESTING BERT HANDLER...")
try:
    sys.path.append('.')  # Add current directory to path
    from bert_handler import BERTHandler
    print("   ✅ BERTHandler imported successfully")
    
    handler = BERTHandler(model_dir, tokenizer_dir, label_file)
    print("   ✅ BERTHandler instantiated")
    
    print("   🔄 Attempting to load models...")
    success = handler.load_models()
    
    if success:
        print("   🎉 BERTHandler.load_models() SUCCESS!")
        print("   ✅ BERT is working!")
    else:
        print("   ❌ BERTHandler.load_models() returned False")
        
except Exception as e:
    print(f"   ❌ BERTHandler failed: {e}")
    print(f"   📝 Error type: {type(e).__name__}")
    import traceback
    print("   📝 Full traceback:")
    traceback.print_exc()

print()
print("=== DIAGNOSTIC COMPLETE ===")
print()
print("📋 NEXT STEPS:")
print("1. Share the output above")
print("2. We'll identify the exact failure point")
print("3. Apply targeted fix")