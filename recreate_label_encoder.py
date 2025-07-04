from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Recreate the label encoder with current numpy version
label_encoder = LabelEncoder()

# Based on your training, these should be your classes:
classes = ['Birth Control', 'Depression', 'Diabetes, Type 2', 'High Blood Pressure']
label_encoder.fit(classes)

# Save with current numpy/joblib versions
print("Recreating label encoder with current numpy version...")
joblib.dump(label_encoder, 'model/label_encoder_fixed.pkl')
print("Label encoder recreated successfully!")

# Test loading
try:
    test_encoder = joblib.load('model/label_encoder_fixed.pkl')
    print(f"Test load successful! Classes: {test_encoder.classes_}")
except Exception as e:
    print(f"Test load failed: {e}")
