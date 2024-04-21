from src.preprocessing import DataProcessing
import pickle
import os

def predict(X):
    # Load the model and scaler from the saved file
    with open(r".\model\model.pickle", 'rb') as f:
        print("Model Imported")
        model, label_encoders, scaler = pickle.load(f)
    dataprocess = DataProcessing()
    dataprocess.cleaning_steps(X)               # Perform Cleaning
    dataprocess.perform_feature_engineering(X)  # Perform Feature Engineering

    # Label Encoding
    for column, label_encoder in label_encoders.items():
        X[column] = label_encoder.transform(X[column])

    X = scaler.transform(X)  # Standardize
    pred = model.predict(X)  # Predict time of delivery
    return pred
