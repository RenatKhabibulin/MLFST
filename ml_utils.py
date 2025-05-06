import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Temporarily commenting out TensorFlow imports to fix compatibility issues
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def train_linear_regression(X, y):
    """
    Train a linear regression model and return predictions and metrics
    
    Parameters:
    - X: features array
    - y: target array
    
    Returns:
    - model: trained LinearRegression model
    - y_pred: predictions
    - metrics: dictionary with MAE, MSE, RMSE, R2
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    return model, y_pred, metrics

def train_logistic_regression(X, y):
    """
    Train a logistic regression model and return predictions and metrics
    
    Parameters:
    - X: features array
    - y: target array (binary)
    
    Returns:
    - model: trained LogisticRegression model
    - y_pred: predictions
    - metrics: dictionary with accuracy and classification report
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return model, y_pred, metrics

def train_random_forest(X, y):
    """
    Train a random forest classifier and return predictions and metrics
    
    Parameters:
    - X: features array
    - y: target array (categorical)
    
    Returns:
    - model: trained RandomForestClassifier model
    - y_pred: predictions
    - metrics: dictionary with accuracy and classification report
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': model.feature_importances_
    }
    
    return model, y_pred, metrics

def train_svm(X, y):
    """
    Train an SVM classifier and return predictions and metrics
    
    Parameters:
    - X: features array
    - y: target array (categorical)
    
    Returns:
    - model: trained SVC model
    - y_pred: predictions
    - metrics: dictionary with accuracy and classification report
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return model, y_pred, metrics

def train_naive_bayes(X, y):
    """
    Train a Naive Bayes classifier and return predictions and metrics
    
    Parameters:
    - X: features array
    - y: target array (categorical)
    
    Returns:
    - model: trained GaussianNB model
    - y_pred: predictions
    - metrics: dictionary with accuracy and classification report
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return model, y_pred, metrics

def perform_kmeans_clustering(X, n_clusters=3):
    """
    Perform K-means clustering and return cluster assignments
    
    Parameters:
    - X: features array
    - n_clusters: number of clusters
    
    Returns:
    - model: trained KMeans model
    - labels: cluster assignments
    - metrics: dictionary with inertia
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
    
    # Calculate metrics
    metrics = {
        'inertia': model.inertia_,
        'cluster_centers': scaler.inverse_transform(model.cluster_centers_)
    }
    
    return model, labels, metrics

def perform_pca(X, n_components=2):
    """
    Perform PCA dimensionality reduction
    
    Parameters:
    - X: features array
    - n_components: number of principal components
    
    Returns:
    - model: trained PCA model
    - X_pca: transformed data
    - metrics: dictionary with explained variance
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = PCA(n_components=n_components)
    X_pca = model.fit_transform(X_scaled)
    
    # Calculate metrics
    metrics = {
        'explained_variance_ratio': model.explained_variance_ratio_,
        'explained_variance': model.explained_variance_,
        'components': model.components_
    }
    
    return model, X_pca, metrics

def create_cnn_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a CNN model for image classification
    This is a simple demonstration model for fire detection
    
    Parameters:
    - input_shape: shape of input images (height, width, channels)
    - num_classes: number of output classes
    
    Returns:
    - model: Information about model architecture (TensorFlow disabled in this environment)
    """
    # TensorFlow is currently disabled due to compatibility issues
    # Returning a description of the architecture instead
    
    architecture = {
        "model_type": "CNN",
        "input_shape": input_shape,
        "num_classes": num_classes,
        "layers": [
            {"type": "Conv2D", "filters": 32, "kernel_size": (3, 3), "activation": "relu"},
            {"type": "MaxPooling2D", "pool_size": (2, 2)},
            {"type": "Conv2D", "filters": 64, "kernel_size": (3, 3), "activation": "relu"},
            {"type": "MaxPooling2D", "pool_size": (2, 2)},
            {"type": "Conv2D", "filters": 128, "kernel_size": (3, 3), "activation": "relu"},
            {"type": "MaxPooling2D", "pool_size": (2, 2)},
            {"type": "Flatten"},
            {"type": "Dense", "units": 128, "activation": "relu"},
            {"type": "Dropout", "rate": 0.5},
            {"type": "Dense", "units": num_classes, "activation": "softmax"}
        ]
    }
    
    return architecture

def create_lstm_model(input_shape=(10, 5), num_classes=2):
    """
    Create an LSTM model for time series prediction
    This is a simple demonstration model for predicting fire spread
    
    Parameters:
    - input_shape: shape of input sequences (timesteps, features)
    - num_classes: number of output classes
    
    Returns:
    - model: Information about model architecture (TensorFlow disabled in this environment)
    """
    # TensorFlow is currently disabled due to compatibility issues
    # Returning a description of the architecture instead
    
    architecture = {
        "model_type": "LSTM",
        "input_shape": input_shape,
        "num_classes": num_classes,
        "layers": [
            {"type": "LSTM", "units": 64, "return_sequences": True},
            {"type": "LSTM", "units": 32},
            {"type": "Dense", "units": 32, "activation": "relu"},
            {"type": "Dropout", "rate": 0.3},
            {"type": "Dense", "units": num_classes, "activation": "softmax"}
        ]
    }
    
    return architecture
