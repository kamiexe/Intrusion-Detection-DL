import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Define features (matches mock dataset)
    numerical_features = ['duration', 'src_bytes', 'dst_bytes', 'count']
    categorical_features = ['protocol_type', 'service', 'flag']
    
    # Encode categorical data
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_data = encoder.fit_transform(df[categorical_features]).toarray()
    
    # Scale numerical data
    scaler = StandardScaler()
    numerical_data = scaler.fit_transform(df[numerical_features])
    
    # Combine features
    X = np.hstack([numerical_data, categorical_data])
    
    # Binary labels
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test