import mlflow
import numpy as np
import mlflow.sklearn
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model():
    
    # Load data
    df = pd.read_csv("data/raw/diamonds.csv")
    
    # Preprocessing
    with open('models/encoders/Cut_encoder.pkl', 'rb') as f:
        cut_enc = pickle.load(f)
    with open('models/encoders/Color_encoder.pkl', 'rb') as f:
        color_enc = pickle.load(f)
    with open('models/encoders/Clarity_encoder.pkl', 'rb') as f:
        clarity_enc = pickle.load(f)
    with open('models/encoders/Polish_encoder.pkl', 'rb') as f:
        polish_enc = pickle.load(f)
    with open('models/encoders/Symmetry_encoder.pkl', 'rb') as f:
        symmetry_enc = pickle.load(f)
    with open('models/encoders/Report_encoder.pkl', 'rb') as f:
        report_enc = pickle.load(f)

    # Prepare features/target
    X = df.drop(columns=['Price'])
    y = df['Price']

    X['Cut'] = cut_enc.transform(X['Cut'])
    X['Color'] = color_enc.transform(X['Color'])
    X['Clarity'] = clarity_enc.transform(X['Clarity'])
    X['Polish'] = polish_enc.transform(X['Polish'])
    X['Symmetry'] = symmetry_enc.transform(X['Symmetry'])
    X['Report'] = report_enc.transform(X['Report'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    import joblib
    joblib.dump(model, "models/production/diamond_price_model.pkl")
    
    return model

# if __name__ == "__main__":
#     train_model()

