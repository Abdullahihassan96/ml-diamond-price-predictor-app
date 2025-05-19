import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime
import logging
from src.monitoring.logging import log_prediction  # New import for MLOps

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Simplified Path Handling for Streamlit Cloud ---
@st.cache_resource
def load_assets():
    """Load model and encoders from the models/ directory with version control support"""
    try:
        # Load model (now from production folder)
        with open('models/production/diamond_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load encoders
        encoders = {}
        encoder_cols = ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']
        
        for col in encoder_cols:
            with open(f'models/encoders/{col}_encoder.pkl', 'rb') as f:
                encoders[col] = pickle.load(f)
        
        logger.info("Successfully loaded model and encoders")
        return model, encoders
        
    except Exception as e:
        logger.error(f"Failed to load model files: {str(e)}", exc_info=True)
        st.error(f"Failed to load model files: {str(e)}")
        st.error("Please check if all model files exist in the models/ folder")
        st.stop()

# Load assets once when app starts
model, encoders = load_assets()

# --- Prediction Function with Logging ---
def make_prediction(input_data):
    """Make prediction and log results"""
    try:
        # Transform categorical features
        encoded_data = input_data.copy()
        for col in ['Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']:
            encoded_data[col] = encoders[col].transform([encoded_data[col]])[0]
        
        # Convert to DataFrame for model
        prediction_input = pd.DataFrame([encoded_data.values()], 
                                      columns=encoded_data.keys())
        
        # Predict
        price = model.predict(prediction_input)[0]
        
        # Log prediction
        log_prediction(
            input_data=input_data,
            prediction=price,
            model_version="1.0"  # Could be dynamic from DVC
        )
        
        return price
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise

# --- Streamlit UI ---
st.title("💎 Diamond Price Predictor (MLOps Version)")
st.write("Enter the details below to get a predicted price.")

# User inputs
carat = st.number_input("Carat Weight", min_value=0.1, max_value=5.0, 
                       step=0.01, value=1.0, help="Weight of the diamond in carats")

# Dynamic select boxes from encoder classes
cut = st.selectbox("Cut Quality", encoders['Cut'].classes_, 
                  help="Quality of the diamond's cut")
color = st.selectbox("Color Grade", encoders['Color'].classes_,
                    help="Diamond color grade (D is best)")
clarity = st.selectbox("Clarity Grade", encoders['Clarity'].classes_,
                      help="Presence of inclusions/flaws")
polish = st.selectbox("Polish Quality", encoders['Polish'].classes_)
symmetry = st.selectbox("Symmetry Grade", encoders['Symmetry'].classes_)
report = st.selectbox("Report Type", encoders['Report'].classes_,
                     help="Grading report authority")

# Predict button
if st.button("Predict Price"):
    with st.spinner("Calculating..."):
        try:
            # Prepare input data (now as dict for better logging)
            input_data = {
                "carat": carat,
                "cut": cut,
                "color": color,
                "clarity": clarity,
                "polish": polish,
                "symmetry": symmetry,
                "report": report,
                "timestamp": datetime.now().isoformat()
            }
            
            # Predict
            price = make_prediction(input_data)
            
            # Display
            st.success(f"Estimated Diamond Price: **${price:,.2f}**")
            st.balloons()
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error("Please check your inputs and try again.")

# --- Model Version Info (Optional MLOps Feature) ---
st.sidebar.markdown("### Model Information")
st.sidebar.write(f"Model loaded: `diamond_price_model.pkl`")
st.sidebar.write("Last updated: 2023-11-15")  # Could be dynamic from DVC