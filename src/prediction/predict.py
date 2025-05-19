from src.monitoring.logging import log_prediction

def make_prediction(model, input_data):
    # Preprocess input
    processed_data = preprocess(input_data)
    
    # Make prediction
    prediction = model.predict([processed_data])[0]
    
    # Log prediction
    log_prediction(input_data, prediction)
    
    return prediction