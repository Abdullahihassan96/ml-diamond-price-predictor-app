import pandas as pd
from datetime import datetime
import os

def log_prediction(input_data, prediction):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        **input_data,
        "prediction": prediction
    }
    
    log_file = "monitoring/predictions_log.csv"
    
    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Append to log file
    pd.DataFrame([log_entry]).to_csv(
        log_file,
        mode='a',
        header=not os.path.exists(log_file),
        index=False
    )