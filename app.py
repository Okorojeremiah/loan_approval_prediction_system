from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import json
import joblib

app = Flask(__name__)

# Load the trained model
model_path = 'model/final_model.pkl'
model_dir = os.path.join(os.path.dirname(__file__), model_path)
loaded_model = joblib.load(open(model_dir, 'rb'))

# Load the column set
columns_path = 'data/columns_set.json'
columns_dir = os.path.join(os.path.dirname(__file__), columns_path)
with open(columns_dir, 'r') as f:
    columns_set = json.loads(f.read())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Create DataFrame from JSON
    df = pd.DataFrame(data, index=[0])

    # Ensure columns match the model's expected columns
    df = df.reindex(columns=columns_set['data_columns'], fill_value=0)

    # Make prediction
    prediction = loaded_model.predict(df)
    
    prediction_list = prediction.tolist()

    return jsonify({'prediction': prediction_list})

if __name__ == '__main__':
    app.run(debug=True)
