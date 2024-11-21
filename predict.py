# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:36:42 2024

@author: Sarra Dehili
"""

import pickle 
from flask import Flask, request, jsonify

model_file = 'model_rf.bin'


with open(model_file, 'rb') as f_in :
    dv, model = pickle.load(f_in)
    
    
app = Flask('air_quality')

@app.route('/predict', methods=['POST'])

def predict():
    
    data = request.get_json()
    
    X = dv.transform([data])
    y_pred = model.predict(X)[0]
    
    label_mapping = {
        0: "Good",
        1: "Moderate",
        2: "Unhealthy for SG",
        3: "Unhealthy",
        4: "Very Unhealthy",
        5: "Hazardous"
    }
    
    aqi_class = label_mapping.get(y_pred, "Unknown")

    result = {
        'Predicted_label': int(y_pred),  
        'Air Quality': aqi_class
        }
        
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
