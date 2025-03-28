import pickle
from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])    
def predict_api():
    print('data')
    data = request.json['data']
    print(data)
    print(data['MedInc'])
    
    qualitydata={
        'MedInc': data['MedInc'],
        'HouseAge': data['HouseAge'],
        'AveRooms': data['AveRooms'],
        'Latitude': data['Latitude'],
        'RPH': data['AveRooms']/data['AveOccup'],
    }
    print(qualitydata)
    
    # Convert input data to numpy array and reshape
    new_data = np.array(list(qualitydata.values())).reshape(1, -1)
    
    # Scale the data
    new_data = scalar.transform(new_data)
    
    # Make prediction
    output = regmodel.predict(new_data)
    
    print(output[0])
    return jsonify({'prediction': float(output[0])})

if __name__ == '__main__':
    app.run(debug=True)
