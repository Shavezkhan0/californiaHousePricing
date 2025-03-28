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
    data = request.json['data']
    print(data)
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

@app.route('/predict', methods=['POST'])
def predict():
    # data= [float(x) for x in request.form.values()]
    # final_input=scalar.transform(np.array(data).reshape(1,-1))
    # print(final_input)
    # output=regmodel.predict(final_input)[0]
    
    try:
        MedInc = float(request.form['MedInc'])
        HouseAge = float(request.form['HouseAge'])
        AveRooms = float(request.form['AveRooms'])
        AveOccup = float(request.form['AveOccup'])
        Latitude = float(request.form['Latitude'])
        RPH = AveRooms / AveOccup

        qualitydata = {
            'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'Latitude': Latitude,
            'RPH': RPH,
        }

        # Convert input data to numpy array and reshape
        new_data = np.array(list(qualitydata.values())).reshape(1, -1)
        # Scale the data
        new_data = scalar.transform(new_data)
        # Make prediction
        output = regmodel.predict(new_data)

        return render_template('home.html', prediction_text='Predicted House Price is : ${:.2f}'.format(output[0]))
    except Exception as e:
        return render_template('home.html', prediction_text='Error: {}'.format(str(e)))


if __name__ == '__main__':
    app.run(debug=True)
