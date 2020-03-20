import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('html/index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_data = read_input()
    features = extract_features(input_data)
    test_data = np.array([value for key, value in features.items()]).reshape(1, -1)
    prediction = model.predict(test_data)
    
    return render_template('html/index.html', prediction_text='Predicted Bike Demand is {}'.format(math.ceil(prediction)))

"""
    Function to read inputted data from the GUI
"""
def read_input():
    fields = ['hr', 'weekday', 'mnth', 'season', 'weathersit', 'hum', 'temp', 'workingday', 'windspeed']
    input_data = dict()
    
    for field in fields:
        input_data[field] = request.form[field]   
    
    return input_data


"""
    Function to extract features from inputted data
"""
def extract_features(input_data):
    features = dict()
    
    # Working Day
    features['workingday'] = int(input_data['workingday'])
    
    # Temperature
    # The values will be derived via (t-t_min)/(t_max-t_min), 
    # t_min=-8, t_max=+39
    t_min = -8
    t_max = 39
    features['temp'] = (int(input_data['temp']) - t_min)/ (t_max - t_min)
    
    # Humidity
    # We will normalize the inputted humidity by 100 (Max possible value)
    # to create the feature hum
    features['hum'] = int(input_data['hum'])/100
    
    # Wind Speed
    # We will normalize the inputted windspeed by 67 (Max possible value)
    # to create the feature windspeed
    features['windspeed'] = int(input_data['windspeed'])/67
    
    # Season
    season_features = ['season_Winter', 'season_Spring', 'season_Fall']
    
    for season_feature in season_features:
        if input_data['season'] in season_feature:
            features[season_feature] = 1
        else:
            features[season_feature] = 0
    
    # Month
    month_features = ['mnth_Mar', 'mnth_Apr', 'mnth_May',
                      'mnth_Jun', 'mnth_Jul', 'mnth_Aug', 'mnth_Sep', 'mnth_Oct',
                      'mnth_Nov', 'mnth_Dec']
    
    for month_feature in month_features:
        if input_data['mnth'] in month_feature:
            features[month_feature] = 1
        else:
            features[month_feature] = 0
    
    # Hour
    hour_features = ['hr_0.0', 'hr_1.0', 'hr_2.0', 'hr_3.0',
                     'hr_4.0', 'hr_5.0', 'hr_6.0', 'hr_7.0', 'hr_8.0', 'hr_9.0',
                     'hr_10.0', 'hr_11.0', 'hr_12.0', 'hr_13.0', 'hr_14.0', 'hr_15.0',
                     'hr_16.0', 'hr_17.0', 'hr_18.0', 'hr_19.0', 'hr_20.0', 'hr_21.0',
                     'hr_22.0', 'hr_23.0']
    
    for hour_feature in hour_features:
        if input_data['hr'] in hour_feature:
            features[hour_feature] = 1
        else:
            features[hour_feature] = 0
    
    # Day of the Week
    day_features = ['weekday_Sun', 'weekday_Mon', 'weekday_Tue',
                    'weekday_Wed', 'weekday_Thu', 'weekday_Fri', 'weekday_Sat']
    
    for day_feature in day_features:
        if input_data['weekday'] in day_feature:
            features[day_feature] = 1
        else:
            features[day_feature] = 0
            
    # Weather Situation
    weather_features = ['weathersit_1.0', 'weathersit_2.0', 'weathersit_3.0']
    
    for weather_feature in weather_features:
        if input_data['weathersit'] in weather_feature:
            features[weather_feature] = 1
        else:
            features[weather_feature] = 0
            
    return features

    
if __name__ == "__main__":
    app.run(debug=True)# -*- coding: utf-8 -*-

