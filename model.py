# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 00:09:56 2020

@author: Ankit Raina
"""

"""
Importing packages
"""
import pandas as pd
import pickle

"""
    Function to create a serialized object of the ML model
"""
def serialize_model():
    # Reading the data set
    data = pd.read_csv('hour.csv')
    # Pre-processing data
    data = preprocess_data(data)
    # Getting the best features
    features = select_features()
    # Create model
    target = "cnt"
    model = create_model(data, features, target)
    # Saving the model to disk
    pickle.dump(model, open('model.pkl','wb'))

"""
    Function to pre-process data
"""
def preprocess_data(data):  
    
    # Setting categorical variables as category type 
    # and renaming the labels for the categories
    categorical_variables = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
    
    for var in categorical_variables:
        data[var] = data[var].astype('category')
        
    # Season
    # 1:winter, 2:spring, 3:summer, 4:fall
    season_labels = ["Winter", "Spring", "Summer", "Fall"]
    data["season"] = data["season"].cat.rename_categories(season_labels)
    
    # Month
    #1: Jan, 2: Feb, 3: Mar, 4: Apr, 5:May, 6: Jun, 7: Jul, 8: Aug, 9: Sep, 10: Oct, 11: Nov, 12: Dec
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    data["mnth"] = data["mnth"].cat.rename_categories(month_labels)
    
    # Day of the Week
    #0: Sunday, 1: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday, 6: Saturday
    week_labels = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
    data["weekday"] = data["weekday"].cat.rename_categories(week_labels)
    
    # Converting binary categorical features into numeric Panda Series
    data["workingday"] = pd.to_numeric(data['workingday'])
    data["holiday"] = pd.to_numeric(data['holiday'])
    
    # Creating dummy variables for categorical variables with multiple categories
    mul_cat_var = categorical_variables = ["season", "mnth", "hr", "weekday", "weathersit"]
    
    for var in mul_cat_var:
        dummies = pd.get_dummies(data[var], prefix=var, dummy_na=True)
        data = data.join(dummies)
    
    data.drop(columns = categorical_variables, inplace=True)
    
    return data

"""
    Function to select the best features
"""
def select_features(): 
    # Best features based on GINI impurity in Extra Tree Regressor algorithm found in our analysis
    features_to_select = ['workingday', 'temp', 'hum', 'windspeed', 'season_Winter',
                          'season_Spring', 'season_Fall', 'mnth_Mar', 'mnth_Apr', 'mnth_May',
                          'mnth_Jun', 'mnth_Jul', 'mnth_Aug', 'mnth_Sep', 'mnth_Oct',
                          'mnth_Nov', 'mnth_Dec', 'hr_0.0', 'hr_1.0', 'hr_2.0', 'hr_3.0',
                          'hr_4.0', 'hr_5.0', 'hr_6.0', 'hr_7.0', 'hr_8.0', 'hr_9.0',
                          'hr_10.0', 'hr_11.0', 'hr_12.0', 'hr_13.0', 'hr_14.0', 'hr_15.0',
                          'hr_16.0', 'hr_17.0', 'hr_18.0', 'hr_19.0', 'hr_20.0', 'hr_21.0',
                          'hr_22.0', 'hr_23.0', 'weekday_Sun', 'weekday_Mon', 'weekday_Tue',
                          'weekday_Wed', 'weekday_Thu', 'weekday_Fri', 'weekday_Sat',
                          'weathersit_1.0', 'weathersit_2.0', 'weathersit_3.0']
    return features_to_select

"""
    Function that creates the ML model object
"""
def create_model(data, features, target):    
    X = data.loc[:, data.columns.isin(features)]
    y = data.loc[:, target]

    # From our analyis, the best model is Gradient Boosting Regressor having hyper-parameters
    # learning_rate=0.5,
    # loss='huber'
    # n_estimators=99
    from  sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(random_state=0, learning_rate=0.5, loss='huber', n_estimators=99)
    model.fit(X, y)
    
    return model

"""
    Main function
"""
if __name__ == "__main__":
    serialize_model()