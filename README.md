# ML-Model-Python-Flask-Service
Python flask service that exposes ML model used for predicting Bike Sharing Demand.

Data: Bike Sharing data provided by Capital BikeShare.

Serialized the Gradient Boosting Regressor model using pickle library in Python and created simple web UI 
and a Python flask service making use of the serialized model exposing the model to predict new unseen test data.
I have hosted the service and UI on Heroku, having the below url:
https://ml-bike-sharing-prediction-api.herokuapp.com/
