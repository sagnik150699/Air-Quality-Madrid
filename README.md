# Air Quality Prediction in Madrid
This repository contains a regression-based model to predict the monthly average air quality in Madrid using historical data. The data includes daily air quality measurements and metadata related to the measurement stations.

## Usage
Run the Jupyter Notebook air_quality_prediction.ipynb to train and evaluate the regression model

##Dataset Overview
The dataset includes:

Metadata about the location and measurement point: PROVINCIA, MUNICIPIO, ESTACION, MAGNITUD, and PUNTO_MUESTREO.
Year and month of the measurement: ANO and MES.
Daily measurements of air quality from D01 to D31.
Verification or validation flags for the corresponding daily measurements from V01 to V31.

## Model
The model used is a Linear Regression model from the scikit-learn library. The target variable is the monthly average air quality derived from the daily measurements.

## Results
The model achieved an RMSE of approximately 55.66 on the test set, indicating the average error of the model's predictions.
