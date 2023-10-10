import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('datos01.csv', delimiter=';')

# Preprocess data
daily_columns = [f"D{i:02}" for i in range(1, 32)]
for column in daily_columns:
    data[column].fillna(data[column].mean(), inplace=True)
data['monthly_avg'] = data[daily_columns].mean(axis=1)
data_cleaned = data.drop(daily_columns + [f"V{i:02}" for i in range(1, 32)], axis=1)

# Feature Engineering
data_cleaned['year_month'] = data_cleaned['ANO']*100 + data_cleaned['MES']

# Features and Target
X = data_cleaned[['PROVINCIA', 'MUNICIPIO', 'ESTACION', 'MAGNITUD', 'ANO', 'MES', 'year_month']]
y = data_cleaned['monthly_avg']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building and Prediction
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)
