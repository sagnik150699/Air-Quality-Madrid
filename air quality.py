

import pandas as pd
# Importing the dataset and preprocessing
dataset = pd.read_csv('C:\\Users\\DATA\Desktop\\AIR QUALITY MADRID\\csvs_per_year\madrid_2002.csv')

dataset.info()
datasets = dataset.drop(['date','station'],axis=1)
dataset.fillna(dataset.mean(), inplace=True)
datasets['Target'] = datasets['CO']
features = datasets
labels = datasets['CO'].values
features = features.values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(X_test)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)

