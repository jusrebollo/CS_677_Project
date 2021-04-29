"""
The date is for Hunter Mountain NY. Using historical data, can we accurately
predict 2021 and determine the best features for determining snowfall.
"""

import numpy
import xlrd
import pandas as pd
import numpy as np
from array import *
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
import seaborn as sns
from scipy import stats

import sklearn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix

# split train into 2010-2020 and test into 2021

# load data
from sklearn.tree import DecisionTreeClassifier

snow_data = pd.read_csv(r'/Users/jreb/PycharmProjects/rebollo_final_project/'
                        r'Hunter_Mountain_data.csv', low_memory=False)

snow_data = snow_data.drop(columns=['lat', 'lon', 'sea_level', 'grnd_level',
                                    'rain_3h','snow_3h', 'timezone'])

# fill Nan with 0
snow_data.fillna(value=0, method=None, axis=None, inplace=True)

# create simplifed date range for graphing
# 1/1/2021 - 4/6/2021
dates = pd.date_range('1/1/2021', periods=2690, freq='H')

# correlation matrix
snow_data.corr()
corrMatrix = snow_data.corr()

# plot correlation matrix
sns.heatmap(corrMatrix, annot=True)
plt.show()

X= snow_data[['temp_max', 'humidity',
               'clouds_all', 'temp_min', 'wind_speed']].values
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(snow_data['snow_1h'].values)


X_train = X[:99556]

X_test = X[99556:]
y_train = y[:99556]
y_test = y[99556:]

# ______________________________________________________________________________
# NB
NB_classifier = GaussianNB().fit(X_train, y_train)

prediction_nb = NB_classifier.predict(X_test)

accuracy_nb = round((accuracy_score(y_test, prediction_nb) * 100), 2)

confusion_nb = (confusion_matrix(y_test, prediction_nb))
print("NB accuracy")
print(accuracy_nb)

# ______________________________________________________________________________
# Decision Tree
tree_classifier = sklearn.tree.DecisionTreeClassifier()
tree_classifier = tree_classifier.fit(X_train, y_train)
prediction_tree = tree_classifier.predict(X_test)

results = le.inverse_transform(prediction_tree)

accuracy_tree = round((accuracy_score(y_test, prediction_tree) * 100), 2)
confusion_tree = (confusion_matrix(y_test, prediction_tree))

print("Decision Tree Classifier")
print("Accuracy")
print(accuracy_tree)

TN_ct = confusion_tree[0][0]
FN_ct = confusion_tree[1][0]
TP_ct = confusion_tree[1][1]
FP_ct = confusion_tree[0][1]

# true_pos_ct = round(((TP_ct / (TP_ct + FN_ct)) * 100), 2)
true_neg_ct = round(((TN_ct / (TN_ct + FP_ct)) * 100), 2)

# ______________________________________________________________________________
# Random Forest

n_best = 3
d_best = 5
model_best = RandomForestClassifier(n_estimators=n_best, max_depth=d_best,
                                    criterion='entropy')
model_best.fit(X_train, y_train)
prediction_rf_best = model_best.predict(X_test)
confusion_rf_best = (confusion_matrix(y_test, prediction_rf_best))
accuracy_rf_best = round((accuracy_score(y_test, prediction_rf_best) * 100), 2)

print("Random Forest Classifier")
print("Accuracy")
print(accuracy_rf_best)

results1 = le.inverse_transform(prediction_rf_best)

best_k = 3
knn1 = KNeighborsClassifier(n_neighbors=best_k)
knn1.fit(X_train, y_train)

y_pred_best = knn1.predict(X_test)
accruacy_knn = round((accuracy_score(y_test, y_pred_best) * 100), 2)
print("Accuracy for knn:", accruacy_knn)

results1 = le.inverse_transform(prediction_rf_best)

# inverse transform to get desired output decoded data
y_test = le.inverse_transform(y_test)
y_pred_best = le.inverse_transform(y_pred_best)
prediction_tree = le.inverse_transform(prediction_tree)
prediction_nb = le.inverse_transform(prediction_nb)


# plots
# decision_tree
fig, ax = plt.subplots()
ax.plot(dates, y_test, '-ok', label="Actual Values")
ax.plot(dates, prediction_tree, '-o', label="Decision Tree Prediction")
plt.title("Decision Tree")
plt.xlabel("Days of the year 2021")
plt.ylabel("Snowfall in Inches")
ax.legend()
plt.show()

# knn
fig, ax = plt.subplots()
ax.plot(dates, y_test, '-ok', label="Actual Values")
ax.plot(dates, y_pred_best, '-o', label="Knn")
plt.title("knn")
plt.xlabel("Days of the year 2021")
plt.ylabel("Snowfall in Inches")
ax.legend()
plt.show()

# Random forest
fig, ax = plt.subplots()
ax.plot(dates, y_test, '-ok', label="Actual Values")
ax.plot(dates, prediction_rf_best, '-o', label="Random Forest")
plt.title("Random Forest")
plt.xlabel("Days of the year 2021")
plt.ylabel("Snowfall in Inches")
ax.legend()
plt.show()

# NB
fig, ax = plt.subplots()
ax.plot(dates, y_test, '-ok', label="Actual Values")
ax.plot(dates, prediction_nb, '-o', label="Naive Bayesian")
plt.title("Naive Bayesian")
plt.xlabel("Days of the year 2021")
plt.ylabel("Snowfall in Inches")
ax.legend()
plt.show()

