import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# read the datasets.csv
audit_risk = pd.read_csv('/Users/me-mac/Desktop/study/AI/Moheb/audit_risk.csv')

x = audit_risk.iloc[:, :-1].values
y = audit_risk.iloc[:, -1:].values


# parsin from string to number then use LabelEncoder
lc_x = LabelEncoder()
x[:,1] =  lc_x.fit_transform(x[:,1])


#change nan values to mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:,])
x = imputer.transform(x[:,])


# splitting X and y into training and testing sets for prediction 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# make numbers closest together by scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# start training by using fit
mlp = MLPClassifier()
mlp.fit(x_train, y_train)


# making predictions on the testing set
y_pred = mlp.predict(x_test)


# comparing (y_test) with (y_pred)
print(" neural_network model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)




