import numpy as np # linear algebra(manipulating arrays)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#-----------------------------------------------------------------------------#

real_estate = pd.read_csv('/Users/me-mac/Desktop/study/AI/Moheb/Real estate.csv')
x = real_estate.iloc[:, :-1].values
y = real_estate.iloc[:,7:].values
real_estate.head()

plt.plot(real_estate)

df_summary=real_estate.describe()
df_summary

#--- identifing the NaN then replace it by mean ---#
x = real_estate.iloc[:, :-1].values
y = real_estate.iloc[:, -1:].values

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer = imputer.fit(x)
imputer = imputer.fit(y)

x = imputer.transform(x)
y = imputer.transform(y)

#--- start to learn the machine ---#

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
#// len(x_train) -> 38, len(x_test) -> 10

#--- make some of linear regression & statistics & prediction ---#

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

# relation for Iris as ma matrix
real_estate.corr()

#--- calc the MSE ---#
mean_squared_error(y_test, y_pred)

# Data Visualization

plt.scatter(x_test, y_test, color="black")
plt.scatter(x_train, y_train)

plt.plot(x_test, y_pred, color="red", linewidth=2)

