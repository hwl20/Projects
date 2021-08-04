'''
- Note this is a project done with heavy reliance on teaching material from Udemy
- Dataset is retrieved from Kaggle and access to this material can be found at: https://www.kaggle.com/c/bike-sharing-demand/data?select=test.csv
- In this exercise the aim is to clean the data before modelling a multiple linear regression 
to predict the total count of bikes rented out during each hour covered by the test set
- Manage to obtain predictions with R^2 = 0.928 and RMSE = 0.381 
on test dataset whilist omitting the first 3 predictions due to us accounting for autocorrelation
on the demand variable that we're looking to predict
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math

# Importing dataframe
os.listdir()
bikes = pd.read_csv("hour.csv")

# Preliminary Analysis and Feature Selection
bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index','date','casual','registered'], axis = 1)

# Checking misising values
bikes_prep.isnull().sum() # No missing values found

# Visualize the datas using pandas histogram
bikes_prep.hist(rwidth = 0.9)
plt.tight_layout()



# Data visualization
# Visualise the continuous features VS demand
plt.figure()
plt.subplot(2,2,1)
plt.title("Temperature VS Demand")
plt.scatter(bikes_prep['temp'], bikes_prep['demand'],s=2,c='g')

plt.subplot(2,2,2)
plt.title("aTemp VS Demand")
plt.scatter(bikes_prep['atemp'], bikes_prep['demand'],s=2, c='b')

plt.subplot(2,2,3)
plt.title("Humidity VS Demand")
plt.scatter(bikes_prep['humidity'], bikes_prep['demand'],s=2, c='m')

plt.subplot(2,2,4)
plt.title("Windspeed VS Demand")
plt.scatter(bikes_prep['windspeed'], bikes_prep['demand'],s=2, c='c')

plt.tight_layout()


print(bikes_prep['season'].unique())
# Visualising the categorical features VS demand
# Create a 3 x 4 subplot

plt.figure()
plt.subplot(3,3,1)
plt.title("Average demand per season")

# Create unique seasons values
cat_list = bikes_prep['season'].unique()
# Create average demand per season using groupby
cat_average = bikes_prep.groupby('season').mean()['demand']
colours = ['g','r','m','b']
plt.bar(cat_list, cat_average, color=colours)


plt.subplot(3,3,2)
plt.title("Average demand per year")
cat_list = bikes_prep['year'].unique()
cat_average = bikes_prep.groupby('year').mean()['demand']
plt.bar(cat_list, cat_average, color=['b','r'])


plt.subplot(3,3,3)
plt.title("Average demand per month")
cat_list = bikes_prep['month'].unique()
cat_average = bikes_prep.groupby('month').mean()['demand']
colors = ['g','r','y','b','m','c','b','gold','lime','pink','slategray','dodgerblue']
plt.bar(cat_list, cat_average, color=colors)


plt.subplot(3,3,4)
plt.title("Average demand per hour")
cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
colors = ['g','r','y','b','m','c','b','gold','lime','pink','slategray','dodgerblue',
          'orange','steelblue','lawngreen','blue',
          'brown','tan','orchid','navy',
          'aqua', 'peru','plum','deeppink']
plt.bar(cat_list, cat_average, color=colors)


plt.subplot(3,3,5)
plt.title("Average demand per holiday")
cat_list = bikes_prep['holiday'].unique()
cat_average = bikes_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list, cat_average, color=['r','g'])


plt.subplot(3,3,6)
plt.title("Average demand per weekday")
cat_list = bikes_prep['weekday'].unique()
cat_average = bikes_prep.groupby('weekday').mean()['demand']
colors = ['g','r','y','b','m','c','b']
plt.bar(cat_list, cat_average, color=colors)


plt.subplot(3,3,7)
plt.title("Average demand per workingday")
cat_list = bikes_prep['workingday'].unique()
cat_average = bikes_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list, cat_average, color=['r','g'])


plt.subplot(3,3,8)
plt.title("Average demand per weather")
cat_list = bikes_prep['weather'].unique()
cat_average = bikes_prep.groupby('weather').mean()['demand']
plt.bar(cat_list, cat_average, color=['r','g','b','y'])

plt.tight_layout()
plt.clf()

plt.figure()
plt.title("Average demand per hour")
cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
colors = ['g','r','y','b','m','c','b']
plt.bar(cat_list, cat_average, color=colors)


# Checking for outliers
bikes_prep['demand'].describe()
bikes_prep['demand'].quantile([0.05,0.1,0.15,0.9,0.95,0.99]) # In terms of percentile


# Checking Multiple Linear Regression Assumptions

# Linearity using correlation coefficient matrix using corr
correlation = bikes_prep[['temp','atemp','humidity','windspeed','demand']].corr()

bikes_prep = bikes_prep.drop(['weekday','year','workingday','atemp','windspeed'], axis = 1)

# Check the autocorrelation in demand using acorr
df1 = pd.to_numeric(bikes_prep['demand'], downcast = 'float') # float32 provided

# Finding out autocorrelation of demand variable
plt.figure()
plt.acorr(df1, maxlags = 12)
# Any value corresponding to >0.8 is considered to be significant


# Log normalise the feature demand
# Log normal distribution means that taking log of the values will give us a normal distribution log(x)= normal distribution
df1 = bikes_prep['demand']
df2 = np.log(df1)

plt.figure()
df1.hist(rwidth=0.9, bins=20)

plt.figure()
df2.hist(rwidth=0.9, bins=20)

bikes_prep['demand'] = np.log(bikes_prep['demand'])

# Autocorrelation in the demand column
t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = bikes_prep['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = bikes_prep['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']

bikes_prep_lag = pd.concat([bikes_prep, t_1,t_2,t_3],axis =1)

bikes_prep_lag = bikes_prep_lag.dropna()


# Create Dummy Variables and drop first using get_dummies
# We need features to be of type category for get_dummies to work

bikes_prep_lag.dtypes

bikes_prep_lag['season'] = bikes_prep_lag['season'].astype('category')
bikes_prep_lag['month'] = bikes_prep_lag['month'].astype('category')
bikes_prep_lag['holiday'] = bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather'] = bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['hour'] = bikes_prep_lag['hour'].astype('category')

bikes_prep_lag = pd.get_dummies(bikes_prep_lag,drop_first=True)


# Split the X and Y datasety into training and testing set

# demand is time dependent or time series
# Hence we cannot randomly drop certain points as we will lose the autocorrelation
Y= bikes_prep_lag[['demand']]
X = bikes_prep_lag.drop(['demand'], axis=1)

# Create the size ofr 70% of the data to be training date
tr_size = 0.7*len(X)
tr_size = int(tr_size)

X_train = X.values[0:tr_size]
X_test = X.values[tr_size:len(X)]
Y_train = Y.values[0:tr_size]
Y_test = Y.values[tr_size:len(Y)]

# Linear Regression
from sklearn.linear_model import LinearRegression

std_reg = LinearRegression() # Create an instance of the linear regression
std_reg.fit(X_train, Y_train)   # Train model

# r-squared value
r2_train = std_reg.score(X_train,Y_train) 
r2_test = std_reg.score(X_test, Y_test)

# Create Y predictions
Y_predict = std_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(Y_test, Y_predict))


# Calculating of RMSLE and compare results
Y_test_e = []
Y_predict_e = []

for i in range(0,len(Y_test)):
    Y_test_e.append(math.exp(Y_test[i]))
    Y_predict_e.append(math.exp(Y_predict[i]))
    
    
# Do the sum of the logs and squares to calculate RMSLE
# rmsle is good for predictions with no negative values abd
# also prediction values with very small and large values
print(min(bikes['demand'])) #1
print(max(bikes['demand'])) #977

log_sq_sum = 0.0
for i in range(0,len(Y_test_e)):
    log_a = math.log(Y_test_e[i] + 1)
    log_p = math.log(Y_predict_e[i] + 1)
    log_diff = (log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff
    
rmsle = math.sqrt(log_sq_sum/len(Y_test))

print("\n",rmsle)
# Decent rmsle score obtained    