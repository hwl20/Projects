'''
Project dataset: https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho?select=Car+details+v3.csv
Date retrieved: 2021-07-20

Business problem: 
    We want to create a model that is able to accurately predict the value of a car. 
    This will allow car buyers to better evaluate the true value of a car such that they are not overpaying based on certain car metrics.
    Also it will allow sellers to better estimate the price they want to list their car especially if they want to ensure they are able to sell their car.
    This has very practical use cases for car buying/selling platforms to provide price suggestion when users list cars on their platforms
    Higher turnover rate will mean more PROFIT for platforms who earn comission from a resale of a car and happy customers which help customer retention as well
        
Dataset details:
-This dataset contains information about used cars based in India last recorded on June 2020
-Preliminary column desription and personal thoughts:
    1. name - name and model of vehicle
    2. year - car manufacture date, older cars should have relatively lower value due to depreciation
    3. selling_price - dependent variable that we are looking to build a model that can accurately predict this
    4. km_driven - indicates the distance that has already been driven, which may imply the current condition of the car
    5. fuel - type of fuel used, diesel cars are generally cheaper than petrol cars
    6. seller_type - Individual sellers should be cheaper than Dealer who would want to collcect a comission
    7. transmission - Manual or Auto, not clear how would this affect price yet
    8. owner - Number of owners a particular car had, more owners car generally have lower prices as it may suggest that there might be some issues in the car (psychological factor)
    9. mileage - fuel efficency, higher efficiency could potentially fetch higher prices
    10. engine - Engine capacity, higher engine capacity should suggest higher selling price
    11. max_power - Power of engine, higher power should suggest higher selling price
    12. torque - Engine rotational force, higher torque should suggest higher selling price
    13. seats- Seats of car, more seats generally corresponds to higher selling price

In here we will follow the data science framework of working with data:
Retrieving data - Kindly done so from kaggle
Data Cleaning
Exploratory Data Analysis to get a better understanding of data
Feature selection 
Model Building
Validating model goodness and effectiveness
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math

###################### Retrieving data ######################
print(os.listdir(),"\n")
df = pd.read_csv("Car details v3.csv")
df.shape
df.head()
# df.info()


###################### Data Cleaning ######################
df.duplicated().any()
df = df.drop_duplicates() # Removed duplicates as I assume they were just double counted

# Null cells
df.isnull().sum()
percentage_null_per_category = df.isnull().sum()/ df.shape[0]*100
df.isnull().any(axis=1).sum()
percentage_null_total = df.isnull().any(axis=1).sum()/df.shape[0]*100
# Since null values only take up 3% of the dataset, I will remove all rows with null values as well
df = df.dropna(axis=0)


# Converting manufacture year to the age of the instead called years_used
df['years_used'] = 2021 - df.year
df = df.drop('year', axis=1)


# Removing the units for data
mileage_units = pd.Series(map(lambda x: x.split(' ')[1], df.mileage)).unique()
fuel_types = list(df.fuel.unique())
fuel_conversion = {fuel_types[0]: 0.832,fuel_types[1]: 0.73, fuel_types[2]: 0.512, fuel_types[3]: 0.714} # Conversion value for litre to kg equivalent was extracted from the web 
def converting_mileage(row):
    '''
    function that converts km/kg to kmpl
    '''
    value = float(row.mileage.split()[0])
    if not row.mileage.endswith('kmpl'):
        f_type = row.fuel
        row['mileage'] = round(value*fuel_conversion[f_type],2)
    else: row['mileage'] = value
    return row  
df = df.apply(converting_mileage, axis=1)

pd.Series(map(lambda x: x.split(' ')[1], df.engine)).unique()
df['engine'] = df['engine'].str.strip("CC")

pd.Series(map(lambda x: x.split(' ')[1], df.max_power)).unique()
df['max_power'] = df['max_power'].str.strip("bhp")


# Setting datatypes for visualizations
category_lst = ['fuel','seller_type','transmission','owner','seats']
df[category_lst] = df[category_lst].astype('category')
df['engine'] = pd.to_numeric(df['engine'])
df['max_power'] = pd.to_numeric(df['max_power'])





###################### Exploratory Data Analysis ######################

# Distribution of continuous variables
plt.figure()
plt.subplot(2,3,1)
plt.title("Selling Price")
plt.hist(df.selling_price, rwidth = 0.9, bins=50)
plt.subplot(2,3,2)
plt.title("KM driven")
plt.hist(df.km_driven, rwidth = 0.9, bins = 50)
plt.subplot(2,3,3)
plt.title("mileage")
plt.hist(df.mileage, rwidth = 0.9)
plt.subplot(2,3,4)
plt.title("engine")
plt.hist(df.engine, rwidth = 0.9)
plt.subplot(2,3,5)
plt.title("max_power")
plt.hist(df.max_power, rwidth = 0.9, bins=20)
plt.subplot(2,3,6)
plt.title("years used")
plt.hist(df.years_used, rwidth = 0.9)
plt.tight_layout()
'''
By observation, 
selling_price, km_driven, max_power, years_used seem to be right skewed
mileage seems to be normally distributed
engine seems to not follow any particular distributon
'''

# Boxplot for continuous variables
# Coloring boxplot color: https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color            
# Color palette: https://matplotlib.org/stable/gallery/color/named_colors.html
plt.figure()
plt.subplot(2,3,1)
plt.title("Selling Price")
plt.boxplot(df.selling_price,
            boxprops=dict(color="g"),
            capprops=dict(color="g"),
            flierprops=dict(color='g', markeredgecolor='g'),
            medianprops=dict(color='g'),
            vert=False)
plt.subplot(2,3,2)
plt.title("KM driven")
plt.boxplot(df.km_driven,
            boxprops=dict(color="r"),
            capprops=dict(color="r"),
            flierprops=dict(color='r', markeredgecolor='r'),
            medianprops=dict(color='r'),
            vert=False)
plt.subplot(2,3,3)
plt.title("mileage")
plt.boxplot(df.mileage,
            boxprops=dict(color="b"),
            capprops=dict(color="b"),
            flierprops=dict(color='b', markeredgecolor='b'),
            medianprops=dict(color='b'),
            vert=False)
plt.subplot(2,3,4)
plt.title("engine")
plt.boxplot(df.engine,
            boxprops=dict(color="y"),
            capprops=dict(color="y"),
            flierprops=dict(color='y', markeredgecolor='y'),
            medianprops=dict(color='y'),
            vert=False)
plt.subplot(2,3,5)
plt.title("max_power")
plt.boxplot(df.max_power,
            boxprops=dict(color="m"),
            capprops=dict(color="m"),
            flierprops=dict(color='m', markeredgecolor='m'),
            medianprops=dict(color='m'),
            vert=False)
plt.subplot(2,3,6)
plt.title("years used")
plt.boxplot(df.years_used,
            boxprops=dict(color="c"),
            capprops=dict(color="c"),
            flierprops=dict(color='c', markeredgecolor='c'),
            medianprops=dict(color='c'),
            vert=False)
plt.tight_layout()
'''
km_driven, selling_price, mileage, max_power has very 
big outliers that is required to be dealt with
'''

# Visualising the continuous features VS demand
plt.figure()
plt.suptitle("Selling Price VS Variables")
plt.subplot(2,3,1)
plt.title("Selling Price VS KM driven")
plt.scatter(df.km_driven, df.selling_price, c='r',s=2)
plt.subplot(2,3,2)
plt.title("Selling Price VS mileage")
plt.scatter(df.mileage, df.selling_price, c='b',s=2)
plt.subplot(2,3,3)
plt.title("Selling Price VS engine")
plt.scatter(df.engine, df.selling_price, c='g',s=2)
plt.subplot(2,3,4)
plt.title("Selling Price VS max_power")
plt.scatter(df.max_power, df.selling_price, c='y',s=2)
plt.subplot(2,3,5)
plt.title("Selling Price VS years used")
plt.scatter(df.years_used, df.selling_price, c='m',s=2)
plt.tight_layout()
'''
km_driven, mileage, max_power seems to require dealing with outliers
km_drive, years_used - negative r/s wrt selling_price
max_power, engine - positive r/s wrt selling_price
mileage - no obvious r/s between the two variables
'''


# Visualising the categorical features VS demand
plt.figure()
plt.subplot(2,3,1)
plt.title("Selling Price VS fuel")
cat_unique = df.fuel.unique()
cat_average = df.groupby("fuel").mean()['selling_price']
plt.bar(cat_unique, cat_average, color="r")
plt.subplot(2,3,2)
plt.title("Selling Price VS seller_type")
cat_unique = df.seller_type.unique()
cat_average = df.groupby("seller_type").mean()['selling_price']
plt.bar(cat_unique, cat_average, color='b')
plt.subplot(2,3,3)
plt.title("Selling Price VS transmission")
cat_unique = df.transmission.unique()
cat_average = df.groupby("transmission").mean()['selling_price']
plt.bar(cat_unique, cat_average, color='g')
plt.subplot(2,3,4)
plt.title("Selling Price VS owner")
cat_unique = df.owner.unique()
cat_average = df.groupby("owner").mean()['selling_price']
plt.bar(cat_unique, cat_average, color='y')
plt.xticks(rotation=45)
plt.subplot(2,3,5)
plt.title("Selling Price VS seats")
cat_unique = df.seats.unique()
cat_average = df.groupby("seats").mean()['selling_price']
plt.bar(cat_unique, cat_average, color='m')
plt.tight_layout()
'''
All categorical variables show that there is variation
for each category of a variable on selling price.
All categorical variables are useful and should be kept
'''

# Correlation between continuous variables
continuous_lst = list(set(df.columns)-set(category_lst)-set(['torque','name']))
corr_table = df[continuous_lst].corr()
'''
No signs of multicollinearity (>+-0.8) amongst our continous variables
However mileage seem to have weak impact on selling_price - something to take note
'''




###################### Feature Selection and Engineering ######################
df_original = df.copy()
df = df_original.copy()
# In this section we will be curating the data to be best suited for modelling purposes
df.apply(lambda x: len(x.unique()), axis = 0)
df = df.drop(['name','torque'], axis=1)     # Drop name and torque because they are categorical variables with too much variation - Not beneficial for building our model

# Setting limits for continuous data - according to boxplot
description = df.describe()
df.selling_price.quantile([0.003,0.5,0.75,0.997])
df.km_driven.quantile([0.003,0.5,0.75,0.997])
df.mileage.quantile([0.003,0.5,0.75,0.997])
df.years_used.quantile([0.003,0.5,0.75,0.997])
df.max_power.quantile([0.003,0.5,0.75,0.997])
df.engine.quantile([0.003,0.5,0.75,0.997])

df = df.loc[df.selling_price<(6000000)]

df = df.loc[df.km_driven<(10**6)]

df = df[(df.mileage<=(35)) & (df.mileage>=0)]

df = df[df.max_power<350]

df = df.loc[df.engine<3000]

# Dummy Encoding for categorical variables
category_lst
df = pd.get_dummies(df, drop_first=True)


# Scaling the data
# selling_price, km_driven, max_power, years_used are right skewed
df_original = df.copy()
df1 = df['selling_price']
df2 = np.log(df1)
plt.figure()
plt.suptitle("selling_price")
plt.subplot(1,2,1)
plt.title("Before normalization")
df1.hist(rwidth=0.9)
plt.subplot(1,2,2)
plt.title("After normalization")
df2.hist(rwidth=0.9)
df['selling_price'] = np.log(df.selling_price)


'''
Website on normalization/standardization: https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
Website explaining when to standardize variables: https://www.listendata.com/2017/04/how-to-standardize-variable-in-regression.html
IT IS NOT REQUIRED TO SCALE INDEPENDENT VARIABLES FOR MLR
'''

###################### Model Building ######################
'''
We will be using a multiple linear regression for building our model here
'''
# Splitting into training and test dataset from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X = df.drop(['selling_price'], axis=1)
Y = df[['selling_price']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

# Mult  iple Linear Regression Model
from sklearn.linear_model import LinearRegression
std_reg = LinearRegression()
model = std_reg.fit(X_train, y_train)
y_predict = model.predict(X_test)




###################### Validating model goodness and effectiveness ######################

# Using k-fold cross validation to see if we have overfitted our data
from sklearn.model_selection import cross_val_score, cross_val_predict
scores = cross_val_score(model, X_train, y_train,cv=6)
print("Cross validated R^2 scores: ", scores)
'''
Generally all above 0.8, which is a good sign there is not overfitting
This means that our model is good in predicing car prices accurately given outside data
'''


# We will gauge the metrics of the main model that we have built
r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test,y_test)
adjusted_r_squared_train = 1 - (1-r2_train)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
adjusted_r_squared_test = 1 - (1-r2_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("r2_train: ",round(r2_train,3),"\nr2_test: ",round(r2_test,3),"\n")
print("adjusted r2_train: ",round(adjusted_r_squared_train,3),"\nadjusted r2_test: ",round(adjusted_r_squared_test,3),"\n")
'''
Removing Mileage predictor results in a lower adjusted r-squared, hence we will keep mileage predictor in our model
'''

# Normalzied RMSE 
from sklearn.metrics import mean_squared_error
y_test_lst = y_test['selling_price'].tolist()
y_predict_lst = list(map(lambda x: x[0], y_predict))
y_test_actual = []
y_predict_actual = []
for i in range(0,len(y_test)):
    y_test_actual.append(math.exp(y_test_lst[i]))
    y_predict_actual.append(math.exp(y_predict_lst[i]))
print("Minimum selling price: ", df_original.selling_price.min())
print("Maximum selling price: ",df_original.selling_price.max())
rmse = math.sqrt(mean_squared_error(y_test_actual, y_predict_actual))
normalized_rmse = rmse/(df_original.selling_price.max()-df_original.selling_price.min())
print(f'Normalized RMSE is {round(normalized_rmse,2)}')



# Main insights
'''
FINAL THOUGHTS AND CONCLUSION OF MODEL:
    
- We have obtained a good r2_test score of 0.849 AND a low NRMSE score of 0.02, and K-fold cross validation show no signs of overfitting
- In conclusion, this means that our model is a great fit for such a predicting car prices for the given metrics
- This means that target audience, mainly car buying/selling platform should find this analysis useful for pricing suggestions to 
sellers which makes prices more competitive and relevant, translating to higher sales rate, which results in better earnings 
for the company

**Notes for improvement
- Car sales are also dependent on time of the year, hence a model that include time series analysis can be included to gauge the
season where car is in higher demand->Charge higher prices OR lower demand-> Charge lower prices
- This dataset provides locale of only one country, hence model may or may not be applicable to all countries. Howeever, my belief
is that with simple fine tuning and feature seclection, this model will work equally well in different demographics

'''
