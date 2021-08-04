# Private Properties Price Prediction Analysis
# Combining the dataframes

# Combining the 6 tables of data obtained from URA website
df1 = pd.read_csv("District 1-5.csv", index_col=0)
df2 = pd.read_csv("District 6-10.csv", index_col=0)
df3 = pd.read_csv("District 11-15.csv", index_col=0)
df4 = pd.read_csv("District 16-20.csv", index_col=0)
df5 = pd.read_csv("District 21-26.csv", index_col=0)
df6 = pd.read_csv("District 27-28.csv", index_col=0)

df = df1.append(df2, ignore_index= True)
df = df.append(df3, ignore_index=True)
df = df.append(df4, ignore_index=True)
df = df.append(df5, ignore_index=True)
df = df.append(df6, ignore_index=True)
df.to_csv('Private Home Data.csv')

# Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import math 

#################### Data Preprocessing ####################
# Preliminary Analysis
# Link to Dataset could be found here: https://docs.google.com/spreadsheets/d/1EGHUpw3v2Fa2vFbGd6_3yyUrTPTKM8vZB_w6V6xxez0/edit?usp=sharing
df = pd.read_csv('Private Home Data.csv',index_col=0)
df_subset = df.head()

# Solving null values
df.isnull().any(axis=0)
df.info()
# Tenure has only 1 null value
missing_row = df[df.isnull().any(axis=1)]
similar_development = df[(df['Project Name']=="EMERY POINT") & (df['Street Name']=="IPOH LANE")]
# Manually Imputing the missing cell since similar development are all 'Freehold'
df['Tenure'].fillna('Freehold', inplace=True)
df.isnull().any()


dup = df[df.duplicated()]
# Over 3% of data are duplciates - Exact same Pricing and metrics


## Labelling the data into a suitable format
df.dtypes
df.apply(lambda x: x.unique().size, axis=0)

types_of_tenures = df.Tenure.unique()
tenure_information = df.Tenure.value_counts()
# Common format: x yrs lease commencing from y
# Dealing with formatting for exceptions
df[df.Tenure=='99 years leasehold']
similar_development = df[(df['Project Name']=="PARKWOOD RESIDENCES") & (df['Street Name']=="YIO CHU KANG ROAD")]
# Upon research, PARKWOOD RESIDENCES has a 99 yrs lease commencing from 2018 - https://www.mysgprop.com/parkwood-residences-oxley/
df['Tenure'].replace('99 years leasehold','99 yrs lease commencing from 2018', inplace=True)
df['Tenure'].replace('110 Yrs From 01/11/2017','110 yrs lease commencing from 2017', inplace=True)
df['isFreehold'] = df.apply(lambda x: 'Freehold' if x.Tenure =='Freehold' else 'Not Freehold', axis=1)
df['Tenure'].replace('Freehold','999999 yrs commencing from 2021', inplace=True)
def remaining_lease(row):
    '''
    Function is used to extract remaining lease term of the development using information from tenure column
    '''
    lst = row['Tenure'].split(' ')
    remain = (int(lst[0])-(date.today().year - int(lst[-1])))
    return remain
df['remaining lease'] = df.apply(remaining_lease, axis=1)


type_of_floor = df['Floor Level'].unique()
df['Floor Level'].value_counts()
test = df[df['Floor Level']=='-']
# Research shows that developments with Floor Level = '-' are generally 01 to 05 stories tall
similar_development = df[(df['Project Name']=="SERENE VIEW MANSIONS") & (df['Street Name']=="LORONG SELANGAT")]
similar_development = df[(df['Project Name']=="SKIES MILTONIA")]
# Such as Serene Views Mansions and Skies Miltonia, hence we classify them to be within the '01 to 05' floor range
df['Floor Level'].replace('-','01 to 05', inplace=True)


## Simple Feature Engineering to remove some less important features
# Checking number of unique values in each column
df_original = df.copy()
cols_to_remove = []
df.apply(lambda x: x.unique().size, axis=0)

# Project Name and Street Name are too specific for our analysis hence we will remove them as well
cols_to_remove.extend(['Project Name','Street Name'])   # Business POV

# Remove Tenure since we have already extracted the data beforehand
cols_to_remove.append('Tenure')                         # Data POV

int((df['Nett Price ($)']=='-').sum())/df.shape[0]
# Over 99% of data has net price as '-' so we will ignore 'Nett Price ($)' and stick with 'Price ($)' as our outcome variable
cols_to_remove.append('Nett Price ($)')                  # Data POV

# Unit Price ($psf) will immediately give away the housing price and is also too specific of a parameter for user to input
cols_to_remove.append('Unit Price ($psf)')              # Business POV

# We are considering the most recent 5 years of housing sales, to keep prices relevant, and since we are not doing a time series analysis, we will omit 'Date of Sale' as well
cols_to_remove.append('Date of Sale')                   # Business POV

df.drop(cols_to_remove, axis=1, inplace=True)






#################### Data Visualization ####################
# Exploring the categorical variables 
lst_of_categorical_variables = ['Type','Postal District','Market Segment','Type of Sale','No. of Units', 'Type of Area','Floor Level', 'isFreehold']
print(" - We have ", len(lst_of_categorical_variables), " categorical variables", sep='')
plt.figure()
plt.suptitle("Categorical Variables part 1")
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title(lst_of_categorical_variables[i])
    plt.hist(df[lst_of_categorical_variables[i]], rwidth=0.9)
plt.tight_layout()

plt.figure()
plt.suptitle("Categorical Variables part 2")
for i in range(4,8):
    plt.subplot(2,2,i-3)
    plt.title(lst_of_categorical_variables[i])
    plt.hist(df[lst_of_categorical_variables[i]], rwidth=0.9)
    plt.xticks(rotation=45)
plt.tight_layout()
'''
Basic analysis on distribution of categorical variables:
    - More condominum than apartmenet type housing
    - Fairly balanced number of sales in each Postal District
    - Most sales occured Outside Central Region (OCR), followed by Rest of Central Region(RCR) and lastly Core Centreal Region(CCR)
    - Both Resale and New Sales of condominiums & apartments are almost the same price
    - Most transaction involve low number of units, understandably so as people usually only buy 1 property at each go
    - Most transaction are of Strata format, because we extracted only condominum and apartment data from website, land may be because a developer bought out entire space for rebuild (a possible for Land)
    - Most homes transacted are under 30 stories as homes in Singapore are usually around that range, rarity are those around 40-50 stories
    - Less Freehold homes are transacted as compared to non-freehold ones
'''

# Exploring the continuous variables 
lst_of_continuous_variables = list(set(df.columns)-set(lst_of_categorical_variables))
print(" - We have ", len(lst_of_continuous_variables), " continuous variables", sep='')
plt.figure()
plt.suptitle("Continuous Variables Distribution Plot")
for i  in range(len(lst_of_continuous_variables)):
    plt.subplot(3,1,i+1)
    plt.title(lst_of_continuous_variables[i])
    plt.hist(df[lst_of_continuous_variables[i]])
plt.tight_layout()

plt.figure()
plt.suptitle("Continuous Variables BoxPlot")
for i  in range(len(lst_of_continuous_variables)):
    plt.subplot(3,1,i+1)
    plt.title(lst_of_continuous_variables[i])
    plt.boxplot(df[lst_of_continuous_variables[i]])
plt.tight_layout()
# Distribution does not show any relationship due to heavy outliers

# We shall clean up some datapoints with rough values
df_continuous = df[lst_of_continuous_variables]
df_continuous.describe()
df_continuous = df_continuous[df_continuous['Area (Sqft)']<10000]
df_continuous = df_continuous[df_continuous['Price ($)']<10**7]
df_continuous['remaining lease'].where(df_continuous['remaining lease'] < 999, 999, inplace=True)
(df_continuous['remaining lease'].value_counts()).sort_index() # We changed approximately 27007 houses from >999 years/freehold status to 999 years

# Replotting the graph after clearing some datapoints
plt.figure()
plt.suptitle("Continuous Variables Distribution Plot")
for i  in range(len(lst_of_continuous_variables)):
    plt.subplot(3,1,i+1)
    plt.title(lst_of_continuous_variables[i])
    plt.hist(df_continuous[lst_of_continuous_variables[i]])
plt.tight_layout()

plt.figure()
plt.suptitle("Continuous Variables BoxPlot")
for i  in range(len(lst_of_continuous_variables)):
    plt.subplot(3,1,i+1)
    plt.title(lst_of_continuous_variables[i])
    plt.boxplot(df_continuous[lst_of_continuous_variables[i]])
plt.tight_layout()
'''
 - Both Area (Sqft) and Price ($) follow a log normal distibution after cleaning up some datapoints,
but remaining lease still look to have 2 ends of the spectrum 
 - This is due to having homes which are of 999/9999/999999/freehold status
 - We might have to adjust this data more should we choose to use this variable later in the model
'''

# Plotting categorical variables against Price ($)
color_palette = ['b','r','g','c','y','m','k','b']
plt.figure()
plt.suptitle("Categorical Variables vs Price ($) part 1")
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title(lst_of_categorical_variables[i])
    cat_unique = df.groupby(lst_of_categorical_variables[i]).mean()['Price ($)'].index
    cat_average = df.groupby(lst_of_categorical_variables[i]).mean()['Price ($)']
    plt.bar(x = cat_unique, height = cat_average, 
            color = color_palette[i])
plt.tight_layout()

plt.figure()
plt.suptitle("Categorical Variables vs Price ($) part 2")
for i in range(4,8):
    plt.subplot(2,2,i-3)
    plt.title(lst_of_categorical_variables[i])
    cat_unique = df.groupby(lst_of_categorical_variables[i]).mean()['Price ($)'].index
    cat_average = df.groupby(lst_of_categorical_variables[i]).mean()['Price ($)']
    plt.bar(x = cat_unique, height = cat_average, 
            color = color_palette[i])
    plt.xticks(rotation=45)
plt.tight_layout()
'''
Quick Analysis:
- Condominium and Apartment have similar pricing
*- Postal District 6 (Town area) seems to have abnormally high Price ($)
- Core Central Region has the highest average housing pricing, as expected
- Resale homes are more expensive than New Homes
- Higher No. of units per transaction means higher price
- Land labelled homes are generally more expensive than Strata ones
- Higher floor points to higer pricing
- Freehold houses are more expensive than non freehold houses

* represent insights that are not expected
'''

# Plotting continuous variables against Price ($)
plt.figure()
plt.suptitle("Continuous Variables vs Price ($)")
continuous_predictors = list(set(lst_of_continuous_variables) - set(['Price ($)']))
for i  in range(len(continuous_predictors)):
    plt.subplot(2,1,i+1)
    plt.title(continuous_predictors[i]+ " vs price")
    plt.scatter(df[continuous_predictors[i]], df['Price ($)'], 
                c = color_palette[i])
plt.tight_layout()
'''
As Area (Sqft) increases, Price increases as well
Again, due to remaining lease having 99999/freehold years status, we get a weird scatterplot
'''

# After dealing with outliers
plt.figure()
plt.suptitle("Continuous Variables vs Price ($)")
continuous_predictors = list(set(lst_of_continuous_variables) - set(['Price ($)']))
for i  in range(len(continuous_predictors)):
    plt.subplot(2,1,i+1)
    plt.title(continuous_predictors[i]+ " vs price")
    plt.scatter(df_continuous[continuous_predictors[i]], df_continuous['Price ($)'], 
                c = color_palette[i])
plt.tight_layout()
'''
Even scaling down the remaining lease does not seem to aid the plot as the range we set for freehold houses are still quite large
Here, we might consider to 
1) scale it down futher (without hurting the integrity of freehold housing)
2) stick to the categorical variable 'isFreehold' to differentiate the 2 categories
'''

corr_table = df[lst_of_continuous_variables].corr()
corr_table_after_outliers = df_continuous[lst_of_continuous_variables].corr()
# Adjusting the remaining lease did help boost the correlation between remaining lease and Price ($), but correlation still weak
plt.figure()
plt.subplot(211)
plt.title("Based on original data")
sns.heatmap(corr_table, annot=True)
plt.subplot(212)
plt.title("After dealing with outliers")
sns.heatmap(corr_table_after_outliers, annot=True)
# No sign of multicollinearity that seem to have to be dealt with



#################### Feature Engineering ####################
# We will start off first by dealing with data that does not fit into our business problem
def return_unique(input_col):
    '''
    function to get unique datapoints in each column
    '''
    return input_col.unique()
for i in range(len(df.columns)):
    if df.columns[i]=="Price ($)":
        continue
    print(df.columns[i],":", list(return_unique(df.iloc[:,i])),'\n')

# We are only interested in single home transaction
df = df[df['No. of Units']==1]                      # BUSINESS POV
df.drop(['No. of Units'], axis=1, inplace = True)   # Since only 1 category left

# We are only interested in non-landed categories
df['Type of Area'].value_counts()
df = df[df['Type of Area']=='Strata']                    # BUSINESS POV + DATA POV
df.drop(['Type of Area'], axis=1, inplace = True)        # Since only 1 category left



# Next we will adjust data according to ensure a good fit for our model later
# Choosing a suitable value to include for freehold housing 
df['remaining lease'].quantile([0.5,0.65,0.6875,0.7,0.95])
test = df["remaining lease"].value_counts().sort_index()
# There is a huge jump from 106 year to 805 years. We can choose >=805 to be scaled down then
# Lets change all those that are >=805 years to be approximately 300 years to create some kind of linear correlation
df_continuous = df.copy()
df_continuous = df_continuous[['Price ($)', 'Area (Sqft)', 'remaining lease']]
df_continuous['remaining lease'].where(df_continuous['remaining lease'] <=800 , 300, inplace=True)
df_continuous.corr()
#test = df_continuous["remaining lease"].value_counts().sort_index()
df.corr()
# correlation of remaining lease does not seem to improve despite reducing the 
# range of values, hence we will just continue to use it first and if necessary, remove this variable later
plt.figure()
plt.suptitle("Continuous Variables BoxPlot")
for i  in range(len(lst_of_continuous_variables)):
    plt.subplot(3,1,i+1)
    plt.title(lst_of_continuous_variables[i])
    plt.boxplot(df[lst_of_continuous_variables[i]])
plt.tight_layout()


# We do this to see if we have succesfully dealt with the abnormally high prices we seen in the earlier plot for Postal District VS Price ($)
plt.figure()
plt.title('Postal District vs Price ($)')
cat_unique = df['Postal District'].unique()
cat_unique.sort()
cat_average = df.groupby('Postal District').mean()['Price ($)']
plt.bar(x = cat_unique, height = cat_average, color = 'b')
# Yes we have - District 6 value is now more normal (arnd 6 million)


# One Hot Encoding Categorical Variables
df.dtypes
lst_of_categorical_variables
lst_of_categorical_variables.remove('No. of Units')
lst_of_categorical_variables.remove('Type of Area')
df[lst_of_categorical_variables] = df[lst_of_categorical_variables].astype('category')
df = pd.get_dummies(df,drop_first=True)

# Log normalizing the 'Price ($)' outcome variable 
df1 = df['Price ($)']
df2 = np.log(df1)
plt.figure()
plt.suptitle("Price ($)")
plt.subplot(1,2,1)
plt.title("Before normalization")
df1.hist(rwidth=0.9)
plt.subplot(1,2,2)
plt.title("After normalization")
df2.hist(rwidth=0.9)
df['Price ($)'] = np.log(df['Price ($)'])
# Outcome variable now looks normalized




#################### Building Preliminary model ####################
model_scores_df = pd.DataFrame(columns=['K-fold Average','r2_test','adjusted r2_test'],
                               index = ['model1','model2','model3','model4','model5'])


# Model 1 - Retaining ALL remaining variables and their ranges (except 'remaining lease' whereby we changed 9999/99999/999999/freehold housings to 999 years)
from sklearn.model_selection import train_test_split
X = df.drop(['Price ($)'], axis=1)
Y = df[['Price ($)']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2021)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model = reg.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score, cross_val_predict
scores = cross_val_score(model, X_train, y_train,cv=6)
print("Cross validated R^2 scores: ", scores)
print("K fold Cross Validation Average Score:", round(scores.mean(),3))

y_predict = model.predict(X_test)
r2_test = model.score(X_test,y_test)
adjusted_r2_test = 1 - (1-r2_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("r2 test: ", round(r2_test,3),"\nadjusted r2 test", round(adjusted_r2_test,3))
model_scores_df.loc['model1'] = [round(scores.mean(),3),round(r2_test,3),round(adjusted_r2_test,3)]



# Model 2 - Removing 'remaining lease' variable
from sklearn.model_selection import train_test_split
X = df.drop(['Price ($)','remaining lease'], axis=1)
Y = df[['Price ($)']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2021)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model = reg.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score, cross_val_predict
scores = cross_val_score(model, X_train, y_train,cv=6)
print("Cross validated R^2 scores: ", scores)
print("K fold Cross Validation Average Score:", round(scores.mean(),3))

y_predict = model.predict(X_test)
r2_test = model.score(X_test,y_test)
adjusted_r2_test = 1 - (1-r2_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("r2 test: ", round(r2_test,3),"\nadjusted r2 test", round(adjusted_r2_test,3))
model_scores_df.loc['model2'] = [round(scores.mean(),3),round(r2_test,3),round(adjusted_r2_test,3)]

'''
No change in all metrics shows that we may just consider removing the 
'remaining lease' variable since it does not value-add our model goodness-of-fit
'''


# Model 3 - Reducing the super high priced housing
df['Original Price'] = df.apply(lambda x: math.exp(x['Price ($)']), axis=1)
df['Original Price'].describe()
df['Original Price'].quantile([0.025, 0.25, 0.5, 0.75, 0.997])
df_model3 = df.copy()
df_model3 = df_model3[df_model3['Original Price'] < df_model3['Original Price'].quantile([0.997]).iloc[0]]
df_model3[['Original Price']].hist(rwidth=0.9, bins = 50) # To check shape after removing huge variations
df.drop(['Original Price'], axis=1, inplace=True)
df_model3.drop(['Price ($)'], axis=1, inplace = True)


df1 = df_model3['Original Price']
df2 = np.log(df1)
plt.figure()
plt.suptitle("Original Price")
plt.subplot(1,2,1)
plt.title("Before normalization")
df1.hist(rwidth=0.9)
plt.subplot(1,2,2)
plt.title("After normalization")
df2.hist(rwidth=0.9)
df_model3['Original Price'] = np.log(df_model3['Original Price'])

from sklearn.model_selection import train_test_split
X = df_model3.drop(['Original Price','remaining lease'], axis=1)
Y = df_model3[['Original Price']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2021)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model = reg.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score, cross_val_predict
scores = cross_val_score(model, X_train, y_train,cv=6)
print("Cross validated R^2 scores: ", scores)
print("K fold Cross Validation Average Score:", round(scores.mean(),3))

y_predict = model.predict(X_test)
r2_test = model.score(X_test,y_test)
adjusted_r2_test = 1 - (1-r2_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("r2 test: ", round(r2_test,3),"\nadjusted r2 test", round(adjusted_r2_test,3))
model_scores_df.loc['model3'] = [round(scores.mean(),3),round(r2_test,3),round(adjusted_r2_test,3)]

'''
Very slight improvement in r2 adn adjusted r2
However, decrease in K-Fold Cross Validation Average Score by the same magnitude, 
suggesting a possible slight overfitting of data
'''


# Model 4 - Reducing the super big area housing
df_model3['Area (Sqft)'].describe()
df_model3['Area (Sqft)'].quantile([0.025, 0.25, 0.995,0.996,0.997, 0.998,0.999])
plt.figure()
plt.suptitle("Continuous Variables BoxPlot")
model3_continuous_variables = ['Area (Sqft)', 'remaining lease','Original Price']
for i  in range(len(model3_continuous_variables)):
    plt.subplot(3,1,i+1)
    plt.title(model3_continuous_variables[i])
    plt.boxplot(df_model3[model3_continuous_variables[i]])
plt.tight_layout()
plt.figure()
plt.title("Area (Sqft) vs price")
plt.axvline(4000, 0, 1, label='4000',color='b')
plt.axvline(5000, 0, 1, label='5000',color='r')
plt.axvline(6000, 0, 1, label='6000',color='k')
plt.scatter(df_model3['Area (Sqft)'], df_model3['Original Price'], 
                c = 'g', alpha=0.2)
plt.tight_layout()
# From graph looks like it make sense to perhaps omit datapoints with Area (Sqft) >5000
df_model4 = df_model3[df_model3['Area (Sqft)']<=5000]

df_corr = df[['Area (Sqft)', 'remaining lease','Price ($)']].corr()
model3_corr = df_model3[['Area (Sqft)', 'remaining lease','Original Price']].corr()
model4_corr = df_model4[['Area (Sqft)', 'remaining lease','Original Price']].corr()
plt.figure()
plt.subplot(311)
plt.title("Before accounting for outliers")
sns.heatmap(df_corr, annot=True)
plt.subplot(312)
plt.title("Removing high priced homes")
sns.heatmap(model3_corr, annot=True)
plt.subplot(313)
plt.title("Removing high priced homes+Large Area homes")
sns.heatmap(model4_corr, annot=True)
plt.tight_layout()
# Seems like accounting for outliers made our model slightly less correlated to the Pricing
# but lets train the model nevertheless. 
# Consdering the outliers are still possible inputs for our business case
# And we have dealt with those outliers which does not fit our business case (etc land developer accquiting land for redevelopment
# where the prices run almost in hundered of million dollars)

from sklearn.model_selection import train_test_split
X = df_model4.drop(['Original Price','remaining lease'], axis=1)
Y = df_model4[['Original Price']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2021)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model = reg.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score, cross_val_predict
scores = cross_val_score(model, X_train, y_train,cv=6)
print("Cross validated R^2 scores: ", scores)
print("K fold Cross Validation Average Score:", round(scores.mean(),3))

y_predict = model.predict(X_test)
r2_test = model.score(X_test,y_test)
adjusted_r2_test = 1 - (1-r2_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("r2 test: ", round(r2_test,3),"\nadjusted r2 test", round(adjusted_r2_test,3))
model_scores_df.loc['model4'] = [round(scores.mean(),3),round(r2_test,3),round(adjusted_r2_test,3)]
'''
Model suggests that by removing both the outliers of high pricing + large area housings,
we will get a model with a better goodness-of-fit. Hence, we should use this current
model
'''

# Model 5- Including the variable 'Unit Price ($psf) which we have removed earlier
df_model5 = df_model4.copy()
df_model5 = pd.merge(df_model5, df_original[['Unit Price ($psf)']], left_index = True, right_index=True)
model5_corr = df_model5[['Area (Sqft)', 'remaining lease','Original Price', 'Unit Price ($psf)']].corr()
# As seen, Unit Price ($psf) actually has a relatively strong correlation to our outcome variable Original Price 
# (aka Price ($) in original df). However, we omitted this factor as it was too specific in our business case
# Nobody will know what amount they want to pay per-square-foot ($psf) and is not a key metric buy homebuyers in the 
# general sense. 
# But lets see its impact on the model 
from sklearn.model_selection import train_test_split
X = df_model5.drop(['Original Price'], axis=1)
Y = df_model5[['Original Price']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2021)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model = reg.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score, cross_val_predict
scores = cross_val_score(model, X_train, y_train,cv=6)
print("Cross validated R^2 scores: ", scores)
print("K fold Cross Validation Average Score:", round(scores.mean(),3))

y_predict = model.predict(X_test)
r2_test = model.score(X_test,y_test)
adjusted_r2_test = 1 - (1-r2_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("r2 test: ", round(r2_test,3),"\nadjusted r2 test", round(adjusted_r2_test,3))
model_scores_df.loc['model5'] = [round(scores.mean(),3),round(r2_test,3),round(adjusted_r2_test,3)]
'''
An almost 0.1 improvement across all metrics!
However, as per stated, due to our business case, this parameter won't be considered despite
its potential in improving the goodness-of-fit our model
'''


print(model_scores_df)
'''
Model 1 - Retaining ALL remaining variables and not excluding any outliers for each variable
Model 2 - Removing variable 'remaining lease' 
Model 3 - Reducing the super high priced housing (Outliers of outcome variable)
Model 4 - Reducing the super high priced housing + Reducing the super big area housing (Outliers)
Model 5 - Including the variable 'Unit Price ($psf) which we have removed earlier


Overall, we will choose model 4, which parameters suit our business case,
and data are accounted for in the best possible way to ensure robustness of our model

So essentially, our final model will be in the form of:
 Price ($) ~ [Dummy_Encode(Type, Postal District, Market Segment, Type of Sale, Floor Level, isFreehold) + Area (Sqft)]
'''


#################### Building Final model ####################
# We are building according to model 4
df_final = df_model4.copy()
from sklearn.model_selection import train_test_split
X = df_final.drop(['Original Price','remaining lease'], axis=1)
Y = df_final[['Original Price']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2021)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model = reg.fit(X_train, y_train)



#################### Model Validation ####################
from sklearn.model_selection import cross_val_score, cross_val_predict
scores = cross_val_score(model, X_train, y_train,cv=6)
print("Below are the metrics for our FINAL model that will be pushed out")
print("Cross validated R^2 scores: ", scores)
print("K fold Cross Validation Average Score:", round(scores.mean(),3))

y_predict = model.predict(X_test)
r2_test = model.score(X_test,y_test)
adjusted_r2_test = 1 - (1-r2_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("r2 test: ", round(r2_test,3),"\nadjusted r2 test", round(adjusted_r2_test,3))

from sklearn.metrics import mean_squared_error
y_test_lst = y_test['Original Price'].tolist()
y_predict_lst = list(map(lambda x: x[0], y_predict))
y_test_actual = []
y_predict_actual = []
for i in range(0,len(y_test)):
    y_test_actual.append(math.exp(y_test_lst[i]))
    y_predict_actual.append(math.exp(y_predict_lst[i]))
print("Minimum testing selling price: ", min(y_test_actual))
print("Maximum testing selling price: ", max(y_test_actual))
rmse = math.sqrt(mean_squared_error(y_test_actual, y_predict_actual))
normalized_rmse = rmse/(max(y_test_actual)-min(y_test_actual))
print(f'Normalized RMSE is {round(normalized_rmse,2)}')



'''
Conclusion:
    - Our model on the test dataset has a value of 0.803, and an adjusted r2 of 0.803 as well,
    which indicates that our model has a high goodness-of-fit
    - Our final model also has a normalized RMSE score of 0.08, indicating that our model has 
    low error and high accuracy
    - The model is able to give a good estimate of housing Price in the range based on the following 
    parameters that will be inputted by user:
                   Type - Apartment OR Condominium
                   Postal District - Rough location                    
                   Market Segment- In Town OR outside of Town
                   Type of Sale - Resale OR New Development
                   Floor Level - Which level category (in steps of 5 etc 1-5, 6-10, 11-15....)
                   isFreehold - Freehold OR Not Freehold
                   Area (Sqft) - Size of housing space (Giving a good gauge of # of rooms)

                                                        
Overall, to confirm this model accuracy, we can test it with data that will be coming out in the following
few months (blinfold data testing).
            
      
Limitations:
Data - 
    - No information on number of rooms, which in Singapore is an important metric when choosing a home,
    though Area (Sqft) does give a gauge on number of rooms

Analysis & Model Building - 
    - We omitted the super high priced housing in our model to improve the fit of our regression model,
    hence our data cannot fully confirm if it can be extrapolated to account for the higher end pricing houses (etc above 11 million)
       ~ However, such houses transacted are of the low numbers and we want to cater to transactions that have higher volume 
    
    - Model also excluded remaining lease indicator, which is a seemingly important factor by thought, but from our analysis show little improvement
    on our model
    
'''
