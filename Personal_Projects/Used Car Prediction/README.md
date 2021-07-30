This file contains the project on predicting Used Car Prices, based off a dataset of car prices in India from Kaggle.

Navigate this page:
 - Raw python code: Used Car Final Version.py
 - Coded on Google Colab: UsedCarGoogleColab.ipynb

    
        
        
Brief Overview:

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
