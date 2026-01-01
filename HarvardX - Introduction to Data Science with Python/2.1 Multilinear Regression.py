# Importing necessary libraries
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from helper import fit_and_plot_linear, fit_and_plot_multi

# Reading the dataset
#df = pd.read_csv("Advertising.csv")
#df.head()

# Creating a DataFrame to store results
#df_results = pd.DataFrame(columns = ["Predictor", "R2 Train", "R2 Test"])

# Linear Regression

# Fitting and plotting linear regression for TV
#fit_and_plot_linear(df[["TV"]])

# Fitting and plotting linear regression for Radio
#fit_and_plot_linear(df[["Radio"]])

# Fitting and plotting linear regression for Newspaper
#fit_and_plot_linear(df[["Newspaper"]])

# Multi Linear Regression

# Fitting and plotting multi linear regression
#fit_and_plot_multi()

### edTest(test_dataframe) ###

# List of predictors
#predictors = ["TV", "Radio", "Newspaper"]

# Loop through each predictor and fit linear regression model
#for i in predictors:
#    r2_train, r2_test = fit_and_plot_linear(df[[i]])
#    df_results = pd.concat([df_results, pd.DataFrame({"Predictor": [i], "R2 Train": [r2_train], "R2 Test": [r2_test]})], ignore_index=True)

# Fit multi linear regression model and store results
#r2_multi_train, r2_multi_test = fit_and_plot_multi()
#df_results = pd.concat([df_results, pd.DataFrame({"Predictor": ["Multi"], "R2 Train": [r2_multi_train], "R2 Test": [r2_multi_test]})], ignore_index=True)

# Display the results
#df_results.head()

# Fitting a Multi-Regression Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Advertising.csv")

for col in ["TV", "Radio", "Newspaper"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
df.head()

# Initialize a list to store the MSE values
mse_list = []

cols = [["TV"], ["Radio"], ["Newspaper"],
        ["TV", "Radio"], ["TV", "Newspaper"], 
        ["Radio", "Newspaper"], ["TV", "Radio", "Newspaper"]]

for i in cols:
    x = df[i]
    y = df["Sales"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
    
    lreg = LinearRegression()
    lreg.fit(x_train, y_train)

    y_pred = lreg.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

# Create and display the results table
t = PrettyTable(['Predictors', 'MSE'])

for i in range(len(mse_list)):
    t.add_row([cols[i], round(mse_list[i], 3)])

print(t)