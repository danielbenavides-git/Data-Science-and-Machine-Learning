import numpy as np
import pandas as pd
import seaborn as sns 
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("colinearity.csv")
pprint(df.head())

X = df.drop(["y"], axis=1)
y = df["y"]

### edTest(test_coeff) ###

linear_coef = []

for i in X:
    x = df[i]
    linreg = LinearRegression()
    linreg.fit(x.values.reshape(-1, 1), y)
    linear_coef.append(linreg.coef_)

pprint(linear_coef)

### edTest(test_multi_coeff) ###

multi_linear = LinearRegression()
multi_linear.fit(X, y)
multi_coef = multi_linear.coef_

print('Simple(one variable) linear regression for each variable:', sep = '\n')

for i in range(4):
    pprint(f'Value of beta{i+1} = {linear_coef[i][0]:.2f}')

print('Multi-Linear regression on all variables')
for i in range(4):
    pprint(f'Value of beta{i+1} = {round(multi_coef[i],2)}')

corrMatrix = df[['x1','x2','x3','x4']].corr() 
sns.heatmap(corrMatrix, annot=True) 
plt.show()