import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# HELPER #

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def get_poly_pred(x_train, x_test, y_train, degree=1):

    # Generate polynomial features on the train data
    x_poly_train= PolynomialFeatures(degree=degree).fit_transform(x_train)

    # Generate polynomial features on the test data
    print(x_train.shape, x_test.shape, y_train.shape)
    x_poly_test= PolynomialFeatures(degree=degree).fit_transform(x_test)

    # Initialize a model to perform polynomial regression
    polymodel = LinearRegression()

    # Fit the model on the polynomial transformed train data
    polymodel.fit(x_poly_train, y_train)

    # Predict on the entire polynomial transformed test data
    y_poly_pred = polymodel.predict(x_poly_test)
    return y_poly_pred

# END HELPER #

df = pd.read_csv("poly.csv")
pprint(df.head())

x = df.drop(["y"], axis=1)
y = df["y"]

fig, ax = plt.subplots()
ax.plot(x,y,'x')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.set_title('$y$ vs $x$')
plt.show();

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=22)

model = LinearRegression()
model.fit(x_train, y_train)

y_lin_pred = model.predict(x_test)

fig, ax = plt.subplots()
ax.plot(x,y,'x', label='data')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.plot(x_test, y_lin_pred, label='linear model predictions')
plt.legend()
plt.show();

guess_degree = 4
y_poly_pred = get_poly_pred(x_train, x_test, y_train, degree=guess_degree)

idx = np.argsort(x_test.values[:,0])

x_test = x_test.iloc[idx]
y_test = y_test.iloc[idx]

y_lin_pred = y_lin_pred[idx]
y_poly_pred= y_poly_pred[idx]

# First plot x & y values using plt.scatter
plt.scatter(x, y, s=10, label="Test Data")

# Plot the linear regression fit curve
plt.plot(x_test,y_lin_pred,label="Linear fit", color='k')

# Plot the polynomial regression fit curve
plt.plot(x_test, y_poly_pred, label="Polynomial fit",color='red', alpha=0.6)

# Assigning labels to the axes
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend()
plt.show()

### edTest(test_poly_predictions) ###
# Calculate the residual values for the polynomial model

poly_residuals = y_test - y_poly_pred

### edTest(test_linear_predictions) ###
# Calculate the residual values for the linear model

lin_residuals = y_test - y_lin_pred

# Plot the histograms of the residuals for the two cases

# Distribution of residuals
fig, ax = plt.subplots(1,2, figsize = (10,4))
bins = np.linspace(-20,20,20)
ax[0].set_xlabel('Residuals')
ax[0].set_ylabel('Frequency')

# Plot the histograms for the polynomial regression
ax[0].hist(poly_residuals, bins, label = 'poly residuals', color='#B2D7D0', alpha=0.6)

# Plot the histograms for the linear regression
ax[0].hist(lin_residuals, bins, label = 'linear residuals', color='#EFAEA4', alpha=0.6)

ax[0].legend(loc = 'upper left')

# Distribution of predicted values with the residuals
ax[1].hlines(0,-75,75, color='k', ls='--', alpha=0.3, label='Zero residual')
ax[1].scatter(y_poly_pred, poly_residuals, s=10, color='#B2D7D0', label='Polynomial predictions')
ax[1].scatter(y_lin_pred, lin_residuals, s = 10, color='#EFAEA4', label='Linear predictions' )
ax[1].set_xlim(-75,75)
ax[1].set_xlabel('Predicted values')
ax[1].set_ylabel('Residuals')
ax[1].legend(loc = 'upper left')
fig.suptitle('Residual Analysis (Linear vs Polynomial)')
plt.show();