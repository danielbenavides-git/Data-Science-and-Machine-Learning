import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("poly.csv")
print(df.head())

x = df["x"].values
y = df["y"].values

# Plot x & y to visually inspect the data

fig, ax = plt.subplots()
ax.plot(x,y,'x')
ax.set_xlabel('$x$ values')
ax.set_ylabel('$y$ values')
ax.set_title('$y$ vs $x$')
plt.show();

model = LinearRegression()
model.fit(x.reshape(-1,1), y)

y_lin_pred = model.predict(x.reshape(-1,1))

### edTest(test_deg) ###

guess_degree = 4

x_poly= PolynomialFeatures(degree=guess_degree).fit_transform(x.reshape(-1,1))

polymodel = LinearRegression(fit_intercept=False)
polymodel.fit(x_poly, y)

y_poly_pred = polymodel.predict(x_poly)

# To visualise the results, we create a linspace of evenly spaced values
# This ensures that there are no gaps in our prediction line as well as
# avoiding the need to create a sorted set of our data.
# Worth examining and understand the code 

# create an array of evenly spaced values
x_l = np.linspace(np.min(x),np.max(x),100).reshape(-1, 1)

# Prediction on the linspace values
y_lin_pred_l = model.predict(x_l)

# PolynomialFeatures on the linspace values
x_poly_l= PolynomialFeatures(degree=guess_degree).fit_transform(x_l)

# Prediction on the polynomial linspace values
y_poly_pred_l = polymodel.predict(x_poly_l)

plt.scatter(x, y, s=10, label="Data")
plt.plot(x_l, y_lin_pred_l, color = "red", label='Linear Fit')
plt.plot(x_l, y_poly_pred_l, color = "purple" , label='Polynomial Fit')

plt.xlabel("x values")
plt.ylabel("y values")
plt.legend()
plt.show()

### edTest(test_poly_predictions) ###

poly_residuals = y - y_poly_pred

### edTest(test_linear_predictions) ###

lin_residuals = y - y_lin_pred

# Plot the histograms of the residuals for the two cases

# Distribution of residuals

fig, ax = plt.subplots(1,2, figsize = (10,4))
bins = np.linspace(-20,20,20)
ax[0].set_xlabel('Residuals')
ax[0].set_ylabel('Frequency')

#Plot the histograms for the polynomial regression
ax[0].hist(poly_residuals, bins,label = "Poly residuals", color='#B2D7D0', alpha=0.6)

#Plot the histograms for the linear regression
ax[0].hist(lin_residuals, bins, label = "Linear residuals", color='#EFAEA4', alpha=0.6)
ax[0].legend(loc = 'upper left')

# Distribution of predicted values with the residuals
ax[1].scatter(y_poly_pred, poly_residuals, s=10)
ax[1].scatter(y_lin_pred, lin_residuals, s = 10 )
ax[1].set_xlim(-75,75)
ax[1].set_xlabel('Predicted values')
ax[1].set_ylabel('Residuals')
ax[1].legend(['Polynomial','Linear'])

fig.suptitle('Residual Analysis (Linear vs Polynomial)')
plt.show();