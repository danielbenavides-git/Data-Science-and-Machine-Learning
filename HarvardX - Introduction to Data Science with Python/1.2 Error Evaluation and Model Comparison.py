import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# EXERCISE 2: Simple KNN Regression

# PART 1: KNN by hand for k = 1

filename = 'Advertising.csv'
df_adv = pd.read_csv(filename)
df_adv.head()

x_true = df_adv.TV.iloc[5:13]
y_true = df_adv.Sales.iloc[5:13]

idx = np.argsort(x_true)

x_true = x_true.iloc[idx].values
y_true = y_true.iloc[idx].values

### edTest(test_findnearest) ###
# Define a function that finds the index of the nearest neighbor 
# and returns the value of the nearest neighbor.  
# Note that this is just for k = 1 where the distance function is 
# simply the absolute value.

def find_nearest(array, value):
    idx = np.abs(array - value).argmin()
    return idx, array[idx]

x = np.linspace(np.min(x_true), np.max(x_true), 100)
y = np.zeros((len(x)))

for i, xi in enumerate(x):
    y[i] = y_true[find_nearest(x_true, x[i])[0]]

plt.plot(x, y, '-.')

plt.plot(x_true, y_true, 'kx')

plt.title('TV vs Sales')
plt.xlabel('TV budget in $1000')
plt.ylabel('Sales in $1000')

plt.show()

# PART 2: KNN for K >= 1 using sklearn

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

data_filename = 'Advertising.csv'
df = pd.read_csv(data_filename)
df.head()

x = df[['TV']].values
y = df['Sales'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=66)

k_value_min = 1
k_value_max = 70

k_list = np.linspace(k_value_min, k_value_max, 70, dtype=int)

fig, ax = plt.subplots(figsize=(10, 6))
j = 0

for k_value in k_list:
    model = KNeighborsRegressor(n_neighbors=int(k_value))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    colors = ['grey','r','b']
    if k_value in [1,10,70]:
        xvals = np.linspace(x.min(),x.max(),100).reshape(-1,1)
        ypreds = model.predict(xvals)
        ax.plot(xvals, ypreds,'-',label = f'k = {int(k_value)}',linewidth=j+2,color = colors[j])
        j+=1
        
ax.legend(loc='lower right',fontsize=20)
ax.plot(x_train, y_train,'x',label='train',color='k')
ax.set_xlabel('TV budget in $1000',fontsize=20)
ax.set_ylabel('Sales in $1000',fontsize=20)
plt.tight_layout()

plt.show()

# EXERCISE 3: Finding the best K in KNN regression

from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

data_filename = 'Advertising.csv'
df = pd.read_csv(data_filename)
df.head()

x = df[['TV']].values
y = df['Sales'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, 
                                                    random_state=66)

k_value_min = 1
k_value_max = 70

k_list = np.linspace(k_value_min, k_value_max, 70, dtype=int)

fig, ax = plt.subplots(figsize=(10, 6))
j = 0

knn_dict = {}

for k_value in k_list:
    model = KNeighborsRegressor(n_neighbors=int(k_value))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    MSE = mean_squared_error(y_test, y_pred)
    knn_dict[k_value] = MSE

    colors = ['grey','r','b']
    if k_value in [1,10,70]:
        xvals = np.linspace(x.min(),x.max(),100).reshape(-1, 1)
        ypreds = model.predict(xvals)
        ax.plot(xvals, ypreds,'-',label = f'k = {int(k_value)}',linewidth=j+2,color = colors[j])
        j+=1
        
ax.legend(loc='lower right',fontsize=20)
ax.plot(x_train, y_train,'x',label='test',color='k')
ax.set_xlabel('TV budget in $1000',fontsize=20)
ax.set_ylabel('Sales in $1000',fontsize=20)
plt.tight_layout()

plt.show()

# Plot a graph which depicts the relation between the k values and MSE

plt.figure(figsize=(8, 6))
plt.plot(list(knn_dict.keys()), list(knn_dict.values()), 'k.-', alpha=0.5, linewidth=2)

plt.xlabel('k', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.title('Test $MSE$ values for different k values - KNN regression', fontsize=20)
plt.tight_layout()

plt.show()

# Best KNN model

min_mse = min(knn_dict.values())
best_model = [key  for (key, value) in knn_dict.items() if value == min_mse]

print("The best k value is ", best_model, "with a MSE of ", round(min_mse),0)

model = KNeighborsRegressor(n_neighbors=best_model[0])
model.fit(x_train,y_train)
y_pred_test = model.predict(x_test)

print(f"The R2 score for your model is {round(r2_score(y_test, y_pred_test),2)}")