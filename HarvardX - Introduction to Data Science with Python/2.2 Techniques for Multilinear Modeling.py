import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("credit.csv")
df.head()

x = df.drop("Balance", axis=1)
y = df["Balance"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

try:
    test_model = LinearRegression().fit(x_train, y_train)
except Exception as e:
    print('Error!:', e)

df.dtypes

### edTest(test_model1) ###

numeric_features = df.select_dtypes(include=[np.number]).columns.drop("Balance")

model1 = LinearRegression().fit(x_train[numeric_features], y_train)

train_score = model1.score(x_train[numeric_features], y_train)
test_score = model1.score(x_test[numeric_features], y_test)
print('Train R2:', round(train_score, 3))
print('Test R2:', round(test_score, 3))

### edTest(test_design) ###

x_train_design = pd.get_dummies(x_train, drop_first=True)
x_test_design = pd.get_dummies(x_test, drop_first=True)
print(x_train_design.head())
print(x_test_design.head())

x_train_design.dtypes

### edTest(test_model2) ###

model2 = LinearRegression().fit(x_train_design, y_train)

train_score = model2.score(x_train_design, y_train)
test_score = model2.score(x_test_design, y_test)
print('Train R2:', round(train_score, 3))
print('Test R2:', round(test_score, 3))

coefs = pd.DataFrame(model2.coef_, index=x_train_design.columns, columns=['beta_value'])
coefs

# sns.barplot(data=coefs.T, orient='h').set(title='Model Coefficients')
# plt.show()

### edTest(test_model3) ###

best_cat_feature = coefs[coefs['beta_value'] == coefs['beta_value'].max()].index[0]

features = ['Income', best_cat_feature]
model3 = LinearRegression()
model3.fit(x_train_design[features], y_train)

beta0 = model3.intercept_
beta1 = model3.coef_[features.index('Income')]
beta2 = model3.coef_[features.index(best_cat_feature)]

coefs = pd.DataFrame([beta0, beta1, beta2], index=['Intercept']+features, columns=['beta_value'])

sns.barplot(data=coefs.T, orient='h').set(title='Model Coefficients');
plt.show()

### edTest(test_prediction_lines) ###

x_space = np.linspace(x['Income'].min(), x['Income'].max(), 1000)

# When categorical feature is true/present (1)
y_hat_yes = beta0 + beta1 * x_space + beta2 * 1
# When categorical feature is false/absent (0)
y_hat_no = beta0 + beta1 * x_space + beta2 * 0

ax = sns.scatterplot(data=pd.concat([x_train_design, y_train], axis=1), x='Income', y='Balance', hue=best_cat_feature, alpha=0.8)
ax.plot(x_space, y_hat_no)
ax.plot(x_space, y_hat_yes)
plt.show()