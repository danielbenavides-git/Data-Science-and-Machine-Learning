# Simple Data Plotting

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Advertising.csv")
df.head()

df_new = df.head(7)
print(df_new)

plt.scatter(df_new['TV'], df_new['Sales'])
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('TV vs Sales')

plt.show()

plt.scatter(df["TV"], df["Sales"])
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('TV vs Sales')

plt.show()