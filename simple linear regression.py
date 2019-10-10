import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv("Salary_Data.csv")
print(data.head(10))

plt.scatter(data["YearsExperience"] ,data["Salary"])
plt.title("Salary based on Experience")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

independent_x = data.iloc[:,:-1].values
dependent_y = data.iloc[:,1].values
print(independent_x)
print(dependent_y)

train_x , test_x , train_y , test_y = train_test_split(independent_x ,dependent_y , test_size=0.3, random_state=0)

lin = LinearRegression()
lin.fit(train_x , train_y)


plt.scatter(train_x , train_y , color = "blue")
plt.plot(train_x , lin.predict(train_x) , color = "yellow")
plt.xlabel("Years")
plt.ylabel("Salary")
plt.title("Simple Linear Regression training data")
plt.show()

plt.scatter(test_x , test_y , color = "blue")
plt.plot(train_x , lin.predict(train_x) , color = "yellow")
plt.xlabel("Years")
plt.ylabel("Salary")
plt.title("Simple Linear Regression testing data")
plt.show()

print(test_y)
y_pred = lin.predict(test_x)
print("*****************************")
print(y_pred)


z = np.array([[11]])

y_pred1 = lin.predict(z)
print(y_pred1)


print("coef" , lin.coef_)
print("intercept" , lin.intercept_)

"""coef [9360.26128619]
intercept 26777.391341197625"""

