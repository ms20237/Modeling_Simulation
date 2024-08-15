import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # import dataset from csv file
dataset = pd.read_csv('batch-yield-and-purity.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print dataset
print(dataset.shape)
print(dataset.head())

# Ploting Scatter Points
plt.scatter(X, y, c='#ef5423', label='purity vs yield')
 
plt.xlabel('yield')
plt.ylabel('purity')
plt.legend()
plt.show()


# Least square
# Mean x and y
mean_X = np.mean(X)
mean_y = np.mean(y)

# Total numbers of values
n = len(X)


# Using the formula to calculate 'm' and 'c'
numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_X) * (y[i] - mean_y)
    denom += (X[i] - mean_X) ** 2
m = numer / denom
c = mean_y - (m * mean_X)
 
# Printing coefficients
print("Coefficients")
print(m, c)


# Plotting Values and Regression Line
 
max_x = np.max(X) + 100
min_x = np.min(X) - 100
 
# Calculating line values x and y
L_X = np.linspace(min_x, max_x, 1000)
L_y = c + m * L_X


 
# Ploting Line
plt.plot(L_X, L_y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, y, c='#ef5423', label='purity vs yield(Least square)')
 
plt.xlabel('yield')
plt.ylabel('purity')
plt.legend()
plt.show()


# Compute least squares error
errors = []
for i in range(n):
    predicted_y = m * X[i] + c
    error = (y[i] - predicted_y) ** 2
    errors.append(error)

# Plot least squares error
plt.scatter(X, errors, c='#ef5423', label='purity vs yield(Least square)')

plt.xlabel('yield')
plt.ylabel('Least Squares Error')
plt.legend()
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Training the simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



# Predict method for set results
y_Pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title('purity vs yield(Least square_Prediction Line)')
plt.xlabel('yield')
plt.ylabel('purity')
plt.show()



# Calculating Root Mean Squares Error
rmse = 0
for i in range(n):
    y_pred = c + m * X[i]
    rmse += (y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/n)
print("RMSE")
print(rmse)

