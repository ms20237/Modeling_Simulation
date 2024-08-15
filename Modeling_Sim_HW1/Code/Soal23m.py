import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO


# import numpy as np
file_path = 'pHdata.dat' 
with open(file_path, 'r') as file: 
    content = file.read() 
    print(type(content))
    
    data = StringIO(content)
    df = pd.read_csv(data, sep='   ', header=None)
    print(df)


# Extract numeric arrays from the DataFrame
t = df.iloc[:, 0].values  # Time step
x1 = df.iloc[:, 1].values  # input 1
x2 = df.iloc[:, 2].values  # input 2
y = df.iloc[:, 3].values  # Output

# Normalize the data
t = (t - np.min(t)) / (np.max(t) - np.min(t))
x1 = (x1 - np.min(x1)) / (np.max(x1) - np.min(x1))
x2 = (x2 - np.min(x2)) / (np.max(x2) - np.min(x2))
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# Create X matrix
X = np.column_stack((np.ones_like(x1), x1**4, x2**4))

# Scatter plot
plt.figure(1)
plt.scatter(x1, y)
plt.xlabel('Acid solution flow in liters')
plt.ylabel('pH of the solution in the tank')

plt.figure(2)
plt.scatter(x2, y)
plt.xlabel('Base solution flow in liters')
plt.ylabel('pH of the solution in the tank')


fig = plt.figure()
ax = plt.axes(projection = '3d')
sctt = ax.scatter3D(x1, x2, y, alpha = 0.8, color = 'blue')
plt.title('data(3D)')
ax.set_xlabel('In(x1)', fontweight = 'bold')
ax.set_ylabel('In(x2)', fontweight = 'bold')
ax.set_zlabel('In(y)', fontweight = 'bold')
plt.show()


## Least Square
theta = np.linalg.pinv(X) @ y

theta1 = np.linalg.lstsq(X, y, rcond=None)[0]

# Parameters
intercept = theta[0]
slope_x1 = theta[1]
slope_x2 = theta[2]

# Plotting
y_fit = intercept + slope_x1 * x1 + slope_x2 * x2

fig = plt.figure()
ax = plt.axes(projection = '3d')

ax.scatter3D(x1, x2, y_fit, alpha = 0.8, color = 'red')
ax.scatter3D(x1, x2, y, alpha = 0.8, color = 'blue')

plt.title('Line of Least square method')
ax.set_xlabel('In(x1)', fontweight = 'bold')
ax.set_ylabel('In(x2)', fontweight = 'bold')
ax.set_zlabel('In(y)', fontweight = 'bold')

plt.show()



# Error
# Calculate prediction values
y_fit = intercept + slope_x1 * x1 + slope_x2 * x2
E = y - y_fit

# Plot Error
plt.figure()
plt.plot(E, 'k', linewidth=1.5)
plt.xlabel('Data point index')
plt.ylabel('Error(y - y-fit)')
plt.title('Error Plot = Actual Value - Fitted Value')
plt.grid()

# Parameters
_lambda = 0.9  # recommended: 0.7 < lambda < 0.9

# Forgetting Factor
theta = np.zeros((X.shape[1], 1))
P = np.eye(X.shape[1]) / _lambda

# Loop through all available data points (up to the length of y)
for i in range(len(y)):
    x_i = X[i, :].reshape(-1, 1)
    y_predicted = x_i.T @ theta
    e = y[i] - y_predicted
    K = P @ x_i / (_lambda + x_i.T @ P @ x_i)
    theta = theta + K * e
    P = (P - K @ x_i.T @ P) / _lambda

# Error
# Calculate prediction values
y_fit = X @ theta
E = y - y_fit

# Plotting Error
plt.figure()
plt.plot(E, 'k', linewidth=1.5)
plt.xlabel('Data point Error')
plt.ylabel('Index')
plt.title('Error Plot = Actual Value - Fitted Value with Forgetting Factor')
plt.grid(True)
plt.show()


## Sliding Window
Window_size = 50
Step_size = 1

num_points = len(y)
num_windows = (num_points - Window_size) // Step_size + 1
intercepts = np.zeros(num_windows)
slope_x1 = np.zeros(num_windows)
slope_x2 = np.zeros(num_windows)

errors = np.zeros((num_windows, Window_size))

# Sliding window least square
for i in range(num_windows):
    start_idx = (i * Step_size)
    end_idx = start_idx + Window_size

    x1_window = x1[start_idx:end_idx]
    x2_window = x2[start_idx:end_idx]
    y_window = y[start_idx:end_idx]

    X_window = np.column_stack((np.ones_like(x1_window), x1_window, x2_window))
    theta_window = np.linalg.pinv(X_window) @ y_window
    intercepts[i] = theta_window[0]
    slope_x1[i] = theta_window[1]
    slope_x2[i] = theta_window[2]

    y_fit_window = X_window @ theta_window

    errors[i, :] = y_window - y_fit_window

# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(211, projection='3d')

for i in range(num_windows):
    x1_window = x1[i * Step_size:i * Step_size + Window_size]
    x2_window = x2[i * Step_size:i * Step_size + Window_size]
    y_fit = intercepts[i] + slope_x1[i] * x1_window + slope_x2[i] * x2_window
    ax1.plot(x1_window, x2_window, y_fit)

ax1.scatter(x1, x2, y, c='r', marker='o')
ax1.set_xlabel('Acid solution flow in liters')
ax1.set_ylabel('Base solution flow in liters')
ax1.set_zlabel('pH of the solution in the tank')
ax1.set_title('Fitted Lines using Sliding Window Least Squares Error')

ax2 = fig.add_subplot(212)
ax2.plot(errors.T, 'k', linewidth=1.5)
ax2.set_xlabel('Data point Error')
ax2.set_ylabel('Error (y - y-fit)')
ax2.set_title('Error (y - y-fit)')
plt.grid(True)
plt.show()

## RLS method for Sliding window
Window_Size = 50
Step_Size = 1

# Parameters
intercepts = np.zeros(len(y) - Window_Size + 1)
slope_x1 = np.zeros(len(y) - Window_Size + 1)
slope_x2 = np.zeros(len(y) - Window_Size + 1)

errors = np.zeros((len(y) - Window_Size + 1, Window_Size))

for i in range(len(y) - Window_Size + 1):
    x1_window = x1[i:i + Window_Size]
    x2_window = x2[i:i + Window_Size]
    y_window = y[i:i + Window_Size]

    P = np.eye(3)
    theta = np.zeros((3, 1))
    window_errors = np.zeros(Window_Size)

    for j in range(Window_Size):
        x = np.array([[1], [x1_window[j]], [x2_window[j]]])
        e = y_window[j] - x.T @ theta
        K = (P @ x) / (1 + x.T @ P @ x)
        theta = theta + K * e
        P = P - K @ x.T @ P
        window_errors[j] = e

    errors[i, :] = window_errors

# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(211, projection='3d')

for i in range(len(y) - Window_Size + 1):
    x1_window = x1[i:i + Window_Size]
    x2_window = x2[i:i + Window_Size]
    y_fit = intercepts[i] + slope_x1[i] * x1_window + slope_x2[i] * x2_window
    ax1.plot(x1_window, x2_window, y_fit, 'r')

ax1.scatter(x1, x2, y, c='b', marker='o')
ax1.set_xlabel('Acid solution flow in liters')
ax1.set_ylabel('Base solution flow in liters')
ax1.set_zlabel('pH of the solution in the tank')
ax1.set_title('Fitted Lines using Sliding Window RLS Method')

ax2 = fig.add_subplot(212)
ax2.plot(errors.T, 'k', linewidth=1.5)
ax2.set_xlabel('Data point index')
ax2.set_ylabel('Error(y - y-fit)')
ax2.set_title('Errors plot:Actual Value - Fitted Value(Sliding Window RLS Method)')
plt.grid(True)
plt.show()
