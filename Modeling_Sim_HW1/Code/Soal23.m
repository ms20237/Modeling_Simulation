close all;
clear;
clc;

%% Importing files and plot
file_Path = 'pHdata.csv';

% Read the CSV file to a table
data_Table = readtable(file_Path);

% Extract numeric arrays from the table
t = data_Table{:, 1}; % Time step
x1 = data_Table{:, 2}; % input 1
x2 = data_Table{:, 3}; %input 2
y = data_Table{:, 4}; % Output

% Normalize the data
t = (t - min(t)) / (max(t) - min(t));
x1 = (x1 - min(x1)) / (max(x1) - min(x1));
x2 = (x2 - min(x2)) / (max(x2) - min(x2));
y = (y - min(y)) / (max(y) - min(y));

% Create X matrix
X = [ones(size(x1)) x1.^4 x2.^4];

% Scatter plot
figure(1);
scatter(x1, y);
xlabel('Acid solution flow in liters');
ylabel('pH of the solution in the tank');
figure(2);
scatter(x2, y);
xlabel('Base solution flow in liters');
ylabel('pH of the solution in the tank');
figure(3);
scatter3(x1, x2, y, 'Filled');
xlabel('Acid solution flow in liters');
ylabel('Base solution flow in liters');
zlabel('pH of the solution in the tank');

%% Least Square
theta = pinv(X) * y;

theta1 = lsqr(X, y);

% Parameters
intercept = theta(1);
slope_x1 = theta(2);
slope_x2 = theta(3);


% Plotting
y_fit = intercept + slope_x1 * x1 + slope_x2 * x2;
figure;
scatter3(x1, x2, y, 'Filled');

hold on;
scatter3(x1, x2, y_fit, 'Filled');
scatter3(x2, x2, y_fit, 'Filled');
xlabel('Acid solution flow in liters');
ylabel('Base solution flow in liters');
zlabel('pH of the solution in the tank');
legend('Actual Data', 'Filled line');
title('Fitted Line using least square Method');
grid on;
hold off;


% Error
% Caculate prediction values
y_fit = intercept + slope_x1 * x1 + slope_x2 * x2;

% Calculate error
E = y - y_fit;

% Plot Error
figure;
plot(E, 'k', 'LineWidth', 1.5);
xlabel('Data point index');
ylabel('Error(y y-fit)');
title('Error Plot = Actual Value - Fitted Value');
grid on;


%% Least Squares with Forgetting Factor
% Parameters
lambda = 0.9; % recomended: 0.7 < lambda < 0.9

% Forgetting Factor
theta = zeros(size(X, 2), 1);
P = eye(size(X, 2)) / lambda;

% Loop through all available data points (up to the length of y)
for i = 1:length(y)
    
   x_i = X(i, :)';
   y_predicted = x_i' * theta;
   e = y(i) - y_predicted;
   K = P * x_i/(lambda + x_i' * P * x_i);
   theta = theta + K * e;
   P = (P - K * x_i' * P)/lambda;
end

% Error
% Caculate prediction values
% y_fit = intercept + slope_x1 * x1 + slope_x2 * x2;
y_fit = X * theta; % Predicted output
% Calculate error
E = y - y_fit;

% Plotting Error
figure;
plot(E, 'k', 'LineWidth', 1.5);
xlabel('Data point Error');
ylabel('Index');
title('Error Plot = Actual Value - Fitted Value with Forgetting Factor');
grid on;

    
%% Sliding Window

Window_size = 50;
Step_size = 1;

num_points = length(y);
num_windows = floor((num_points - Window_size)/Step_size) + 1;
intercepts = zeros(num_windows, 1);
slope_x1 = zeros(num_windows, 1);
slope_x2 = zeros(num_windows, 1);

errors = zeros(num_windows, Window_size);

% Sliding window least square
for i = 1:num_windows
   start_idx = (i-1)*Step_size + 1;
   end_idx = start_idx + Window_size - 1;
   
   x1_window = x1(start_idx:end_idx);
   x2_window = x2(start_idx:end_idx);
   y_window = y(start_idx:end_idx);
   
   X = [ones(size(x1_window)) x1_window x2_window];
   theta = pinv(X) * y_window;
   intercept(i) = theta(1);
   slope_x1(i) =  theta(2);
   slope_x2(i) =  theta(3);
   
   y_fit = X * theta;
   
   errors(i,:) = y_window - y_fit;
end    

% Plotting
figure;

subplot(2,1,1);
hold on;
for i = 1:num_windows
    x1_window = x1((i-1) * Step_size + 1: (i-1) * Step_size + Window_size);
    x2_window = x2((i-1) * Step_size + 1: (i-1) * Step_size + Window_size);

    y_fit = intercept(i) + slope_x1(i) * x1_window + slope_x2(i) * x2_window;
    plot3(x1_window, x2_window, y_fit);
end

scatter3(x1, x2, y, 'o');
xlabel('Acid solution flow in liters');
ylabel('Base solution flow in liters');
zlabel('pH of the solution in the tank');
title('Fitted Lines using Sliding Window Least Squares Error');
grid on;
hold off;

subplot(2, 1, 2);
plot(errors', 'k', 'LineWidth', 1.5);
xlabel('Data point Error');
ylabel('Error (y - y-fit)');
title('Error (y - y-fit)');
grid on;

%% RLS method for Sliding window
Window_Size = 50;
Step_Size = 1;

% Parameters
intercepts = zeros(length(y) - Window_Size + 1, 1);
slope_x1 = zeros(length(y) - Window_Size + 1, 1);
slope_x2 = zeros(length(y) - Window_Size + 1, 1);

errors = zeros(length(y) - Window_Size + 1, Window_Size);

for i = 1:length(y) - Window_Size + 1
    x1_window = x1(i:i+Window_Size-1);
    x2_window = x2(i:i+Window_Size-1);
    y_window = y(i:i+Window_Size-1);
    
    P = eye(3);
    theta = zeros(3, 1);
    window_errors = zeros(1, Window_Size);
    
    for j = 1:Window_Size
        x = [1; x1_window(j); x2_window(j)];
        e = y_window(j) - x' * theta;
        K = (P * x)/(1 + x' * P * x);
        theta = theta + K * e;
        P = (P - K * x' * P);
        window_errors(j) = e;
    end
    
    errors(i, :) = window_errors;
    
end    

% Plotting
figure;

subplot(2, 1, 1);
hold on;
for i = 1:length(y) - Window_Size + 1
    x1_window = x1(i:i+Window_Size-1);
    x2_window = x2(i:i+Window_Size-1);
    
    y_fit = intercept(i) + slope_x1(i) * x1_window + slope_x2(i) * x2_window;
    plot3(x1_window, x2_window, y_fit, 'r');
    
end

scatter3(x1, x2, y, 'Filled');
xlabel('Acid solution flow in liters');
ylabel('Base solution flow in liters');
zlabel('pH of the solution in the tank');
title('Fitted Lines using Sliding Window RLS Method');
grid on;
hold off;

subplot(2, 1, 2);
plot(errors', 'k', 'LineWidth', 1.5);
xlabel('Data point index');
ylabel('Error(y - y-fit)');
title('Errors plot:Actual Value - Fitted Value(Sliding Window RLS Method)');
grid on;
