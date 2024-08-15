close all;
clear;
clc

%% Importing files and plot
file_Path = 'batch-yield-and-purity.csv';

% Read the CSV file to a table
data_Table = readtable(file_Path);

% Extract numeric arrays from the table
x = data_Table{:, 1}; % Inputs
y = data_Table{:, end}; % Outputs

% Normalize the data
x = (x - min(x)) / (max(x) - min(x));
y = (y - min(y)) / (max(y) - min(y));

% Create X matrix
X = [ones(size(x)) x];

% Scatter plot
figure(1);
scatter(x, y);

%% Least Square
theta = pinv(X) * y;

thetal = lsqr(X,y);

% Parameters
intercept = theta(1);
slope_x = theta(2);

% Plotting
y_fit = intercept + slope_x * x;
figure;
scatter(x, y,'filled');
hold on;
scatter(x, y_fit, 'Filled');
scatter(x, y_fit, 'Filled');
legend('Actual Data', 'Filled line');
title('Best Line');
grid on;
hold off;


% Error
y_fit = intercept + slope_x * x;

% Calculate error
E = y - y_fit;

% Plot Error
figure;
plot(E, 'LineWidth', 1.5);
title('Error Plot = Actual Value - Fitted Value');
grid on;
