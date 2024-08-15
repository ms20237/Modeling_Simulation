close all;
clear;
clc;

%% Importing files and plot
file_Path = 'ballbeam.dat';

% Read the CSV file to a table
data_Table = readtable(file_Path);

% Extract numeric arrays from the table
u = data_Table{:, 1}; % Input 
y = data_Table{:, end}; % Output


% Step 1: Calculate correlation
corr_coef = corr(u, y);
disp(['Correlation coefficient between input and output: ', num2str(corr_coef)]);


% Prepare the data for system identification
data = iddata(y, u, 0.001); % 0.001 is the sample time


% Use N4SID method to identify the system with a maximum order of 20
sys = n4sid(data, 20);

% Display the identified system
disp('Identified System:');
disp(sys);




% Extract state-space matrices from the identified system
[A, B, C, D] = ssdata(sys);

% Process and measurement noise covariances (you may need to tune these)
Q = eye(size(A)) * 0.01; % Process noise covariance
R = 0.01; % Measurement noise covariance

% Initial state and covariance guess
x0 = zeros(size(A, 1), 1); % Initial state guess
P0 = eye(size(A)); % Initial state covariance guess

% Kalman filter initialization
kalmanFilter = kalman(A, B, C, D, Q, R, P0, x0);

% Initialize state and output arrays for storing the filter estimates
x_estimated = zeros(size(A, 1), length(u));
y_estimated = zeros(size(y));

% Kalman filter implementation
for i = 1:length(u)
    % Prediction step
    kalmanFilter = kalman_predict(kalmanFilter, u(i));
    
    % Correction step
    [kalmanFilter, x_estimated(:, i)] = kalman_correct(kalmanFilter, y(i));
    y_estimated(i) = C * x_estimated(:, i) + D * u(i); % Estimated output
end

% Plot the results
t = (0:length(u)-1) * 0.001; % Time vector
figure;
subplot(2,1,1);
plot(t, y, 'b', t, y_estimated, 'r--');
xlabel('Time (s)');
ylabel('Output');
legend('Measured', 'Estimated');
title('Output Estimation');

subplot(2,1,2);
plot(t, x_estimated);
xlabel('Time (s)');
ylabel('State');
legend('State 1', 'State 2'); % Add legend entries for each state
title('State Estimation');

