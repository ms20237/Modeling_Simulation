close all;
clear;
clc;


% Parameters
N = 1000;  % Number of samples
Ts = 1;    % Sampling time
variance = 0.2;

% Generate PRBS signal
u = idinput(N, 'prbs', [0 1], [-1 1]);

% Generate Gaussian noise
e = sqrt(variance) * randn(N, 1);

% Define the system using idpoly
A = [1 8/10];  % Coefficients for y(t) and y(t-1)
B = [0 0.5];   % Coefficients for u(t-1), notice the zero for the delay
C = 1;         % Coefficient for the noise e(t)

% Create the idpoly model
sys = idpoly(A, B, C, [], [], Ts);

% Create data object for simulation
data = iddata([], u, Ts);

% Simulate the output (without noise)
y_sim = sim(sys, data);

% Extract the simulated output data
y_data = y_sim.OutputData;

% Add noise to the simulated output
y = y_data + e;

% Plot the results
figure;
subplot(3, 1, 1);
plot(u);
title('PRBS Signal');
xlabel('Time');
ylabel('u(t)');

subplot(3, 1, 2);
plot(e);
title('Gaussian Noise');
xlabel('Time');
ylabel('e(t)');

subplot(3, 1, 3);
plot(y);
title('Output Signal y(t)');
xlabel('Time');
ylabel('y(t)');

