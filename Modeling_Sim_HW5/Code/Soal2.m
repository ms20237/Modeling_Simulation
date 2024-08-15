close all;
clear;
clc;

% Define the differential equation
f = @(t, x) x^3 * t;

% Initial conditions
x0 = 1;
t0 = 0;
t_final = 1; % Define the final time
h = 0.01; % Step size

% Number of steps
N = (t_final - t0) / h;

% Preallocate arrays for efficiency
t = zeros(1, N+1);
x = zeros(1, N+1);

% Set initial values
t(1) = t0;
x(1) = x0;

%% Euler Method

% Euler method
for i = 1:N
    x(i+1) = x(i) + h * f(t(i), x(i));
    t(i+1) = t(i) + h;
end

% Plot the result
figure(1);
plot(t, x);
title('Euler Method');
xlabel('t');
ylabel('x(t)');
grid on;


%% Taylor Method
dfdt = @(t, x) x^3; % Partial derivative with respect to t
dfdx = @(t, x) 3 * x^2 * t; % Partial derivative with respect to x

% Taylor method
for i = 1:N
    x(i+1) = x(i) + h * f(t(i), x(i)) + 0.5 * h^2 * (dfdt(t(i), x(i)) + f(t(i), x(i)) * dfdx(t(i), x(i)));
    t(i+1) = t(i) + h;
end

% Plot the result
figure(2);
plot(t, x);
title('Taylor Method');
xlabel('t');
ylabel('x(t)');
grid on;


%% Runge-Kutta Method (4th order)
% Runge-Kutta method (4th order)
for i = 1:N
    k1 = h * f(t(i), x(i));
    k2 = h * f(t(i) + 0.5*h, x(i) + 0.5*k1);
    k3 = h * f(t(i) + 0.5*h, x(i) + 0.5*k2);
    k4 = h * f(t(i) + h, x(i) + k3);
    x(i+1) = x(i) + (1/6)*(k1 + 2*k2 + 2*k3 + k4);
    t(i+1) = t(i) + h;
end

% Plot the result
figure(3);
plot(t, x);
title('Runge-Kutta Method');
xlabel('t');
ylabel('x(t)');
grid on;

%% Compare methods
% Define the differential equation and its derivatives
f = @(t, x) x^3 * t;
dfdt = @(t, x) x^3; % Partial derivative with respect to t
dfdx = @(t, x) 3 * x^2 * t; % Partial derivative with respect to x

% Initial conditions
x0 = 1;
t0 = 0;
t_final = 1; % Define the final time
h = 0.01; % Step size

% Number of steps
N = (t_final - t0) / h;

% Preallocate arrays for efficiency
t = zeros(1, N+1);
x_euler = zeros(1, N+1);
x_taylor = zeros(1, N+1);
x_rk4 = zeros(1, N+1);

% Set initial values
t(1) = t0;
x_euler(1) = x0;
x_taylor(1) = x0;
x_rk4(1) = x0;

% Euler method
for i = 1:N
    x_euler(i+1) = x_euler(i) + h * f(t(i), x_euler(i));
    t(i+1) = t(i) + h;
end

% Taylor method
for i = 1:N
    x_taylor(i+1) = x_taylor(i) + h * f(t(i), x_taylor(i)) + 0.5 * h^2 * (dfdt(t(i), x_taylor(i)) + f(t(i), x_taylor(i)) * dfdx(t(i), x_taylor(i)));
end

% Runge-Kutta method (4th order)
for i = 1:N
    k1 = h * f(t(i), x_rk4(i));
    k2 = h * f(t(i) + 0.5*h, x_rk4(i) + 0.5*k1);
    k3 = h * f(t(i) + 0.5*h, x_rk4(i) + 0.5*k2);
    k4 = h * f(t(i) + h, x_rk4(i) + k3);
    x_rk4(i+1) = x_rk4(i) + (1/6)*(k1 + 2*k2 + 2*k3 + k4);
end

% Plot the results
figure(4);
plot(t, x_euler, 'r', t, x_taylor, 'g', t, x_rk4, 'b');
legend('Euler Method', 'Taylor Method', 'Runge-Kutta Method');
title('Comparison of Numerical Methods');
xlabel('t');
ylabel('x(t)');
grid on;
