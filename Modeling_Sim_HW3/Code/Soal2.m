close all;
clear;
clc;


% Define the parameters for chirp signal
t1 = 1; % Time of f1(s)
fs = 500;  % Sampling Frequency (Hz)
t = 0:1/fs:t1; % Time vector from 0 to 2 seconds with a sampling frequency of 1 kHz
f0 = 0; % Starting frequency (Hz)
f1 = 250; % Ending frequency (Hz)


% Generate the chirp signal
u = chirp(t, f0, t1, f1);
u = transpose(u);

% Plot the chirp signal
figure(1);
plot(t, u);
xlabel('Time');
ylabel('Amplitude');
title('Chirp Signal');


% Define the parameters for white noise signal
N = 2001;  % Number of samples
variance = 0.25;

% Generate white noise with mean 0 and variance equal to the desired variance
e = sqrt(variance) * randn(length(t), 1); % White Sigal

input_data = [u e];

% Plot the white noise signal
figure(2);
plot(e);
xlabel('Time');
ylabel('Amplitude');
title('White Noise Signal (Variance = 0.25)');


% Model
% Define the parameters for Model
A = [1 2];
B = 1;
C = [1 0.5];
model = idpoly(A, B, [], [], C, 0, 1/fs);


% Output Signal
y_sim = sim(model, input_data);


idata = iddata(y_sim, u, 0.001); 


% Plot the Output Signal
figure(3);
plot(t, y_sim);
xlabel('Time');
ylabel('Amplitude');
title('Output Signal');


% Plotting combined for better comparision
figure(4);
plot(t, u, t, y_sim);
legend('Chirp Signal', 'System Output');
title('Chirp Input and System Output');
xlabel('Time(s)');
ylabel('Amplitude');





% My Model
y = zeros(size(u));
for i = 3:length(u)
    y(i) = u(i) + 2*u(i-1) + e(i) + 0.5*e(i-1);
end

% Models
% ARX
arx_model = arx(iddata(y, u, 1), [2 1 1]);
disp('ARX:');
disp(arx_model.B);

% BJ
bj_model = bj(iddata(y, u, 1), [2 1 1 0 1]);
disp('BJ:');
disp(bj_model.B);

% OE
oe_model = oe(iddata(y, u, 1), [2 1 1]);
disp('OE:');
disp(oe_model.B);

% Plotting
figure;
subplot(2,1,1);
plot(t, u, 'b', t, y, 'r--');
legend('Input Signal', 'Output Signal');
xlabel('Time (s)');
ylabel('Signal');
title('Input and Output Signals');

subplot(2,1,2);
spectrogram(y, hamming(128), 120, 128, 1000, 'yaxis');
title('Frequency Spectrum');