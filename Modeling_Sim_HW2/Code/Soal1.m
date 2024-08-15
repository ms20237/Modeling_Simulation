close all;
clear;
clc;

%% Load dataset
dataset = importdata('ESL.arff');
x = dataset.data(:, 1:4);
y = dataset.data(:, 5);
n = size(x, 2);

%% Forward selestion method
best_regressors = [];
best_theta = [];
best_rss = inf;
n = size(x, 2);

for k = 1:n
    combinations = nchoosek(1:n, k);
    for i = 1:size(combinations, 1)
        selected_regressors = combinations(i, :);
        X_selected = x(:, selected_regressors);
        theta = pinv(X_selected) * y;
        
        y_fit = X_selected * theta;
        rss = sum((y - y_fit).^2);
        if rss < best_rss
            best_rss = rss;
            best_regressors = selected_regressors;
            best_theta = theta;
        end    
    end    
end    

fprintf('Selected Regressors: %s\n', num2str(best_regressors));
fprintf('Corresponding Parameters: \n');
disp(best_theta);
