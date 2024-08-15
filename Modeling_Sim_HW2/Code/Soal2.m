close all;
clear;
clc;

%% Load dataset
dataset = importdata('ESL.arff');
x = dataset.data(:, 1:4);
y = dataset.data(:, 5);
n = size(x, 2);

%% Backward Elimination
selected_regressors = 1:n;
theta_hat = [];

while numel(selected_regressors) > 0
    X_selected = x(:, selected_regressors);
    theta = pinv(X_selected) * y;
    y_fit = X_selected * theta;
    rss = sum((y - y_fit).^2);
    theta_hat = [theta_hat; theta'];
    regressors_to_remove = [];
    best_rss = rss;
    
    for i = 1:numel(selected_regressors)
        temp_selected_regressors = selected_regressors(selected_regressors);
        temp_X_selected = x(:, temp_selected_regressors);
        temp_theta = pinv(temp_X_selected) * y;
        temp_Y_fit = temp_X_selected * temp_theta;
        temp_rss = sum((y - y_fit).^2);
        
        if temp_rss < best_rss
            best_rss = temp_rss;
            regressors_to_remove = selected_regressors(i);
        end
    end
    
    if ~isempty(regressors_to_remove)
        selected_regressors = selected_regressors(selected_regressors);
    else 
        break; 
    end
end

fprintf('Selected Regressors: %s\n', num2str(selected_regressors));
fprintf('Corresponding Parameters: \n');
disp(theta_hat);
   