% Evaluation script for the collaborative filtering problem. Loads data
% matrix, splits known values into training and testing sets, and computes
% the MSE of the predicted to the true known entries of the test data.
%
% Loads data from Data.mat and calls PredictMissingValues.m.
clear all;

% Setup
%rand('seed', 1);  % fix random seed for reproducibility

% Constants
filename = 'Data.mat';
prc_trn = 0.5;  % percentage of training data
nil = 0;  % missing value indicator

% Load data
L = load(filename);
X = L.X;

% Split intro training and testing index sets
idx = find(X ~= nil); 
n = numel(idx);

n_trn = round(n*prc_trn);
rp = randperm(n);
idx_trn = idx(rp(1:n_trn));
idx_tst = idx(rp(n_trn+1:end));

% Build training and testing matrices
X_trn = ones(size(X))*nil;
X_trn(idx_trn) = X(idx_trn);  % add known training values

X_tst = ones(size(X))*nil;
X_tst(idx_tst) = X(idx_tst);  % add known training values

global k lambda learning_rate;
k = 10;
lambda = 18;
learning_rate = 0.1;

% Loop through epocs until convergence or overfitting
rmse1 = [];
rmse2 = [];
for e = 1:100
    display(['Epoch: ' num2str(e)]);
    
    % Predict the missing values here!
    tic;
    X_pred = PredictMissingValues(X_trn, nil);
    toc
    
    % Compute MSE
    rmse1 = [rmse1 sqrt(mean((X_tst(X_tst ~= nil) - X_pred(X_tst ~= nil)).^2))];  % error on known test values
    rmse2 = [rmse2 sqrt(mean((X_trn(X_trn ~= nil) - X_pred(X_trn ~= nil)).^2))];  % error on known test values
    
    figure(1)
    plot(rmse1);
    hold on
    plot(rmse2, 'r');
    hold off
    axis([-inf, inf, 0, 1.5])
    drawnow;

    disp(['Root of Mean-squared error (test): ' num2str(rmse1(end))]);
    disp(['Root of Mean-squared error (train): ' num2str(rmse2(end))]);
end

