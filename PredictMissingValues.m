function X_pred = PredictMissingValues(X, nil)
% Predict missing entries in matrix X based on known entries. Missing
% values in X are denoted by the special constant value nil.

% Use baseline or SVD?
alg = 3; % 0 = Baseline, 1 = SVD

switch(alg)
    case 5
        %EMNMF
        X_pred = PredictMissingValuesEMNMF(X, nil);
    case 4
        %WNMF
        X_pred = PredictMissingValuesWNMF(X, nil);
    case 3
        % ALS
        X_pred = PredictMissingValuesALS(X, nil);
    case 2
        % SGD
        X_pred = PredictMissingValuesSGD(X, nil);
    case 1
        % SVD
        X_pred = PredictMissingValuesSVD(X, nil);
    case 0
        % Baseline
        X_pred = PredictMissingValuesBaseline(X, nil);

end