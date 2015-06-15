function X_pred = PredictMissingValuesWNMF(X, nil)

k = 10;

X(X == nil) = nan;

min_val = min(min(X));
X = X - min_val;

option = struct('distance','kl', 'iter', 200);
[A,Y,numIter,tElapsed,finalResidual] = wnmfrule(X,k, option);

X_pred = A*Y;

X_pred = X_pred + min_val;