function X_pred = PredictMissingValuesBaseline(X, nil)

X_pred = X;

% Baseline algorithm (just predict with mean of each item)
X(X==nil) = 0;
means = mean(X,1);

for i = 1:size(X,2)
    X_pred(X_pred(:,i) == nil,i) = means(i);
end

