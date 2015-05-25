function X_pred = PredictMissingValuesSVD(X, nil)

% First replace missing values using baseline
X = PredictMissingValuesBaseline(X, nil);

% Compute SVD
[U, D, V] = svd(X);
%U = U*sqrt(D);
%V = sqrt(D)*V;

% Shrink dimensionality
%plot(diag(D));
%pause;

k = 6;

D = D(1:k,1:k);
U = U(:,1:k);
V = V(:,1:k);

% Reconstruct X
X_pred = U*D*V';

end