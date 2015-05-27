function [ X_pred ] = PredictMissingValuesEMNMF( X, nil )
%PREDICTMISSINGVALUESEMNMF Summary of this function goes here
%   Detailed explanation goes here

%p%ersistent U Z X_tmp;
%g%lobal k;

k = 6;

% X = X';
min_val = min(min(X(X~=nil)));
X(X~=nil) = X(X~=nil) - min_val;

% if(isempty(U))
    U = rand(size(X, 1), k);
    Z = rand(k, size(X, 2));

    X_tmp = PredictMissingValuesBaseline(X, nil);
% end

for i = 1:300

U = U .* (X_tmp*Z')./(U*(Z*Z'));
Z = Z .* (U'*X_tmp)./(U'*U*Z);

X_tmp = U*Z;
X_tmp(X ~= nil) = X(X~=nil);

end

% Run WNMF
X(X == nil) = nan;
option = struct('distance','ls', 'iter', 1000, 'Y', Z, 'A', U);
[A,Y,numIter,tElapsed,finalResidual] = wnmfrule(X,k, option);

X_pred = (A*Y) + min_val;

end

