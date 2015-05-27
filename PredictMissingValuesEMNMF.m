function [ X_pred ] = PredictMissingValuesEMNMF( X, nil )
%PREDICTMISSINGVALUESEMNMF Summary of this function goes here
%   Detailed explanation goes here

persistent U Z X_tmp;
global k;

% k = 6;

% X = X';
X(X~=nil) = X(X~=nil) + 10;

if(isempty(U))
    U = rand(size(X, 1), k);
    Z = rand(k, size(X, 2));
%     U = normr(U);
%     Z = normr(Z);
%     X_tmp = U*Z;
    X_tmp = PredictMissingValuesBaseline(X, nil);
end

% for i = 1:50

% X_tmp(X ~= nil) = X(X~=nil);

U = U .* (X_tmp*Z')./(U*Z*Z');
Z = Z .* (U'*X_tmp)./(U'*U*Z);
% U = normr(U);
% Z = normr(Z);

% X_tmp = U*Z;

% end

X_pred = (U*Z) - 10;

end

