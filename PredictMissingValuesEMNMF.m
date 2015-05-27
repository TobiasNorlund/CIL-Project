function [ X_pred ] = PredictMissingValuesEMNMF( X, nil )
%PREDICTMISSINGVALUESEMNMF Summary of this function goes here
%   Detailed explanation goes here

%p%ersistent U Z X_tmp;
%g%lobal k;

k = 6;

% X = X';
X(X~=nil) = X(X~=nil);

%if(isempty(U))
    U = 2*rand(size(X, 1), k);
    Z = 2*rand(k, size(X, 2));
%     U = normr(U);
%     Z = normr(Z);
    X_tmp = U*Z;
%end

for i = 1:50

X_tmp(X ~= nil) = X(X~=nil);

U = U .* (X_tmp*Z')./(U*Z*Z');
Z = Z .* (U'*X_tmp)./(U'*U*Z);
% U = normr(U);
% Z = normr(Z);

X_tmp = U*Z;

end

X_pred = X_tmp;

end

