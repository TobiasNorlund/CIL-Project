function [ X_pred ] = PredictMissingValuesEMNMF( X, nil )
%PREDICTMISSINGVALUESEMNMF Summary of this function goes here
%   Detailed explanation goes here

persistent U Z X_tmp;
global k;

k = 6;

% X = X';
min_val = min(min(X(X~=nil)));
X(X~=nil) = X(X~=nil) - min_val;

 if(isempty(U))
    U = rand(size(X, 1), k);
    Z = rand(k, size(X, 2));

    X_tmp = PredictMissingValuesBaseline(X, nil);
 end

%for i = 1:10
%i
U = U .* (X_tmp*Z')./(U*(Z*Z'));
Z = Z .* (U'*X_tmp)./(U'*U*Z);

X_tmp = U*Z;
X_tmp(X ~= nil) = X(X~=nil);

%end

Y = Z;
A = U;

% Run WNMF
%X(X == nil) = nan;
%option = struct('distance','ls', 'iter', 100, 'Y', Z, 'A', U);
%[A,Y,numIter,tElapsed,finalResidual] = wnmfrule(X,k, option);

X_pred = (A*Y) + min_val;

end

