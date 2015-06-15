function X_pred = PredictMissingValuesWNMF(X, nil)

global k
persistent Y A W

xnil = X == nil;
xnotnil = X ~= nil;

if(isempty(Y))
    Y=rand(k,size(X,2));
    Y=max(Y,eps);
    A=X/Y;
    A=max(A,eps);
    
    W=X==nil;
    X(W)=0;
    W=~W;
end

%k = 10;

%min_val = min(min(X));
%X = X - min_val;

%option = struct('distance','ls', 'iter', 100);
%[A,Y,numIter,tElapsed,finalResidual] = wnmfrule(X,k, option);

A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
A=max(A,eps);
Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
Y=max(Y,eps);

X_pred = A*Y;
X_pred(X_pred < min(min(X(xnotnil)))) = min(min(X(xnotnil)));
X_pred(X_pred > max(max(X(xnotnil)))) = max(max(X(xnotnil)));