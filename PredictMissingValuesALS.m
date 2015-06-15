function X_pred = PredictMissingValuesALS(X, nil)

persistent movieReps userReps movieBias userBias mu
global k;
global lambda;

%k = 6;
%lambda = 10;

xnil = X == nil;
xnotnil = X ~= nil;

X0 = X;
X0(xnil) = 0;

% Initialize k-dim representations
if(isempty(userReps) && isempty(movieReps))
    mu = mean2(X0);
    movieReps = 0.5*(rand(k, size(X,2))-0.5); %0.1*ones(k, size(X,2));
    movieBias = mean(X0,1)-mu; %zeros(1, size(X,2));
    userReps = 0.5*(rand(k, size(X,1))-0.5); %0.1*ones(k, size(X,1));
    userBias = mean(X0,2)'-mu; %zeros(1, size(X,1));
end

% Train representations
%for e = 1:10

% Update mu
mov_bs_tmp = repmat(movieBias,size(X,1),1);
res_tmp = X - userReps'*movieReps;
res = res_tmp - mov_bs_tmp - repmat(userBias',1,size(X,2));
res(xnil) = 0;
mu = sum(sum(res))/sum(sum(xnotnil));

% Update movie bias
res_tmp = X - userReps'*movieReps;
usr_bs_tmp = repmat(userBias',1,size(X,2));
bs_tmp = mu*ones(size(X));
res = res_tmp - usr_bs_tmp - bs_tmp;
res(xnil) = 0;
N = sum(xnotnil,1);
movieBias = sum(res,1)./(N*(1+lambda));

% Update user bias
mov_bs_tmp = repmat(movieBias,size(X,1),1);
bs_tmp = mu*ones(size(X));
res = res_tmp - mov_bs_tmp - bs_tmp;
res(xnil) = 0;
N = sum(xnotnil,2);
N(N==0) = 1;
userBias = sum(res,2)./(N*(1+lambda*2));
userBias = userBias';

% Optimize movie representations
for movie_idx = 1:size(X,2)

    i = X(:, movie_idx) ~= nil;
    Q = userReps(:,i)';
    y = X(i,movie_idx) - userBias(i)' - movieBias(movie_idx) - mu;
    movieReps(:,movie_idx) = (Q'*Q + lambda*eye(k))\Q'*y;        
end

% Optimize user representations
for user_idx = 1:size(X,1)

    i = X(user_idx, :) ~= nil;
    Q = movieReps(:,i)';
    y = X(user_idx,i)' - userBias(user_idx) - movieBias(i)' - mu;
    userReps(:,user_idx) = (Q'*Q + lambda*eye(k))\Q'*y;        
end

% Display debug info

display(['Movie bias norm: ' num2str(norm(movieBias))]);
display(['User bias norm: ' num2str(norm(userBias))]);
display(['Mu: ' num2str(mu)]);
%end

% Use new representation to predict ratings
X_pred = userReps'*movieReps + repmat(userBias',1,size(X,2)) + repmat(movieBias,size(X,1),1) + mu*ones(size(X));
X_pred(X_pred < min(min(X(xnotnil)))) = min(min(X(xnotnil)));
X_pred(X_pred > max(max(X(xnotnil)))) = max(max(X(xnotnil)));
%X_pred = round(X_pred);
