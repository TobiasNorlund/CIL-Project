function X_pred = PredictMissingValuesALS(X, nil)

%p%ersistent movieReps userReps movieBias userBias mu
%g%lobal k;
%g%lobal lambda;

k = 5;
lambda = 100;

xnil = X == nil;
xnotnil = X ~= nil;

% Initialize k-dim representations
%if(isempty(userReps) && isempty(movieReps))
    movieReps = 0.1*ones(k, size(X,2));%randn(k, size(X,2));
    movieBias = zeros(1, size(X,2));
    userReps = 0.1*ones(k, size(X,1));%randn(k, size(X,1));
    userBias = zeros(1, size(X,1));
    mu = 0;
%end

% Train representations
for e = 1:20

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

% Update movie bias
res_tmp = X - userReps'*movieReps;
usr_bs_tmp = repmat(userBias',1,size(X,2));
bs_tmp = mu*ones(size(X));
res = res_tmp - usr_bs_tmp - bs_tmp;
res(xnil) = 0;
N = sum(xnotnil,1);
movieBias = sum(res,1)./(N*(1+1));

% Update user bias
mov_bs_tmp = repmat(movieBias,size(X,1),1);
bs_tmp = mu*ones(size(X));
res = res_tmp - mov_bs_tmp - bs_tmp;
res(xnil) = 0;
N = sum(xnotnil,2);
userBias = sum(res,2)./(N*(1+80));
userBias = userBias';

% Update mu
res = res_tmp - mov_bs_tmp - repmat(userBias',1,size(X,2));
res(xnil) = 0;
mu = sum(sum(res))/sum(sum(xnotnil));

end
% norm(movieBias)
% norm(userBias)
% mu

% Use new representation to predict ratings
X_pred = userReps'*movieReps + repmat(userBias',1,size(X,2)) + repmat(movieBias,size(X,1),1) + mu*ones(size(X));
