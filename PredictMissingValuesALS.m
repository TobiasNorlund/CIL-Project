function X_pred = PredictMissingValuesALS(X, nil)

persistent movieReps userReps movieBias userBias
global k;
global lambda;

xnotnil = X ~= nil;

% Initialize k-dim representations
if(isempty(userReps) && isempty(movieReps))
    movieReps = 0.1*ones(k, size(X,2));%randn(k, size(X,2));
    movieBias = zeros(1, size(X,2));
    userReps = 0.1*ones(k, size(X,1));%randn(k, size(X,1));
    userBias = zeros(1, size(X,1));
end

% Train representations

% Optimize movie representations
for movie_idx = 1:size(X,2)

    i = X(:, movie_idx) ~= nil;
    Q = userReps(:,i)';
    y = X(i,movie_idx);
    movieReps(:,movie_idx) = (Q'*Q + lambda*eye(k))\Q'*y;        
end

% Optimize user representations
for user_idx = 1:size(X,1)

    i = X(user_idx, :) ~= nil;
    Q = movieReps(:,i)';
    y = X(user_idx,i)';
    userReps(:,user_idx) = (Q'*Q + lambda*eye(k))\Q'*y;        
end

% Use new representation to predict ratings
X_pred = userReps'*movieReps;
