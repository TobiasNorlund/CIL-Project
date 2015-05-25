function X_pred = PredictMissingValuesSGD(X, nil)

global k;
global learning_rate;
global lambda;

xnil = X == nil;
xnotnil = X ~= nil;

% Initialize k-dim representations
movieReps = randn(k, size(X,2));
userReps = randn(k, size(X,1));

% Train representations
[users, movies] = find(X ~= nil);
rmse = [];
for e = 1:10
    
for i = 1:length(users)
    
    qi = movieReps(:,movies(i));
    pi = userReps(:,users(i));
    
    display([num2str(i) ': ' num2str(norm(qi))])
    
    err = X(users(i), movies(i)) - qi' * pi;
    
    movieReps(:,movies(i)) = qi + learning_rate * (err * pi - lambda * qi);
    userReps(:,users(i))  = pi + learning_rate * (err * qi - lambda * pi);
    
    if(mod(i, 5000) == 0)
        X_tmp = userReps'*movieReps;
        rmse = [rmse sqrt(mean((X(xnotnil) - X_tmp(xnotnil)).^2))];
        plot(rmse)
        axis([-inf, inf, 0, 6])
        drawnow;
    end
end

end

% Use new representation to predict ratings
[users, movies] = find(ones(size(X)));
for i = 1:length(users)
    
    X(users(i), movies(i)) = userReps(:,users(i))' * movieReps(:,movies(i));
    
end

% 
% mean2(X(xnil))
% std2(X(xnil))
X_pred = X;
