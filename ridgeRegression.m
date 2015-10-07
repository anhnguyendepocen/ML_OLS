function [ beta ] = ridgeRegression(y, tX, lambda)
% Computes beta using ridge regression
    D = size(tX,2);
    beta = (tX'*tX + lambda*eye(D)) \(tX'*y);
end

