function [ beta ] = leastSquares(y,tX)
    % Compute the beta using ordinary least squares
    beta = (tX'*tX) \(tX'*y);
end

