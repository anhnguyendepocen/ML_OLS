function beta = logisticRegression(y,tX,alpha)
    % makes sure that the loop finishes
    maxIters = 5000;
    
    % initialization
    beta = zeros(size(tX, 2), 1);

    for k = 1:maxIters
        % computes the gradient
        g = computeGradientLR(y,tX,beta);

        % updates beta
        beta = beta + alpha * g;
        % convergence criterion
        if (norm(g)/norm(beta) <  1e-3)
            break;
        end
    end
end
 
function[g] = computeGradientLR (y, tX, beta)
    g = -(tX' * (sigma(tX * beta) - y))/length(y);
end