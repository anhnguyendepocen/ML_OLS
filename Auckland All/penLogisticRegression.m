function beta = penLogisticRegression(y,tX,alpha, lambda)
    % makes sure that the loop finishes
    maxIters = 5000;
    
    % initialization
    beta = zeros(size(tX, 2), 1);

    for k = 1:maxIters
        % computes the gradient
        g = -(tX' * (sigma(tX * beta) - y))/length(y) + lambda * beta;

        % updates beta
        beta = beta + alpha * g;

        % convergence criterion
        if (norm(g)*alpha/norm(beta) <  1e-3)
            break;
        end
    end
end
