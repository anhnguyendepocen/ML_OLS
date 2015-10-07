function beta = leastSquaresGD(y, tX, alpha)
%Computes beta using gradient descent

  % initialize
  beta = zeros([size(tX, 2) 1]);
  maxIters = 5000;

  % iterate
  for k = 1:maxIters
    % gradient
    g = computeGradient(y, tX, beta);

    % step update
    beta = beta - alpha.*g;
   
    % convergence criterion
    if (norm(g)/(norm(beta)*norm(alpha)) <  1e-5)
       break
    end
  end
end

function g = computeGradient(y,tX,beta)
    N = length(y);
    e = y - tX*beta;
    g = -1/N * tX'*e;
end