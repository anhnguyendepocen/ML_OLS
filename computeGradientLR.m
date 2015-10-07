function[g] = computeGradientLR (y, tX, beta)
g = -(tX' * (sigma(tX * beta) - y))/length(y);
%norm(g)