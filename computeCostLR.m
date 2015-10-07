function[L] = computeCostLR (y, tX, beta)
L = 0;
for i = 1:length(y)
    L = L + y(i)*tX(i,:)*beta/length(y) - log(1+exp(tX(i,:)*beta))/length(y);
end;