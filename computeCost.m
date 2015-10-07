function[L] = computeCost (y, tX, beta)    
e = y - tX*beta;

%MSE
L = e'*e/(2*length(y));

%MAE
%{
L = sum(abs(e));
%}