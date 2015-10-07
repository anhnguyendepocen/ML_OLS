function[L] = MSE (y, y2)
e = y - y2;

L = e'*e/(2*length(y));