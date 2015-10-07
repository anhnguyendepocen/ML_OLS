function[s] = sigma (X)
s = exp(X - log(1+exp(X)));