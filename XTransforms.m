function[tX] = XTransforms (X)
tX = [ones(length(X), 1) X];

%removing useless dimensions
%tX = 

%adding interactions found while doing Te

ds = [1750 2073];
NL = 49;
for d=1:length(ds) 
    tX = [tX tX(:,floor(ds(d)/NL)+1) .* tX(:, mod(ds(d), NL)+1)];
end
%{
ds = [2005 2001 220, 1956 2197 2045 2241 2193 1952 1804];
NL = 49;
for d=1:1
    tX = [tX  tX(:, mod(ds(d), NL)+1).^tX(:,floor(ds(d)/NL)+1)];
end
%}