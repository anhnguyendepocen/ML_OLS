function[rX] = XTransformsLR (X)
tX = [ones(length(X), 1) X];

rX = tX;

%adding interactions found while doing Te
    
    ds = [169 191 384 205 491 120 131 373 183 513 292 380 182 490 35 277];
    NL = 23;
    for d=1:6 
      rX = [rX tX(:,floor(ds(d)/NL)+1) .* tX(:, mod(ds(d), NL)+1)];
    end

    
%normalize everything
for i=2:size(rX,2)
    mu = mean(rX(:,i));
    sd = std(rX(:,i));
    rX(:,i) = (rX(:,i) - mu)./sd;
end
