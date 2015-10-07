function loss = los01(y, yhat)
    loss = length(y(y~=yhat)) / length(y);
end