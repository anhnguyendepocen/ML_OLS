function loss = los01(y, yhat)
    loss = length(y(y~=yhat)) / length(y)
end

function err = rmselogistic(y, phat)
    N = length(y);
    err = sqrt((1/N)*(y-phat)'*(y-phat));
end

function err = logLoss( y, phat )
    N = length(y);
    err = (-1/N) * sum(y .* log(phat) + (1-y).*log(1-phat));
end