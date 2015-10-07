function err = logLoss(y, yhat)
    epss=0.001; %arbitrary value, may be model tuning parameter  
    yhat=min(max(yhat,epss),1-epss);
    N = length(y);
    err = (-1/N) * sum(y .* log(yhat) + (1-y).* log(1-yhat));
end