function err = rmselogistic(y, phat)
    N = length(y);
    err = sqrt((1/N)*(y-phat)'*(y-phat));
end

