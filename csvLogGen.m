% CSV gen
clear all;
load('Auckland_classification.mat')

y = y_train;

%moves the -1 class in y to 0

for i=1:length(y)
    if (y(i) == -1)
        y(i) = 0;
    end
end

for i=1:size(X_train,2)
    mu = mean(X_train(:,i));
    sd = std(X_train(:,i));
    X_train(:,i) = (X_train(:,i) - mu)./sd;
end

for i=1:size(X_test,2)
    mu = mean(X_test(:,i));
    sd = std(X_test(:,i));
    X_test(:,i) = (X_test(:,i) - mu)./sd;
end

tX = XTransformsLR(X_train);
tXTest = XTransformsLR(X_test);

betaLR = logisticRegression(y, tX, 1, 1e-3);

yTrF = sigma(tX * betaLR);
rmselogistic(y, yTrF)

pOut = sigma(tXTest * betaLR);

csvwrite('predictions_classification.csv',pOut)