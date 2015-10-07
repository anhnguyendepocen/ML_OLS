% CSV gen
clear all;
load('Auckland_regression.mat')

%center and normalize all gaussian distributed 1->35
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

yTr = y_train;

%transforms on X
XTr = XTransforms(X_train);
XTe =  XTransforms(X_test);

labels = yTr > 5100;
betaSplit = logisticRegression(labels, XTr, 1.3, 7e-3);
classedTr = sigma(XTr * betaSplit) < 0.5;
classedTe = sigma(XTe * betaSplit) < 0.5;

XTr0 = XTr(find (classedTr == 0), :);
yTr0 = yTr(find (classedTr == 0), :);
XTe0 = XTe(find (classedTe == 0), :);

XTr1 = XTr(find (classedTr == 1), :);
yTr1 = yTr(find (classedTr == 1), :);
XTe1 = XTe(find (classedTe == 1), :);

%regression
betaOLS0 = leastSquares(yTr0, XTr0);
betaOLS1 = leastSquares(yTr1, XTr1);

%predictions
yTrF0 = XTr0 * betaOLS0;
yTrF1 = XTr1 * betaOLS1;

yTeF0 = XTe0 * betaOLS0;
yTeF1 = XTe1 * betaOLS1;

%puts the predictions in the same y
yTrF(find (classedTr == 0), 1) = yTrF0;
yTrF(find (classedTr == 1), 1) = yTrF1;

yTeF(find (classedTe == 0), 1) = yTeF0;
yTeF(find (classedTe == 1), 1) = yTeF1;


% training RMSE
%OLS
sqrt(2*MSE(yTr, yTrF))
                
csvwrite('predictions_regression.csv',yTeF)

%[1 3000] * leastSquares(mean(logLossTe)', [ones(size(dimension', 1), 1) dimension'])