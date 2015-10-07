%% init
clear all;
load('Auckland_regression.mat');

%{

%}


%center and normalize everything
for i=1:48
    mu = mean(X_train(:,i));
    sd = std(X_train(:,i));
    X_train(:,i) = (X_train(:,i) - mu)./sd;
end



%main loop: we get 3/10 as test and 7/10 as train to keep the same ratio as
%with the final test


y = y_train;

%transforms on X
tX = XTransforms(X_train);




    
%dimension test
%dimension = [0 35*49+1:49*49-1];
dimension = 0:2;

for i=1:20
    i
    %k-fold init
K = 3;
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end
    for l = 1:length(dimension)
        %l
        %changing the size of the samples the see the evolution of the error
        y = y_train;

        %transforms on X
        tX = XTransforms(X_train);



        %resets arrays
        clear yTrF;
        clear yTeF;


        k = 1;
        for a = 1:K
            
            %separating in train and test groups
            idxTe = idxCV(k,:);
            idxTr = idxCV([1:k-1 k+1:end],:); 
            idxTr = idxTr(:);
            yTe = y(idxTe);
            XTe = tX(idxTe,:);
            yTr = y(idxTr);
            XTr = tX(idxTr,:);
        
            %transforms
            %[XTr, XTe] = customTrans(XTr, XTe, yTr, dimension(l));
            
            %splits the data into 2 classes <5000 and >5000
            labels = yTr > 5100;
            betaSplit = logisticRegression(labels, XTr, 1, 7e-3);
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


            % training and test MSE
            %OLS
            mseTrSubOLS(k) = sqrt(2*MSE(yTr, yTrF));
            % testing MSE using least squares
            mseTeSubOLS(k) = sqrt(2*MSE(yTe, yTeF));
            k = k+1;
        end
        mseTrOLS(i,l) = mean(mseTrSubOLS);
        mseTeOLS(i,l) = mean(mseTeSubOLS);

        %std
        mseTrOLSsd(i,l) = std(mseTrSubOLS);
        mseTeOLSsd(i,l) = std(mseTeSubOLS);
    end

end
%%
ax = axes();
%errorbar(dimension, mean(mseTrOLS), mean(mseTrOLSsd), 'blue');
plot(dimension(:,2:end), mean(mseTrOLS(:,2:end)), 'blue');
hold on;
%errorbar(dimension, mean(mseTeOLS), mean(mseTeOLSsd), 'red');
plot(dimension(:,2:end), mean(mseTeOLS(:,2:end)), 'red');

%no transformation
plot([0 5000], [mean(mseTeOLS(:,1)) mean(mseTeOLS(:,1))], '-.');
%set(ax, 'XScale', 'log');
xlim([1700 2500]);
ylim([0 1000]);
legend('train', 'test');
hx = xlabel('interaction');
hy = ylabel('RMSE error');

%% 
% the following code makes the plot look nice and increase font size etc.
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;

hold off;
w = 30; h = 20;
set(gcf, 'PaperPosition', [0 0 w h]); %Position plot at left hand corner with width w and height h.
set(gcf, 'PaperSize', [w h]); %Set the paper to have width w and height h.
 saveas(gcf, 'bla', 'pdf') %Save figure
%%
%main loop: we get 3/10 as test and 7/10 as train to keep the same ratio as
%with the final test


sampleSize = 850:50:1400;

for i=1:100
    i
    for l = 1:length(sampleSize)
        l
        %changing the size of the samples the see the evolution of the error
        y = y_train;

        %transforms on X
        tX = XTransforms(X_train);

        tX = tX(1:sampleSize(l), :);
        y = y(1:sampleSize(l), :);

        %resets arrays
        clear idxCV;
        clear yTrF;
        clear yTeF;

        %k-fold init
        K = 3;
        N = size(y,1);
        idx = randperm(N);
        Nk = floor(N/K);
        for k = 1:K
            idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
        end

        k = 1;
        for a = 1:K
                %separating in train and test groups
        idxTe = idxCV(k,:);
        idxTr = idxCV([1:k-1 k+1:end],:); 
        idxTr = idxTr(:);
        yTe = y(idxTe);
        XTe = tX(idxTe,:);
        yTr = y(idxTr);
        XTr = tX(idxTr,:);

            %splits the data into 2 classes <5000 and >5000
            labels = yTr > 5100;
            betaSplit = logisticRegression(labels, XTr, 1, 7e-3);
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


            % training and test MSE
            %OLS
            mseTrSubOLS(k) = sqrt(2*MSE(yTr, yTrF));
            % testing MSE using least squares
            mseTeSubOLS(k) = sqrt(2*MSE(yTe, yTeF));
            k = k+1;
        end
        mseTrOLS(i,l) = mean(mseTrSubOLS);
        mseTeOLS(i,l) = mean(mseTeSubOLS);

        %std
        mseTrOLSsd(i,l) = std(mseTrSubOLS);
        mseTeOLSsd(i,l) = std(mseTeSubOLS);
    end

end
%%
ax = axes();
errorbar(sampleSize, mean(mseTrOLS), mean(mseTrOLSsd), 'blue');
hold on;
errorbar(sampleSize, mean(mseTeOLS), mean(mseTeOLSsd), 'red');
%set(ax, 'XScale', 'log');
ylim([0 1000]);
legend('train', 'test');
hx = xlabel('sample size');
hy = ylabel('RMSE error');

%% 
% the following code makes the plot look nice and increase font size etc.
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;

hold off;
w = 30; h = 20;
set(gcf, 'PaperPosition', [0 0 w h]); %Position plot at left hand corner with width w and height h.
set(gcf, 'PaperSize', [w h]); %Set the paper to have width w and height h.
 saveas(gcf, 'bla', 'pdf') %Save figure
%%

%checks if RR is needed => no

%{
y = y_train;

%transforms on X
tX = [ones(length(X_train), 1) X_train];


%k-fold init
K = 10;
N = size(y,1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

lambda = logspace(-4,4,20);

for l = 1:length(lambda)
    k = 1;
    for a = 1:K
        for b= 1:K
            for c = 1:K
                if (a ~= b && a ~= c && b ~= c)
                    %separating in train and test groups
                    tmp = sort ([a b c]);
                    idxTe = idxCV([tmp(1) tmp(2) tmp(3)], :);
                    idxTe = idxTe(:);
                    idxTr = idxCV([1:tmp(1)-1 tmp(1)+1:tmp(2)-1 tmp(2)+1:tmp(3)-1 tmp(3)+1:end], :); 
                    idxTr = idxTr(:);
                    yTe = y(idxTe);
                    XTe = tX(idxTe,:);
                    yTr = y(idxTr);
                    XTr = tX(idxTr,:);

                    %regression
                    betaRR = ridgeRegression(yTr, XTr, lambda(l));
                    betaOLS = leastSquares(yTr, XTr);
                    
                     % training and test MSE
                     %OLS
                     mseTrSubOLS(k) = sqrt(2*computeCost(yTr, XTr, betaOLS));
                    % testing MSE using least squares
                    mseTeSubOLS(k) = sqrt(2*computeCost(yTe, XTe, betaOLS));
                    
                     %RR
                    mseTrSubRR(k) = sqrt(2*computeCost(yTr, XTr, betaRR));
                    % testing MSE using least squares
                    mseTeSubRR(k) = sqrt(2*computeCost(yTe, XTe, betaRR));
                    
                    
                    k = k+1;
                end
            end
        end
    end
    mseTrOLS(l) = mean(mseTrSubOLS);
    mseTeOLS(l) = mean(mseTeSubOLS);
    
    %std
    mseTrOLSsd(l) = std(mseTrSubOLS);
    mseTeOLSsd(l) = std(mseTeSubOLS);

    
    mseTrRR(l) = mean(mseTrSubRR);
    mseTeRR(l) = mean(mseTeSubRR);
end


semilogx(lambda, mseTrRR, 'red');
hold on;
semilogx(lambda, mseTrOLS, 'yellow');
semilogx(lambda, mseTeRR);
semilogx(lambda, mseTeOLS, 'green');
%}
        
%{
for i=1:size(X_train, 2)
    subplot(7,7,i);
    hist(X_train(:,i))
end
%}

%{
X = zeros(length(X_train), size(X_train, 2)*2);


for i=1:size(X_train, 2)
    X(:,2*i-1) = X_train(:,i);
    X(:,2*i) = X_train(:,i).^2;
end
%}


%co = corrcov(cov(X));

%%
%plot x et y for each variable to check if there is correlation between x
%and y


for i=1:size(X_train, 2)
    subplot(7,7,i);
    plot(X_train(:,i), y_train, '*')
end


%%
%clairement pas des x^2 (terrible corr avec y)


%tests for the crossed variables
%{
X = zeros(length(X_train), 35*(13+1));
for i = 1:48
    for j=1:48
        for k=1:48
            %for l=1:48
                tmp = corr(y_train, X_train(:,i).*X_train(:,j).*X_train(:,k));%.*X_train(:,l));
                if (tmp>0.3)
                    fprintf('%f, %i, %i, %i\n', tmp,i,j,k);%,l);
                end
            %end
        end
    end
end
%}
