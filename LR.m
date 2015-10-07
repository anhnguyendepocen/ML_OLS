clear all;
load('Auckland_classification.mat')


% normalize features (store the mean and variance)
%center and normalize all gaussian distributed 1->35
for i=1:size(X_train,2)
    mu = mean(X_train(:,i));
    sd = std(X_train(:,i));
    X_train(:,i) = (X_train(:,i) - mu)./sd;
end



%moves the -1 class in y to 0

for i=1:length(y_train)
    if (y_train(i) == -1)
        y_train(i) = 0;
    end
end

y = y_train;



for i=1:100
    i
               
    dimension = 1200:50:1500;
    for l = 1:length(dimension)
    
        %transforms on X
        tX = XTransformsLR(X_train);   
            
        tX = tX(1:dimension(l), :);
        y = y_train(1:dimension(l), :);
        
        % a = mseTeOLS; b = sortrows(a, 1)
        %k-fold init
        K = 2;
        N = size(y,1);
        idx = randperm(N);
        Nk = floor(N/K); 
        clear idxCV;

        for k = 1:K
            idx(1+(k-1)*Nk:k*Nk);
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
            %PCA test
            %[XTr, XTe] = XSortLR(XTr, XTe, yTr, dimension(l));
            %[XTr, XTe] = XSortLessAutoCor(XTr, XTe, yTr, dimension(l));
            %[XTr, XTe] = customTransLR(XTr, XTe, yTr, dimension(l));
            %regression
            betaLR = logisticRegression(yTr, XTr, 1, 1e-3);
            yTrF = sigma(XTr * betaLR)>0.5;
            yTeF = sigma(XTe * betaLR)>0.5;

            %{
            mean(yTe)
            mean(yTeF)
            %}

            %phat(k, :) = sigma(XTe * betaLR);

            % training and test MSE
            %OLS
            mseTrSubLR(k) = rmselogistic(yTr, sigma(XTr * betaLR));
            % testing MSE using least squares
            mseTeSubLR(k) = rmselogistic(yTe, sigma(XTe * betaLR));

            %sigma(XTe * betaLR)
            los01TrSub(k) = los01(yTr, sigma(XTr * betaLR)>0.5);
            los01TeSub(k) = los01(yTe, sigma(XTe * betaLR)>0.5);

            logLossTrSub(k) = logLoss(yTr, sigma(XTr * betaLR));
            logLossTeSub(k) = logLoss(yTe, sigma(XTe * betaLR));

            k = k+1;
        end


        mseTrLR(i,l) = mean(mseTrSubLR);
        mseTeLR(i,l) = mean(mseTeSubLR);
        mseTrLRsd(i,l) = std(mseTrSubLR);
        mseTeLRsd(i,l) = std(mseTeSubLR);

        
        los01Tr(i,l) = mean(los01TrSub);
        los01Te(i,l) = mean(los01TeSub);
        los01Trsd(i,l) = std(los01TrSub);
        los01Tesd(i,l) = std(los01TeSub);

        logLossTr(i,l) = mean(logLossTrSub);
        logLossTe(i,l) = mean(logLossTeSub);
        logLossTrsd(i,l) = std(logLossTrSub);
        logLossTesd(i,l) = std(logLossTeSub);
    end
end

%% plot
ax = axes();

%{
plot(dimension, mseTrLR, 'Color', [0 0 1]);
hold on;
plot(dimension, mseTeLR, 'Color', [0.2 1 1]);

plot(dimension, los01Tr, 'Color', [1 0 0]);
plot(dimension, los01Tr, 'Color', [1 0.5 0]);

plot(dimension, logLossTr, 'Color', [0 1 0]);
plot(dimension, logLossTe, 'Color', [0.5 1 0]);
%}


errorbar(dimension, mean(mseTrLR), mean(mseTrLRsd), 'Color', [0 0 1]);
hold on;
errorbar(dimension, mean(mseTeLR), mean(mseTeLRsd), 'Color', [0.2 1 1]);

errorbar(dimension, mean(los01Tr), mean(los01Trsd), 'Color', [1 0 0]);
errorbar(dimension, mean(los01Te), mean(los01Tesd), 'Color', [1 0.5 0]);

errorbar(dimension, mean(logLossTr), mean(logLossTrsd), 'Color', [0 1 0]);
errorbar(dimension, mean(logLossTe), mean(logLossTesd), 'Color', [0.5 1 0]);

%set(ax, 'XScale', 'log');
%set(ax, 'YScale', 'log');

legend('RMSE train', 'RMSE test', '01 Loss train', '01 Loss test', 'log Loss train', 'log Loss test', 'location', 'NorthEast');
hx = xlabel('sample size');
hy = ylabel('');
xlim([1195 1525]);
%xlim([0.9e-2 1.1e3]);

%% 
% the following code makes the plot look nice and increase font size etc.
set(gca,'fontsize',15,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;

hold off;
w = 25; h = 20;
set(gcf, 'PaperPosition', [0 0 w h]); %Position plot at left hand corner with width w and height h.
set(gcf, 'PaperSize', [w h]); %Set the paper to have width w and height h.
 saveas(gcf, 'bla', 'pdf') %Save figure

%%
sampleSize = 600:200:1400;
for l = 1:length(sampleSize)
    y = y_train;

    %transforms on X
    tX = [ones(length(X_train), 1) X_train];

    tX = tX(1:sampleSize(l), :);
    y = y(1:sampleSize(l), :);
    
    clear idxCV;

    %k-fold init
    K = 10;
    N = size(y,1);
    idx = randperm(N);
    Nk = floor(N/K); 
    for k = 1:K
        idx(1+(k-1)*Nk:k*Nk)
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
    
    % a = mseTeOLS; b = sortrows(a, 1)
    l
    k = 1;
    for a = 1:K
        for b= a+1:K
            for c = b+1:K
                %separating in train and test groups
                idxTe = idxCV([a b c], :);
                idxTe = idxTe(:);
                idxTr = idxCV([1:a-1 a+1:b-1 b+1:c-1 c+1:end], :); 
                idxTr = idxTr(:);
                yTe = y(idxTe);
                XTe = tX(idxTe,:);
                yTr = y(idxTr);
                XTr = tX(idxTr,:);
                %PCA test
                %[XTr, XTe] = XSort(XTr, XTe, yTr, dimension(l));
                %[XTr, XTe] = XSortLessAutoCor(XTr, XTe, yTr, dimension(l));
                %[XTr, XTe] = customTrans(XTr, XTe, yTr, dimension(l));
                %regression
                betaLR = logisticRegression(yTr, XTr, 5);

                % training and test MSE
                %OLS
                mseTrSubLR(k) = sqrt(2*computeCost(yTr, XTr, betaLR));
                % testing MSE using least squares
                mseTeSubLR(k) = sqrt(2*computeCost(yTe, XTe, betaLR));

                k = k+1
            end
        end
    end
    mseTrLR(l,:) = mean(mseTrSubLR);
    mseTeLR(l, :) = mean(mseTeSubLR);

    %std
    mseTrLRsd(l) = std(mseTrSubLR);
    mseTeLRsd(l) = std(mseTeSubLR);
end

%% plot
errorbar(sampleSize, mseTrLR, mseTrLRsd, 'blue');
hold on;
errorbar(sampleSize, mseTeLR, mseTeLRsd, 'red');
legend('train', 'test');
hx = xlabel('sample size');
hy = ylabel('MSE error');

% the following code makes the plot look nice and increase font size etc.
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;

hold off;
w = 25; h = 10;
set(gcf, 'PaperPosition', [0 0 w h]); %Position plot at left hand corner with width w and height h.
set(gcf, 'PaperSize', [w h]); %Set the paper to have width w and height h.
 saveas(gcf, 'sampleSizeTestLR', 'pdf') %Save figure


%%
% algorithm parametes
maxIters = 5000;
alpha = 7;
% initialize
%beta = leastSquares(y, tX);
beta = zeros(size(tX, 2), 1);

for k = 1:maxIters
    % INSERT YOUR FUNCTION FOR COMPUTING GRADIENT
    
    g = computeGradientLR(y,tX,beta);
    
    % INSERT YOUR FUNCTION FOR COMPUTING COST FUNCTION
    L = computeCostLR (y, tX, beta);

    % INSERT GRADIENT DESCENT UPDATE TO FIND BETA
    beta = beta + alpha * g;

    % INSERT CODE FOR CONVERGENCE
    if (norm(g)*alpha/norm(beta) <  1e-6)
        break;
    end

    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;

    % print
    %fprintf('%.2f  %.2f %.2f %.2f\n', L, beta(1), beta(2), beta(3));

    %{
    % Overlay on the contour plot
    % For this to work you first have to run grid Search
    subplot(121);
    plot(beta(1), beta(2), 'o', 'color', 0.7*[1 1 1], 'markersize', 12);
    pause(.5) % wait half a second

    % visualize function f on the data
    subplot(122);
    x = [1.2:.01:2]; % height from 1m to 2m 
    x_normalized = (x - meanX)./stdX;
    f = beta(1) + beta(2).*x_normalized;
    plot(height, weight,'.');
    hold on;
    plot(x,f,'r-');
    hx = xlabel('x');
    hy = ylabel('y');
    hold off;
    %}
end
k

%%

for i=1:size(X_train, 2)
    subplot(7,7,i);
    plot(X_train(:,i), y_train, '*')
end

