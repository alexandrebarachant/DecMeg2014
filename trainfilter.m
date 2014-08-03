function [V,P] = trainfilter(Xtrain,Ytrain,Nfilter)

    % Calculates the average evoqued potential for each class
    P1 = mean(Xtrain(:,:,Ytrain==1),3);
    P0 = mean(Xtrain(:,:,Ytrain==0),3);
    
    % Estimates the covariances matrices
    % of the class 1
    C1 = cov(P1');
    % of the class 0
    C0 = cov(P0');
    % of the signal
    C = cov(Xtrain(:,:)');

    % Calcultates the spatial filters
    % for the class 1
    [V1,~] = eig(C\C1);
    % for the class 0
    [V0,~] = eig(C\C0);
    
    % agregates for return arguments
    P = cat(1,V1(:,1:Nfilter)'*P1,V0(:,1:Nfilter)'*P0);
    V = cat(2,V1(:,1:Nfilter),V0(:,1:Nfilter));
   