function [Tte,P,V,C] = extract_features(Xtest,Xtrain,Ytrain,Nfilters)

   
    % training Data 
    % ---------------------------------------------------------------------
    % train filters from training data
    [V,P] = trainfilter(Xtrain,Ytrain,Nfilters);
    % apply filters on training data
    X2 = applyfilters(Xtrain,V);
    % estimate special form covariance matrices for training data
    COVtr = covariances_p300(X2,P);
    % Calculate the Riemannian mean for the training covariance matrices
    C = mean_covariances(COVtr,'riemann');
    
    
    % test Data 
    % ---------------------------------------------------------------------    
    % apply filters on test data
    X2 = applyfilters(Xtest,V);
    % estimate special form covariance matrices for test data
    COVte = covariances_p300(X2,P);    
    
    % Extract features
    % ---------------------------------------------------------------------
    Tte = Tangent_space(COVte,C);
    