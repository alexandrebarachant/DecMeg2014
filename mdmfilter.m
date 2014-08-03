function Ytest = mdmfilter(Xtest,Xtrain,Ytrain,Nfilters)

    % training data 
    % ---------------------------------------------------------------------
    % train filters on training data
    [V,P] = trainfilter(Xtrain,Ytrain,Nfilters);
    % apply filters on training data
    X2 = applyfilters(Xtrain,V);
    % estimates special for covariance matrices
    COVtr = covariances_p300(X2,P);
        
    % test data 
    % ---------------------------------------------------------------------
    % apply filters on test data
    X2 = applyfilters(Xtest,V);
    % estimates special form covariance matrices
    COVte = covariances_p300(X2,P);

    
    % Classification
    % ---------------------------------------------------------------------
    % MDM classification
    Ytest = mdm(COVte,COVtr,Ytrain,'riemann','riemann');
    

    