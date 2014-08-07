%% Load Data of All training subjects

disp('--------------------------------------------------------------------');
disp('Load preprocessed data for the training subjects');

clear all;
sujet = 1:16;
Xtrain = [];
Ytrain = [];
sujetidx = [];

for user = sujet
     if user < 10
        Nuser = ['0' num2str(user)];
    else
        Nuser = num2str(user);
    end
    disp(['Load data for subject ' Nuser]);
   
    % load prepocessed data
    load(['./preproc/train' Nuser '.mat']);
    
    % Agregates data
    Xtrain = cat(3,Xtrain,X);
    Ytrain = cat(1,Ytrain,double(y));
    sujetidx = cat(1,sujetidx,user*ones(size(X,3),1));
end


%% Extract Features
disp('--------------------------------------------------------------------');
disp('Extract training features');
acc = [];
Nfilters = 4;

% allocate variables
V = zeros(size(Xtrain,1),2*Nfilters,length(sujet));
P = zeros(2*Nfilters,size(Xtrain,2),length(sujet));
Cr = zeros(4*Nfilters,4*Nfilters,length(sujet));

feattr = zeros(4*Nfilters*(4*Nfilters+1)/2,size(Xtrain,3),length(sujet));

for s=sujet
    
    disp(['Extract features for subject ' num2str(s)]);
    
    % find indices for the current subjects
    sujettr = sujetidx==s;
    
    % Extract features for each subjects
    [feattr(:,:,s),P(:,:,s),V(:,:,s),Cr(:,:,s)] = extract_features(Xtrain,Xtrain(:,:,sujettr),Ytrain(sujettr),Nfilters);
    
end

% Reshape features
feattr = permute(feattr,[1 3 2]);
feattr = reshape(feattr,[],size(feattr,3)); 

%% train classifyer
disp('--------------------------------------------------------------------');
disp('Train Classifyer');

% train lasso
[BFinal,FitInfoFinal] = lasso(feattr',Ytrain,'Lambda',0.005,'Alpha',0.95);

% clear Training data
clear Xtrain
clear Ytrain
clear feattr

%% Apply on test subjects
disp('--------------------------------------------------------------------');
disp('Processing test subjects');

sujettest = 17:23;

out = [];
truc = [];

for user = sujettest
    % load test subject data 
    if user < 10
        Nuser = ['0' num2str(user)];
    else
        Nuser = num2str(user);
     end
    
    disp(['Load data for subject ' Nuser]);
   
    load(['./preproc/test' Nuser '.mat']);

    %----------------------------------------------------------------------
    % Generic model
    
    tmp = [];
    % Create generic features
    for s=sujet
        % for each training subjest
        % apply spatial filters
        X2 = applyfilters(X,V(:,:,s));
        % estimate covariance matrices
        COVte = covariances_p300(X2,P(:,:,s));
        % Tangent space mapping
        tmp = cat(3,tmp,Tangent_space(COVte,Cr(:,:,s)));
        
    end
    % reshape the features
    tmp = permute(tmp,[1 3 2]);
    featte = reshape(tmp,[],size(tmp,3)); 
    
    % apply the generic classifier
    y_pred = [ones(size(featte,2),1) featte'] * [FitInfoFinal.Intercept;BFinal];
    yte = zeros(size(y_pred));
    yte(y_pred>=mean(y_pred))= 1;
    
    % --------------------------------------------------------------------
    % Unsupervised training
    yteold = zeros(size(X,3),1);
    % first iteration
    yte = mdmfilter(X,X,yte,Nfilters);
    niter = 0;
    % loop until convergence
    while (mean(yte==yteold)~=1 && (niter<10));
        niter = niter + 1;
        yteold=yte;
        yte = mdmfilter(X,X,yte,Nfilters); 
        disp(mean(yte==yteold))
    end
    out = cat(1,out,yte);
    id = user*1000 + (0:(size(X2,3)-1))';
    truc = cat(1,truc,id);
    
end
% Write submission file
csvwrite('submission.csv',[truc out])