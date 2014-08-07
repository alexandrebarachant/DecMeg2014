%% create preproc folder
mkdir preproc
%% all sujet train

clear all;
sujet = 1:16;

for user = sujet
     if user < 10
        Nuser = ['0' num2str(user)];
    else
        Nuser = num2str(user);
    end
    disp(Nuser);
   
    % load raw data
    load(['./data/train_subject' Nuser '.mat']);
    X = double(X);

    % filter
    [b,a] = butter(6,[1 20]/(sfreq/2));
    X2 = filter(b,a,X(:,:,0.5*sfreq:end),[],3);
    
    clear X
    X = permute(X2,[2,3,1]);
    % save preprocessed data
    save(['preproc/train' Nuser '.mat'],'X','y');
end

%% all sujet test

clear all;
sujet = 17:23;

for user = sujet
     if user < 10
        Nuser = ['0' num2str(user)];
    else
        Nuser = num2str(user);
    end
    disp(Nuser);
   
    load(['./data/test_subject' Nuser '.mat']);
    X = double(X);

    % filter
    [b,a] = butter(6,[1 20]/(sfreq/2));
    X2 = filter(b,a,X(:,:,0.5*sfreq:end),[],3);
    
    clear X
    X = permute(X2,[2,3,1]);
    
    save(['preproc/test' Nuser '.mat'],'X');
end

%% 
disp('Preprocessing Done, run final_submission.m next');