function [ Xout ] = applyfilters(X,V)
    
    Xout = zeros(size(V,2),size(X,2),size(X,3));
    % for each trial, apply the spatial filter
    for k=1:size(X,3)
        Xout(:,:,k) = V'*X(:,:,k);
    end
end

