function [predicted xhatClass xhat residuals] = classifyProjection(A,y, class_selector, cdim)

xhat = zeros(size(A,2),1);
residuals = zeros(1,length(class_selector));
minRes = Inf;
for j=1:length(class_selector)
    %pick cdim training samples from that class
    samples = false(size(class_selector{j}));
    n = sum(class_selector{j});
    startPos = find(class_selector{j},1);
    pick = randpermk(n,cdim);
    samples(startPos-1+pick) = true;
    A_sub = A(:,samples);
    %project
    x_hat = A_sub\y;
    res = norm(y - A_sub*x_hat);
    residuals(j) = res;
    if minRes > res
        minRes = res;
        predicted = j;
        xhatClass = zeros(size(A,2),1);
        xhatClass(samples) = x_hat;      
    end
    xhat(samples) = x_hat;
end