function predicted = classifyProjection(A,y, class_selector, cdim)

residuals = zeros(1,length(class_selector));
for j=1:length(class_selector)
    %pick cdim training samples from that class
    samples = false(size(class_selector{j}));
    n = sum(class_selector{j});
    startPos = find(class_selector{j},1);
    pick = randperm(n,cdim);
    samples(startPos-1+pick) = true;
    A_sub = A(:,samples);
    %project
    x_hat = A_sub\y;
    residuals(j) = norm(y - A_sub*x_hat);
end
[m, predicted] = min(residuals);