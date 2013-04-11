function predicted_class = classifyL1Fista(A,y, class_selector,...
    STOPPING_TIME, maxTime)

x_hat = SolveFISTA(A,y, 'stoppingCriterion', STOPPING_TIME,...
    'maxtime', maxTime, 'maxiteration', 1e6);

residuals = zeros(1,length(class_selector));
for j=1:length(class_selector)
    x_aug = x_hat.*class_selector{j};
    residuals(j) = norm(y - A*x_aug);
end
[m, predicted_class] = min(residuals);
