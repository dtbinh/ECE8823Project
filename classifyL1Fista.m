function [predicted_class xhatClass x_hat] = classifyL1Fista(A,y, class_selector,...
    STOPPING_TIME, maxTime)
    
x_hat = SolveFISTA(A,y, 'stoppingCriterion', STOPPING_TIME,...
    'maxtime', maxTime, 'maxiteration', 1e6);
    
minRes = Inf;
for j=1:length(class_selector)
    x_aug = x_hat.*class_selector{j};
    res = norm(y - A*x_aug);
    if minRes>res
        minRes = res;
        predicted_class = j;
        xhatClass = x_aug;
    end
end
