function [predicted_class xhatClass x_hat] = classifyL2(A,y, class_selector)
    
x_hat = A\y;
    
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
