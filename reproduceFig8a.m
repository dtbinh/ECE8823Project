%reproduce figure 8.a of Wright paper with random faces and compare to our
%simple projection classifier

clearvars; close all; clc;
load extended_yale_B.mat

%OPTIONS
%how many different training/testing data combinations
numTrainComb = 100;
%how many tests per combination
numTestsPerform = 10;
%how many features (random faces)
numFeats = [30 56 120 504]; %0 means no dimensionality reduction, use raw images
%classifier
ctype1 = 'srcFista';
p.STOPPING_TIME = -2;
p.maxTime = 8;

% ctype = 'srcOMP';
% p.K = 10; %max number of iterations
% p.tol = 1e-10; %stop if residual is smaller

ctype2 = 'projection';
p.cdim = 9; %taining samples per class. if cdim>numFeat, this classifier 
           %doesnt make sense because the system is underdetermined and the
           %residual is expected to become very small for all classes
           
for numFeat = numFeats
    numFeat
detectionRateComb1 = zeros(1,numTrainComb);
detectionRateComb2 = zeros(1,numTrainComb);
for combs = 1:numTrainComb
    if ~mod(combs,10)
        combs
    end
                
%DATASETS AND LABELS
% create training and testing sets
train_indices = false(1,length(gnd));
classes = unique(gnd);
flip = int8(-1);
for cc = classes'
    n = sum(gnd==cc);
    startPos = find(gnd == cc,1);
    if ~mod(n,2)
        indices = randperm(n,n/2);
    else
        indices = randperm(n,(n+flip)/2);
        flip = flip*(-1);
    end
        train_indices(startPos-1+indices) = true;
end
test_indices = ~train_indices;

A_train = fea(:,train_indices);
train_labels = gnd(train_indices);

A_test = fea(:,test_indices);
test_labels = gnd(test_indices);

% get cardinality of training, testing and label sets
num_train = length(train_labels);
num_test = length(test_labels);
num_labels = length(classes);

% compute selector operators, i.e. where the coefficients 
% associated to each class are stored 
z1 = false(num_train,1);
class_selector = cell(1,num_labels);
for i=1:num_labels
  inds = find(train_labels == classes(i));
  z_temp = z1; 
  z_temp(inds) = true; 
  class_selector{i} = z_temp;
end

% TESTS
check_arr = randperm(num_test, numTestsPerform);
accuracy1 = zeros(1,num_test);
accuracy2 = zeros(1,num_test);
for k=1:length(check_arr)
  i = check_arr(k);
  %random faces
  if numFeat > 0
      R = randn(numFeat, h*w);
      y = R*A_test(:,i);
      A = R*A_train;
  else
      y = A_test(:,i);
      A = A_train;
  end
  
  % classify1
  switch ctype1
      case 'srcFista'
          predicted = classifyL1Fista(A,y, class_selector,...
              p.STOPPING_TIME, p.maxTime);
      case 'srcOMP'
          predicted = classifyL1OMP(A,y, class_selector,...
              p.K, p.tol);
      case 'projection'
          predicted = classifyProjection(A,y, class_selector, p.cdim);
  end 
  predicted_class1 = classes(predicted);
  
  % classify2
  switch ctype2
      case 'srcFista'
          predicted = classifyL1Fista(A,y, class_selector,...
              p.STOPPING_TIME, p.maxTime);
      case 'srcOMP'
          predicted = classifyL1OMP(A,y, class_selector,...
              p.K, p.tol);
      case 'projection'
          predicted = classifyProjection(A,y, class_selector, p.cdim);
  end 
  predicted_class2 = classes(predicted);

  actual_class = test_labels(i);
  
  if predicted_class1 == actual_class
    accuracy1(i) = 1;
  end
  if predicted_class2 == actual_class
    accuracy2(i) = 1;
  end
end    
detectionRateComb1(combs) = sum(accuracy1)/numTestsPerform;
detectionRateComb2(combs) = sum(accuracy2)/numTestsPerform;
end
detectionRate1 = mean(detectionRateComb1);
detectionRate2 = mean(detectionRateComb2);
save(['testDim' num2str(numFeat) '.mat'],'detectionRate1', 'detectionRate2','numFeat');
end

