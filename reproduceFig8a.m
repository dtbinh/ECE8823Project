%reproduce figure 8.a of Wright paper with random faces and compare to our
%simple projection classifier

clearvars; close all; clc;
load extended_yale_B.mat

dbstop error

%% OPTIONS
%how many different training/testing data combinations for testing
numTrainComb = 1000;
%how many tests per combination
numTestsPerform = 1;
%how many features (random faces)
numFeats = 56;%[30 56 120 504]; %0 means no dimensionality reduction, use raw images
%# training samples per class
numTrainSamples = 29; %0 means use 1/2 for training and 1/2 for testing
%use regular test image = 0
%use linear combination of training data as test image =1 
bUseLinComb = 0;
%if bUseLinComb = 1, how many training samples should we use to build y?
numTrainForY = 10;
%Stop and run analyzing scripts when L1 succeeds but Projection fails
bAnalyze = 1;
%classifier
ctype1 = 'srcFista';
p.STOPPING_TIME = -2;
p.maxTime = 8;

ctype2 = 'projection';
p.cdim = 29; %how many taining samples per class to project on. if 
%cdim>numFeat, this classifier doesnt make sense because the system is
%underdetermined and the residual is expected to become very small for all
%classes     

%other not so useful classifiers
%
%ctype= 'guess';
%
% ctype = 'srcOMP';
% p.K = 10; %max number of iterations
% p.tol = 1e-10; %stop if residual is smaller
%
%ctype = 'srcL2';

%% Initializations
detectionRates1 = zeros(1,length(numFeats));
detectionRates2 = zeros(1,length(numFeats));
for nf = 1:length(numFeats)
    disp('================')
    disp(num2str(numFeats(nf)))
    detectionRateComb1 = zeros(1,numTrainComb);
    detectionRateComb2 = zeros(1,numTrainComb);
for combs = 1:numTrainComb
    if ~mod(combs,100)
        disp(num2str(combs))
    end
                
%% DATASETS AND LABELS
% create training and testing sets
train_indices = false(1,length(gnd));
classes = unique(gnd);

if numTrainSamples == 0 %take 1/2 for training and 1/2 for testing
    flip = int8(-1);
    for cc = classes'
        n = sum(gnd==cc);
        startPos = find(gnd == cc,1);
        if ~mod(n,2)
            indices = randpermk(n,n/2);
        else
            indices = randpermk(n,(n+flip)/2);
            flip = flip*(-1);
        end
        train_indices(startPos-1+indices) = true;
    end
else %take a fixed number of samples from each class for training
    for cc = classes'
        n = sum(gnd==cc);
        startPos = find(gnd == cc,1);
        indices = randpermk(n,numTrainSamples);
        train_indices(startPos-1+indices) = true;
    end
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

%% TESTS
check_arr = randpermk(num_test, numTestsPerform);
accuracy1 = zeros(1,numTestsPerform);
accuracy2 = zeros(1,numTestsPerform);
for k=1:length(check_arr)
    %get full test image yfull
    if bUseLinComb %linear combination of training data
        %pick a true class
        iclass = randpermk(length(classes),1);
        actual_class = classes(iclass);
        %build linear combination
        Atrue = A_train(:,class_selector{iclass});
        pick = randpermk(size(Atrue,2),numTrainForY);
        randCoeff = randn(1,numTrainForY);
        yfull = Atrue(:,pick)*randCoeff;
    else       
        i = check_arr(k);
        actual_class = test_labels(i);
        yfull = A_test(:,i);      
    end
    
    %random faces?
    if numFeats(nf) > 0
        R = randn(numFeats(nf), h*w);
        y = R*yfull;
        A = R*A_train;
    else
        y = yfull;
        A = A_train;
    end
    
    % classify using 1
   switch ctype1
      case 'srcFista'
          [predicted xhatClass xhat] = classifyL1Fista(A,y, class_selector,...
              p.STOPPING_TIME, p.maxTime);
      case 'srcOMP'
          [predicted xhatClass xhat] = classifyL1OMP(A,y, class_selector,...
              p.K, p.tol);
      case 'projection'
          [predicted xhatClass xhat residualsP] = classifyProjection(A,y, class_selector, p.cdim);
      case 'srcL2'
          [predicted xhatClass xhat] = classifyL2(A,y, class_selector);
      otherwise
          predicted = 1; %random guess
          xhat = zeros(num_train,1);
  end 
  predicted_class1 = classes(predicted)
  xhat1 = xhat; %full xhat
  xhatClass1 = xhatClass; %xhat restricted to predicted class
  
  % classify2
  switch ctype2
      case 'srcFista'
          [predicted xhatClass xhat] = classifyL1Fista(A,y, class_selector,...
              p.STOPPING_TIME, p.maxTime);
      case 'srcOMP'
          [predicted xhatClass xhat] = classifyL1OMP(A,y, class_selector,...
              p.K, p.tol);
      case 'projection'
          [predicted xhatClass xhat residualsP] = classifyProjection(A,y, class_selector, p.cdim);
      case 'srcL2'
          [predicted xhatClass xhat] = classifyL2(A,y, class_selector);
      otherwise
          predicted = 1; %random guess
          xhat = zeros(num_train,1);
  end 
  predicted_class2 = classes(predicted)
  xhat2 = xhat; %all projections in one vector (not very meaningful)
  xhatClass2 = xhatClass; %xhat restricted to predicted class
  
  actual_class

  if predicted_class1 == actual_class
    accuracy1(i) = 1;
  end
  if predicted_class2 == actual_class
    accuracy2(i) = 1;
  end
  
  %% debug what's going wrong when L1 succeeds, but projection fails
  if bAnalyze && (predicted_class1 == actual_class) && (predicted_class2 ~= actual_class)
      
     figure(1);hold off;  
     plot(y)
     hold on
     plot(A*xhatClass1,'r')
     plot(A*xhat1,'m')
     plot(A*xhatClass2,'g')
     title('test image and reconstructions')
     legend('test image y', ['reconstructed ' ctype1 ' (predicted class only)'],...
         ['reconstructed ' ctype1 ' (all entries of xhat)'], ['reconstructed ' ctype2 ' (from predicted class)'])
     %-------------
     figure(2);hold off;
     plot(xhat1,'r')
     hold on
     plot(xhatClass2,'b')
     title('compare the xhats')
     legend(ctype1, ctype2);     
     %-------------
     normxhat1 = zeros(1,num_labels);
     residuals1 = zeros(1,num_labels);
     for ii = 1:num_labels
         xtest1 = xhat1.*class_selector{ii};
         normxhat1(ii) = norm(xtest1);
         residuals1(ii) = norm(y-A*xtest1);
     end
     [residualsSort1 ind1] = sort(residuals1);
          
     figure(3);hold off;
     plot(1:num_labels, residualsSort1,'o')
     hold on
     plot(find(classes(ind1)==actual_class), residualsSort1(classes(ind1)==actual_class),'xr')
     legend('residuals from classes', 'true class')
     title('residuals from different classes for L1')
     %-------------
     normxhat2 = zeros(1,num_labels);
     for ii = 1:num_labels
         xtest = xhat2.*class_selector{ii};
         normxhat2(ii) = norm(xtest);
     end
     [residualsSortP ind] = sort(residualsP);
     
     ResMatrix2 = [residualsSortP; classes(ind)'; normxhat2(ind); (classes(ind)==actual_class)'];
     
     legend('residuals from classes', 'true class')
     figure(4);hold off;
     plot(1:num_labels, residualsSortP,'o')
     hold on
     plot(find(classes(ind)==actual_class), residualsSortP(classes(ind)==actual_class),'xr')
     legend('residuals from classes', 'true class')
     title('residuals from different classes for Projection')
     
     %--------------
     displayFace(yfull,h,w,5) 
     title('test image')
    
     stop = 1; %set debug marker to stop
  end
  
end    
detectionRateComb1(combs) = mean(accuracy1);
detectionRateComb2(combs) = mean(accuracy2);
end

detectionRates1(nf) = mean(detectionRateComb1);
detectionRates2(nf)  = mean(detectionRateComb2);
end
save(['test' datestr(now,'yyyy_mm_dd_HH:MM') '.mat'],'detectionRates1', 'detectionRates2','numFeats');

