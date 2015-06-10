%% Create Speed Features from Histogram

load('distance_driverByTrip.mat')
s = outcell;

n_speedFeatures = length(myCenters);
speedFeatures1 = nan(n_trip,n_speedFeatures,n_drivers);

tic
for i_driver = 1:n_drivers
    for i_path = 1:n_paths
        stack_i_driver = cell2mat(s(i_driver,i_path));
        [nelements,centers] = hist(stack_i_driver,myCenters);
        frequencies = nelements/length(stack_i_driver);

        % Stack the features
        speedFeatures1(i_path,:,i_driver) = frequencies;
    end
end
toc % 8 seconds outcell

clear stack_i_driver outcell s

save('SpeedFeatures1.mat','speedFeatures1')

%% Clasification Tree

X = [speedFeatures1(:,:,1);speedFeatures1(:,:,2)];
y = [ones(200,1);zeros(200,1)];

tic
cTree = fitctree(X,y, ...
    'AlgorithmForCategorical','Exact');
toc
resubLoss(cTree)

%% Logistic Regression

tic
mdl_logReg = fitglm(X,y,'linear','Distribution','Binomial');
toc
mdl_logReg.plotResiduals


%% Logistic Regression Cross Validation Error

% Syntax:
% mcr = crossval('mcr',X,y,'Predfun',predfun)

MyFunction_LogisticReg = @(XTRAIN, ytrain,XTEST) ...
    round(predict(fitglm(XTRAIN,ytrain,'linear','Distribution','Binomial'),XTEST));

mcr_LogReg = crossval('mcr',X,y,'predfun',MyFunction_LogisticReg,'kfold',10); 
display(mcr_LogReg)


%% Loop Through 1 Driver

preds = zeros(400,n_drivers);

for i_driver = 1:(n_drivers-1)
    
    X = [speedFeatures1(:,:,i_driver);speedFeatures1(:,:,i_driver+1)];
    y = [ones(200,1);zeros(200,1)];
    
    mdl_logReg = fitglm(X,y,'linear','Distribution','Binomial');
    
    preds(:,i_driver) = 2 * mdl_logReg.Fitted.Probability - 1;
       
end

pred_logreg1 = sum(preds,2)>0;


%% Loop Through All Drivers

predsCell = cell(n_drivers,n_drivers);
y = [ones(200,1); zeros(200,1)];
zerovec = zeros(400,1);

tic
for i_driver = 1:n_drivers
    for j_driver = 1:n_drivers
        if i_driver >= j_driver
            predsCell{i_driver,j_driver} = zerovec;
        else
            X = [speedFeatures1(:,:,i_driver);speedFeatures1(:,:,j_driver)];
            mdl_logReg = fitglm(X,y,'linear','Distribution','Binomial');
            predsCell{i_driver,j_driver} = 2 * mdl_logReg.Fitted.Probability - 1;
        end
    end
end
toc

predmatUpperRight = cell2mat(predsCell);
predmatLowerLeft  = cell2mat(predsCell');

pred_logreg_temp = (sum(predmatUpperRight,2) + sum(predmatLowerLeft,2))>0;

ind = repmat(y,n_drivers,1) == 1;
pred_logreg = pred_logreg_temp(ind);
