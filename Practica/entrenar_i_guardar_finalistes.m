% entrena i desa els dos models finalistes a ./models
% finalistes: svm rbf ecoc i ensemble bagging d'arbres

clear; clc;
rng(42);

outDir = fullfile(pwd, 'models');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% carregar dades
if exist('TaulaFinal', 'var') ~= 1
    if exist('TaulaFinal.mat', 'file')
        load('TaulaFinal.mat', 'TaulaFinal');
    else
        error('falta TaulaFinal.mat. executa proba2.m per generar-lo.');
    end
end

% separar predictors i etiqueta
y = categorical(TaulaFinal.Clase);
classes = categories(y);
X = TaulaFinal;
if ismember('Clase', X.Properties.VariableNames)
    X.Clase = [];
end

% metadades per reproduibilitat
meta = struct();
meta.createdAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');
meta.nObs = height(X);
meta.nPredictors = width(X);
meta.predictorNames = X.Properties.VariableNames;
meta.classOrder = classes;
meta.rngSeed = 42;

% model 1: svm rbf ecoc
mdl_svm_rbf = train_svm_rbf_ecoc(X, y);
modelInfo = struct();
modelInfo.name = 'svm rbf amb ecoc one-vs-all (kernel scale auto)';
modelInfo.internalName = 'svm_rbf_ecoc';
modelInfo.template = 'templateSVM(KernelFunction=rbf, KernelScale=auto, Standardize=false)';
modelInfo.coding = 'onevsall';
save(fullfile(outDir, 'svm_rbf_ecoc_final.mat'), 'mdl_svm_rbf', 'meta', 'modelInfo');

% model 2: ensemble bagging
mdl_ens_bag = train_ensemble_bag(X, y);
modelInfo = struct();
modelInfo.name = "ensemble bagging d'arbres (random forest style), 200 arbres";
modelInfo.internalName = 'ensemble_bag';
modelInfo.template = 'templateTree(MinLeafSize=5) + fitcensemble(Method=Bag, NumLearningCycles=200)';
save(fullfile(outDir, 'ensemble_bag_final.mat'), 'mdl_ens_bag', 'meta', 'modelInfo');

disp('fet. models desats a:');
disp(outDir);

% --- helpers ---
function mdl = train_svm_rbf_ecoc(X, y)
    t = templateSVM('KernelFunction','rbf', 'Standardize', false, 'KernelScale','auto');
    mdl = fitcecoc(X, y, 'Learners', t, 'Coding', 'onevsall');
end

function mdl = train_ensemble_bag(X, y)
    t = templateTree('MinLeafSize', 5);
    mdl = fitcensemble(X, y, 'Method', 'Bag', 'Learners', t, 'NumLearningCycles', 200);
end
