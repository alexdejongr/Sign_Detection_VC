% filepath: /home/alex/Escritorio/VC/Practica_VC/Practica/entrena_i_avaluar_5_models.m

rng(42);

outDir = fullfile('..', 'Resultats', 'Experiments');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% carregar dades
if exist('TaulaFinal', 'var') ~= 1
    if exist('TaulaFinal.mat', 'file')
        load('TaulaFinal.mat', 'TaulaFinal');
    else
        error('falta TaulaFinal.mat. desa TaulaFinal des de proba2 o carrega-la al workspace.');
    end
end

y = categorical(TaulaFinal.Clase);
X = TaulaFinal;
if ismember('Clase', X.Properties.VariableNames)
    X.Clase = [];
end

kfold = 5;
% si alguna classe es massa petita, la k-fold estratificada pot fallar
tabY = countcats(y);
if any(tabY < kfold)
    cvp = cvpartition(numel(y), 'KFold', kfold);
else
    cvp = cvpartition(y, 'KFold', kfold);
end

classes = categories(y);
catOrder = categorical(classes, classes);

outFile = fullfile(outDir, 'resultats_5_models.txt');
fid = fopen(outFile, 'w');

fprintf(fid, "resultats 5 models (script)\n");
fprintf(fid, "data: %s\n", datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, "mostres: %d\n", height(X));
fprintf(fid, "classes: %d\n", numel(classes));
fprintf(fid, "kfold: %d\n\n", kfold);

models = {
    struct('name','svm_linear_ecoc', 'train', @train_svm_linear_ecoc)
    struct('name','svm_rbf_ecoc',    'train', @train_svm_rbf_ecoc)
    struct('name','ensemble_bag',    'train', @train_ensemble_bag)
    struct('name','logistic_ecoc',   'train', @train_logistic_ecoc)
    struct('name','lda',             'train', @train_lda)
};

for mi = 1:numel(models)
    mname = models{mi}.name;
    trainFcn = models{mi}.train;

    fprintf("entrenant %s...\n", mname);
    fprintf(fid, "model: %s\n", mname);

    % inicialitza amb missing i assegura mateixes categories que y
    yPred = categorical(repmat(missing, size(y)), classes);
    trainTimeSum = 0;
    predTimeSum = 0;

    for fold = 1:kfold
        idxTr = training(cvp, fold);
        idxTe = test(cvp, fold);

        Xt = X(idxTr,:);
        yt = y(idxTr);
        Xv = X(idxTe,:);
        yv = y(idxTe); %#ok<NASGU>

        t0 = tic;
        mdl = trainFcn(Xt, yt);
        trainTimeSum = trainTimeSum + toc(t0);

        t1 = tic;
    yhat = predict(mdl, Xv);
    yhat = categorical(yhat, classes);
        predTimeSum = predTimeSum + toc(t1);

        yPred(idxTe) = yhat;
    end

    acc = mean(yPred == y);
    err = 1 - acc;
    totalCost = sum(yPred ~= y);

    % ordre robust: inclou totes les labels definides a y
    [C, order] = confusionmat(y, yPred, 'Order', catOrder);

    rowSum = sum(C, 2);
    recall = diag(C) ./ max(rowSum, 1);

    colSum = sum(C, 1)';
    precision = diag(C) ./ max(colSum, 1);

    f1 = 2 .* (precision .* recall) ./ max(precision + recall, eps);
    macroF1 = mean(f1, 'omitnan');
    balAcc = mean(recall, 'omitnan');

    avgTrainTime = trainTimeSum / kfold;
    obsPerSec = height(X) / max(predTimeSum, eps);

    fprintf(fid, "accuracy: %.4f\n", acc);
    fprintf(fid, "error rate: %.4f\n", err);
    fprintf(fid, "total cost (num errors): %d\n", totalCost);
    fprintf(fid, "balanced accuracy: %.4f\n", balAcc);
    fprintf(fid, "macro f1: %.4f\n", macroF1);
    fprintf(fid, "avg training time per fold (s): %.4f\n", avgTrainTime);
    fprintf(fid, "total prediction time (s): %.4f\n", predTimeSum);
    fprintf(fid, "prediction speed (aprox): %.1f obs/sec\n", obsPerSec);

    % figures: confusion (raw, row-normalized) + per-class recall bar
    fig1 = figure('Color','w','Visible','off','Position',[100 100 1400 900]);
    imagesc(C);
    axis image;
    colormap(fig1, turbo);
    colorbar;
    title(sprintf('confusion matrix (raw) - %s', mname), 'Interpreter', 'none');
    xticks(1:numel(order)); yticks(1:numel(order));
    xticklabels(cellstr(order)); yticklabels(cellstr(order));
    xtickangle(45);
    xlabel('pred'); ylabel('true');
    set(gca,'FontSize',12,'TickLength',[0 0]);
    % anota valors
    for r = 1:size(C,1)
        for c = 1:size(C,2)
            val = C(r,c);
            if val == 0
                continue;
            end
            text(c, r, num2str(val), 'HorizontalAlignment','center', 'FontSize',9, 'Color','k');
        end
    end
    exportgraphics(fig1, fullfile(outDir, sprintf('confusion_raw_%s.png', mname)), 'Resolution', 200);
    close(fig1);

    Cn = C ./ max(sum(C,2), 1);
    Cp = 100 * Cn;
    fig2 = figure('Color','w','Visible','off','Position',[100 100 1400 900]);
    imagesc(Cp);
    axis image;
    colormap(fig2, turbo);
    colorbar;
    caxis([0 100]);
    title(sprintf('confusion matrix (row norm, %%) - %s', mname), 'Interpreter', 'none');
    xticks(1:numel(order)); yticks(1:numel(order));
    xticklabels(cellstr(order)); yticklabels(cellstr(order));
    xtickangle(45);
    xlabel('pred'); ylabel('true');
    set(gca,'FontSize',12,'TickLength',[0 0]);
    % anota percentatges
    for r = 1:size(Cp,1)
        for c = 1:size(Cp,2)
            val = Cp(r,c);
            if val < 0.05
                continue;
            end
            text(c, r, sprintf('%.1f', val), 'HorizontalAlignment','center', 'FontSize',9, 'Color','k');
        end
    end
    exportgraphics(fig2, fullfile(outDir, sprintf('confusion_row_norm_%s.png', mname)), 'Resolution', 200);
    close(fig2);

    fig3 = figure('Color','w','Visible','off','Position',[100 100 1400 700]);
    b = bar(recall);
    b.FaceColor = [0.2 0.5 0.9];
    ylim([0 1]);
    grid on;
    title(sprintf('recall per classe - %s', mname), 'Interpreter', 'none');
    xticks(1:numel(classes));
    xticklabels(classes);
    xtickangle(45);
    ylabel('recall');
    set(gca,'FontSize',12);
    exportgraphics(fig3, fullfile(outDir, sprintf('recall_per_class_%s.png', mname)), 'Resolution', 200);
    close(fig3);

    fprintf(fid, "\nclasses (ordre):\n%s\n", strjoin(cellstr(classes), ", "));

    fprintf(fid, "\nconfusion matrix (rows=true, cols=pred):\n");
    for r = 1:size(C,1)
        fprintf(fid, "%s\n", strjoin(string(C(r,:)), "\t"));
    end

    fprintf(fid, "\nconfusion matrix normalitzada per fila (%%):\n");
    for r = 1:size(Cn,1)
        fprintf(fid, "%s\n", strjoin(string(round(100*Cn(r,:)*10)/10), "\t"));
    end

    fprintf(fid, "\nper class metrics:\n");
    fprintf(fid, "classe\trecall\tprecision\tf1\n");
    for ci = 1:numel(classes)
        fprintf(fid, "%s\t%.4f\t%.4f\t%.4f\n", classes{ci}, recall(ci), precision(ci), f1(ci));
    end

    fprintf(fid, "\n\n");
end

fclose(fid);
disp("fet. fitxer resultats: " + outFile);
disp("fet. figures a: " + outDir);

function mdl = train_svm_linear_ecoc(X, y)
    t = templateSVM('KernelFunction','linear', 'Standardize', false);
    mdl = fitcecoc(X, y, 'Learners', t, 'Coding', 'onevsall');
end

function mdl = train_svm_rbf_ecoc(X, y)
    t = templateSVM('KernelFunction','rbf', 'Standardize', false, 'KernelScale','auto');
    mdl = fitcecoc(X, y, 'Learners', t, 'Coding', 'onevsall');
end

function mdl = train_ensemble_bag(X, y)
    t = templateTree('MinLeafSize', 5);
    mdl = fitcensemble(X, y, 'Method', 'Bag', 'Learners', t, 'NumLearningCycles', 200);
end

function mdl = train_logistic_ecoc(X, y)
    t = templateLinear('Learner', 'logistic');
    mdl = fitcecoc(X, y, 'Learners', t, 'Coding', 'onevsall');
end

function mdl = train_lda(X, y)
    mdl = fitcdiscr(X, y, 'DiscrimType', 'linear');
end