% test_dataset_extern
% avalua un model entrenat sobre el dataset extern (Test_Images) utilitzant el mateix pipeline de features que proba2.m
% 
% entrada
% - carpeta: ./Test_Images (subcarpetes = classes)
% - model: ./models/svm_rbf_ecoc_final.mat o ./models/ensemble_bag_final.mat
% - paràmetres: Dades_Model.mat (mu, sigma, coeff_pca, mu_pca, k)
%
% sortida
% - ./Resultats/Experiments/extern_<model>_resultats.txt
% - png: confusion_raw / confusion_row_norm / recall_per_class

% IMPORTANT: no fem `clear` perquè permet passar paràmetres (modelFile, externalRoot, outDir)
% des de batch abans de fer `run('test_dataset_extern.m')`.
clc;
rng(42);

% paths
if ~exist('externalRoot', 'var') || isempty(externalRoot)
    % per defecte, el dataset extern és un nivell a sobre del projecte
    externalRoot = fullfile(pwd, '..', 'Test_Images');
end
if ~exist('outDir', 'var') || isempty(outDir)
    outDir = fullfile('..', 'Resultats', 'Experiments', 'dataset_extra');
end
modelsDir = fullfile(pwd, 'models');
ruta_dataset = externalRoot;
if ~exist(outDir, 'dir'); mkdir(outDir); end

% tria de model (si no s'ha passat des de fora)
if ~exist('modelFile', 'var') || isempty(modelFile)
    modelFile = fullfile(modelsDir, 'svm_rbf_ecoc_final.mat');
end

if ~exist(ruta_dataset, 'dir')
    % tolera el cas Test_images vs Test_Images
    ruta_alt = fullfile(fileparts(ruta_dataset), 'Test_images');
    if exist(ruta_alt, 'dir')
        ruta_dataset = ruta_alt;
    end
end
if ~exist(ruta_dataset, 'dir')
    error('no existeix la carpeta del dataset extern (esperada: ../Test_Images): %s', ruta_dataset);
end
if ~exist(modelFile, 'file')
    error('no existeix el model: %s', modelFile);
end
if ~exist('Dades_Model.mat', 'file')
    error('falta Dades_Model.mat. executa proba2.m abans.');
end

% carregar paràmetres del pipeline
S = load('Dades_Model.mat', 'mu', 'sigma', 'coeff_pca', 'mu_pca', 'k');
mu = S.mu;
sigma = S.sigma;
coeff_pca = S.coeff_pca;
mu_pca = S.mu_pca;
k = S.k;
sigma(sigma == 0) = 1;

% carregar model
M = load(modelFile);
[mdl, ~] = getModelFromMat(M, modelFile);

% nom estable per outputs: basat en el nom del fitxer .mat
[~, modelNameFromFile] = fileparts(modelFile);
modelNameFromFile = regexprep(modelNameFromFile, '_final$', '');
modelName = modelNameFromFile;

% carregar dataset extern
imds = imageDatastore(ruta_dataset, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
numImages = numel(imds.Files);

Features = zeros(numImages, 1771);
Labels = cell(numImages, 1);

nom_fitxer_errors = fullfile(outDir, sprintf('extern_%s_errores_identificacion.txt', modelName));
fid = fopen(nom_fitxer_errors, 'w');
fprintf(fid, 'Carpeta\tNom_Imatge\n');

disp(['extern: s''han trobat ', num2str(numImages), ' imatges. processant...']);
reset(imds);

for i = 1:numImages
    [im, info] = read(imds);
    im = im2uint8(im);

    Labels{i} = char(info.Label);

    im_masked = segmentar(im);

    im_gray = rgb2gray(im_masked);
    bw = imbinarize(im_gray, 0.01);

    if ~any(bw(:))
        [~, nom_arxiu, ext] = fileparts(info.Filename);
        fprintf(fid, '%s\t%s\n', char(info.Label), [nom_arxiu, ext]);
        Features(i, :) = zeros(1, 1771);
        continue;
    end

    stats = regionprops(bw, 'Area', 'Perimeter', 'Solidity', 'BoundingBox','Extent','Eccentricity');
    [~, idx] = max([stats.Area]);
    blob = stats(idx);

    compacitat = (blob.Perimeter ^ 2) / max(blob.Area, eps);
    solidesa = blob.Solidity;
    extent = blob.Extent;
    ecc = blob.Eccentricity;

    rect = blob.BoundingBox;
    im_crop = imcrop(im_masked, rect);

    im_hsv_crop = rgb2hsv(im_crop);
    H_c = im_hsv_crop(:,:,1);
    S_c = im_hsv_crop(:,:,2);

    mask_valid = max(im_crop, [], 3) > 0;
    np = sum(mask_valid(:));
    total_pixels = max(np, 1);

    p_red = sum(mask_valid(:) & ((H_c(:)>0.85)|(H_c(:)<0.15)) & (S_c(:)>0.25));
    p_blue = sum(mask_valid(:) & (H_c(:)>0.50)&(H_c(:)<0.75) & (S_c(:)>0.25));
    p_yellow = sum(mask_valid(:) & (H_c(:)>0.12)&(H_c(:)<0.20) & (S_c(:)>0.25));

    pct_red = p_red / total_pixels;
    pct_blue = p_blue / total_pixels;
    pct_yellow = p_yellow / total_pixels;

    im_resized = imresize(im_crop, [64, 64]);
    hog_vector = extractHOGFeatures(im_resized, 'CellSize', [8 8]);

    w_shape = 6;
    w_color = 2;
    w_hog   = 0.3;

    Features(i,:) = [
        w_shape * [compacitat, solidesa, extent, ecc], ...
        w_color * [pct_red, pct_blue, pct_yellow], ...
        w_hog   * hog_vector
    ];
end

fclose(fid);

idx_brossa = all(Features == 0, 2);
Labels_clean = Labels(~idx_brossa);
Features_clean = Features(~idx_brossa, :);

n_fallades = sum(idx_brossa);
disp(['extern: eliminades ', num2str(n_fallades), ' imatges per fallada de segmentació']);

% pca + z-score (com proba2/test_individual)
X_shape_color = Features_clean(:, 1:7);
X_hog = Features_clean(:, 8:end);

X_hog_centered = X_hog - mu_pca;
X_hog_pca = X_hog_centered * coeff_pca(:, 1:k);
Features_final = [X_shape_color, X_hog_pca];

Features_Norm = (Features_final - mu) ./ sigma;

TaulaExt = array2table(Features_Norm);

noms_manuals = {'Compacitat', 'Solidesa', 'Extent', 'Excentricidad', 'Pct_Red', 'Pct_Blue', 'Pct_Yellow'};
noms_pca = cell(1, k);
for j = 1:k
    noms_pca{j} = ['PCA_' num2str(j)];
end
TaulaExt.Properties.VariableNames = [noms_manuals, noms_pca];

% predicció
% respecta ordre de predictors si el model el requereix
TaulaExt_forPred = TaulaExt;
if isfield(M, 'meta') && isfield(M.meta, 'predictorNames')
    try
        TaulaExt_forPred = TaulaExt_forPred(:, M.meta.predictorNames);
    catch
        % si no coincideixen els noms, mantenim l'ordre actual
    end
end

t0 = tic;
[yPred, scores] = predict(mdl, TaulaExt_forPred); %#ok<ASGLU>
predTime = toc(t0);

% mètriques
yTrue = categorical(string(Labels_clean));
yPred = categorical(string(yPred));

% força el mateix ordre de classes que el dataset extern
classesTrue = categories(yTrue);
order = categorical(classesTrue, classesTrue);
[C, order] = confusionmat(yTrue, yPred, 'Order', order);

acc = mean(yPred == yTrue);
err = 1 - acc;
totalCost = sum(yPred ~= yTrue);

rowSum = sum(C, 2);
recall = diag(C) ./ max(rowSum, 1);
colSum = sum(C, 1)';
precision = diag(C) ./ max(colSum, 1);
f1 = 2 .* (precision .* recall) ./ max(precision + recall, eps);
macroF1 = mean(f1, 'omitnan');
balAcc = mean(recall, 'omitnan');
obsPerSec = numel(yTrue) / max(predTime, eps);

% guardar txt
outTxt = fullfile(outDir, sprintf('extern_%s_resultats.txt', modelName));
fout = fopen(outTxt, 'w');
fprintf(fout, 'resultats extern (%s)\n', modelName);
fprintf(fout, 'data: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fout, 'mostres (tot): %d\n', numImages);
fprintf(fout, 'mostres (valides): %d\n', numel(yTrue));
fprintf(fout, 'eliminades per segmentacio: %d\n\n', n_fallades);

fprintf(fout, 'accuracy: %.4f\n', acc);
fprintf(fout, 'error rate: %.4f\n', err);
fprintf(fout, 'total cost (num errors): %d\n', totalCost);
fprintf(fout, 'balanced accuracy: %.4f\n', balAcc);
fprintf(fout, 'macro f1: %.4f\n', macroF1);
fprintf(fout, 'prediction time (s): %.4f\n', predTime);
fprintf(fout, 'prediction speed (aprox): %.1f obs/sec\n\n', obsPerSec);

fprintf(fout, 'classes (ordre):\n%s\n\n', strjoin(cellstr(order), ', '));

fprintf(fout, 'confusion matrix (rows=true, cols=pred):\n');
for r = 1:size(C,1)
    fprintf(fout, '%s\n', strjoin(string(C(r,:)), '\t'));
end

Cn = C ./ max(sum(C,2), 1);
fprintf(fout, '\nconfusion matrix normalitzada per fila (%%):\n');
for r = 1:size(Cn,1)
    fprintf(fout, '%s\n', strjoin(string(round(100*Cn(r,:)*10)/10), '\t'));
end

fprintf(fout, '\nper class metrics:\n');
fprintf(fout, 'classe\trecall\tprecision\tf1\n');
for ci = 1:numel(order)
    fprintf(fout, '%s\t%.4f\t%.4f\t%.4f\n', string(order(ci)), recall(ci), precision(ci), f1(ci));
end

fclose(fout);

% figures
fig1 = figure('Color','w','Visible','off','Position',[100 100 1400 900]);
imagesc(C);
axis image;
colormap(fig1, turbo);
colorbar;
title(sprintf('extern confusion (raw) - %s', modelName), 'Interpreter', 'none');
xticks(1:numel(order)); yticks(1:numel(order));
xticklabels(cellstr(order)); yticklabels(cellstr(order));
xtickangle(45);
xlabel('pred'); ylabel('true');
set(gca,'FontSize',12,'TickLength',[0 0]);
for r = 1:size(C,1)
    for c = 1:size(C,2)
        val = C(r,c);
        if val == 0; continue; end
        text(c, r, num2str(val), 'HorizontalAlignment','center', 'FontSize',9, 'Color','k');
    end
end
exportgraphics(fig1, fullfile(outDir, sprintf('extern_confusion_raw_%s.png', modelName)), 'Resolution', 200);
close(fig1);

Cp = 100 * Cn;
fig2 = figure('Color','w','Visible','off','Position',[100 100 1400 900]);
imagesc(Cp);
axis image;
colormap(fig2, turbo);
colorbar;
caxis([0 100]);
title(sprintf('extern confusion (row norm, %%) - %s', modelName), 'Interpreter', 'none');
xticks(1:numel(order)); yticks(1:numel(order));
xticklabels(cellstr(order)); yticklabels(cellstr(order));
xtickangle(45);
xlabel('pred'); ylabel('true');
set(gca,'FontSize',12,'TickLength',[0 0]);
for r = 1:size(Cp,1)
    for c = 1:size(Cp,2)
        val = Cp(r,c);
        if val < 0.05; continue; end
        text(c, r, sprintf('%.1f', val), 'HorizontalAlignment','center', 'FontSize',9, 'Color','k');
    end
end
exportgraphics(fig2, fullfile(outDir, sprintf('extern_confusion_row_norm_%s.png', modelName)), 'Resolution', 200);
close(fig2);

fig3 = figure('Color','w','Visible','off','Position',[100 100 1400 700]);
b = bar(recall);
b.FaceColor = [0.2 0.5 0.9];
ylim([0 1]);
grid on;
title(sprintf('extern recall per classe - %s', modelName), 'Interpreter', 'none');
xticks(1:numel(order));
xticklabels(cellstr(order));
xtickangle(45);
ylabel('recall');
set(gca,'FontSize',12);
exportgraphics(fig3, fullfile(outDir, sprintf('extern_recall_per_class_%s.png', modelName)), 'Resolution', 200);
close(fig3);

disp('fet. resultats externs:');
disp(outTxt);

% ---- helpers ----
function [mdl, modelName] = getModelFromMat(M, modelFile)
    if isfield(M, 'mdl_svm_rbf')
        mdl = M.mdl_svm_rbf;
        modelName = 'svm_rbf_ecoc';
        return;
    end
    if isfield(M, 'mdl_ens_bag')
        mdl = M.mdl_ens_bag;
        modelName = 'ensemble_bag';
        return;
    end

    % fallback (si el nom de variable canvia)
    fns = fieldnames(M);
    for i = 1:numel(fns)
        if contains(fns{i}, 'mdl')
            mdl = M.(fns{i});
            modelName = regexprep(fns{i}, '^mdl_', '');
            return;
        end
    end

    error('no he trobat cap variable de model dins %s', modelFile);
end

function im_masked = segmentar(im)
    im_hsv = rgb2hsv(im);
    H = im_hsv(:,:,1);
    S = im_hsv(:,:,2);
    V = im_hsv(:,:,3);

    mask_red = ((H > 0.90) | (H < 0.12)) & (S > 0.15) & (V > 0.15);
    mask_blue = (H > 0.57) & (H < 0.77) & (S > 0.25) & (V > 0.15);
    mask_yellow = (H > 0.11) & (H < 0.19) & (S > 0.25) & (V > 0.25);

    bw_raw = mask_red | mask_blue | mask_yellow;

    se_connect = strel('disk', 3);
    bw_connected = imdilate(bw_raw, se_connect);

    bw_filled = imfill(bw_connected, 'holes');

    bw_shrunk = imerode(bw_filled, se_connect);

    se_clean = strel('disk', 2);
    bw_clean = imopen(bw_shrunk, se_clean);

    bw_final = bwareafilt(bw_clean, [150, 999999]);

    stats = regionprops(bw_final, 'BoundingBox', 'PixelIdxList', 'Extent');
    bw_filtrada = false(size(bw_final));
    found_candidate = false;

    for t = 1:length(stats)
        caixa = stats(t).BoundingBox;
        w_box = caixa(3);
        h_box = caixa(4);
        aspect_ratio = w_box / max(h_box, eps);

        es_proporcionat = (aspect_ratio > 0.4) && (aspect_ratio < 2.2);
        te_consistencia = stats(t).Extent > 0.25;

        if es_proporcionat && te_consistencia
            bw_filtrada(stats(t).PixelIdxList) = true;
            found_candidate = true;
        end
    end

    if found_candidate
        bw_final = bwareafilt(bw_filtrada, 1);
    elseif any(bw_final(:))
        bw_final = bwareafilt(bw_final, 1);
    end

    R = im(:,:,1); R(~bw_final) = 0;
    G = im(:,:,2); G(~bw_final) = 0;
    B = im(:,:,3); B(~bw_final) = 0;
    im_masked = cat(3, R, G, B);
end
