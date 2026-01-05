clearvars -except modelo trainedModel; clc; close all;

% carregar paràmetres del pipeline (pca + z-score)
if exist('Dades_Model.mat', 'file')
    load('Dades_Model.mat', 'mu', 'sigma', 'coeff_pca', 'mu_pca', 'k');
else
    error('No es troba Dades_Model.mat');
end

% carregar únicament el model definitiu
scriptDir = fileparts(mfilename('fullpath'));
modelsDir = fullfile(scriptDir, 'models');

modelFile = fullfile(modelsDir, 'svm_rbf_ecoc_final.mat');
if ~exist(modelFile, 'file')
    error(['No es troba el model definitiu: ' modelFile]);
end

S = load(modelFile);
fn = fieldnames(S);
if isempty(fn)
    error(['El fitxer no conté variables: ' modelFile]);
end
model = S.(fn{1});
[~, modelName, ~] = fileparts(modelFile);

% seleccionar imatge
[arxiu, ruta] = uigetfile({'*.jpg;*.png;*.bmp','Imatges'}, 'Selecciona una imatge');
if isequal(arxiu, 0), return; end

try
    im_original = imread(fullfile(ruta, arxiu));
catch
    error(['El archivo ', arxiu, ' no es JPG/PNG válido. ' ...
           'Convierte la imagen a JPEG real.']);
end
im = im2uint8(im_original);

% segmentació (1 cop)
im_masked = segmentar_universal(im);
im_gray = rgb2gray(im_masked);
bw = imbinarize(im_gray, 0.01);

disp(['Arxiu: ', arxiu]);

res = struct('model', modelName, 'pred', '', 'score', 0, 'time', 0);

if ~any(bw(:))
    % cas sense detecció
    res.pred = 'NO DETECTAT';
    res.score = 0;
    res.time = 0;
else
    % features (1 cop) i després predicció per cada model
    stats = regionprops(bw, 'Area', 'Perimeter', 'Solidity', 'BoundingBox','Extent','Eccentricity');
    [~, idx] = max([stats.Area]);
    blob = stats(idx);

    compacitat = (blob.Perimeter ^ 2) / max(blob.Area, eps);
    solidesa = blob.Solidity;
    extent = blob.Extent;
    ecc = blob.Eccentricity;

    rect = blob.BoundingBox;
    im_crop = imcrop(im_masked, rect);

    mask_valid = max(im_crop, [], 3) > 0;
    np = sum(mask_valid(:));
    total_pixels = max(np, 1);

    im_hsv_crop = rgb2hsv(im_crop);
    H_c = im_hsv_crop(:,:,1); S_c = im_hsv_crop(:,:,2);

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
    w_hog = 0.3;

    feat_shape_color = [w_shape * [compacitat, solidesa, extent, ecc], ...
                        w_color * [pct_red, pct_blue, pct_yellow]];
    feat_hog_weighted = w_hog * hog_vector;

    if ~exist('k', 'var') || isempty(k)
        k = 150;
    end

    feat_hog_centered = feat_hog_weighted - mu_pca;
    feat_hog_pca = feat_hog_centered * coeff_pca(:, 1:k);
    VectorFinal = [feat_shape_color, feat_hog_pca];

    sigma_safe = sigma;
    sigma_safe(sigma_safe == 0) = 1;
    VectorNorm = (VectorFinal - mu) ./ sigma_safe;
    T_test = array2table(VectorNorm);

    noms_manuals = {'Compacitat', 'Solidesa', 'Extent', 'Excentricidad', ...
                    'Pct_Red', 'Pct_Blue', 'Pct_Yellow'};
    noms_pca = cell(1, k);
    for j = 1:k
        noms_pca{j} = ['PCA_' num2str(j)];
    end
    T_test.Properties.VariableNames = [noms_manuals, noms_pca];

    tic;
    try
        if isstruct(model) && isfield(model, 'predictFcn')
            [pred, scores] = model.predictFcn(T_test);
        else
            if isa(model, 'ClassificationSVM') && ~isempty(model.PredictorNames) ...
                    && length(model.PredictorNames) == width(T_test)
                T_test.Properties.VariableNames = model.PredictorNames;
            end
            [pred, scores] = predict(model, T_test);
        end
        res.time = toc;
        res.pred = char(pred);
        res.score = max(scores);
    catch ME
        res.time = toc;
        res.pred = 'Error Prediccio';
        res.score = 0;
        disp(['[' modelName '] ' ME.message]);
    end
end

% visualització (original + masked + predicció)
figure('Color','w', 'Position', [100 100 1050 550]);

subplot(1, 4, 1);
imshow(im_original);
title('Imatge Original');

subplot(1, 4, 2);
imshow(im_masked);
title('Segmentació');

subplot(1, 4, 3);
imshow(im_masked);
title({['Model: ' res.model], ...
       ['Pred: ' res.pred], ...
       ['Conf: ' num2str(res.score*100, '%.2f') '%']}, ...
       'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');

% panell buit per mantenir el layout 1x4 (si vols, el podem eliminar)
subplot(1, 4, 4);
axis off;

disp(['Model: ', res.model]);
disp(['  Prediccio: ', res.pred]);
disp(['  Confianca: ', num2str(res.score*100, '%.2f'), '%']);
disp(['  Temps: ', num2str(res.time, '%.4f'), ' s']);

function im_masked = segmentar_universal(im)
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
        w_box = caixa(3); % Canviat nom per evitar conflictes
        h_box = caixa(4); % Canviat nom per evitar conflictes
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
