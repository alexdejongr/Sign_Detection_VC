        clearvars -except modelo trainedModel; clc; close all;

% carregar paràmetres del pipeline (pca + z-score)
if exist('Dades_Model.mat', 'file')
    load('Dades_Model.mat', 'mu', 'sigma', 'coeff_pca', 'mu_pca', 'k');
else
    error('No es troba Dades_Model.mat');
end

% seleccionar el model actiu
if exist('modelo', 'var')
    model_actiu = modelo;
elseif exist('trainedModel', 'var')
    model_actiu = trainedModel;
else
    error('No hi ha cap model al workspace');
end

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

tic; % Inici cronometre

% segmentació
im_masked = segmentar_universal(im);

% si surt tot negre, no hi ha senyal fiable
im_gray = rgb2gray(im_masked);
bw = imbinarize(im_gray, 0.01);

if ~any(bw(:))
    resultat_text = 'NO DETECTAT';
    score_max = 0;
else
    % seleccionar el blob més gran (candidat principal)
    stats = regionprops(bw, 'Area', 'Perimeter', 'Solidity', 'BoundingBox','Extent','Eccentricity');
    [~, idx] = max([stats.Area]); 
    blob = stats(idx);
    
    compacitat = (blob.Perimeter ^ 2) / max(blob.Area, eps);
    solidesa = blob.Solidity;
    extent = blob.Extent;
    ecc = blob.Eccentricity;
    
    rect = blob.BoundingBox;
    im_crop = imcrop(im_masked, rect);
    
    % percentatge de colors només sobre píxels no negres
    mask_valid = max(im_crop, [], 3) > 0;
    np = sum(mask_valid(:));
    if np == 0
        total_pixels = 1;
    else
        total_pixels = np;
    end
    
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

    % construir vector de features igual que al training
    w_shape = 6; 
    w_color = 2; 
    w_hog = 0.3;
    
    feat_shape_color = [w_shape * [compacitat, solidesa, extent, ecc], ...
                        w_color * [pct_red, pct_blue, pct_yellow]];
    
    feat_hog_weighted = w_hog * hog_vector;
    
    if ~exist('k', 'var') || isempty(k)
        k = 150;
    end

    % la pca requereix centrar amb mu_pca abans de projectar
    feat_hog_centered = feat_hog_weighted - mu_pca;
    feat_hog_pca = feat_hog_centered * coeff_pca(:, 1:k);
    
    VectorFinal = [feat_shape_color, feat_hog_pca];
    
    sigma_safe = sigma;
    sigma_safe(sigma_safe == 0) = 1;
    VectorNorm = (VectorFinal - mu) ./ sigma_safe;

    % el model espera una taula amb els mateixos predictors
    T_test = array2table(VectorNorm);
    
    noms_manuals = {'Compacitat', 'Solidesa', 'Extent', 'Excentricidad', ...
                    'Pct_Red', 'Pct_Blue', 'Pct_Yellow'};
    noms_pca = cell(1, k);
    for j = 1:k
        noms_pca{j} = ['PCA_' num2str(j)];
    end
    
    % si el svm té noms de predictors, provar de respectar-los
    if isa(model_actiu, 'ClassificationSVM') && ~isempty(model_actiu.PredictorNames)
        if length(model_actiu.PredictorNames) == width(T_test)
            T_test.Properties.VariableNames = model_actiu.PredictorNames;
        else
            T_test.Properties.VariableNames = [noms_manuals, noms_pca];
        end
    else
        T_test.Properties.VariableNames = [noms_manuals, noms_pca];
    end

    try
        if isstruct(model_actiu) && isfield(model_actiu, 'predictFcn')
            [pred, scores] = model_actiu.predictFcn(T_test);
        else
            [pred, scores] = predict(model_actiu, T_test);
        end
        
        resultat_text = char(pred);
        score_max = max(scores);
        
    catch ME
        resultat_text = 'Error Prediccio';
        score_max = 0;
        disp(ME.message);
    end
end

temps = toc;

% visualització
figure('Color','w', 'Position', [100 100 1000 500]);

subplot(1, 2, 1);
imshow(im_original);
title('Imatge Original');

subplot(1, 2, 2);
imshow(im_masked);
title({['PREDICCIO: ' resultat_text], ...
       ['Confianca: ' num2str(score_max*100, '%.2f') '%']}, ...
       'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

disp(['Arxiu: ', arxiu]);
disp(['Prediccio: ', resultat_text]);
disp(['Confianca: ', num2str(score_max*100, '%.2f'), '%']);
disp(['Temps: ', num2str(temps, '%.3f'), ' s']);

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