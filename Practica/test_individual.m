clearvars -except modelo trainedModel; clc; close all;

% =========================================================================
% 1. CARREGA DE DADES DE NORMALITZACIÓ I PCA
% =========================================================================
if exist('Dades_Model.mat', 'file')
    load('Dades_Model.mat', 'mu', 'sigma', 'coeff_pca');
    disp('Dades del model (PCA i Normalitzacio) carregades correctament.');
else
    error('Error: No es troba l''arxiu Dades_Model.mat. Executa l''script principal primer.');
end

% Detectar quin tipus de model tenim actiu
if exist('modelo', 'var')
    model_actiu = modelo;
    disp('Utilitzant model: SVM Manual (modelo)');
elseif exist('trainedModel', 'var')
    model_actiu = trainedModel;
    disp('Utilitzant model: Classification Learner (trainedModel)');
else
    error('Error: No hi ha cap model carregat al Workspace.');
end

% =========================================================================
% 2. SELECCIÓ D'IMATGE
% =========================================================================
[arxiu, ruta] = uigetfile({'*.jpg;*.png;*.bmp', 'Imatges'}, 'Selecciona una imatge');
if isequal(arxiu, 0), return; end

im_original = imread(fullfile(ruta, arxiu));
im = im2uint8(im_original);

tic; % Inici cronometre

% =========================================================================
% 3. SEGMENTACIÓ
% =========================================================================
im_masked = segmentar_universal(im);

% Comprovacio rapida si la imatge es negra
im_gray = rgb2gray(im_masked);
bw = imbinarize(im_gray, 0.01);

if ~any(bw(:))
    resultat_text = 'NO DETECTAT';
    score_max = 0;
else
    % =====================================================================
    % 4. EXTRACCIÓ DE CARACTERÍSTIQUES
    % =====================================================================
    
    % --- A. FORMA ---
    stats = regionprops(bw, 'Area', 'Perimeter', 'Solidity', 'BoundingBox','Extent','Eccentricity');
    [~, idx] = max([stats.Area]); 
    blob = stats(idx);
    
    compacitat = (blob.Perimeter ^ 2) / blob.Area;
    solidesa = blob.Solidity;
    extent = blob.Extent;
    ecc = blob.Eccentricity;
    
    % --- B. COLOR (HOC) ---
    rect = blob.BoundingBox;
    im_crop = imcrop(im_masked, rect);
    
    mask_valid = max(im_crop, [], 3) > 0;
    if sum(mask_valid(:)) == 0, total_pixels = 1; else, total_pixels = sum(mask_valid(:)); end
    
    im_hsv_crop = rgb2hsv(im_crop);
    H_c = im_hsv_crop(:,:,1); S_c = im_hsv_crop(:,:,2);
    
    p_red = sum(mask_valid(:) & ((H_c(:)>0.85)|(H_c(:)<0.15)) & (S_c(:)>0.25));
    p_blue = sum(mask_valid(:) & (H_c(:)>0.50)&(H_c(:)<0.75) & (S_c(:)>0.25));
    p_yellow = sum(mask_valid(:) & (H_c(:)>0.12)&(H_c(:)<0.20) & (S_c(:)>0.25));
    
    pct_red = p_red / total_pixels;
    pct_blue = p_blue / total_pixels;
    pct_yellow = p_yellow / total_pixels;
    
    % --- C. TEXTURA (HOG) ---
    im_resized = imresize(im_crop, [64, 64]);
    hog_vector = extractHOGFeatures(im_resized, 'CellSize', [8 8]);
    
    % =====================================================================
    % 5. CONSTRUCCIÓ DEL VECTOR I PCA
    % =====================================================================
    
    % Pesos (Han de ser els mateixos que al training!)
    w_shape = 6; 
    w_color = 2; 
    w_hog = 0.3;
    
    % 1. Apliquem pesos
    feat_shape_color = [w_shape * [compacitat, solidesa, extent, ecc], ...
                        w_color * [pct_red, pct_blue, pct_yellow]];
    
    feat_hog_weighted = w_hog * hog_vector;
    
    % 2. Projecció PCA (nomes a la part HOG)
    % Nota: coeff_pca ja conte la transformacio per a les 150 variables
    feat_hog_pca = feat_hog_weighted * coeff_pca(:, 1:150);
    
    % 3. Concatenacio Final
    VectorFinal = [feat_shape_color, feat_hog_pca];
    
    % 4. Normalitzacio Z-Score (usant mu i sigma carregats)
    VectorNorm = (VectorFinal - mu) ./ sigma;
    
    % =====================================================================
    % 6. PREPARACIÓ DE LA TAULA I PREDICCIÓ
    % =====================================================================
    T_test = array2table(VectorNorm);
    
    % Assignacio de noms de variables per evitar error del SVM
    noms_manuals = {'Compacitat', 'Solidesa', 'Extent', 'Excentricidad', ...
                    'Pct_Red', 'Pct_Blue', 'Pct_Yellow'};
    noms_pca = cell(1, 150);
    for j = 1:150, noms_pca{j} = ['PCA_' num2str(j)]; end
    
    % Intentem usar els noms del model si existeixen, sino els genèrics
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

% =========================================================================
% 7. VISUALITZACIÓ
% =========================================================================
figure('Color','w', 'Position', [100 100 1000 500]);

subplot(1, 2, 1);
imshow(im_original);
title('Imatge Original');

subplot(1, 2, 2);
imshow(im_masked);
title({['PREDICCIO: ' resultat_text], ...
       ['Confianca: ' num2str(score_max*100, '%.2f') '%']}, ...
       'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

disp('-----------------------------------------');
disp(['Arxiu:    ', arxiu]);
disp(['Prediccio: ', resultat_text]);
disp(['Confianca: ', num2str(score_max*100, '%.2f'), '%']);
disp(['Temps:     ', num2str(temps, '%.3f'), ' s']);
disp('-----------------------------------------');


% =========================================================================
% FUNCIÓ DE SEGMENTACIÓ FINAL
% =========================================================================
function im_masked = segmentar_universal(im)
    % 1. Convertim a HSV
    im_hsv = rgb2hsv(im);
    H = im_hsv(:,:,1);
    S = im_hsv(:,:,2);
    V = im_hsv(:,:,3);
    
    % Vermell tolerant
    mask_red = ((H > 0.90) | (H < 0.12)) & (S > 0.15) & (V > 0.15);
    % Blau tolerant
    mask_blue = (H > 0.57) & (H < 0.77) & (S > 0.25) & (V > 0.15);
    % Groc "intel·ligent" (Tancat a 0.19 per evitar fulles verdes)
    mask_yellow = (H > 0.11) & (H < 0.19) & (S > 0.25) & (V > 0.25);
    
    bw_raw = mask_red | mask_blue | mask_yellow;
    
    % Pas A: DILATAR (Connectar)
    se_connect = strel('disk', 3); 
    bw_connected = imdilate(bw_raw, se_connect);
    
    % Pas B: OMPLIR (Sòlid)
    bw_filled = imfill(bw_connected, 'holes');
    
    % Pas C: ERODIR (Recuperar mida original)
    bw_shrunk = imerode(bw_filled, se_connect);
    
    % Pas D: NETEJA FINAL
    se_clean = strel('disk', 2);
    bw_clean = imopen(bw_shrunk, se_clean);
    
    % 4. FILTRATGE
    bw_final = bwareafilt(bw_clean, [150, 999999]); 
    
    % 5. FORMA I RESCAT
    stats = regionprops(bw_final, 'BoundingBox', 'PixelIdxList', 'Extent');
    bw_filtrada = false(size(bw_final));
    found_candidate = false;
    
    for k = 1:length(stats)
        caixa = stats(k).BoundingBox;  
        w_box = caixa(3); % Canviat nom per evitar conflictes
        h_box = caixa(4); % Canviat nom per evitar conflictes
        aspect_ratio = w_box / h_box;
        
        es_proporcionat = (aspect_ratio > 0.4) && (aspect_ratio < 2.2);
        te_consistencia = stats(k).Extent > 0.25; 
        
        if es_proporcionat && te_consistencia
            bw_filtrada(stats(k).PixelIdxList) = true;
            found_candidate = true;
        end
    end
    
    if found_candidate
        bw_final = bwareafilt(bw_filtrada, 1);
    elseif any(bw_final(:))
        % Fail-safe: Si falla la forma, rescatem la taca de color més gran
        bw_final = bwareafilt(bw_final, 1);
    end

    im_masked = im;
    R = im(:,:,1); R(~bw_final) = 0;
    G = im(:,:,2); G(~bw_final) = 0;
    B = im(:,:,3); B(~bw_final) = 0;
    im_masked = cat(3, R, G, B);
end