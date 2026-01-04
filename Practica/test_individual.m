clearvars -except modelo coeff_pca trainedModel Features TaulaFinal; clc; close all;

% =========================================================================
%                       CONFIGURACIÓ INICIAL
% =========================================================================

% 1. RECUPERAR DADES DE NORMALITZACIÓ
% Necessitem la Mitjana (mu) i Desviació (sigma) de les dades originals.
if exist('Features', 'var')
    mu = mean(Features);
    sigma = std(Features);
    disp('✅ Dades de normalització: Recuperades de "Features".');
elseif exist('TaulaFinal', 'var')
    % Si només tens la TaulaFinal (ja normalitzada), intentem fer enginyeria inversa
    % o assumim que tens les dades crues en una altra variable.
    % L'ideal és tenir 'Features'. Si no, avisem.
    disp('⚠️ ALERTA: No trobo "Features" (dades crues). Si "TaulaFinal" ja està normalitzada,');
    disp('   això no serà exacte. Intentant recuperar dades...');
    raw_data = TaulaFinal{:, 1:end-1}; 
    mu = mean(raw_data); % Això serà 0 si ja està normalitzat (malament)
    sigma = std(raw_data); % Això serà 1 (malament)
    if abs(mean(mu)) < 0.1
        error('❌ ERROR CRÍTIC: La variable "Features" no hi és i "TaulaFinal" sembla ja normalitzada. Necessito les dades originals per calcular la mitjana real.');
    end
else
    error('❌ ERROR: Necessito la variable "Features" al Workspace.');
end

% 2. DETECTAR EL MODEL
if exist('modelo', 'var')
    model_actiu = modelo;
    disp('✅ Model detectat: "modelo"');
elseif exist('trainedModel', 'var')
    model_actiu = trainedModel;
    disp('✅ Model detectat: "trainedModel"');
else
    error('❌ ERROR: No trobo "modelo" ni "trainedModel" al Workspace.');
end

% 3. SELECCIONAR IMATGE
[arxiu, ruta] = uigetfile({'*.jpg;*.png;*.bmp', 'Imatges'}, 'Selecciona una imatge per test');
if isequal(arxiu, 0), return; end

im_original = imread(fullfile(ruta, arxiu));
im = im2uint8(im_original);

% =========================================================================
%                       PROCESSAMENT (PIPELINE)
% =========================================================================
tic; % Inici cronòmetre

% --- PAS A: SEGMENTACIÓ ROBUSTA (UNIVERSAL) ---
[im_masked, bw] = segmentar_universal(im);

% Validació
if ~any(bw(:))
    resultat_text = 'NO-SENYAL (No detectat)';
    score_max = 0;
    color_titol = 'red';
else
    % --- PAS B: EXTRACCIÓ DE CARACTERÍSTIQUES ---
    
    % 1. FORMA
    stats = regionprops(bw, 'Area', 'Perimeter', 'Solidity', 'BoundingBox','Extent','Eccentricity');
    [~, idx] = max([stats.Area]); 
    blob = stats(idx);
    
    compacitat = (blob.Perimeter ^ 2) / blob.Area;
    solidesa = blob.Solidity;

    extent = blob.Extent;
    ecc = blob.Eccentricity;
    
    % 2. COLOR (HOC)
    rect = blob.BoundingBox;
    im_crop = imcrop(im_masked, rect);
    
    im_hsv_crop = rgb2hsv(im_crop);
    H_c = im_hsv_crop(:,:,1);
    S_c = im_hsv_crop(:,:,2);
    
    mask_valid = max(im_crop, [], 3) > 0;
    total_pixels = sum(mask_valid(:)) + 1;
    
    % Llindars HSV (els mateixos de l'entrenament)
    p_red = sum(mask_valid(:) & ((H_c(:)>0.85)|(H_c(:)<0.15)) & (S_c(:)>0.25));
    p_blue = sum(mask_valid(:) & (H_c(:)>0.50)&(H_c(:)<0.75) & (S_c(:)>0.25));
    p_yellow = sum(mask_valid(:) & (H_c(:)>0.12)&(H_c(:)<0.20) & (S_c(:)>0.25));
    
    pct_red = p_red / total_pixels;
    pct_blue = p_blue / total_pixels;
    pct_yellow = p_yellow / total_pixels;
    
    % 3. TEXTURA (HOG)
    im_resized = imresize(im_crop, [64, 64]);
    hog_vector = extractHOGFeatures(im_resized, 'CellSize', [8 8]);
    
    % VECTOR BRUT FINAL (1769 característiques)
    w_shape = 6;
    w_color = 2;
    w_hog   = 0.3;

    VectorBrut = [
        w_shape * [compacitat, solidesa, extent, ecc], ...
        w_color * [pct_red, pct_blue, pct_yellow], ...
        w_hog   * hog_vector
    ];
    % --- PAS C: NORMALITZACIÓ I PREDICCIÓ ---
    
    % 1. Normalitzar (Z-Score)
    VectorNorm = (VectorBrut - mu) ./ sigma;
    
    % 2. Convertir a Taula i ARREGLAR NOMS (El teu error estava aquí)
    T_testAux = array2table(VectorNorm);

    X_shape_color = T_testAux(:,1:8);
    X_hog = T_testAux{:,9:end};

    X_hog_pca = X_hog * coeff_pca(:,1:150);

    X_hog_pca = array2table(X_hog_pca);

    T_test = [X_shape_color, X_hog_pca];
    
    % Generem els noms EXACTES que espera el model
    % (Les 5 primeres manuals, la resta "FeaturesX")
    noms_columnes = cell(1, 158);
    noms_columnes{1} = 'Compacitat';
    noms_columnes{2} = 'Solidesa';
    noms_columnes{3} = 'Extent';
    noms_columnes{4} = 'Excentricidad';
    noms_columnes{5} = 'Pct_Red';
    noms_columnes{6} = 'Pct_Blue';
    noms_columnes{7} = 'Pct_Yellow';
    for k = 8:158
        noms_columnes{k} = ['Features_final' num2str(k)];
    end
    
    % Assignem els noms a la taula
    T_test.Properties.VariableNames = noms_columnes;
    
    % 3. Predicció
    try
        if isstruct(model_actiu) && isfield(model_actiu, 'predictFcn')
            % Cas 'trainedModel' (Exportat de l'App)
            [pred, scores] = model_actiu.predictFcn(T_test);
        else
            % Cas 'modelo' (Objecte SVM directe)
            % Comprovem si el model vol noms diferents per seguretat
            if isa(model_actiu, 'ClassificationSVM') && ~isempty(model_actiu.PredictorNames)
                 % Si el model té noms guardats, els fem servir prioritàriament
                 if length(model_actiu.PredictorNames) == width(T_test)
                     T_test.Properties.VariableNames = model_actiu.PredictorNames;
                 end
            end
            [pred, scores] = predict(model_actiu, T_test);
        end
        
        resultat_text = char(pred);
        score_max = max(scores);
        color_titol = 'blue';
        
    catch ME
        resultat_text = 'ERROR PREDICCIÓ';
        score_max = 0;
        color_titol = 'red';
        disp(['❌ ERROR: ' ME.message]);
    end
end
temps = toc;

% =========================================================================
%                       VISUALITZACIÓ
% =========================================================================
figure('Color','w', 'Name', 'Resultat Test Individual', 'Position', [100 100 900 400]);

% Imatge Original
subplot(1,2,1); 
imshow(im_original); 
title('Imatge Original', 'FontSize', 12);

% Resultat amb Màscara
subplot(1,2,2); 
imshow(im_masked); 
title({['PREDICCIÓ: ' resultat_text], ...
       ['Confiança: ' num2str(score_max, '%.4f')]}, ...
       'Color', color_titol, 'FontSize', 16, 'Interpreter', 'none', 'FontWeight', 'bold');

disp('=========================================');
disp(['📂 Fitxer:   ', arxiu]);
disp(['🤖 Resultat: ', resultat_text]);
disp(['📊 Score:    ', num2str(score_max, '%.4f')]);
disp(['⏱️ Temps:    ', num2str(temps, '%.3f'), ' s']);
disp('=========================================');




function im_masked = segmentar_universal(im)
    % 1. Convertim a HSV
    im_hsv = rgb2hsv(im);
    H = im_hsv(:,:,1);
    S = im_hsv(:,:,2);
    V = im_hsv(:,:,3);
    
    % Vermell tolerant
    mask_red = ((H > 0.90) | (H < 0.12)) & (S > 0.15) & (V > 0.20);
    % Blau tolerant
    mask_blue = (H > 0.58) & (H < 0.77) & (S > 0.25) & (V > 0.15);
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