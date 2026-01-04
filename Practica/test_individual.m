clearvars -except modelo trainedModel Features TaulaFinal; clc; close all;

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
    stats = regionprops(bw, 'Area', 'Perimeter', 'Solidity', 'BoundingBox');
    [~, idx] = max([stats.Area]); 
    blob = stats(idx);
    
    compacitat = (blob.Perimeter ^ 2) / blob.Area;
    solidesa = blob.Solidity;
    
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
    VectorBrut = [compacitat, solidesa, pct_red, pct_blue, pct_yellow, hog_vector];
    
    % --- PAS C: NORMALITZACIÓ I PREDICCIÓ ---
    
    % 1. Normalitzar (Z-Score)
    VectorNorm = (VectorBrut - mu) ./ sigma;
    
    % 2. Convertir a Taula i ARREGLAR NOMS (El teu error estava aquí)
    T_test = array2table(VectorNorm);
    
    % Generem els noms EXACTES que espera el model
    % (Les 5 primeres manuals, la resta "FeaturesX")
    noms_columnes = cell(1, 1769);
    noms_columnes{1} = 'Compacitat';
    noms_columnes{2} = 'Solidesa';
    noms_columnes{3} = 'Pct_Red';
    noms_columnes{4} = 'Pct_Blue';
    noms_columnes{5} = 'Pct_Yellow';
    for k = 6:1769
        noms_columnes{k} = ['Features' num2str(k)];
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


% =========================================================================
%      FUNCIÓ DE SEGMENTACIÓ MILLORADA (UNIVERSAL)
% =========================================================================
function [im_masked, bw_final] = segmentar_universal(im)
    % 1. PRE-PROCESSAMENT FÍSIC (Matemàtica RGB)
    im_double = im2double(im);
    R = im_double(:,:,1);
    G = im_double(:,:,2);
    B = im_double(:,:,3);
    
    % A) Contrast BLAU (Obligatori) -> Elimina el cel (té vermell)
    contrast_blau = max(0, B - R); 
    
    % B) Contrast VERMELL PUR (Prohibició) -> Elimina núvols (tenen verd)
    contrast_vermell_pur = max(0, R - G); 
    
    % C) Contrast GROC (Obres) -> Elimina cel blau
    contrast_groc = max(0, (R + G) - B);
    
    % 2. SEGMENTACIÓ HSV
    im_hsv = rgb2hsv(im);
    H = im_hsv(:,:,1);
    S = im_hsv(:,:,2);
    
    % --- MÀSCARES COMBINADES ---
    
    % -> VERMELL: Exigim R > G per matar núvols
    mask_red_hsv = ((H > 0.90) | (H < 0.12)) & (S > 0.15);
    mask_red = mask_red_hsv & (contrast_vermell_pur > 0.05);
    
    % -> GROC: Estàndard amb filtre càlid
    mask_yellow_hsv = (H > 0.11) & (H < 0.18) & (S > 0.20);
    mask_yellow = mask_yellow_hsv & (contrast_groc > 0.10);
    
    % -> BLAU: Saturació ALTA (>0.35) per matar cel clar
    mask_blue_hsv = (H > 0.55) & (H < 0.75) & (S > 0.35);
    mask_blue = mask_blue_hsv & (contrast_blau > 0.10);
    
    % 3. UNIÓ I MORFOLOGIA
    bw_raw = mask_red | mask_yellow | mask_blue;
    
    se_connect = strel('disk', 2); 
    bw = imdilate(bw_raw, se_connect);
    bw = imfill(bw, 'holes');
    bw = imerode(bw, se_connect);
    
    bw = imopen(bw, strel('disk', 3));
    
    % Filtre de mida
    bw_final_cand = bwareafilt(bw, [150, 999999]);
    
    % 4. SELECCIÓ ROBUSTA
    stats = regionprops(bw_final_cand, 'BoundingBox', 'Area', 'PixelIdxList', 'Solidity', 'Extent');
    bw_good = false(size(bw_final_cand));
    found = false;
    
    for k = 1:length(stats)
        aspect = stats(k).BoundingBox(3) / stats(k).BoundingBox(4);
        % Aspect ratio raonable i objecte sòlid (no branques)
        if (aspect > 0.45 && aspect < 2.2) && (stats(k).Solidity > 0.5)
            bw_good(stats(k).PixelIdxList) = true;
            found = true;
        end
    end
    
    % Lògica de Rescat
    if found
        bw_final = bwareafilt(bw_good, 1);
    elseif any(bw_final_cand(:))
        % Si la forma és dolenta però hi ha color correcte, ho salvem
        bw_final = bwareafilt(bw_final_cand, 1);
    else
        bw_final = bw_final_cand; 
    end
    
    % 5. RESULTAT FINAL
    im_masked = im;
    R_out = im(:,:,1); R_out(~bw_final) = 0;
    G_out = im(:,:,2); G_out(~bw_final) = 0;
    B_out = im(:,:,3); B_out(~bw_final) = 0;
    im_masked = cat(3, R_out, G_out, B_out);
end