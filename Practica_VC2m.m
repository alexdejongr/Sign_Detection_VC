clear; clc; close all;

ruta_dataset = './'; 
imds = imageDatastore(ruta_dataset, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Inicialitzem variables
numImages = numel(imds.Files);

% --- TOTAL FEATURES: 1769 ---
% 2 (Forma) + 3 (Color HOC) + 1764 (HOG)
Features = zeros(numImages, 1769);
Labels = cell(numImages, 1);

% Obrim el fitxer de text per escriure els errors
nom_fitxer_errors = 'errores_identificacion.txt';
fid = fopen(nom_fitxer_errors, 'w');
fprintf(fid, 'Carpeta\tNom_Imatge\n'); 
fprintf(fid, '-----------------------------------\n');

disp(['S''han trobat ', num2str(numImages), ' imatges. Processant...']);
reset(imds); 

for i = 1:numImages
    % Llegim imatge i info 
    [im, info] = read(imds);
    im = im2uint8(im);
    
    % Guardem l'etiqueta 
    Labels{i} = info.Label;
    
    % --- 1. SEGMENTACIÓ ROBUSTA (Nova Funció Sandwich) ---
    im_masked = segmentar(im);
    
    % Comprovar si està buida
    im_gray = rgb2gray(im_masked);
    bw = imbinarize(im_gray, 0.01); 
    
    % Si la segmentació falla (tot negre)
    if ~any(bw(:))
        [~, nom_arxiu, ext] = fileparts(info.Filename);
        nom_complet = [nom_arxiu, ext];
        nom_carpeta = char(info.Label);
        
        fprintf(fid, '%s\t%s\n', nom_carpeta, nom_complet);
        % disp(['Avís: Error a ', nom_carpeta, '/', nom_complet]);
        
        % Posem zeros
        Features(i, :) = zeros(1, 1769);
        continue; 
    end
    
    % --- 2. EXTRACTOR DE FORMA (COMPACITAT/SOLIDESA) ---
    stats = regionprops(bw, 'Area', 'Perimeter', 'Solidity', 'BoundingBox');
    [~, idx] = max([stats.Area]); % Objecte més gran
    blob = stats(idx);
    
    compacitat = (blob.Perimeter ^ 2) / blob.Area;
    solidesa = blob.Solidity;
    
    % Retall per a Color i HOG
    rect = blob.BoundingBox;
    im_crop = imcrop(im_masked, rect);
    
    % --- 3. EXTRACTOR DE COLOR (HOC) ---
    im_hsv_crop = rgb2hsv(im_crop);
    H_c = im_hsv_crop(:,:,1);
    S_c = im_hsv_crop(:,:,2);
    
    mask_valid = max(im_crop, [], 3) > 0; % Ignorem el fons negre
    total_pixels = sum(mask_valid(:)) + 1;
    
    p_red = sum(mask_valid(:) & ((H_c(:)>0.85)|(H_c(:)<0.15)) & (S_c(:)>0.25));
    p_blue = sum(mask_valid(:) & (H_c(:)>0.50)&(H_c(:)<0.75) & (S_c(:)>0.25));
    p_yellow = sum(mask_valid(:) & (H_c(:)>0.12)&(H_c(:)<0.20) & (S_c(:)>0.25));
    
    pct_red = p_red / total_pixels;
    pct_blue = p_blue / total_pixels;
    pct_yellow = p_yellow / total_pixels;
    
    % --- 4. EXTRACTOR DE TEXTURA (HOG) ---
    im_resized = imresize(im_crop, [64, 64]);
    hog_vector = extractHOGFeatures(im_resized, 'CellSize', [8 8]);
    
    % Guardem tot junt (1769 variables)
    Features(i, :) = [compacitat, solidesa, pct_red, pct_blue, pct_yellow, hog_vector];
end

% Tanquem fitxer errors
fclose(fid);

% Neteja final de la taula
filas_cero = all(Features == 0, 2);
total_ceros = sum(filas_cero);
disp(['S''han eliminat ', num2str(total_ceros), ' imatges que han fallat.']);

idx_brossa = all(Features == 0, 2);
Features(idx_brossa, :) = [];
Labels(idx_brossa) = [];

% Creació de la Taula Final
TaulaFinal = array2table(Features);

% Noms de columnes clau
TaulaFinal.Properties.VariableNames{1} = 'Compacitat';
TaulaFinal.Properties.VariableNames{2} = 'Solidesa';
TaulaFinal.Properties.VariableNames{3} = 'Pct_Red';
TaulaFinal.Properties.VariableNames{4} = 'Pct_Blue';
TaulaFinal.Properties.VariableNames{5} = 'Pct_Yellow';

TaulaFinal.Clase = string(Labels);

% Normalització
TaulaFinal{:, 1:end-1} = normalize(TaulaFinal{:, 1:end-1});

disp('Procés finalitzat. TaulaFinal creada.');
% Usem size(..., 1) per evitar conflictes amb la variable 'height'
disp(['Total mostres vàlides: ', num2str(size(TaulaFinal, 1))]);


function [im_masked, bw_final] = segmentar(im)
    % 1. PRE-PROCESSAMENT FÍSIC (RGB MATH)
    im_double = im2double(im);
    R = im_double(:,:,1);
    G = im_double(:,:,2);
    B = im_double(:,:,3);
    
    % A) Contrast BLAU (Obligatori)
    % El senyal blau és profund. El cel blau té blanc (vermell).
    contrast_blau = max(0, B - R); 
    
    % B) Contrast VERMELL PUR (Prohibició/Perill)
    % CLAU PER AL CEL: Els núvols i el cel brillant tenen R i G alts.
    % El senyal vermell té R alt i G baix. Restant G eliminem núvols.
    contrast_vermell_pur = max(0, R - G); 
    
    % C) Contrast GROC (Obres)
    % El groc és la suma de R + G.
    contrast_groc = max(0, (R + G) - B);
    
    % 2. SEGMENTACIÓ HSV
    im_hsv = rgb2hsv(im);
    H = im_hsv(:,:,1);
    S = im_hsv(:,:,2);
    V = im_hsv(:,:,3);
    
    % --- MÀSCARES COMBINADES ---
    
    % -> VERMELL: Exigim que R sigui superior a G per evitar cels brillants
    mask_red_hsv = ((H > 0.90) | (H < 0.12)) & (S > 0.15);
    mask_red = mask_red_hsv & (contrast_vermell_pur > 0.05);
    
    % -> GROC: Usem el filtre càlid estàndard
    mask_yellow_hsv = (H > 0.11) & (H < 0.18) & (S > 0.20);
    mask_yellow = mask_yellow_hsv & (contrast_groc > 0.10);
    
    % -> BLAU: Pugem la saturació mínima a 0.35 per matar el cel "rentat"
    mask_blue_hsv = (H > 0.55) & (H < 0.75) & (S > 0.35);
    mask_blue = mask_blue_hsv & (contrast_blau > 0.10);
    
    % 3. UNIÓ I MORFOLOGIA
    bw_raw = mask_red | mask_yellow | mask_blue;
    
    % Neteja acurada (Sandwich suau per no enganxar arbres)
    se_connect = strel('disk', 2); 
    bw = imdilate(bw_raw, se_connect);
    bw = imfill(bw, 'holes');
    bw = imerode(bw, se_connect);
    
    % Obertura per treure soroll de branques fines
    bw = imopen(bw, strel('disk', 3));
    
    % Filtre de mida mínima
    bw_final_cand = bwareafilt(bw, [150, 999999]);
    
    % 4. SELECCIÓ DEL MILLOR CANDIDAT (ESTRATÈGIA ROBUSTA)
    stats = regionprops(bw_final_cand, 'BoundingBox', 'Area', 'PixelIdxList', 'Solidity', 'Extent');
    bw_good = false(size(bw_final_cand));
    found = false;
    
    for k = 1:length(stats)
        aspect = stats(k).BoundingBox(3) / stats(k).BoundingBox(4);
        
        % Criteris de forma:
        % 1. Proporció: Ni molt pla ni molt alt (0.45 - 2.2)
        % 2. Solidesa: El senyal ha de ser "massís" (> 0.5), no una branca dispersa
        if (aspect > 0.45 && aspect < 2.2) && (stats(k).Solidity > 0.5)
            bw_good(stats(k).PixelIdxList) = true;
            found = true;
        end
    end
    
    % LÒGICA DE RESCAT (FAIL-SAFE)
    if found
        % Si tenim formes bones, agafem la més gran
        bw_final = bwareafilt(bw_good, 1);
    elseif any(bw_final_cand(:))
        % Si el filtre de forma ho ha matat tot (ex: senyal parcialment tapat),
        % però teníem color correcte, recuperem la taca de color més gran.
        % És millor passar-li al model una taca deformada que una imatge negra.
        bw_final = bwareafilt(bw_final_cand, 1);
    else
        % No s'ha trobat res
        bw_final = bw_final_cand; 
    end
    
    % 5. MASCARAR LA IMATGE ORIGINAL
    im_masked = im;
    R_out = im(:,:,1); R_out(~bw_final) = 0;
    G_out = im(:,:,2); G_out(~bw_final) = 0;
    B_out = im(:,:,3); B_out(~bw_final) = 0;
    im_masked = cat(3, R_out, G_out, B_out);
end