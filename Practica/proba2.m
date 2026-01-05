clear; clc; close all;

% carregar dataset
ruta_dataset = './'; 
imds = imageDatastore(ruta_dataset, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

numImages = numel(imds.Files);

% 4 Forma + 3 Color + 1764 HOG
Features = zeros(numImages, 1771);
Labels = cell(numImages, 1);

% guardar imatges on falla la segmentació
nom_fitxer_errors = 'errores_identificacion.txt';
fid = fopen(nom_fitxer_errors, 'w');
fprintf(fid, 'Carpeta\tNom_Imatge\n'); 

disp(['S''han trobat ', num2str(numImages), ' imatges. Processant...']);
reset(imds); 

for i = 1:numImages
    % Llegir imatge i etiqueta
    [im, info] = read(imds);
    im = im2uint8(im);
    
    Labels{i} = info.Label;
    
    % Segmentació
    im_masked = segmentar(im);
    
    im_gray = rgb2gray(im_masked);
    bw = imbinarize(im_gray, 0.01); 
    
    if ~any(bw(:))
        [~, nom_arxiu, ext] = fileparts(info.Filename);
        nom_complet = [nom_arxiu, ext];
        nom_carpeta = char(info.Label);
        
        fprintf(fid, '%s\t%s\n', nom_carpeta, nom_complet);
        Features(i, :) = zeros(1, 1771);
        continue; 
    end
    
    % seleccionar el blob més gran
    stats = regionprops(bw, 'Area', 'Perimeter', 'Solidity', 'BoundingBox','Extent','Eccentricity');
    [~, idx] = max([stats.Area]);
    blob = stats(idx);
    
    compacitat = (blob.Perimeter ^ 2) / max(blob.Area, eps);
    solidesa = blob.Solidity;

    extent = blob.Extent;
    ecc = blob.Eccentricity;
    
    rect = blob.BoundingBox;
    im_crop = imcrop(im_masked, rect);
    
    % Color només sobre píxels no negres
    im_hsv_crop = rgb2hsv(im_crop);
    H_c = im_hsv_crop(:,:,1);
    S_c = im_hsv_crop(:,:,2);
    
    mask_valid = max(im_crop, [], 3) > 0;
    np = sum(mask_valid(:));
    if np == 0
        total_pixels = 1;
    else
        total_pixels = np;
    end
    
    p_red = sum(mask_valid(:) & ((H_c(:)>0.85)|(H_c(:)<0.15)) & (S_c(:)>0.25));
    p_blue = sum(mask_valid(:) & (H_c(:)>0.50)&(H_c(:)<0.75) & (S_c(:)>0.25));
    p_yellow = sum(mask_valid(:) & (H_c(:)>0.12)&(H_c(:)<0.20) & (S_c(:)>0.25));
    
    pct_red = p_red / total_pixels;
    pct_blue = p_blue / total_pixels;
    pct_yellow = p_yellow / total_pixels;
    
    im_resized = imresize(im_crop, [64, 64]);
    hog_vector = extractHOGFeatures(im_resized, 'CellSize', [8 8]);
    
    % Construir vector de features
    w_shape = 6;
    w_color = 2;
    w_hog   = 0.3;

    Features(i,:) = [
        w_shape * [compacitat, solidesa, extent, ecc], ...
        w_color * [pct_red, pct_blue, pct_yellow], ...
        w_hog   * hog_vector
    ];
end

filas_cero = all(Features == 0, 2);

total_ceros = sum(filas_cero);
disp(['S''han eliminat ', num2str(total_ceros), ' imatges que han fallat.']);

idx_brossa = all(Features == 0, 2);
Features(idx_brossa, :) = [];
Labels(idx_brossa) = [];

% Prova: PCA
X_shape_color = Features(:, 1:7);
X_hog = Features(:, 8:end);

k = 150;
[coeff_pca, score, ~, ~, ~, mu_pca] = pca(X_hog);
X_hog_pca = score(:, 1:k);
Features_final = [X_shape_color, X_hog_pca];

% Tanquem fitxer errors
fclose(fid);

% pca només a la part hog (guardar mu_pca per al test)
mu = mean(Features_final, 1);
sigma = std(Features_final, 0, 1);
sigma(sigma == 0) = 1;
Features_Norm = (Features_final - mu) ./ sigma;
TaulaFinal = array2table(Features_Norm);

noms_manuals = {'Compacitat', 'Solidesa', 'Extent', 'Excentricidad', ...
                'Pct_Red', 'Pct_Blue', 'Pct_Yellow'};
noms_pca = cell(1, k);
for j = 1:k
    noms_pca{j} = ['PCA_' num2str(j)];
end
TaulaFinal.Properties.VariableNames = [noms_manuals, noms_pca];

TaulaFinal.Clase = string(Labels);

% guarda la taula amb predictors i etiqueta per experiments
save('TaulaFinal.mat','TaulaFinal');

% desar paràmetres per reproduir el pipeline a test
save('Dades_Model.mat', 'mu', 'sigma', 'coeff_pca', 'mu_pca', 'k');

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