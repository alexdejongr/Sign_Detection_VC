ruta_dataset = './'; 
imds = imageDatastore(ruta_dataset, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
% Inicialitzem variables
numImages = numel(imds.Files);
Features = zeros(numImages, 1766);
Labels = cell(numImages, 1);

disp(['S''han trobat ', num2str(numImages), ' imatges. Processant...']);
reset(imds); 

for i = 1:numImages
    %  Llegim imatge i info 
    [im, info] = read(imds);
    im = im2uint8(im);
    
    % Guardem l'etiqueta 
    Labels{i} = info.Label;
    
    % Segmentació
    im_masked = segmentar(im);
    
    % Extracció característiques
    im_gray = rgb2gray(im_masked);
    bw = imbinarize(im_gray, 0.01); 
    
    % Si la segmentació falla (tot negre), posem zeros per no trencar el codi
    if ~any(bw(:))
        Features(i, :) = zeros(1, 1766);
        disp('Avís: Imatge buida o no detectada.');
        continue; 
    end
    
    % RegionProps
    stats = regionprops(bw, 'Area', 'Perimeter', 'Solidity', 'BoundingBox');
    [~, idx] = max([stats.Area]); % Objecte més gran
    blob = stats(idx);
    
    compacitat = (blob.Perimeter ^ 2) / blob.Area;
    solidesa = blob.Solidity;
    
    % HOG sobre el retall
    rect = blob.BoundingBox;
    im_crop = imcrop(im_masked, rect);
    im_resized = imresize(im_crop, [64, 64]);
    
    
    hog_vector = extractHOGFeatures(im_resized, 'CellSize', [8 8]);
    
    % Guardem a la matriu general
    Features(i, :) = [compacitat, solidesa, hog_vector];
end

TaulaFinal = array2table(Features);
% Renombrem només les dues primeres variables que sabem quines són
TaulaFinal.Properties.VariableNames{1} = 'Compacitat';
TaulaFinal.Properties.VariableNames{2} = 'Solidesa';
TaulaFinal.Clase = string(Labels);
disp('Procés finalitzat.');
disp(['Total mostres: ', num2str(height(TaulaFinal))]);

function im_masked = segmentar(im)
    
    im_hsv = rgb2hsv(im);
    H = im_hsv(:,:,1);
    S = im_hsv(:,:,2);
    V = im_hsv(:,:,3);
    
    
    mask_red = ((H > 0.92) | (H < 0.08)) & (S > 0.3) & (V > 0.2);
    
    mask_blue = (H > 0.58) & (H < 0.75) & (S > 0.6) & (V > 0.25);
    
    mask_yellow = (H > 0.12) & (H < 0.28) & (S > 0.25) & (V > 0.3);
    % unio mascares
    
    bw_raw = mask_red | mask_blue | mask_yellow;
    
    % morfologia neteja
    bw_filled = imfill(bw_raw, 'holes');
    
    ee = strel('disk', 2);
    bw_open = imopen(bw_filled, ee);
    
    se_gran = strel('disk', 4);
    bw_final = imclose(bw_open, se_gran);
    % Neteja inicial de soroll molt petit 
    bw_final = bwareafilt(bw_final, [300, 999999]); 
    
    
    %analitzem propietats geometrices
    stats = regionprops(bw_final, 'BoundingBox', 'Area', 'PixelIdxList');
    
    % Creem una imatge negra buida per anar posant els "bons" candidats
    bw_filtrada = false(size(bw_final));
    
    for k = 1:length(stats)
        caixa = stats(k).BoundingBox;  
        width = caixa(3);
        height = caixa(4);
        
        aspect_ratio = width / height;
        
        % criteri forma
        es_proporcionat = (aspect_ratio > 0.5) && (aspect_ratio < 1.8);
        
       
        
        if es_proporcionat
            % Si compleix la forma, l'afegim a la mascara bona
            bw_filtrada(stats(k).PixelIdxList) = true;
        end
    end
    
    % entre els que tenen forma de senyal, agafem el mes gran
    
    if any(bw_filtrada(:))
        bw_final = bwareafilt(bw_filtrada, 1);
    else
        % per no retornar una imatge negra, tornem el pla gran
        bw_final = bwareafilt(bw_final, 1);
    end
    
    %nomes ensenya el que hem seleccionat com a mascara i tot lu altre negre
    im_masked = im;
    R = im(:,:,1); R(~bw_final) = 0;
    G = im(:,:,2); G(~bw_final) = 0;
    B = im(:,:,3); B(~bw_final) = 0;
    im_masked = cat(3, R, G, B);
    
end

% netejar la taula final borrant totes les imatges que no ha pogut
% segmentar

filas_cero = all(Features == 0, 2);

total_ceros = sum(filas_cero);

disp(['hi ha ', num2str(total_ceros), 'files 0']);

idx_brossa = all(Features == 0, 2);

Features(idx_brossa, :) = [];

Labels(idx_brossa) = [];



filas_cero = all(Features == 0, 2);

total_ceros = sum(filas_cero);

disp(['hi ha ', num2str(total_ceros), 'files 0']);

TaulaFinal = array2table(Features);
TaulaFinal.Properties.VariableNames{1} = 'Compacitat';
TaulaFinal.Properties.VariableNames{2} = 'Solidesa';
TaulaFinal.Clase = string(Labels);

