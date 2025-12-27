ruta_dataset = './'; 
imds = imageDatastore(ruta_dataset, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Inicialitzem variables
numImages = numel(imds.Files);

%%%%%%%%%afegim a les carectistiques el HOC

Features = zeros(numImages, 1769);
Labels = cell(numImages, 1);

% Obrim el fitxer de text per escriure els errors
nom_fitxer_errors = 'errores_identificacion.txt';
fid = fopen(nom_fitxer_errors, 'w');
fprintf(fid, 'Carpeta\tNom_Imatge\n'); % Escrivim la capçalera
fprintf(fid, '-----------------------------------\n');

disp(['S''han trobat ', num2str(numImages), ' imatges. Processant...']);
reset(imds); 

for i = 1:numImages
    % Llegim imatge i info 
    [im, info] = read(imds);
    im = im2uint8(im);
    
    % Guardem l'etiqueta 
    Labels{i} = info.Label;
    
    % Segmentació
    im_masked = segmentar(im);
    
    % Extracció característiques per comprovar si està buida
    im_gray = rgb2gray(im_masked);
    bw = imbinarize(im_gray, 0.01); 
    
    % Si la segmentació falla (tot negre)
    if ~any(bw(:))
        % Guardem el nom al TXT
        [~, nom_arxiu, ext] = fileparts(info.Filename);
        nom_complet = [nom_arxiu, ext];
        nom_carpeta = char(info.Label);
        
        fprintf(fid, '%s\t%s\n', nom_carpeta, nom_complet);
        disp(['Avís: Error a ', nom_carpeta, '/', nom_complet]);
        
        % --- CANVI 2: Posem zeros amb la nova mida (1769) ---
        Features(i, :) = zeros(1, 1769);
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
    
    % --- CANVI 3: EXTRACCIÓ DE COLOR (HOC) ---
    % Analitzem el color del retall per saber si és vermell, blau o groc
    im_hsv_crop = rgb2hsv(im_crop);
    H_c = im_hsv_crop(:,:,1);
    S_c = im_hsv_crop(:,:,2);
    
    % Màscara vàlida (ignorem el fons negre del crop)
    mask_valid = max(im_crop, [], 3) > 0;
    total_pixels = sum(mask_valid(:)) + 1; % +1 per evitar divisió per 0
    
    % Comptem píxels de cada color
    p_red = sum(mask_valid(:) & ((H_c(:)>0.85)|(H_c(:)<0.15)) & (S_c(:)>0.25));
    p_blue = sum(mask_valid(:) & (H_c(:)>0.50)&(H_c(:)<0.75) & (S_c(:)>0.25));
    p_yellow = sum(mask_valid(:) & (H_c(:)>0.12)&(H_c(:)<0.20) & (S_c(:)>0.25));
    
    pct_red = p_red / total_pixels;
    pct_blue = p_blue / total_pixels;
    pct_yellow = p_yellow / total_pixels;
    
    % Redimensionem a 64x64 (Important per mantenir vector fix)
    im_resized = imresize(im_crop, [64, 64]);
    
    % Extracció HOG (Vector de 1764)
    hog_vector = extractHOGFeatures(im_resized, 'CellSize', [8 8]);
    
    % --- CANVI 4: Guardem TOT a la matriu general ---
    % [Compacitat, Solidesa, %Vermell, %Blau, %Groc, Vector HOG]
    Features(i, :) = [compacitat, solidesa, pct_red, pct_blue, pct_yellow, hog_vector];
end

% Tanquem el fitxer de text
fclose(fid);
disp(['Llistat d''errors guardat a: ', nom_fitxer_errors]);

% Neteja final de la taula
filas_cero = all(Features == 0, 2);
total_ceros = sum(filas_cero);
disp(['S''han eliminat ', num2str(total_ceros), ' imatges que han fallat.']);

idx_brossa = all(Features == 0, 2);
Features(idx_brossa, :) = [];
Labels(idx_brossa) = [];

% Creació de la Taula Final
TaulaFinal = array2table(Features);

% --- CANVI 5: Noms de les columnes actualitzats ---
TaulaFinal.Properties.VariableNames{1} = 'Compacitat';
TaulaFinal.Properties.VariableNames{2} = 'Solidesa';
TaulaFinal.Properties.VariableNames{3} = 'Pct_Red';
TaulaFinal.Properties.VariableNames{4} = 'Pct_Blue';
TaulaFinal.Properties.VariableNames{5} = 'Pct_Yellow';

TaulaFinal.Clase = string(Labels);

% Normalització (Important per al PCA)
TaulaFinal{:, 1:end-1} = normalize(TaulaFinal{:, 1:end-1});

disp('Procés finalitzat. TaulaFinal creada.');
disp(['Total mostres vàlides: ', num2str(height(TaulaFinal))]);


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
    
    % analitzem propietats geometrices
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
    
    % només ensenya el que hem seleccionat com a mascara
    im_masked = im;
    R = im(:,:,1); R(~bw_final) = 0;
    G = im(:,:,2); G(~bw_final) = 0;
    B = im(:,:,3); B(~bw_final) = 0;
    im_masked = cat(3, R, G, B);
end