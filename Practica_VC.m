ruta_dataset = 'C:\Users\alexd\Desktop\VC'; 

imds = imageDatastore(ruta_dataset, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Inicialitzem variables
Features = [];
Labels = {}; 

disp(['S''han trobat ', num2str(length(imds.Files)), ' imatges. Processant...']);

reset(imds); 
while hasdata(imds)
    %  Llegim imatge i info 
    [im, info] = read(imds);
    
    % Guardem l'etiqueta 
    Labels{end+1, 1} = info.Label;
    
    % Segmentació
    im_masked = segmentar(im);
    
    % Extracció característiques
    im_gray = rgb2gray(im_masked);
    bw = imbinarize(im_gray, 0.01); 
    
    % Si la segmentació falla (tot negre), posem zeros per no trencar el codi
    if ~any(bw(:))
       
        dummy_hog = zeros(1, 1764); 
        Features = [Features; 0, 0, dummy_hog];
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
    Features = [Features; compacitat, solidesa, hog_vector];
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
    mask_yellow = (H > 0.11) & (H < 0.19) & (S > 0.4) & (V > 0.2);
    
    bw_raw = mask_red | mask_blue | mask_yellow;
    
    % Morfologia
    bw_filled = imfill(bw_raw, 'holes');
    ee = strel('disk', 2);
    bw_open = imopen(bw_filled, ee);
    se_gran = strel('disk', 4);
    bw_final = imclose(bw_open, se_gran);
    bw_final = bwareafilt(bw_final, 1); 
    
    % Màscara final (Posem a negre el fons)
    im_masked = im;
    R = im(:,:,1); R(~bw_final) = 0;
    G = im(:,:,2); G(~bw_final) = 0;
    B = im(:,:,3); B(~bw_final) = 0;
    im_masked = cat(3, R, G, B);
    
end
