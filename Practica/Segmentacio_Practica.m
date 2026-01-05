
% --- SELECTOR D'IMATGE ---
[arxiu, ruta] = uigetfile({'*.jpg;*.png;*.bmp', 'Imatges'}, 'Selecciona una imatge per test');
if isequal(arxiu, 0), return; end

im = imread(fullfile(ruta, arxiu));

im_masked = segmentar(im);
figure;
montage({im, im_masked});
title('Original vs Procesada');

function im_masked = segmentar(im)
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