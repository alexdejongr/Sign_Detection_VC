im = imread("limit/002_0001.png");
figure,imshow(im)

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
figure,imshow(bw_filled)
ee = strel('disk', 2);
bw_open = imopen(bw_filled, ee);

se_gran = strel('disk', 4);
bw_final = imclose(bw_open, se_gran);
% Neteja inicial de soroll molt petit 
bw_final = bwareafilt(bw_final, [300, 999999]); 

figure,imshow(bw_final)
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

figure, imshow(im_masked);
