im = imread("vianant\035_1_0006.png"); % Assegura't que la ruta és correcta
figure, imshow(im); title('Original');

im_hsv = rgb2hsv(im);
H = im_hsv(:,:,1);
S = im_hsv(:,:,2);
V = im_hsv(:,:,3);

% --- CANVI 1: LLINDARS MÉS TOLERANTS ---
% Ampliem el rang de Hue (0.92->0.90 i 0.08->0.12)
% Baixem drasticament la Saturació (0.3->0.15) i el Valor (0.2->0.15)
mask_red = ((H > 0.90) | (H < 0.12)) & (S > 0.15) & (V > 0.15);

% També baixem una mica els blaus per si de cas
mask_blue = (H > 0.55) & (H < 0.77) & (S > 0.25) & (V > 0.15);

mask_yellow = (H > 0.11) & (H < 0.19) & (S > 0.25) & (V > 0.25);
% unio mascares
bw_raw = mask_red | mask_blue | mask_yellow;

% --- CANVI 2: ESTRATÈGIA DE CONNEXIÓ ---
% El problema és que amb poca llum el cercle es trenca.
% NO facis imopen primer (això esborra els trossets). 
% Fes imdilate primer per "enganxar" els trossos.

se_connect = strel('disk', 3); 
bw_connected = imdilate(bw_raw, se_connect);

% Ara omplim forats
bw_filled = imfill(bw_connected, 'holes');
bw_shrunk = imerode(bw_filled, se_connect);

% Ara sí, netegem el soroll exterior (Erosió suau)
se_clean = strel('disk', 2);
bw_clean = imopen(bw_shrunk, se_clean);

% Neteja per mida (Baixem el mínim a 150 per seguretat)
bw_final = bwareafilt(bw_clean, [150, 999999]); 

% --- CANVI 3: FILTRE DE FORMA MÉS RELAXAT ---
stats = regionprops(bw_final, 'BoundingBox', 'Area', 'PixelIdxList', 'Extent');
bw_filtrada = false(size(bw_final));

found = false;
for k = 1:length(stats)
    caixa = stats(k).BoundingBox;  
    width = caixa(3);
    height = caixa(4);
    aspect_ratio = width / height;
    
    % Ampliem el rang d'aspect ratio (per senyals vistos de costat)
    es_proporcionat = (aspect_ratio > 0.4) && (aspect_ratio < 2.2);
    
    % Extent: Un senyal sòlid (cercle/quadrat) ocupa bastant de la seva caixa
    te_consistencia = stats(k).Extent > 0.25; 

    if es_proporcionat && te_consistencia
        bw_filtrada(stats(k).PixelIdxList) = true;
        found = true;
    end
end

% "Fail-safe": Si després de filtrar no queda res, però teníem detecció inicial,
% agafem la taca més gran (millor això que una imatge negra)
if ~found && any(bw_final(:))
    disp('AVÍS: Forma dubtosa, però recuperant el blob més gran per no perdre la imatge.');
    bw_final = bwareafilt(bw_final, 1);
elseif found
    bw_final = bw_filtrada;
else
    % Si realment no hi havia res des del principi
    bw_final = bwareafilt(bw_final, 1); 
end

% Visualització final
im_masked = im;
R = im(:,:,1); R(~bw_final) = 0;
G = im(:,:,2); G(~bw_final) = 0;
B = im(:,:,3); B(~bw_final) = 0;
im_masked = cat(3, R, G, B);

figure, imshow(im_masked); title('Resultat Millorat');