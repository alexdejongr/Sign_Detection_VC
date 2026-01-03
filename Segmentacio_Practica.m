

% --- 1. CARREGAR IMATGE ---
[arxiu, ruta] = uigetfile({'*.*'}, 'Selecciona la imatge del senyal 80');
if isequal(arxiu,0), return; end
im = imread(fullfile(ruta, arxiu));

figure('Name', 'Analisi Final', 'Color', 'w');
subplot(2,3,1); imshow(im); title('Imatge Original');

% --- 2. FILTRES FÍSICS (MATH) ---
im_double = im2double(im); 
R = im_double(:,:,1);
G = im_double(:,:,2);
B = im_double(:,:,3);

% A) FILTRE BLAU ESTRICTE (Obligatori)
% El cel sol tenir saturació baixa. El senyal té saturació alta.
contrast_blau = max(0, B - R); 

% B) NOUS FILTRES VERMELLS (Anti-Núvols)
% En lloc de sumar G, el restem. Això mata el blanc i el gris del cel.
% Senyal Vermell = R alt, G baix.
contrast_vermell_pur = max(0, R - G); 

% C) FILTRE GROC (Per si de cas)
% El groc sí que necessita R i G alts.
contrast_groc = max(0, (R + G) - B); 

subplot(2,3,4); imshow(contrast_blau, [0 0.5]); title('Contrast Blau (B-R)');
subplot(2,3,5); imshow(contrast_vermell_pur, [0 0.3]); title('Contrast Vermell (R-G)');

% --- 3. SEGMENTACIÓ HSV ESTRICTA ---
im_hsv = rgb2hsv(im);
H = im_hsv(:,:,1);
S = im_hsv(:,:,2);
V = im_hsv(:,:,3);

% -> MÀSCARA VERMELLA (R - G és la clau aquí)
% El to vermell (H) és important, però el contrast R-G mana.
mask_red_hsv = ((H > 0.90) | (H < 0.12)) & (S > 0.15); 
mask_red = mask_red_hsv & (contrast_vermell_pur > 0.05); % 0.05 és suficient si restem G

% -> MÀSCARA GROGA (Obres)
mask_yellow_hsv = (H > 0.11) & (H < 0.18) & (S > 0.20);
mask_yellow = mask_yellow_hsv & (contrast_groc > 0.10); 

% -> MÀSCARA BLAVA (Obligatori) - ELIMINAR CEL
% AQUI ESTÀ EL TRUC DEL CEL: 
% Si és cel, la Saturació (S) sol ser baixa (< 0.3). La pintura és > 0.3.
% Pugem la S mínima per al blau a 0.35.
mask_blue_hsv = (H > 0.55) & (H < 0.75) & (S > 0.35); 
mask_blue = mask_blue_hsv & (contrast_blau > 0.10);

% UNIÓ
bw_raw = mask_red | mask_yellow | mask_blue;

% --- 4. MORFOLOGIA (Neteja) ---
se_connect = strel('disk', 2); % Reduït a 2 per no ajuntar arbres amb el senyal
bw_connected = imdilate(bw_raw, se_connect);
bw_filled = imfill(bw_connected, 'holes');
bw_shrunk = imerode(bw_filled, se_connect);
bw_clean = imopen(bw_shrunk, strel('disk', 3)); % Neteja forta de branques

% Filtre final per mida
bw_final_cand = bwareafilt(bw_clean, [150, 999999]); 

% --- 5. VISUALITZACIÓ ---
im_masked = im;
R_out = im(:,:,1); R_out(~bw_final_cand) = 0;
G_out = im(:,:,2); G_out(~bw_final_cand) = 0;
B_out = im(:,:,3); B_out(~bw_final_cand) = 0;
im_masked = cat(3, R_out, G_out, B_out);

subplot(2,3,[2 3]); imshow(im_masked); title('RESULTAT FINAL');
subplot(2,3,6); imshow(bw_final_cand); title('Màscara');