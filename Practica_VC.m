
% STOP:        Forma OCTOGONAL única. Tot vermell. Text "STOP" blanc.
% Vermella

% LIMIT:       Cercle vermell. Conté números (30, 40, 60...) negres al centre.

% D_PROHIBIDA: Cercle vermell. Rectangle blanc horitzontal al mig. Molt sòlida.

% NO_GIRAR:    Cercle vermell. Fletxa negra ratllada per una barra vermella.

% NO_SOROLL:   Cercle vermell. Corneta negra ratllada. HOG complex.

% NO_APARCAR:  Cercle. Vora vermella, fons blau i una X vermella (o barra
% diagonal). 

% VIANANT:     Triangle majoritariament groc . Vora negre. Silueta personona
% negre 


% D_OBLIGAT:   Cercle blau. Fletxa blanca gran (recte, gir o bifurcada).
% ZONA_BICI:   Cercle blau. Silueta blanca de bicicleta (dues rodes circulars).
% ZONA_COTXE:  Cercle blau. Silueta blanca de cotxe vista de front (simètrica).


%funcio per segmentar
function im_masked = segmentar(im)
   
    
    im_hsv = rgb2hsv(im);
    H = im_hsv(:,:,1);
    S = im_hsv(:,:,2);
    V = im_hsv(:,:,3);
    
    
    mask_red = ((H > 0.92) | (H < 0.08)) & (S > 0.3) & (V > 0.2);
    
    mask_blue = (H > 0.58) & (H < 0.75) & (S > 0.6) & (V > 0.25);
    
    mask_yellow = (H > 0.11) & (H < 0.19) & (S > 0.4) & (V > 0.2);
    
    % unio mascares
    
    bw_raw = mask_red | mask_blue | mask_yellow;
    
    % morfologia neteja
    bw_filled = imfill(bw_raw, 'holes');
    
    ee = strel('disk', 2);
    bw_open = imopen(bw_filled, ee);
    
    se_gran = strel('disk', 4);
    bw_final = imclose(bw_open, se_gran);
    
    
    bw_final = bwareafilt(bw_final, 1); 
    
    
    im_masked = im;
    % Posem a negre tot el que no sigui la màscara
    R = im(:,:,1); R(~bw_final) = 0;
    G = im(:,:,2); G(~bw_final) = 0;
    B = im(:,:,3); B(~bw_final) = 0;
    im_masked = cat(3, R, G, B);
    
    
    figure,imshow(im_masked)
end


ruta_dataset = 'C:\Users\alexd\Desktop\VC'; 

% llemgim carpetes
imds = imageDatastore(ruta_dataset, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Inicialitzem variables buides
Features = [];
Labels = {};

disp(['S''han trobat ', num2str(length(imds.Files)), ' imatges. Processant...']);

% Bucle per totes les fotos
reset(imds); 

while hasdata(imds)
    % Llegim la imatge i la informació 
    [im, info] = read(imds);
    
    
    

    %% segmentacio :
    im_masked = segmentar(im);

    %% extraccio carecteristiques

   
end

% Taula Final per el learner
TaulaFinal = array2table(Features, 'VariableNames', {'Compacitat', 'Solidesa', 'HOG_Mean'});
TaulaFinal.Clase = Labels;

disp('Procés finalitzat.');
disp(['Total mostres per entrenar: ', num2str(height(TaulaFinal))]);