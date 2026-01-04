ruta_dataset = './'; 
imds = imageDatastore(ruta_dataset, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Inicialitzem variables
numImages = numel(imds.Files);


% 1. HOC (Color): 3 valors (Red, Blue, Yellow)
% 2. HOG GLOBAL: 1764 valors (Per capturar formes generals encara que hi hagi soroll)
% Total: 1767 features
TotalFeatures = 3 + 1764; 

Features = zeros(numImages, TotalFeatures);
Labels = cell(numImages, 1);

disp(['Iniciant extracció de features GLOBALS per a ', num2str(numImages), ' imatges...']);
reset(imds); 

for i = 1:numImages
    % Llegim imatge
    [im, info] = read(imds);
    im = im2uint8(im);
    
    % Guardem l'etiqueta 
    Labels{i} = info.Label;
    
    % --- PART 1: HOC GLOBAL (Histograma de Color) ---
    % A diferència de l'anterior, aquí mirem TOTA la imatge sense retallar
    im_hsv = rgb2hsv(im);
    H = im_hsv(:,:,1);
    S = im_hsv(:,:,2);
    
    total_pixels = numel(H); % Total de píxels de la imatge
    
    % Comptem píxels de cada color en tota la imatge
    % (Ajustem llindars per ser més tolerants ja que hi ha fons)
    p_red = sum(((H(:)>0.90)|(H(:)<0.10)) & (S(:)>0.30));
    p_blue = sum((H(:)>0.55)&(H(:)<0.75) & (S(:)>0.30));
    p_yellow = sum((H(:)>0.12)&(H(:)<0.20) & (S(:)>0.30));
    
    pct_red = p_red / total_pixels;
    pct_blue = p_blue / total_pixels;
    pct_yellow = p_yellow / total_pixels;
    
    % --- PART 2: HOG GLOBAL ---
    % Redimensionem la imatge sencera a 64x64 per tenir un vector fix
    % Això "aixafa" la imatge, però manté l'estructura general del senyal
    im_resized = imresize(im, [64, 64]);
    
    % Convertim a gris per al HOG
    if size(im_resized, 3) == 3
        im_gray_resized = rgb2gray(im_resized);
    else
        im_gray_resized = im_resized;
    end
    
    % Extracció HOG (Vector de 1764)
    % Usem un CellSize una mica més gran [8 8] per ser més genèrics
    hog_vector = extractHOGFeatures(im_gray_resized, 'CellSize', [8 8]);
    
    % --- GUARDEM ---
    % [Pct_Red, Pct_Blue, Pct_Yellow, Vector HOG]
    Features(i, :) = [pct_red, pct_blue, pct_yellow, hog_vector];
    
    % Barra de progrés simple
    if mod(i, 100) == 0
        disp(['Processades ', num2str(i), ' / ', num2str(numImages)]);
    end
end

% Creació de la Taula Final
TaulaFallback = array2table(Features);

% Noms de les columnes
TaulaFallback.Properties.VariableNames{1} = 'Global_Pct_Red';
TaulaFallback.Properties.VariableNames{2} = 'Global_Pct_Blue';
TaulaFallback.Properties.VariableNames{3} = 'Global_Pct_Yellow';
% La resta són HOG automàticament

TaulaFallback.Clase = string(Labels);
vars_numeriques = TaulaFallback.Properties.VariableNames(1:end-1);
TaulaFallback{:, vars_numeriques} = normalize(TaulaFallback{:, vars_numeriques});
disp('-----------------------------------');
disp('Procés finalitzat. Variable "TaulaFallback" creada.');
