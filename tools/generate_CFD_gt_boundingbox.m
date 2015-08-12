% ------------------------------------------------------------------------
% Generate the training gt bounding box of the CFD datasets.
% We should run this matlab script in the fast-rcnn/tools directory
% The output is the matlab .mat files
% Written by Tingwu Wang, 11.08.2015, as a junior RA in CUHK, MMLAB
% ------------------------------------------------------------------------

addpath '../lib/matlab_lib/'
debug = false;

CFD_root_dir = '../data/CFD/';
CFD_image_dir = '../data/CFD/image/';
CFD_boundingbox_output_dir = [CFD_root_dir 'bounding_box/'];

% the index for delivering the type into the training three types
upper = [2, 5, 6, 7, 23];
lower_type2 = [12, 14, 17, 19];
lower_type1 = [13, 21];  % leggings
whole = [8];

% load the dataset gt diles, we got a 'all_category_name', 23X1 cell
% and a 'all_colors_name', 13X1 cell, and a big file called "fashion_dataset"
% a 1x2682 cell
if ~(exist('all_category_name', 'var') && ...
        exist('all_category_name', 'var') ...
        && exist('all_category_name', 'var'))
    load([CFD_root_dir 'fashon_parsing_data.mat'])
end

for i_Image = 1: 1: length(fashion_dataset)
    % process the image one by one!
    image_name = fashion_dataset{i_Image}.img_name;
    if mod(i_Image, 100) == 1
        fprintf(['    Generating the gt bounding box for' ...
            ' the %d th image in the CFD dataset\n'], i_Image)
    end
    output_mat_name = [image_name(1:find(image_name == '.')) 'mat'];
    
    % convert the super pixel annotation into the classic label annotation
    image_data = fashion_dataset{i_Image}.segmentation;
    mapping_label = fashion_dataset{i_Image}.category_label;
    image_data(:, :) = mapping_label(image_data(:, :) + 1);
    
    % generate the bounding box using the old fashion way, add one for the
    % compatibility
    coordinates = zeros(4, 5);
    coordinates(1, :) = [1, get_boundingbox(upper + 1, image_data, true)];
    % the lower 1 body (stockings etc.)
    coordinates(2, :) = [2, get_boundingbox(lower_type1 + 1, image_data, true)];
    % the lower 2 body (pants etc.)
    coordinates(3, :) = [2, get_boundingbox(lower_type2 + 1, image_data, true)];
    % the whole body (dress)
    coordinates(4, :) = [3, get_boundingbox(whole + 1, image_data, true)];
    
    if debug
        color_map = ['g', 'b', 'b', 'r'];
        imshow([CFD_image_dir fashion_dataset{i_Image}.img_name])
        for i_coor = 1: 1: length(coordinates(:, 1))
            if coordinates(i_coor, 4) - coordinates(i_coor, 2) <= 0 ...
                    || coordinates(i_coor, 5) - coordinates(i_coor, 3) <=0
                continue
            end
                rectangle('Position', ...
                    [coordinates(i_coor, 2), coordinates(i_coor, 3), ...
                    coordinates(i_coor, 4) - coordinates(i_coor, 2), ...
                    coordinates(i_coor, 5) - coordinates(i_coor, 3)], ...
                    'LineWidth', 3, 'EdgeColor',color_map(i_coor));
        end
    end
    save([CFD_boundingbox_output_dir output_mat_name], 'coordinates');
end
