% ------------------------------------------------------------------------
% Generate the training gt bounding box of the CCP datasets.
% We should run this matlab script in the fast-rcnn/tools directory
% The output is the matlab .mat files
% Written by Tingwu Wang, 11.08.2015, as a junior RA in CUHK, MMLAB
% ------------------------------------------------------------------------

addpath '../lib/matlab_lib/'
debug = false;

% the input and output dir
CCP_root_dir = '../data/CCP/';
CCP_image_dir = '../data/CCP/photos/';
CCP_boundingbox_output_dir = [CCP_root_dir 'bounding_box/'];
pix_annotation_dir = [CCP_root_dir 'annotations/pixel-level/'];

% a dataset transition
upper = [5, 6, 12, 14, 25, 27, 39, 47, 49, 50, 52, 55, 56, 11, 23];
lower_type1 = [28, 46, 54];
lower_type2 = [31, 32, 41, 26, 43];
whole = [15, 36];

% get the image name
image_list = dir(pix_annotation_dir);

for i_Image = 1: 1: length(image_list)
    if i_Image <= 2 % the first two are not image name
        continue;
    end
    if mod(i_Image, 100) == 1
        fprintf('    Testing the %d th image in the test set\n', i_Image - 2)
    end
    % try to read the gt files, if not exsit, generate them
    % load(image_annotation) % the variable: 'tags'
    pixel_file = [pix_annotation_dir '/' image_list(i_Image).name];
    
    coordinates = zeros(4, 5);
    coordinates(1, :) = [1, get_boundingbox(upper, pixel_file, false)];
    % the lower 1 body (stockings etc.)
    coordinates(2, :) = [2, get_boundingbox(lower_type1, pixel_file, false)];
    % the lower 2 body (pants etc.)
    coordinates(3, :) = [2, get_boundingbox(lower_type2, pixel_file, false)];
    % the whole body (dress)
    coordinates(4, :) = [3, get_boundingbox(whole, pixel_file, false)];
    
    % read the results file
    image_name = image_list(i_Image).name;
    image_name = [image_name(1:end-4) '.jpg'];
    output_mat_name = [image_name(1:find(image_name == '.')) 'mat'];

    if debug
        color_map = ['g', 'b', 'b', 'r'];
        imshow([CCP_image_dir image_name])
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
    save([CCP_boundingbox_output_dir output_mat_name], 'coordinates');

end