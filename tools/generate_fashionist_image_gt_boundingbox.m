% ------------------------------------------------------------------------
% @Brief
% This matlab script is used to generate the images and the training gt 
% bounding box of the fashionista datasets.
% We should run this matlab script in the fast-rcnn/tools directory
% The output is the matlab .mat files in the fashionista/groundtruth
% and the .jpg files in the fashionista/
% Written by Tingwu Wang, 11.08.2015, as a junior RA in CUHK, MMLAB
% ------------------------------------------------------------------------

addpath(genpath('../lib/matlab_lib/'));
debug = true;

fashionista_root_dir = '../data/Fashionista/';
fashionista_image_output_dir = '../data/Fashionista/images/';
fashionista_boundingbox_output_dir = [fashionista_root_dir 'bounding_box/'];

% addpath dependency for the image decoder and the image encoder
addpath [fashionista_root_dir 'lib/encoder/']

% the index for delivering the type into the training three types
upper = [4, 5, 12, 13, 17, 18, 23, 24, 26, 34, 35, 36, 37];
lower_type2 = [3, 9, 16, 19];
lower_type1 = [2, 20, 32]; % leggings
whole = [8, 15, 44, 53];

% load the fashionista dataset gt files, the `truths` is the important 
% data source. The `truths` and `predictions` struct has the following
% fields.  index       Index of the sample.
%    image       JPEG-encoded image data.
%    pose        Struct of pose annotation in PARSE box format [Yang 11].
%    annotation  Struct of superpixel annotation.
%    id          Internal use.
% the size of the dataset is 685

if ~(exist('truths', 'var') && ...
        exist('test_index', 'var') ...
        && exist('predictions_paperdoll', 'var'))
    load([fashionista_root_dir 'fashionista_v0.2.1.mat'])
end

for i_Image = 1: 1: length(truths)
    % process the image one by one!
    image_name = [num2str(i_Image) '.jpg'];
    
    % save the raw data into the jpg files
    raw_image_data = truths(i_Image).image;
    jpg_image_data = imdecode(raw_image_data, 'jpg');
    imwrite(jpg_image_data, [fashionista_image_output_dir image_name])

    % first 
    if mod(i_Image, 10) == 1
        fprintf(['    Generating the gt bounding box for' ...
            ' the %d th image in the CFD dataset\n'], i_Image)
    end
    output_mat_name = [image_name(1:find(image_name == '.')) 'mat'];
    
    % convert the super pixel annotation into the classic label annotation
    image_data = imdecode(truths(i_Image).annotation.superpixel_map, 'PNG');
    image_data = double(image_data);
    mapping_label = truths(i_Image).annotation.superpixel_labels;
    image_data(:, :) = mapping_label(image_data(:, :));
    
    % generate the bounding box using the old fashion way, add 1 for the
    % compatibility of the old get_boundingbox function
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
        imshow(jpg_image_data)
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
    save([fashionista_boundingbox_output_dir output_mat_name], 'coordinates');
end
