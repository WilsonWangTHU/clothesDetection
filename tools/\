% ------------------------------------------------------------------------
% Generate the training gt bounding box of the CFD datasets.
% We should run this matlab script in the fast-rcnn/tools directory
%
% Written by Tingwu Wang, 11.08.2015, as a junior RA in CUHK, MMLAB
% ------------------------------------------------------------------------

CFD_root_dir = '../data/CFD/';
CFD_image_dir = '../data/CFD/image';
CFD_boundingbox_output_dir = [CFD_root_dir 'bounding_box/']

% the index for delivering the type into the training three types
upper = [5, 6, 12, 14, 25, 27, 39, 47, 49, 50, 52, 55, 56, 11, 23];
lower_type1 = [28, 46, 54];
lower_type2 = [31, 32, 41, 26, 43];
whole = [15, 36];

% load the dataset gt diles, we got a 'all_category_name', 23X1 cell
% and a 'all_colors_name', 13X1 cell, and a big file called "fashion_dataset"
% a 1x2682 cell
load([CFD_root_dir 'fashon_parsing_data.mat'])

for i_Image = 1: 1: length(fashion_dataset)
    % process the image one by one!
    image_name = fashion_dataset{i_Image}.img_name;
    if mod(i_Image, 100) == 1
        fprintf(['    Generating the gt bounding box for' ...
            ' the %d th image in the CFD dataset\n', i_Image - 2)
    end
    % try to read the gt files, if not exsit, generate them
    if exist([boundingbox_gt_dir '/' image_list(i_Image).name], 'file') ~= 2
        % load(image_annotation) % the variable: 'tags'
        pixel_file = [pix_annotation_dir '/' image_list(i_Image).name];
        
        % the upper body
        coordinates_u = get_boundingbox(upper, pixel_file);
        
        % the lower 1 body (stockings etc.)
        coordinates_l1 = get_boundingbox(lower_type1, pixel_file);
        
        % the lower 2 body (pants etc)
        coordinates_l2 = get_boundingbox(lower_type2, pixel_file);
        save([boundingbox_gt_dir '/' image_list(i_Image).name], ...
            'coordinates_u', 'coordinates_l1', 'coordinates_l2');
    else
        load([boundingbox_gt_dir '/' image_list(i_Image).name]);
    end
    
    % read the results file
    image_name = image_list(i_Image).name;
    image_name = [image_name(1:end-4) '.jpg'];
    results = get_float_text_results( ...
        [forever21_result_dir '/' image_name float_ext]);
    results_cls = get_int_text_results( ...
        [forever21_result_dir '/' image_name int_ext]);
    
    [sgl_number_gt, sgl_number_pst_detection, ...
        sgl_number_detection, sgl_number_recall] = ...
        precision_test([coordinates_u; coordinates_l1; coordinates_l2], ...
        results, results_cls, cfd_threshhold, IOU_threshhold, 'forever21');
    number_gt = number_gt + sgl_number_gt;
    number_pst_detection = number_pst_detection + sgl_number_pst_detection;
    number_detection = number_detection + sgl_number_detection;
    number_recall = number_recall + sgl_number_recall;
end
