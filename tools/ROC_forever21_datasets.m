% ------------------------------------------------------------------------
% Generate the results of the ROC curve on forever21 datasets.
% The methods tested include the 'Fast-RCNN', 'Pose detector'.
%
% Written by Tingwu Wang, 20.7.2015, as a junior RA in CUHK, MMLAB
% ------------------------------------------------------------------------

function [number_gt, number_pst_detection, number_detection] = ...
    ROC_forever21_datasets(upper, lower_type1, lower_type2, ...
    whole, cfd_threshhold, IOU_threshhold)
% the result variable
number_gt = 0; number_pst_detection = 0; number_detection = 0;

% prepare the path directory
[mat_file_path] = fileparts(mfilename('fullpath'));
rcnn_forever21_result_dir = [mat_file_path '/../data/results/forever21'];
%image_dir = [mat_file_path '/../data/CCP/photos'];
%image_annotation_dir = [mat_file_path '/../data/CCP/annotations/image-level'];
pix_annotation_dir = [mat_file_path '/../data/CCP/annotations/pixel-level'];
boundingbox_gt_dir = [mat_file_path '/../data/CCP/annotations/boundingbox'];

float_ext = '_floatResults';
int_ext = '_intResults';

% get the image name
image_list = dir(pix_annotation_dir);

for i_Image = 1: 1: length(image_list)
    if i_Image <= 2 % the first two are not image name
        continue;
    end
    
    fprintf('    Testing the %d th image in the test set\n', i_Image - 2)

    % try to read the gt files, if not exsit, generate them
    if exist([boundingbox_gt_dir '/' image_list(i_Image).name],'file') ~= 2
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
        [rcnn_forever21_result_dir '/' image_name float_ext]);
    results_cls = get_int_text_results( ...
        [rcnn_forever21_result_dir '/' image_name int_ext]);
    
    [sgl_number_gt, sgl_number_pst_detection, sgl_number_detection] = ...
        precision_test([coordinates_u; coordinates_l1; coordinates_l2], ...
        results, results_cls, cfd_threshhold, IOU_threshhold);
    number_gt = number_gt + sgl_number_gt;
    number_pst_detection = number_pst_detection + sgl_number_pst_detection; 
    number_detection = number_detection + sgl_number_detection;
    
end


end
