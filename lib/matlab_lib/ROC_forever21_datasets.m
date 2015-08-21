% ------------------------------------------------------------------------
% Generate the results of the ROC curve on forever21 datasets.
% The methods tested include the 'Fast-RCNN', 'Pose detector'.
%
% Written by Tingwu Wang, 20.07.2015, as a junior RA in CUHK, MMLAB
% Updated in 20, 08, 2015, add the curve for each class
% ------------------------------------------------------------------------

function [number_gt, number_pst_detection, ...
    number_detection, number_recall] = ...
    ROC_forever21_datasets(method, cfd_threshhold, IOU_threshhold, model_version)
% the result variable
number_gt = 0; number_pst_detection = 0; 
number_detection = 0; number_recall = 0;

% prepare the path directory
switch method
    case 'fast-RCNN',
        [mat_file_path] = fileparts(mfilename('fullpath'));
        forever21_result_dir = [mat_file_path '/../../data/results/forever21'];
    case 'pose',
        [mat_file_path] = fileparts(mfilename('fullpath'));
        forever21_result_dir = [mat_file_path '/../../data/results/forever21_pose'];
    otherwise
        fprintf('The method doesnt exist! Check again!\n')
        fprintf('Use `fast-RCNN` or `pose` !\n')
        error('Program exit')
end
if model_version == 2
    forever21_result_dir = [forever21_result_dir + '_version2'];
end

%image_dir = [mat_file_path '/../data/CCP/photos'];
%image_annotation_dir = [mat_file_path '/../data/CCP/annotations/image-level'];
boundingbox_gt_dir = [mat_file_path '/../../data/CCP/bounding_box'];

float_ext = '_floatResults';
int_ext = '_intResults';

% get the image name
image_list = dir(boundingbox_gt_dir);

for i_Image = 1: 1: length(image_list)
    if i_Image <= 2 % the first two are not image name
        continue;
    end
    if mod(i_Image, 100) == 1
        fprintf('    Testing the %d th image in the test set\n', i_Image - 2)
    end
    
%     % try to read the gt files, if not exsit, generate them
%     if exist([boundingbox_gt_dir '/' image_list(i_Image).name], 'file') ~= 2
%         % load(image_annotation) % the variable: 'tags'
%         pixel_file = [pix_annotation_dir '/' image_list(i_Image).name];
%         
%         % the upper body
%         coordinates_u = get_boundingbox(upper, pixel_file);
%         
%         % the lower 1 body (stockings etc.)
%         coordinates_l1 = get_boundingbox(lower_type1, pixel_file);
%         
%         % the lower 2 body (pants etc)
%         coordinates_l2 = get_boundingbox(lower_type2, pixel_file);
%         save([boundingbox_gt_dir '/' image_list(i_Image).name], ...
%             'coordinates_u', 'coordinates_l1', 'coordinates_l2');
%     else
%         load([boundingbox_gt_dir '/' image_list(i_Image).name]);
%     end
    
    % read the gt results in the bounding_box directory
    coordinates = load([boundingbox_gt_dir image_list(i_Image).name]);

    % read the results file
    image_name = image_list(i_Image).name;
    image_name = [image_name(1:end-4) '.jpg'];
    results = get_float_text_results( ...
        [forever21_result_dir '/' image_name float_ext]);
    results_cls = get_int_text_results( ...
        [forever21_result_dir '/' image_name int_ext]);
    
    
    [sgl_number_gt, sgl_number_pst_detection, ...
        sgl_number_detection, sgl_number_recall] = ...
        precision_test(coordinates, ...
        results, results_cls, cfd_threshhold, IOU_threshhold, 'forever21');
    
    number_gt = number_gt + sgl_number_gt;
    number_pst_detection = number_pst_detection + sgl_number_pst_detection;
    number_detection = number_detection + sgl_number_detection;
    number_recall = number_recall + sgl_number_recall;
end


end
