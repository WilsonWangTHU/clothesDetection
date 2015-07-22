% ------------------------------------------------------------------------
% Generate the results of the ROC curve on forever21 datasets.
% The methods tested include the 'Fast-RCNN', 'Pose detector'.
%
% Written by Tingwu Wang, 20.7.2015, as a junior RA in CUHK, MMLAB
% ------------------------------------------------------------------------

function [number_gt, number_pst_detection, ...
    number_detection, number_recall] = ...
    ROC_JD_datasets(method, upper, lower_type1, lower_type2, ...
    whole, cfd_threshhold, IOU_threshhold)
% the result variable
number_gt = 0; number_pst_detection = 0;
number_detection = 0; number_recall = 0;

% basic experiment parameters
number_category = 26;

% prepare the path directory
switch method
    case 'fast-RCNN',
        [mat_file_path] = fileparts(mfilename('fullpath'));
        forever21_result_dir = [mat_file_path '/../../data/results/Jingdong'];
        gt_results_dir = [mat_file_path '/../../data/clothesDataset/test'];
    case 'pose',
        fprintf('The pose method is not available! Check again!\n')
        fprintf('Use `fast-RCNN` or `pose` !\n')
        error('Program exit')
    otherwise
        fprintf('The method doesnt exist! Check again!\n')
        fprintf('Use `fast-RCNN` or `pose` !\n')
        error('Program exit')
end

% the extension of the class
float_ext = 'floatResults';
int_ext = 'intResults';

% the directory function
for i_category = 1: 1: number_category
    
    % read the test results!
    float_results_file = ...
        fopen([forever21_result_dir '/' num2str(i_category) float_ext]);
    intt_results_file = ...
        fopen([forever21_result_dir '/' num2str(i_category) int_ext]);
    
    % get the number of test image this class
    results = get_float_text_results(float_results_file);
    results_cls = get_int_text_results(intt_results_file);
    if length(results(:, 1))

    
    % read the image index and the label index
    f = fopen([gt_results_dir '/' num2str(i_category) ...
        '/newGUIDMapping.txt'], r);
    
    for 
    tline = fgets(fileID);
    
    
    if mod(i_Image, 100) == 1
        fprintf('    Testing the %d th image in the test set\n', i_Image - 2)
    end
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
        [forever21_result_dir '/' image_name float_ext]);
    results_cls = get_int_text_results( ...
        [forever21_result_dir '/' image_name int_ext]);
    
    [sgl_number_gt, sgl_number_pst_detection, ...
        sgl_number_detection, sgl_number_recall] = ...
        precision_test([coordinates_u; coordinates_l1; coordinates_l2], ...
        results, results_cls, cfd_threshhold, IOU_threshhold);
    
    % add the results
    number_gt = number_gt + sgl_number_gt;
    number_pst_detection = number_pst_detection + sgl_number_pst_detection;
    number_detection = number_detection + sgl_number_detection;
    number_recall = number_recall + sgl_number_recall;
end


end
