% ------------------------------------------------------------------------
% Generate the results of the ROC curve on forever21 datasets.
% The methods tested include the 'Fast-RCNN', 'Pose detector'.
%
% Written by Tingwu Wang, 20.7.2015, as a junior RA in CUHK, MMLAB
% Update:
%   The test for multilabel attributive is added on 3.8.2015
% ------------------------------------------------------------------------

function [number_gt, number_pst_detection, ...
    number_detection, number_recall, ...
    number_gt_attr, number_pst_detection_attr, ...
    number_detection_attr, number_recall_attr, ...
    cat_number_gt, cat_number_pst_detection, ...
    cat_number_detection, cat_number_recall] = ...
    ROC_JD_datasets(method, cfd_threshhold, IOU_threshhold, ...
    plot_for_each_category, multi_label_test, multi_label_softmax)

if nargin < 5
    multi_label_test = false;
end

% the result variable
number_gt = 0; number_pst_detection = 0;
number_detection = 0; number_recall = 0;

number_gt_attr = zeros(3, 1); number_pst_detection_attr = zeros(3, 1);
number_detection_attr = zeros(3, 1); number_recall_attr = zeros(3, 1);

% record the three class if necessary
three_number_gt = zeros(3, 1); three_number_pst_detection = zeros(3, 1);
three_number_detection = zeros(3, 1); three_number_recall = zeros(3, 1);

% record the 26 class if necessary
category_number_gt = zeros(26, 1);
category_number_pst_detection = zeros(26, 1);
category_number_detection = zeros(26, 1);
category_number_recall = zeros(26, 1);

% basic experiment parameters
number_category = 26;

% transmit the 26 class into the 3 class, upper, lower and whole
twentysix2three = [ones(1, 7), 3, ones(1, 2), 3, ones(1, 8), 3, ...
    2 * ones(1, 6)];


% prepare the path directory
switch method
    case 'fast-RCNN',
        [mat_file_path] = fileparts(mfilename('fullpath'));
        jd_result_dir = [mat_file_path '/../../data/results/Jingdong'];
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
if multi_label_test
    jd_result_dir = [jd_result_dir '_multi_label'];
    if multi_label_softmax
        jd_result_dir = [jd_result_dir '_softmax'];
    end
end


% the extension of the class
float_ext = 'floatResults';
int_ext = 'intResults';
label_ext = '.clothInfo';

% the class map list
type_classes = {'风衣', '毛呢大衣', '羊毛衫/羊绒衫', ...
    '棉服/羽绒服',  '小西装/短外套', '西服', '夹克', '旗袍', '皮衣', '皮草', ...
    '婚纱', '衬衫', 'T恤', 'Polo衫', '开衫', '马甲', '男女背心及吊带', '卫衣', ...
    '雪纺衫', '连衣裙', '半身裙', '打底裤', '休闲裤', '牛仔裤', ...
    '短裤', '卫裤/运动裤'};
texture_classes = {'一致色', '横条纹', '纵条纹', '豹纹斑马纹', '格子', ...
    '圆点', '乱花', 'LOGO及印花图案', '其他'
    };
neckband_classes = {'圆领', 'V领', '翻领', '立领', '高领', '围巾领', ...
    '一字领', '大翻领西装领', '连帽领', '其他'};
sleeve_classes = {'短袖', '中袖', '长袖'};

attr_length = [length(texture_classes), ...
    length(neckband_classes), length(sleeve_classes)];
% the directory function
for i_category = 1: 1: number_category
    
    % read the test results!
    float_results_file = [jd_result_dir '/' num2str(i_category) float_ext];
    int_results_file = [jd_result_dir '/' num2str(i_category) int_ext];
    
    % get the number of test image this class
    results = get_float_text_results(float_results_file, multi_label_test);
    results_cls = get_int_text_results(int_results_file);
    if length(results(:, 1)) ~= length(results_cls(:, 1))
        error('The result sizes are not matched! Check the dimension!\n');
    end
    
    % read the image index and the label index
    image_name_file = fopen([gt_results_dir '/' num2str(i_category) ...
        '/newGUIDMapping.txt'], 'r');
    
    % process the image one by one
    i_image = 1;  % this variable record the number of image in this categ
    tline = fgets(image_name_file);
    while tline ~= -1
        
        if mod(i_image, 50) == 1
            fprintf('    Testing the %d th image in the %d cat\n', ...
                i_image, i_category)
        end
        
        % cur_image_name = tline(8: find(tline == '"') - 1);
        cur_label_name = [tline(find(tline == '"') + 1: end - 1), label_ext];
        
        if ~multi_label_test
            [gt_position, class, ~] = read_cloth_xml( ...
                [gt_results_dir '/' num2str(i_category) ...
                '/Label/' cur_label_name], type_classes, ...
                texture_classes, neckband_classes, sleeve_classes, ...
                multi_label_test);
        else
            [gt_position, class, multi_label_class] = read_cloth_xml( ...
                [gt_results_dir '/' num2str(i_category) ...
                '/Label/' cur_label_name], type_classes, ...
                texture_classes, neckband_classes, sleeve_classes, ...
                multi_label_test);
        end
        class = twentysix2three(class);
        
        % get the class detection results of this image, we send in the
        % gt_position and its class, and the results' position and class
        % and the cfd_threshhold
        [sgl_number_gt, sgl_number_pst_detection, ...
            sgl_number_detection, sgl_number_recall] = ...
            precision_test([gt_position, class], ...
            results(10 * i_image - 9: 10 * i_image, 1 : 5), ...
            results_cls(10 * i_image - 9: 10 * i_image, :), ...
            cfd_threshhold(1), IOU_threshhold, 'JD');
        
        number_gt = number_gt + sgl_number_gt;
        number_pst_detection = number_pst_detection + ...
            sgl_number_pst_detection;
        number_detection = number_detection + sgl_number_detection;
        number_recall = number_recall + sgl_number_recall;
        
        if multi_label_test
            % first we compute the max class for each sub attributive
            % groups. Preprocess the gt_label and the test results
            multi_attributive = find(multi_label_class == 1);
            % get the predicted id
            results_cls_confidence = zeros(10, 6);
            for ibox = 1: 1: 10
                [max_texture_cfd, max_texture_Id] = ...
                    max(results(10 * i_image - 10 + ibox, ...
                    6 : 6 + attr_length(1) - 1));
                [max_neckband_cfd, max_neckband_Id] = ...
                    max(results(10 * i_image - 10 + ibox, ...
                    6 + attr_length(1): 6 + attr_length(1) ...
                    + attr_length(2) - 1));
                max_neckband_Id = max_neckband_Id + attr_length(1);
                [max_sleeve_cfd, max_sleeve_Id] = ...
                    max(results(10 * i_image - 10 + ibox, ...
                    6 + attr_length(1) + attr_length(2) : ...
                    6 + attr_length(1) + attr_length(2) ...
                    + attr_length(3) - 1));
                max_sleeve_Id = max_sleeve_Id + attr_length(1) + ...
                    attr_length(2);
                results_cls_confidence(ibox, :) = [...
                    max_texture_Id, max_texture_cfd, ...
                    max_neckband_Id, max_neckband_cfd, ...
                    max_sleeve_Id, max_sleeve_cfd];
            end
            
            [sgl_number_gt_atti, sgl_number_pst_detection_atti, ...
                sgl_number_detection_atti, sgl_number_recall_atti] = ...
                attributive_test(gt_position, multi_attributive, ...
                results(10 * i_image - 9: 10 * i_image, 1 : 4), ...
                results_cls_confidence, ...
                cfd_threshhold(2), IOU_threshhold, attr_length, 'JD');
            
            number_gt_attr = number_gt_attr + sgl_number_gt_atti;
            number_pst_detection_attr = number_pst_detection_attr + ...
                sgl_number_pst_detection_atti;
            number_detection_attr = number_detection_attr + ...
                sgl_number_detection_atti;
            number_recall_attr = number_recall_attr + sgl_number_recall_atti;
        end
        
        if plot_for_each_category == true
            three_number_gt(twentysix2three(i_category)) = ...
                three_number_gt(twentysix2three(i_category)) + ...
                sgl_number_gt;
            three_number_pst_detection(twentysix2three(i_category)) = ...
                three_number_pst_detection(twentysix2three(i_category)) + ...
                sgl_number_pst_detection;
            three_number_detection(twentysix2three(i_category)) = ...
                three_number_detection(twentysix2three(i_category)) + ...
                sgl_number_detection;
            three_number_recall(twentysix2three(i_category)) = ...
                three_number_recall(twentysix2three(i_category)) + ...
                sgl_number_recall;
            
            category_number_gt(i_category) = ...
                category_number_gt(i_category) + sgl_number_gt;
            category_number_pst_detection(i_category) = ...
                category_number_pst_detection(i_category) + ...
                sgl_number_pst_detection;
            category_number_detection(i_category) = ...
                category_number_detection(i_category) + sgl_number_detection;
            category_number_recall(i_category) = ...
                category_number_recall(i_category) + sgl_number_recall;
        end
        
        i_image = i_image + 1;
        tline = fgets(image_name_file);  % read the next line
    end
    
end

cat_number_gt = [three_number_gt; category_number_gt];
cat_number_pst_detection = [three_number_pst_detection; ...
    category_number_pst_detection];
cat_number_detection = [three_number_detection; category_number_detection];
cat_number_recall = [three_number_recall; category_number_recall];