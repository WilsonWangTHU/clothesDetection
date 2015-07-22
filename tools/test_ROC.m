% ------------------------------------------------------------------------
% Generate the results of the ROC curve on JD datasets, forever21 datasets.
% The methods tested include the 'Fast-RCNN', 'Pose detector'.
%
% Written by Tingwu Wang, 20.7.2015, as a junior RA in CUHK, MMLAB
% ------------------------------------------------------------------------

% set this on when we use 50% as the training set, and the rest 50% as test
% set in the JD datasets
test_type_1000 = true;
which test_ROC.m
% set the dataset you want to test
JD_datasets = false;
forever21_datasets = true;
method = 'pose';
%method = 'fast-RCNN'; % 'pose'

% some basic test parameters
step_number = 50;
switch method
    case 'fast-RCNN',
        min_cfd = 0.60;
        max_cfd = 0.99;
    case 'pose',
        min_cfd = -1.1;
        max_cfd = -0.4;
    otherwise,
        fprintf('Error! You could pose detector `pose`, or `fast-rcnn`')
        error('Program exit')
end
IOU_threshhold = 0.5;  % the IOU threshhold that defines currect detections

% a dataset transition
upper = [5, 6, 12, 14, 25, 27, 39, 47, 49, 50, 52, 55, 56, 11, 23];
lower_type1 = [28, 46, 54];
lower_type2 = [31, 32, 41, 26, 43];
whole = [15, 36];

% dependency
[mat_file_path] = fileparts(mfilename('fullpath'));
if isdir(mat_file_path)
    addpath(mat_file_path)
    addpath([mat_file_path '/../lib/matlab_lib/'])
else
    addpath([pwd mat_file_path])
    addpath([pwd mat_file_path '/../lib/matlab_lib/'])
end
number_gt = zeros(step_number, 1);
number_pst_detection = zeros(step_number, 1);
number_detection = zeros(step_number, 1);
number_recall = zeros(step_number, 1);
count = 1;

if forever21_datasets == true
    for cfd_threshhold = linspace(min_cfd, max_cfd, step_number)
        fprintf('Running the test when the cfd is %f\n', cfd_threshhold)
        [number_gt(count), number_pst_detection(count), ...
            number_detection(count), number_recall(count)] = ...
            ROC_forever21_datasets(method, upper, lower_type1, ...
            lower_type2, whole, cfd_threshhold, IOU_threshhold);
        count = count + 1;
    end
    
    % plot the results
    figure
    plot(number_recall ./ number_gt, number_pst_detection./number_detection)
    title(['The precision vs recall figure using' method 'method'])
    xlabel('Recall Rate')
    ylabel('Precision Rate')
    
end

if JD_datasets == true
    for cfd_threshhold = linspace(min_cfd, max_cfd, step_number)
        fprintf('Running the test when the cfd is %f\n', cfd_threshhold)
        [number_gt(count), number_pst_detection(count), ...
            number_detection(count), number_recall(count)] = ...
            ROC_JD_datasets(upper, lower_type1, ...
            lower_type2, whole, cfd_threshhold, IOU_threshhold);
        count = count + 1;
    end
    % plot the results
    figure
    plot(number_recall ./ number_gt, number_pst_detection./number_detection)
    title(['The precision vs recall figure using' method 'method'])
    xlabel('Recall Rate')
    ylabel('Precision Rate')
end