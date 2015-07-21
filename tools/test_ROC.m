% ------------------------------------------------------------------------
% Generate the results of the ROC curve on JD datasets, forever21 datasets.
% The methods tested include the 'Fast-RCNN', 'Pose detector'.
%
% Written by Tingwu Wang, 20.7.2015, as a junior RA in CUHK, MMLAB
% ------------------------------------------------------------------------

% set this on when we use 50% as the training set, and the rest 50% as test
% set in the JD datasets
test_type_1000 = true;

% some basic test parameters
step_number = 50;
min_cfd = 0.60;
max_cfd = 0.99;
IOU_threshhold = 0.5;  % the IOU threshhold that defines currect detections

% set the dataset you want to test
JD_datasets = false;
forever21_datasets = true;

% a dataset transition
upper = [5, 6, 12, 14, 25, 27, 39, 47, 49, 50, 52, 55, 56, 11, 23];
lower_type1 = [28, 46, 54];
lower_type2 = [31, 32, 41, 26, 43];
whole = [15, 36];

% dependency
[mat_file_path] = fileparts(mfilename('fullpath'));
addpath(mat_file_path)

number_gt = zeros(step_number, 1);
number_pst_detection = zeros(step_number, 1);
number_detection = zeros(step_number, 1);
count = 1;

if forever21_datasets == true
    for cfd_threshhold = linspace(min_cfd, max_cfd, step_number)
        fprintf('Running the test when the cfd is %f\n', cfd_threshhold)
        [number_gt(count), number_pst_detection(count), ...
            number_detection(count)] = ...
            ROC_forever21_datasets(upper, lower_type1, ...
            lower_type2, whole, cfd_threshhold, IOU_threshhold);
        count = count + 1;
    end
end
