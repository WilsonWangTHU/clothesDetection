% ------------------------------------------------------------------------
% Generate the results of the ROC curve on JD datasets, forever21 datasets.
% The methods tested include the 'Fast-RCNN', 'Pose detector'.
%
% Written by Tingwu Wang, 20.7.2015, as a junior RA in CUHK, MMLAB
% ------------------------------------------------------------------------


clc, clear;
% set this on when we use 50% as the training set, and the rest 50% as test
% set in the JD datasets
test_type_1000 = true;
which test_ROC.m
% set the dataset you want to test
JD_datasets = true;
forever21_datasets = false;
%method = 'pose';
method = 'fast-RCNN'; % 'pose'
plot_for_each_category = true;

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

cat_number_gt = zeros(step_number, 3 + 26);
cat_number_pst_detection = zeros(step_number, 3 + 26);
cat_number_detection = zeros(step_number, 3 + 26);
cat_number_recall = zeros(step_number, 3 + 26);

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
    title(['The precision vs recall figure using ' method ' method in forever21'])
    xlabel('Recall Rate')
    ylabel('Precision Rate')
    
end

if JD_datasets == true
    for cfd_threshhold = linspace(min_cfd, max_cfd, step_number)
        fprintf('Running the test when the cfd is %f\n', cfd_threshhold)
        if plot_for_each_category == false
            [number_gt(count), number_pst_detection(count), ...
                number_detection(count), number_recall(count),~,~,~,~] = ...
                ROC_JD_datasets(method, cfd_threshhold, IOU_threshhold, ...
                plot_for_each_category);
        else
            [number_gt(count), number_pst_detection(count), ...
                number_detection(count), number_recall(count), ...
                cat_number_gt(count, :), ...
                cat_number_pst_detection(count, :), ...
                cat_number_detection(count, :),...
                cat_number_recall(count, :)] = ...
                ROC_JD_datasets(method, cfd_threshhold, IOU_threshhold, ...
                plot_for_each_category);
        end
        count = count + 1;
    end
    % plot the results
    figure
    plot(number_recall ./ number_gt, ...
        number_pst_detection ./ number_detection)
    title(['The precision vs recall figure using ' method ' method in JD'])
    xlabel('Recall Rate')
    ylabel('Precision Rate')
    if plot_for_each_category == true
        for i = 1: 1: 26 + 3
            h = figure;
            plot(cat_number_recall(:, i) ./ cat_number_gt(:, i), ...
                cat_number_pst_detection(:, i)./ cat_number_detection(:, i))
            title(['The precision vs recall figure using ' method ' method in JD'])
            xlabel('Recall Rate')
            ylabel('Precision Rate')
            map = ['upper', 'lower', 'whole'];
            if i > 3
                file_name = ['/media/Elements/twwang/fast-rcnn/data/' ...
                    'PR_fig_JD_cat_' num2str(i) '.fig'];
            else
                file_name = ['/media/Elements/twwang/fast-rcnn/data/' ...
                    'PR_fig_JD_cat_' map(i) '.fig'];
            end
            grid on; box on;
            savefig(h, file_name)
        end
        % plot them together
        h = figure;
        color = ['r', 'g', 'b'];
        for i = 1: 1: 3
            plot(cat_number_recall(:, i) ./ cat_number_gt(:, i), ...
                cat_number_pst_detection(:, i)./ cat_number_detection(:, i), ...
                color(i))
            title(['The precision vs recall figure using ' method ' method in JD'])
            xlabel('Recall Rate')
            ylabel('Precision Rate')
            map = ['upper', 'lower', 'whole'];
            hold on;
            grid on; box on;
        end
        file_name = ['/media/Elements/twwang/fast-rcnn/data/' ...
            'PR_fig_JD_together.fig'];
        legend('upper', 'lower', 'whole')
        
        savefig(h, file_name)
        
    end
    
end

