% ------------------------------------------------------------------------
% Generate the results of the ROC curve on JD datasets, forever21 datasets.
% The methods tested include the 'Fast-RCNN', 'Pose detector'.
%
% Written by Tingwu Wang, 20.7.2015, as a junior RA in CUHK, MMLAB
% Update:
%   Add the multilabel_test in 4.8.2015
% ------------------------------------------------------------------------


clc, clear;
% set in the JD datasets set the dataset you want to test
JD_datasets = true;
forever21_datasets = false;

% set the testing method
method = 'fast-RCNN'; % 'pose'

% setting for whether to plot or not
plot_for_each_category = true;

% set this on when we use 50% as the training set, and the rest 50% as test
test_type_1000 = false;

% set this on when considering the multi_label attributive test
multilabel_test = true;

% set this on when testing the softmax attributive retrieval
multilabel_softmax = false;
sfm_min_cdf = 0;
sfm_max_cdf = 9.9;
% some basic test parameters
step_number = 90;
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
    result_save_path = [mat_file_path '/../data/results/result_curve/'];
else
    addpath([pwd mat_file_path])
    addpath([pwd mat_file_path '/../lib/matlab_lib/'])
    result_save_path = [pwd mat_file_path '/../data/results/result_curve/'];
end

fprintf('The results will be saved to %s\n', result_save_path);

number_gt = zeros(step_number, 1);
number_pst_detection = zeros(step_number, 1);
number_detection = zeros(step_number, 1);
number_recall = zeros(step_number, 1);

% there are three types of attributives
number_gt_attr = zeros(step_number, 3);
number_pst_detection_attr = zeros(step_number, 3);
number_detection_attr = zeros(step_number, 3);
number_recall_attr = zeros(step_number, 3);

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
        
        % two different threshhold for the class and attributive are sent
        % in the test function respectively
        if multilabel_softmax == false
            united_cfd_threshhold = [cfd_threshhold, ...
                (cfd_threshhold - min_cfd) / (max_cfd - min_cfd) * ...
                (sfm_max_cdf - sfm_min_cdf) + sfm_min_cdf];
        else
            united_cfd_threshhold = [cfd_threshhold, cfd_threshhold];
        end
        
        if plot_for_each_category == false
            if multilabel_test == true
                [number_gt(count), number_pst_detection(count), ...
                    number_detection(count), number_recall(count), ...
                    number_gt_attr(count, :), number_pst_detection_attr(count, :), ...
                    number_detection_attr(count, :), number_recall_attr(count, :), ...
                    ~,~,~,~] ...
                    = ROC_JD_datasets(method, united_cfd_threshhold, ...
                    IOU_threshhold, plot_for_each_category, multilabel_test);
            else
                [number_gt(count), number_pst_detection(count), ...
                    number_detection(count), number_recall(count), ...
                    ~,~,~,~,~,~,~,~] ...
                    = ROC_JD_datasets(method, united_cfd_threshhold, ...
                    IOU_threshhold, plot_for_each_category, multilabel_test);
            end
        else
            if multilabel_test == true
                [number_gt(count), number_pst_detection(count), ...
                    number_detection(count), number_recall(count), ...
                    number_gt_attr(count, :), number_pst_detection_attr(count, :), ...
                    number_detection_attr(count, :), number_recall_attr(count, :), ...
                    cat_number_gt(count, :), ...
                    cat_number_pst_detection(count, :), ...
                    cat_number_detection(count, :),...
                    cat_number_recall(count, :)] = ...
                    ROC_JD_datasets(method, united_cfd_threshhold, ...
                    IOU_threshhold, plot_for_each_category, multilabel_test);
            else
                [number_gt(count), number_pst_detection(count), ...
                    number_detection(count), number_recall(count), ...
                    ~,~,~,~, ...
                    cat_number_gt(count, :), ...
                    cat_number_pst_detection(count, :), ...
                    cat_number_detection(count, :),...
                    cat_number_recall(count, :)] = ...
                    ROC_JD_datasets(method, united_cfd_threshhold, ...
                    IOU_threshhold, plot_for_each_category, multilabel_test);
            end
        end
        count = count + 1;
    end
    % plot the results
%     figure
%     plot(number_recall ./ number_gt, ...
%         number_pst_detection ./ number_detection)
%     title(['The precision vs recall figure using ' method ' method in JD'])
%     xlabel('Recall Rate')
%     ylabel('Precision Rate')
    if plot_for_each_category == true
%         for i = 1: 1: 26 + 3
%             h = figure;
%             plot(cat_number_recall(:, i) ./ cat_number_gt(:, i), ...
%                 cat_number_pst_detection(:, i)./ cat_number_detection(:, i))
%             title(['The precision vs recall figure using ' method ' method in JD'])
%             xlabel('Recall Rate')
%             ylabel('Precision Rate')
%             map = ['upper', 'lower', 'whole'];
%             if i > 3
%                 file_name = [result_save_path ...
%                     'PR_fig_JD_cat_' num2str(i) '.fig'];
%             else
%                 file_name = [result_save_path ...
%                     'PR_fig_JD_cat_' map(i) '.fig'];
%             end
%             grid on; box on;
%             savefig(h, file_name)
%         end
%         % plot the class detection results of three big class together
%         h = figure;
%         color = ['r', 'g', 'b'];
%         for i = 1: 1: 3
%             plot(cat_number_recall(:, i) ./ cat_number_gt(:, i), ...
%                 cat_number_pst_detection(:, i)./ cat_number_detection(:, i), ...
%                 color(i))
%             title(['The precision vs recall figure using ' method ' method in JD'])
%             xlabel('Recall Rate')
%             ylabel('Precision Rate')
%             map = ['upper', 'lower', 'whole'];
%             hold on;
%             grid on; box on;
%         end
%         file_name = [result_save_path ...
%             'PR_fig_JD_together.fig'];
%         legend('upper', 'lower', 'whole')
%         
%         savefig(h, file_name)
%         
        % plot the detection results of three attributive 
        h = figure;
        color = ['r', 'g', 'b'];
        for i = 1: 1: 3
            plot(number_recall_attr(:, i) ./ number_gt_attr(:, i), ...
                number_pst_detection_attr(:, i)./ number_detection_attr(:, i), ...
                color(i))
            title(['The precision vs recall of attributive figure using ' method ' method in JD'])
            xlabel('Recall Rate')
            ylabel('Precision Rate')
            hold on;
            grid on; box on;
        end
        file_name = [result_save_path ...
            'PR_Attributive_fig_JD_together.fig'];
        legend('Texture', 'Neckband', 'sleeve')
        savefig(h, file_name)
        
    end
    
end

