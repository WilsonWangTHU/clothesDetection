clear; clc;

% --------------------------------------------------------
% Single Image Demo for Cloth Detector
%
% Input: Folder of demo images
% Output: Bounding boxes of pre-defined object classes
%
% Written by Ziwei Liu, 2015/08/10
% --------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------------- Configuration ------------------------------------

% Set Dir 
dir_lib = '../../lib/';
dir_matlab = '../';

dir_data = '../../data/';
file_img = [dir_data, 'Demo/000004.jpg'];
file_proposal = [dir_data, 'Demo/000004_boxes.mat'];

dir_nets = '../../nets/';
file_net = [dir_nets, 'VGG16_FRCN_VOC/test.prototxt'];

dir_models = '../../models/';
file_model = [dir_models, 'Models_FT_VOC_bbox/vgg16_fast_rcnn_iter_40000.caffemodel'];
pixel_mean = reshape([102.9801, 115.9465, 122.7717], [1 1 3]);

% Add Libraries
addpath(genpath(dir_matlab));

% Set Hyper-parameters
flag_use_gpu = true;

scales = [];

height_data = 224;
width_data = 224;

T_nms_bbox = 0.3;
flag_nms_percls = true;

T_prob_cls = 0.8;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------------- Pipeline -----------------------------------------

% CNN Initialization
tic;
caffe('init', file_net, file_model, 'test');
if flag_use_gpu
	caffe('set_model_gpu');
else
	caffe('set_model_cpu');
end
toc;
disp('CNN Initialization Completed ...');

% Load Img
img_cur = imread(file_img);

% Calculate corresponding Object Proposals
tic;
load(file_proposal);
proposals_cur = single(boxes) + 1; % account for 1-based indexing in MATLAB 
clear boxes;
toc;
disp('Object Proposals Calculation Completed ...');

% Transform Input
tic;
[data_cur, rois_cur] = transformInput(img_cur, proposals_cur, pixel_mean, scales);
toc;
disp('Input Transformation Completed ...');

% CNN Feed Forward
tic;
blobs_in = cell{2, 1};
blobs_in{1} = data_cur;
blobs_in{2} = rois_cur;

blobs_out = caffe('forward', blobs_in);

probs_cls_cur = squeeze(blobs_out{1})';
bbox_delta_cur = squeeze(blobs_out{2})';
toc;
disp('CNN Feed Forward Completed ...');

% Transform Output
bbox_cur = transformOutput(proposals_cur, bbox_delta_cur);

% Post-process Bounding Boxes (NMS)
tic;
detection_cur = nmsBBox(bbox_cnn, prob_cnn, T_nms_bbox, flag_nms_percls); % non-maximum suppression
toc;
disp('Post-processing Completed ...');

% Visualize Results
num_cls = length(detection_cur);

for id_cls = 2:num_cls

	detection_cls = detection_cur{id_cls};
	indices_thres = find(detection_cls(:, 5) > T_prob_cls);
	showBoxes(img_cur, detection_cls(indices_thres, :));
	pause;

end

%----------------------------------- The End -------------------------------------------