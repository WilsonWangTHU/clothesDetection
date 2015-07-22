clear; clc;

if isunix()
  addpath mex_unix;
elseif ispc()
  addpath mex_pc;
end

% set directory
dir_img = '../../../Dataset/Fashionista_extend/';
dir_img_src = [dir_img, 'images/'];
dir_img_dst = [dir_img, 'selected/']; mkdir(dir_img_dst);
file_model = './PARSE_model.mat';

T_score = -0.5;

% load model
load(file_model);

imlist = dir([dir_img_src, '*.jpg']);

for i = 1:length(imlist)
    
    % load and display image
    im = imread([dir_img_src, imlist(i).name]);

    im_downsampled = imresize(im, 1/4, 'bilinear');
    
    % call detect function
    tic;
    boxes = detect_fast(im_downsampled, model, min(model.thresh,-1));
    dettime = toc; % record cpu time
    boxes = nms(boxes, .1); % nonmaximal suppression
    
    num_boxes = size(boxes, 1);
    score_cur = -10;

    if num_boxes == 1
    	score_cur = boxes(end);
    	if score_cur > T_score
    		imwrite(im, [dir_img_dst, imlist(i).name]);
    	end
    end

    fprintf('Processing Img %d, Score = %.2f, Detection took %.1f seconds\n', i, score_cur, dettime);
    
end