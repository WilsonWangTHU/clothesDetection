clear; clc;

addpath(genpath('./utilities/'));

% Initialization

% file_bbox_gt_enlarge = './bbox/list_bbox_gt_enlarge.txt';
% file_bbox_gt_leftprofile = './bbox/list_bbox_gt_leftprofile.txt';
% file_bbox_gt_rightprofile = './bbox/list_bbox_gt_rightprofile.txt';
% file_bbox_gt_face = './bbox/list_bbox_gt_face.txt';
% file_bbox_gt_skin = './bbox/list_bbox_gt_skin.txt';
% file_bbox_gt_tight = './bbox/list_bbox_gt_tight.txt';

file_bbox_gt_enlarge = './bbox_meitu/list_bbox_gt_enlarge.txt';
file_bbox_gt_leftprofile = './bbox_meitu/list_bbox_gt_leftprofile.txt';
file_bbox_gt_rightprofile = './bbox_meitu/list_bbox_gt_rightprofile.txt';
file_bbox_gt_face = './bbox_meitu/list_bbox_gt_face.txt';
file_bbox_gt_skin = './bbox_meitu/list_bbox_gt_skin.txt';
file_bbox_gt_tight = './bbox_meitu/list_bbox_gt_tight.txt';

method = 'acf';
% num_img = 39829;
num_img = 1000;
% num_img = 3876;
T = 0.5;

switch method

case 'attr'

	% file_bbox_method = './bbox/list_bbox_attr.txt';
	file_bbox_method = './bbox_meitu/list_bbox_attr_meitu.txt';
	score_set = 7:-0.2:1.5;
	score_default = 2.2;

case 'acf'

	% file_bbox_method = './bbox/list_bbox_acf.txt';
	file_bbox_method = './bbox_meitu/list_bbox_acf_meitu.txt';
	score_set = 100:-5:-10;
	score_default = -1;

case 'dpm'

	% file_bbox_method = './bbox/list_bbox_dpm.txt';
	file_bbox_method = './bbox_meitu/list_bbox_dpm_meitu.txt';
	score_set = 3:-0.1:-3;
	score_default = 0;

case 'surf'

	file_bbox_method = './bbox/list_bbox_surf.txt';
	score_set = 5:-0.2:0;
	score_default = 1;

case 'frontal'

	file_bbox_method = './bbox/list_bbox_frontal.txt';
	score_set = 0.5:-0.05:0;
	score_default = 0.1;

case 'multiview'

	file_bbox_method = './bbox/list_bbox_multiview.txt';	
	score_set = 10:-0.5:1;
	score_default = 3;

case 'spider'

	file_bbox_method = './bbox/list_bbox_spider.txt';	
	score_set = 80:-5:0;
	score_default = 10;

end

fid_bbox_method = fopen(file_bbox_method, 'rt');

fid_bbox_gt_enlarge = fopen(file_bbox_gt_enlarge, 'rt');
lines = textscan(fid_bbox_gt_enlarge, ['%s', repmat(' %d', [1, 4])]);
fclose(fid_bbox_gt_enlarge);
list_name = lines{1};
list_bbox_gt_enlarge = [lines{2}, lines{3}, lines{4}, lines{5}];

fid_bbox_gt_leftprofile = fopen(file_bbox_gt_leftprofile, 'rt');
lines = textscan(fid_bbox_gt_leftprofile, ['%s', repmat(' %d', [1, 4])]);
fclose(fid_bbox_gt_leftprofile);
list_name = lines{1};
list_bbox_gt_leftprofile = [lines{2}, lines{3}, lines{4}, lines{5}];

fid_bbox_gt_rightprofile = fopen(file_bbox_gt_rightprofile, 'rt');
lines = textscan(fid_bbox_gt_rightprofile, ['%s', repmat(' %d', [1, 4])]);
fclose(fid_bbox_gt_rightprofile);
list_name = lines{1};
list_bbox_gt_rightprofile = [lines{2}, lines{3}, lines{4}, lines{5}];

fid_bbox_gt_face = fopen(file_bbox_gt_face, 'rt');
lines = textscan(fid_bbox_gt_face, ['%s', repmat(' %d', [1, 4])]);
fclose(fid_bbox_gt_face);
list_name = lines{1};
list_bbox_gt_face = [lines{2}, lines{3}, lines{4}, lines{5}];

fid_bbox_gt_skin = fopen(file_bbox_gt_skin, 'rt');
lines = textscan(fid_bbox_gt_skin, ['%s', repmat(' %d', [1, 4])]);
fclose(fid_bbox_gt_skin);
list_name = lines{1};
list_bbox_gt_skin = [lines{2}, lines{3}, lines{4}, lines{5}];

fid_bbox_gt_tight = fopen(file_bbox_gt_tight, 'rt');
lines = textscan(fid_bbox_gt_tight, ['%s', repmat(' %d', [1, 4])]);
fclose(fid_bbox_gt_tight);
list_name = lines{1};
list_bbox_gt_tight = [lines{2}, lines{3}, lines{4}, lines{5}];

num_level = length(score_set);

true_positives = zeros(1, num_level);
false_positives = zeros(1, num_level);
total_positives = zeros(1, num_level);

% Get True Positives & False Positives

for id_img = 1:num_img

	name_cur = fscanf(fid_bbox_method, '%s', 1);

	num_bbox = fscanf(fid_bbox_method, '%d', 1);

	bbox_gt_enlarge = list_bbox_gt_enlarge(id_img, :);
	bbox_gt_leftprofile = list_bbox_gt_leftprofile(id_img, :);
	bbox_gt_rightprofile = list_bbox_gt_rightprofile(id_img, :);
	bbox_gt_face = list_bbox_gt_face(id_img, :);
	bbox_gt_skin = list_bbox_gt_skin(id_img, :);
	bbox_gt_tight = list_bbox_gt_tight(id_img, :);

	bbox_cur = zeros(num_bbox, 5);
	true_positives_cur = zeros(1, num_level);

	for id_bbox = 1:num_bbox
		
		bbox_cur(id_bbox, 1:4) = fscanf(fid_bbox_method, '%d', 4);
		bbox_cur(id_bbox, 5) = fscanf(fid_bbox_method, '%f', 1);	

		for id_level = 1:num_level
			
			threshold = score_set(id_level);
		
			if bbox_cur(id_bbox, 5) > threshold
				total_positives(id_level) = total_positives(id_level) + 1;
				
				[flag_true_positive_enlarge, ~] = ...
                    decideOverlap(bbox_cur(id_bbox, 1:4), bbox_gt_enlarge, T);
				[flag_true_positive_leftprofile, ~] = ...
                    decideOverlap(bbox_cur(id_bbox, 1:4), bbox_gt_leftprofile, T);
				[flag_true_positive_rightprofile, ~] = ...
                    decideOverlap(bbox_cur(id_bbox, 1:4), bbox_gt_rightprofile, T);
				[flag_true_positive_face, ~] = ...
                    decideOverlap(bbox_cur(id_bbox, 1:4), bbox_gt_face, T);
				[flag_true_positive_skin, ~] = ...
                    decideOverlap(bbox_cur(id_bbox, 1:4), bbox_gt_skin, T);
				[flag_true_positive_tight, ~] = ...
                    decideOverlap(bbox_cur(id_bbox, 1:4), bbox_gt_tight, T);

				flag_true_positive = flag_true_positive_enlarge ...
                    | flag_true_positive_leftprofile ...
                    | flag_true_positive_rightprofile ...
                    | flag_true_positive_face ...
                    | flag_true_positive_skin ...
                    | flag_true_positive_tight;

				true_positives_cur(id_level) = ...
                    true_positives_cur(id_level) | flag_true_positive;
				false_positives(id_level) = ...
                    false_positives(id_level) + (~flag_true_positive);

			end
		
		end

	end

	true_positives = true_positives + true_positives_cur;

	disp(['Processing Img ', num2str(id_img), '...']);

end

fclose(fid_bbox_method);

true_positive_rates = single(true_positives ./ num_img);