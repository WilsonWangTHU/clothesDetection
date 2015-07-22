% ----------------------------------------------------------------------
%   generate the test results of upper body and lower body by using the
%   pose detector, the image set is the
%   18.7.2015
% ----------------------------------------------------------------------


% add the dependency of the pose detector
[mat_file_path] = fileparts(mfilename('fullpath'));
addpath([mat_file_path '/../lib/matlab_lib/pose_detector'])
addpath([mat_file_path '/../lib/matlab_lib'])

if isunix()
    addpath([mat_file_path '/../lib/matlab_lib/pose_detector/mex_unix'])
elseif ispc()
    addpath([mat_file_path '/../lib/matlab_lib/pose_detector/mex_pc'])
end

load('PARSE_model');

% dependency
addpath(mat_file_path)
addpath([mat_file_path '/../lib/matlab_lib/'])
if isdir(mat_file_path)
    addpath([mat_file_path '/../lib/matlab_lib/'])
else
    addpath([pwd mat_file_path '/../lib/matlab_lib/'])
end

% get the image name
image_dir = [mat_file_path '/../data/CCP/photos'];
pix_annotation_dir = [mat_file_path '/../data/CCP/annotations/pixel-level'];
image_list = dir(pix_annotation_dir);

% output directory
output_dir = [mat_file_path '/../data/results/forever21_pose'];
float_ext = '_floatResults';
int_ext = '_intResults';

for i_Image = 1: 1: length(image_list)
    
    if i_Image <= 2
        continue;
    end
    
    image_name = image_list(i_Image).name;  % the name of the image
    image_name = [image_name(1 : end-4) '.jpg'];
    
    % the result writer
    float_writer = fopen([output_dir '/' image_name float_ext], 'w');
    int_writer = fopen([output_dir '/' image_name int_ext], 'w');
    
    % write the data header, it is written to fit in the sensetime API
    fprintf(float_writer,'%d\n', 10);
    fprintf(int_writer,'%d\n', 10);
    fprintf(float_writer,'%d\n', 5);
    fprintf(int_writer,'%d\n', 1);
    
    % resize the image first, as DPM is slow when the image is big
    im = imread([image_dir '/' image_name]);
    im_size = size(im);
    im = imresize(im, [180 180 / im_size(1) * im_size(2) ]);
    
    % call detect function
    tic;
    boxes = detect_fast(im, model, min(model.thresh,-1.1));
    
    boxes = nms(boxes, .1); % nonmaximal suppression
    
    % save the results
    box_size = size(boxes);
    
    %showboxes(im, boxes(1,:),colorset); % show the best detection
    
    % get the upper body and lower body
    upper = [3, 8, 9, 15, 20, 21];
    lower = [10, 11, 12, 13, 14, 22, 23, 24, 25, 26];
    
    for i_detect = 1: 1: 5
        if i_detect > length(boxes(:, 1))
            % no more detection, write -1
            write_text_results(float_writer, int_writer, ...
                [-1, -1, -1, -1], -1, -1);
            write_text_results(float_writer, int_writer, ...
                [-1, -1, -1, -1], -1, -1);
            continue
        end
        
        % upper body
        xmin = 10000;
        ymin = 10000;
        xmax = 0;
        ymax = 0;
        for iBox = 1: 1: length(upper)
            i = upper(iBox);
            xmin = min(boxes(i_detect, i * 4 - 3), xmin);
            xmax = max(boxes(i_detect, i * 4 - 1), xmax);
            ymin = min(boxes(i_detect, i * 4 - 2), ymin);
            ymax = max(boxes(i_detect, i * 4), ymax);
        end
        write_text_results(float_writer, int_writer, ...
            [xmin, ymin, xmax, ymax] * im_size(1) / 180, ...
            boxes(i_detect, end), 1);
        
        % lower body
        xmin = 1000000;
        ymin = 1000000;
        xmax = 0;
        ymax = 0;
        for iBox = 1: 1: length(lower)
            i = lower(iBox);
            xmin = min(boxes(i_detect, i * 4 - 3), xmin);
            xmax = max(boxes(i_detect, i * 4 - 1), xmax);
            ymin = min(boxes(i_detect, i * 4 - 2), ymin);
            ymax = max(boxes(i_detect, i * 4), ymax);
        end
        
        write_text_results(float_writer, int_writer, ...
            [xmin, ymin, xmax, ymax] * im_size(1) / 180, ...
            boxes(i_detect, end), 2);
    end
    
    % get the frame image
    dettime = toc; % record cpu time
    
    fprintf('detection took %.1f seconds on %d th image %s\n', dettime, i_Image - 2, image_name);
    
end
