% -------------------------------------------------------------
%   In this file, we generate the bounding box from the pixel
%   annotation.
%   The boxes are then saved for future uses
%
%   Written by Tingwu Wang, 21.7.2015
% -------------------------------------------------------------

function coordinates = get_boundingbox(index, pixel_file, the_data_flag)
debug = 0;  % set this on to see the different parts for debugging
if ~exist('the_data_flag', 'var')
    the_data_flag = false;
end

if the_data_flag
    groundtruth = pixel_file;
else
    load(pixel_file)  % the groud truth file is written as: groundtruth
end

% the overall box is defined by this 4 variables
xmax = 0;
ymax = 0;
xmin = 10000;
ymin = 10000;

for i_index = 1: 1: length(index)
    [rows, cols] = find(groundtruth == index(i_index) - 1);
    
    if debug == 1
        xmax = 0;
        ymax = 0;
        xmin = 10000;
        ymin = 10000;
        load('label_list.mat')
    end
    
    if debug == 1
        if length(cols) >= 2
            xmax = max(max(cols), xmax);
            xmin = min(min(cols), xmin);
        end
        if length(rows) >= 2
            ymax = max(max(rows), ymax);
            ymin = min(min(rows), ymin);
        end
    else
        if length(cols) >= 2
            xmax = max(max(cols), xmax);
            xmin = min(min(cols), xmin);
        end
        if length(rows) >= 2
            ymax = max(max(rows), ymax);
            ymin = min(min(rows), ymin);
        end
    end
    
    if debug == 1 && length(rows) >= 2 && length(cols) >= 2
        xmin, ymin, xmax, ymax
        fprintf('the class is %s\n', label_list{index(i_index)})
        
        
        rectangle('Position', [xmin ymin xmax-xmin ymax-ymin], ...
            'LineWidth', 3, 'EdgeColor','r');
    end
end
coordinates = [xmin, ymin, xmax, ymax];
