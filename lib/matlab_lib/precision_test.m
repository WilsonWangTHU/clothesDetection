function [number_gt, number_pst_detection, ...
    number_detection, number_recall] = ...
    precision_test(gt_coordinates, results, results_cls, ...
    cfd_threshhold, IOU_threshhold, dataset_name)

number_gt = 0;
number_pst_detection = 0;
number_detection = 0;
number_recall = 0;


% preprocess, check the size of input. Use map_class to denote class
if strcmp(dataset_name, 'forever21')
    % validate the size of input, two datasets are different
    co_size = size(gt_coordinates);
    if co_size(1) ~= 3
        error('The size is unmatched, Upper, Lower1, Lower2 means 3!')
    end
    
    map_class = [1, 2, 2];  % it is a mapping, as the 2, 3 are both in class 2
    for i = 1: 1: co_size(1)
        cur_gt_coordinates = zeros(4, 1);
        cur_gt_coordinates(:) = gt_coordinates(i, :);
        
        index = find(results_cls == map_class(i));      % get the index of
        index = index(results(index, 5) > cfd_threshhold);  % the according class
        
        % accumulate the number of total test, note that cls 2,3 are the same
        if i ~= 3; number_detection = number_detection + length(index); end
        
        % no gt of this class in this image
        if cur_gt_coordinates(1) >= cur_gt_coordinates(3) || ...
                cur_gt_coordinates(2) >= cur_gt_coordinates(4)
            continue
        end
        
        % test for this class
        number_gt = number_gt + 1;
        
        this_class_number_pst_detection = IOU_test( ...
            cur_gt_coordinates, results(index, :), IOU_threshhold);
        number_pst_detection = number_pst_detection + ...
            this_class_number_pst_detection;
        
        % if this gt is detected at least once as a validate proposals, the
        % recall is added by one
        if this_class_number_pst_detection > 0
            number_recall = number_recall + 1;
        end
    end
else
    if strcmp(dataset_name, 'JD')
        % validate the size of input, two datasets are different
        co_size = size(gt_coordinates);
        if co_size(1) ~= 1 || co_size(2) ~= 5
            error('The size is unmatched, JD datasets need 1X6 vec !')
        end
        % calculate the number of gt
        number_gt = 1;
        
        % calculate the number of detection
        index = find(results_cls ~= -1);  % get the index of positive class
        index = index(results(index, 5) > cfd_threshhold);
        number_detection = length(index);
        
        % calculate the number of positive detection
        index = find(results_cls == gt_coordinates(5));  % get gt class
        index = index(results(index, 5) > cfd_threshhold);
        
        number_pst_detection = IOU_test( ...
            gt_coordinates(1:4), results(index, :), IOU_threshhold);
        if number_pst_detection == 1; number_recall = 1; end
    end
end
% now the real part after the preprocess
end

function pst_detection = IOU_test( ...
    coordinate, det_coordinate, IOU_threshhold)

pst_detection = 0;
gt_size = (coordinate(4) - coordinate(2)) * (coordinate(3) - coordinate(1));

for i_coord = 1: 1: length(det_coordinate(:, 1))
    cur_coordinates = zeros(4, 1);  % process the detection coord one by
    % one
    cur_coordinates(:) = det_coordinate(i_coord, 1 : 4);
    
    det_size = (cur_coordinates(4) - cur_coordinates(2)) * ...
        (cur_coordinates(3) - cur_coordinates(1));
    
    overlaps = ...
        (min(cur_coordinates(4), coordinate(4)) - ...
        max(cur_coordinates(2), coordinate(2))) * ...
        (min(cur_coordinates(3), coordinate(3)) - ...
        max(cur_coordinates(1), coordinate(1)));
    if overlaps / (gt_size + det_size - overlaps) > IOU_threshhold
        pst_detection = pst_detection + 1;
    end
end

end
