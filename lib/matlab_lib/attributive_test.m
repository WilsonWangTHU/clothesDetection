function [number_gt, number_pst_detection, ...
    number_detection, number_recall] = ...
    attributive_test(gt_coordinates, gt_classes, predicted_coordinates, ...
    results_class_cfd, cfd_threshhold, IOU_threshhold, attr_length,...
    dataset_name)

% the results_class_cfd is like this: 
% cls1, cls1_cfd, cls2, cls2_cfd, cls3, cls3_cfd. 
number_gt = zeros(3, 1);
number_pst_detection = zeros(3, 1);
number_detection = zeros(3, 1);
number_recall = zeros(3, 1);

% preprocess, check the size of input. Use map_class to denote class
if strcmp(dataset_name, 'forever21')
    % validate the size of input, two datasets are different
    error('Not implemented yet')
end
if strcmp(dataset_name, 'JD')
    % validate the size of input, two datasets are different
    if length(gt_coordinates) ~= 4
        error('The size is unmatched, JD datasets need a at least 1X4 vec!')
    end
    record_map = zeros(3, 1);
    if ~isempty(gt_classes)
        for i = 1: 1: length(gt_classes)
            % calculate the number of gt, and the number of detection
            if gt_classes(i) <= attr_length(1)
                current_cls = 1;  % the texture
                
            else
                if gt_classes(i) <= attr_length(2) + attr_length(1)
                    current_cls = 2;  % the neckband
                else
                    current_cls = 3;  % the sleeve
                end
            end
            % we still need to take out the detection when there is no 
            % according gt! we use this variables
            record_map(current_cls) = 1;
            % get the number of gt
            number_gt(current_cls) = 1;
            
            % get the number of detection
            detection_id = find(results_class_cfd(:, ...
                2 * current_cls) > cfd_threshhold);
            number_detection(current_cls) = length(detection_id);
            
            if ~isempty(detection_id)
                % get the number of positive detection                
                number_pst_detection(current_cls) = ...
                    IOU_test(gt_coordinates, ...
                    predicted_coordinates(detection_id, :), IOU_threshhold);
            end
                        
            % get the number of recall
            if number_pst_detection(current_cls) ~= 0; 
                number_recall(current_cls) = 1;
            end
        end
    end
    
    % the number_detection for class which has no gt
    for i = 1: 1: length(record_map)
        % this class has no gt, but we still need the detection
        if record_map(i) == 0 
            detection_id = find(results_class_cfd(:, ...
                2 * i) > cfd_threshhold);
            number_detection(i) = length(detection_id);
        end
    end
    
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