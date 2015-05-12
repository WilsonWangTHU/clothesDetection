function get_all_proposals(path, train_test_flag)

if ~exist('train_test_flag', 'var') || isempty(train_test_flag) || train_test_flag == false
    % simply process the main directory
    boxes = get_proposal_by_path(path);
else
    boxesTrain = get_proposal_by_path([path '/train']);
    boxesTest = get_proposal_by_path([path '/test']);
end


function boxes = get_proposal_by_path(path)

image_path = dir(path);
boxes = cell(1, length(image_path) - 2);

for i = 1: length(image_path)
    if image_path(i).isdir == 1
        continue
    end
    im = imread([path '/' image_path(i).name]);
    
    boxes{i - 2} = selective_search_boxes(im, true);
end
fprintf('saving the proposal mat at %s', [path '/../' 'boxes.mat'])
save([path '/../' 'boxes.mat'], 'boxes')
