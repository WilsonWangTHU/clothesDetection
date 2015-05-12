path = '/home/wtw/fast-rcnn/lib/selective_search_ijcv_with_python/1/image_50'
image_path = dir(path)
boxes = {}
for i = 1: length(image_path)
    if image_path(i).isdir == 1
        continue
    end
    im = imread([path '/' image_path(i).name]);
    
    boxes{i - 2} = selective_search_boxes(im, true);
    
end
