function [gt_position, class]= read_cloth_xml(xml_file, class_map)
    xml_struct = xml2struct(xml_file);
    
    % read the cloth type
    child = xml_struct.Children;
    type = child(2).Attributes.Value;
    class = find(strcmp(class_map, type));
    
    % get the positions
    y1 = str2double(child(2).Children(2).Attributes(2).Value);
    x1 = str2double(child(2).Children(2).Attributes(3).Value);
    x2 = str2double(child(2).Children(2).Attributes(4).Value);
    y2 = str2double(child(2).Children(2).Attributes(5).Value);
    
    gt_position = zeros(1, 4);
    gt_position(1) = min(x1, x2);
    gt_position(2) = min(y1, y2);
    gt_position(3) = max(x1, x2);
    gt_position(4) = max(y1, y2);

end