function [gt_position, type_class, multi_label_class] = ...
    read_cloth_xml(xml_file, class_map, ...
    texture_classes_map, neckband_classes_map, sleeve_classes_map, ...
    multi_label_test)

gt_position = zeros(1, 4);
type_class = 0;
multi_label_class = zeros(length(texture_classes_map) + ...
    length(neckband_classes_map) + length(sleeve_classes_map), 1);

if ~multi_label_test
    xml_struct = xml2struct(xml_file);
    
    child = xml_struct.Children;
    
    % read the cloth type
    if ~strcmp(child(2).name, 'clothClass')
        error('The format of the xml file is wrong here at %s', xml_file)
    end
    type = child(2).Attributes.Value;
    type_class = find(strcmp(class_map, type));
    
    % get the positions
    y1 = str2double(child(2).Children(2).Attributes(2).Value);
    x1 = str2double(child(2).Children(2).Attributes(3).Value);
    x2 = str2double(child(2).Children(2).Attributes(4).Value);
    y2 = str2double(child(2).Children(2).Attributes(5).Value);
    
    gt_position(1) = min(x1, x2);
    gt_position(2) = min(y1, y2);
    gt_position(3) = max(x1, x2);
    gt_position(4) = max(y1, y2);
else
    xml_struct = xml2struct(xml_file);
    child = xml_struct.Children;
    
    for i_child = 1: 1: length(child)
        switch child(i_child).Name
            case '#text', continue;
            case 'clothClass',
                type = child(i_child).Attributes.Value;
                type_class = find(strcmp(class_map, type));
                
                % get the positions
                y1 = str2double(child(i_child).Children(2).Attributes(2).Value);
                x1 = str2double(child(i_child).Children(2).Attributes(3).Value);
                x2 = str2double(child(i_child).Children(2).Attributes(4).Value);
                y2 = str2double(child(i_child).Children(2).Attributes(5).Value);
                
                gt_position(1) = min(x1, x2);
                gt_position(2) = min(y1, y2);
                gt_position(3) = max(x1, x2);
                gt_position(4) = max(y1, y2);
            case 'clothTexture',
                texture_id = find(strcmp(texture_classes_map, ...
                    child(i_child).Attributes.Value), 1);
                if isempty(texture_id); continue; end
                multi_label_class(texture_id) = 1;
            case 'clothNeckband',
                texture_id = find(strcmp(neckband_classes_map, ...
                    child(i_child).Attributes.Value), 1);
                if isempty(texture_id); continue; end
                multi_label_class(texture_id + ...
                    length(texture_classes_map)) = 1;
            case 'clothSleeve', 
                texture_id = find(strcmp(sleeve_classes_map, ...
                    child(i_child).Attributes.Value), 1);
                if isempty(texture_id); continue; end
                multi_label_class(texture_id + ...
                    length(neckband_classes_map) + ...
                    length(texture_classes_map)) = 1;
        end
    end
    
end
end