function result = get_float_text_results(file, multi_label_test)

fileID = fopen(file);

% get the number of proposals and dimentions
tline = fgets(fileID);
proposal_num = cell2mat(textscan(tline,'%d\n'));

tline = fgets(fileID);
dimention = cell2mat(textscan(tline,'%d\n'));

if dimention ~= 5 && multi_label_test == false
    fprintf('The dimension is %d, but we need 5\n', dimention);
    error('There is a mismatch of dimention! Error!\n')
end

if dimention ~= 5 + 22 && multi_label_test
    fprintf('The dimension is %d, but we need 27\n', dimention);
    error('There is a mismatch of dimention! Error!\n')
end

% read the results of the test
result = zeros(proposal_num, dimention);
for i = 1: 1: proposal_num
    tline = fgets(fileID);
    if multi_label_test
        reading_para = [repmat('%f ', 1, 26) '%f\n'];
        result(i, :) = cell2mat(textscan(tline, reading_para));
    else
        result(i, :) = cell2mat(textscan(tline, '%f %f %f %f %f\n'));
    end
end
fclose(fileID);