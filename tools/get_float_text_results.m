function result = get_float_text_results(file)

fileID = fopen(file);

% get the number of proposals and dimentions
tline = fgets(fileID);
proposal_num = cell2mat(textscan(tline,'%d\n'));

tline = fgets(fileID);
dimention = cell2mat(textscan(tline,'%d\n'));
if dimention ~= 5
    printf('There is a mismatch of dimention! Error!\n')
    exit
end

% read the results of the test
result = zeros(proposal_num, dimention);
for i = 1: 1: proposal_num
    tline = fgets(fileID);
    result(i, :) = cell2mat(textscan(tline,'%f %f %f %f %f'));
end
fclose(fileID);