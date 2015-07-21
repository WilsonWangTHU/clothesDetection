function data = get_float_text_results(file)

fileID = fopen(file);

% get the number of proposals and dimentions
tline = fgets(fileID);
proposal_num = cell2mat(textscan(tline,'%d\n'));

tline = fgets(fileID);
dimention = cell2mat(textscan(tline,'%d\n'));
if dimention ~= 5
textscan(fileID,'%d')

textscan(fileID,'%f %f %f %f %f');
fclose(fid);