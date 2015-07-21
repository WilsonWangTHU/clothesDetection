function data = read_text_matrix(file)

fileID = fopen(file, 'r');
textscan(fileID,'%d')

textscan(fileID,'%d')

textscan(fileID,'%f %f %f %f %f');
fclose(fid);