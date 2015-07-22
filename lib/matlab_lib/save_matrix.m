function save_matrix(data, file, type)

fid = fopen(file, 'wb');
row  = size(data ,1);
col  = size(data ,2);
fwrite(fid, row, 'int32');
fwrite(fid, col, 'int32');
fwrite(fid, data', type);
fclose(fid);