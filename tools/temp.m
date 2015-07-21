

float_ext = ['_floatResults'];
int_ext = ['_intResults'];
[mat_file_path] = fileparts(mfilename('fullpath'));
rcnn_forever21_result_dir = [mat_file_path '/../data/results/forever21'];

a = ([rcnn_forever21_result_dir '/' '0001.jpg' float_ext]);
fid = fopen(a);

tline = fgets(fid);
while ischar(tline)
    disp(tline)
    tline = fgets(fid);
end

fclose(fid);
