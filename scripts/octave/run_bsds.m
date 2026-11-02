% Run the MFE Toolbox's bsds.m on one exported case, under GNU Octave.
%
% usage: octave-cli run_bsds.m CASE_DIR MFE_BOOTSTRAP_DIR SHADOW_DIR
%
% bsds.m stays unedited. For the identical-index run, SHADOW_DIR goes ahead on the path, so its
% stationary_bootstrap.m returns the indices this library used.

args = argv();
case_dir = args{1};
mfe_dir = args{2};
shadow_dir = args{3};

meta = csvread(fullfile(case_dir, 'meta.csv'));
t = meta(1);
k = meta(2);
b = meta(3);
w = meta(4);
seed = meta(5);

fid = fopen(fullfile(case_dir, 'd.bin'), 'rb', 'ieee-le');
d = fread(fid, [t, k], 'double');
fclose(fid);

% Relative performance is a "good", so bsds takes its negative against a zero benchmark.
bench = zeros(t, 1);
models = -d;

addpath(mfe_dir);
rand('twister', seed);
[c, u, l] = bsds(bench, models, b, w, 'STUDENTIZED');
[rc_c, rc_u, rc_l] = bsds(bench, models, b, w, 'STANDARD');

global OLAB_INDICES
fid = fopen(fullfile(case_dir, 'indices.bin'), 'rb', 'ieee-le');
OLAB_INDICES = fread(fid, [t, b], 'int32=>double');
fclose(fid);

addpath(shadow_dir);
rehash();
[same_c, same_u, same_l] = bsds(bench, models, b, w, 'STUDENTIZED');
[~, same_rc_u] = bsds(bench, models, b, w, 'STANDARD');

out = fopen(fullfile(case_dir, 'bsds.csv'), 'w');
fprintf(out, 'c,u,l,rc_u,same_c,same_u,same_l,same_rc_u\n');
fprintf(out, '%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n', ...
        c, u, l, rc_u, same_c, same_u, same_l, same_rc_u);
fclose(out);
