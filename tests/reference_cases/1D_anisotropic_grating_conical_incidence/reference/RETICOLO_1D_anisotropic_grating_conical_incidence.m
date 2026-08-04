addpath(genpath('E:\Taesang Yun\reticolo\reticolo'));

QUICK = false;

if QUICK
    lambda_list = (500:20:700)*1e-9;
else
    lambda_list = (500:1:700)*1e-9;
end
N = numel(lambda_list);

period = 1e-6;
n_incident_medium    = 1;
n_transmitted_medium = 1;

% Conical mount: theta out of normal AND delta out of the x-z plane, so the
% incident wave has a k_y component.  A 1D grating under conical incidence
% couples the two polarizations, so res1 is called in its full conical form
% (k_parallel + angle_delta) and res2 returns both TEinc and TMinc from a single
% solve - there is no res0(pol) scalar path here.
angle_theta0 = 30;
angle_delta0 = -30;
k_parallel = n_incident_medium * sin(angle_theta0*pi/180);

nn = 100;

n_x = 1.5;
n_y = 2.0;
n_z = 2.5;

% A 1D texture takes a scalar index, so the anisotropic cells are drawn with a
% placeholder and res1 swaps in the diagonal tensor through
% parm.res1.change_index = {[n_placeholder, nx, ny, nz]}.
n_placeholder = 89.99999;

xb = (250:250:1000)*1e-9;
L1 = [1, n_placeholder, n_placeholder, 1];
L2 = [n_placeholder, n_placeholder, n_placeholder, n_placeholder];

textures = {n_incident_medium, {xb, L1}, {xb, L2}, n_transmitted_medium};

profile = {[0, 500e-9, 500e-9, 0], [1, 2, 3, 4]};

parm = res0;
parm.res1.champ = 1;
parm.res1.change_index = { [n_placeholder, n_x, n_y, n_z] };

POLS = {'TE', 'TM'};
NP = numel(POLS);

out_files = cell(1, NP);
for ip = 1:NP
    out_files{ip} = sprintf( ...
        'RETICOLO_1D_anisotropic_grating_conical_incidence_%s.txt', POLS{ip});
    fid = fopen(out_files{ip}, 'w');
    fprintf(fid, 'lambda_nm, R, T\n');
    fclose(fid);
end

fprintf(['theta = %g deg, delta = %g deg (k_parallel = %.6f), ' ...
    '(nx, ny, nz) = (%g, %g, %g), nn = %d, %d wavelengths (%.0f-%.0f nm)\n'], ...
    angle_theta0, angle_delta0, k_parallel, n_x, n_y, n_z, nn, N, ...
    lambda_list(1)*1e9, lambda_list(end)*1e9);

n_ok = 0; n_fail = 0;

for i = 1:N
    wavelength = lambda_list(i);

    R = nan(1, NP);
    T = nan(1, NP);

    try
        aa = res1(wavelength, period, textures, nn, k_parallel, ...
            angle_delta0, parm);
        ef = res2(aa, profile);

        eff = {ef.TEinc_top_reflected.efficiency, ...
               ef.TEinc_top_transmitted.efficiency; ...
               ef.TMinc_top_reflected.efficiency, ...
               ef.TMinc_top_transmitted.efficiency};

        for ip = 1:NP
            if ~isempty(eff{ip, 1}); R(ip) = sum(eff{ip, 1}); end
            if ~isempty(eff{ip, 2}); T(ip) = sum(eff{ip, 2}); end
        end
        n_ok = n_ok + 1;
    catch err
        fprintf('  FAILED lambda=%dnm : %s\n', ...
            round(wavelength*1e9), err.message);
        n_fail = n_fail + 1;
    end

    for ip = 1:NP
        fid = fopen(out_files{ip}, 'a');
        fprintf(fid, '%.17g, %.17g, %.17g\n', wavelength*1e9, R(ip), T(ip));
        fclose(fid);
    end
end

retio;
fprintf('\n%d solved, %d failed\n', n_ok, n_fail);
fprintf('Saved: %s , %s\n', out_files{1}, out_files{2});
