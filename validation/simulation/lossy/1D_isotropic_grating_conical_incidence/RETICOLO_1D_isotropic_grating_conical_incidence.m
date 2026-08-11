% Complex r/t amplitude coefficients and single-wavelength spatial fields for 1D_isotropic_grating_conical_incidence.
%
% Exports the diffracted AMPLITUDES per order, not the summed efficiencies. res2 already
% returns amplitude_TE / amplitude_TM per order; an efficiency-only script never reads them.
%
% Normalization: RETICOLO normalizes amplitude to the energy flux through the xy plane, so
% efficiency = abs(amplitude).^2. meent keeps that factor outside its amplitude, so the Python
% side multiplies by sqrt(Re(kz/(n_top*cos(theta)))) before comparing. See _compare.py.

script_dir = fileparts(mfilename('fullpath'));
% reticolo_setup.m lives at the simulation root; find it without counting levels.
d = script_dir;
while ~isfile(fullfile(d, 'reticolo_setup.m'))
    parent = fileparts(d);
    if strcmp(parent, d); error('reticolo_setup.m not found above %s', script_dir); end
    d = parent;
end
addpath(d);
restore_workdir = reticolo_setup(script_dir);

QUICK = false;

if QUICK
    lambda_list = 500.5e-9:20e-9:700.5e-9;
else
    % General-wavelength validation grid, offset from exact modal cutoffs.
    lambda_list = 500.5e-9:1e-9:700.5e-9;
end
N = numel(lambda_list);

n_incident_medium    = 1;
n_transmitted_medium = 1;

angle_theta0 = 30.0;
angle_delta0 = -30.0;      % meent phi = +30
k_parallel   = n_incident_medium * sin(angle_theta0*pi/180);

nn = 100;

% Lossy: the indices below are the CONJUGATE of the meent ones. meent absorbs
% with n - i*k, RETICOLO with n + i*k. Getting this backwards silently turns
% the medium into a gain medium and R + T exceeds 1.
period = 1e-6;

n_grating = 2 + 0.1i;
n_slab    = 2 + 0.1i;

xb = 250e-9:250e-9:1000e-9;
L1 = [1, n_grating, n_grating, 1];
L2 = [n_slab, n_slab, n_slab, n_slab];

textures = {n_incident_medium, {xb, L1}, {xb, L2}, n_transmitted_medium};
profile  = {[0, 500e-9, 500e-9, 0], [1, 2, 3, 4]};

POLS = {'TE', 'TM'};
DIRS = {'r', 't'};
NP = numel(POLS);
ND = numel(DIRS);

parm = res0;
parm.res1.champ = 1;

RUN_COMPLEX_COEFFICIENTS = true;
RUN_FIELD_PROFILE = true;

if RUN_COMPLEX_COEFFICIENTS
out_files = cell(NP, ND);
for ip = 1:NP
    for id = 1:ND
        out_files{ip, id} = sprintf('RETICOLO_1D_isotropic_grating_conical_incidence_%s_%s.txt', POLS{ip}, DIRS{id});
        fid = fopen(out_files{ip, id}, 'w');
        fprintf(fid, 'wavelength_m, order_x, order_y, te_re, te_im, tm_re, tm_im, efficiency\n');
        fclose(fid);
    end
end

fprintf(['theta = %g deg, delta = %g deg (k_parallel = %.6f), nn = %s, ' ...
    '%d wavelengths (%.6g-%.6g m)\n'], angle_theta0, angle_delta0, k_parallel, ...
    mat2str(nn), N, lambda_list(1), lambda_list(end));

n_ok = 0; n_fail = 0; n_rows = 0; n_empty = 0;

for i = 1:N
    wavelength = lambda_list(i);

    try
        aa = res1(wavelength, period, textures, nn, k_parallel, angle_delta0, parm);
        ef = res2(aa, profile);

        blocks = {ef.TEinc_top_reflected, ef.TEinc_top_transmitted; ...
                  ef.TMinc_top_reflected, ef.TMinc_top_transmitted};

        for ip = 1:NP
            for id = 1:ND
                b = blocks{ip, id};
                % Never skip silently: a block that is empty for an unexpected reason must
                % say so rather than leaving a short file that looks like agreement.
                if isempty(b) || ~isfield(b, 'order') || isempty(b.order)
                    if i == 1
                        fprintf('  NO ORDERS: %s %s\n', POLS{ip}, DIRS{id});
                    end
                    n_empty = n_empty + 1;
                    continue
                end
                if ~isfield(b, 'amplitude_TE') || ~isfield(b, 'amplitude_TM')
                    error('res2 returned no amplitude_TE/amplitude_TM.');
                end

                ord = b.order;
                if size(ord, 2) < 2
                    ord = [ord(:), zeros(size(ord, 1), 1)];
                end
                ampTE = b.amplitude_TE(:);
                ampTM = b.amplitude_TM(:);
                effic = b.efficiency(:);

                fid = fopen(out_files{ip, id}, 'a');
                for k = 1:numel(ampTE)
                    fprintf(fid, '%.17g, %d, %d, %.17g, %.17g, %.17g, %.17g, %.17g\n', ...
                        wavelength, round(ord(k, 1)), round(ord(k, 2)), ...
                        real(ampTE(k)), imag(ampTE(k)), ...
                        real(ampTM(k)), imag(ampTM(k)), effic(k));
                    n_rows = n_rows + 1;
                end
                fclose(fid);
            end
        end
        n_ok = n_ok + 1;
    catch err
        fprintf('  FAILED wavelength=%.6g m : %s\n', wavelength, err.message);
        n_fail = n_fail + 1;
    end
end

fprintf('\n%d solved, %d failed, %d order-rows written, %d empty blocks\n', ...
    n_ok, n_fail, n_rows, n_empty);
for ip = 1:NP
    for id = 1:ND
        fprintf('Saved: %s\n', out_files{ip, id});
    end
end

end  % RUN_COMPLEX_COEFFICIENTS

if RUN_FIELD_PROFILE
    FIELD_WAVELENGTH_M = 600e-9;
    request_file = fullfile(script_dir, 'field_wavelength_m.txt');
    if isfile(request_file)
        requested_wavelength = str2double(strtrim(fileread(request_file)));
        if ~isscalar(requested_wavelength) || ~isfinite(requested_wavelength) || requested_wavelength <= 0
            error('Invalid wavelength in %s.', request_file);
        end
        FIELD_WAVELENGTH_M = requested_wavelength;
    end
    FIELD_BUFFER_TOP = 500e-9;
    FIELD_BUFFER_BOTTOM = 500e-9;
    FIELD_RES_Z_PER_LAYER = 31;
    if isscalar(period)
        FIELD_RES_X = 81;
        FIELD_RES_Y = 1;
    else
        FIELD_RES_X = 41;
        FIELD_RES_Y = 33;
    end

    field_profile = {[FIELD_BUFFER_TOP, profile{1}(2:end-1), FIELD_BUFFER_BOTTOM], profile{2}};
    reticolo_export_field('1D_isotropic_grating_conical_incidence', FIELD_WAVELENGTH_M, period, textures, ...
        nn, k_parallel, angle_delta0, parm, field_profile, ...
        FIELD_RES_X, FIELD_RES_Y, FIELD_RES_Z_PER_LAYER);
end  % RUN_FIELD_PROFILE

retio;
