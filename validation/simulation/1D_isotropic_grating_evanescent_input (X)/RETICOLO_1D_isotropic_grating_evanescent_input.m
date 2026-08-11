% Evanescent-input experiment: per-order amplitudes AND output fields.
%
% Same grating as ../1D_isotropic_grating_propagating_input, but the incident wave is
% evanescent: k_parallel = 1.2 > n_incident, so sin(theta) > 1 and the zeroth order is
% past the light line. meent reaches the same state through a complex theta,
% theta = asin(1.2) = pi/2 + 0.622363i.
%
% WHY THIS CASE NEEDS ITS OWN SCRIPT
%
% With an evanescent incident wave the diffraction *efficiency* stops being defined: the
% incident wave carries no z-directed flux, so "fraction of incident power" has a zero
% denominator. meent reports R + T = 4.17 here, which is not a solver error - the quantity
% simply has no meaning. Every efficiency-based check is therefore inapplicable,
% and the coefficients are the only thing left to compare.
%
% That also breaks the usual normalization bridge between the two codes: RETICOLO's
% amplitude is normalized to energy flux, which is degenerate for the incident wave here.
% So this script exports BOTH
%
%   amplitude_TE / amplitude_TM   - the flux-normalized coefficients (as in the other case)
%   E, H                          - output field components at the origin
%
% IMPORTANT: b.E and b.H are built from RETICOLO's normalized modal fields and scattering
% amplitudes. They are not automatically independent of the incident normalization. A valid
% comparison must divide RETICOLO and meent fields by the same nonzero incident-field
% component and express them in the same field basis. The current Python comparison therefore
% does not consume these columns; they are exported as ingredients for that future mapping.
%
% SELF-DIAGNOSIS
%
% A first run reported "201 solved, 0 failed, 0 order-rows written" - res1/res2 succeeded but
% the per-order blocks came back empty and the writer skipped them without saying why. The
% script now probes a single wavelength first. If no diffracted orders come back it dumps what
% res2 actually returned, side by side with a propagating incidence that is known to work, and
% stops instead of grinding through the whole sweep producing nothing.

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..'));
restore_workdir = reticolo_setup(script_dir);

QUICK = false;

if QUICK
    lambda_list = 500.5e-9:20e-9:700.5e-9;
else
    % General-wavelength validation grid, offset from exact modal cutoffs.
    lambda_list = 500.5e-9:1e-9:700.5e-9;
end
N = numel(lambda_list);

period = 1e-6;
n_incident_medium    = 1;
n_transmitted_medium = 1;

% Evanescent incidence. Not n*sin(angle) of a real angle - passed directly, and > 1 on
% purpose. res1 feeds it to retinit as the Bloch wavevector, which has no <= 1 restriction.
k_parallel   = 1.2;
angle_delta0 = -30;      % meent phi = +30

% Reference incidence used only by the diagnostic dump: a normal propagating angle, the
% configuration the companion case already validates.
k_parallel_ref = 0.5;

nn = 100;

n_grating = 2.0;
n_slab    = 2.0;

xb = 250e-9:250e-9:1000e-9;
L1 = [1, n_grating, n_grating, 1];
L2 = [n_slab, n_slab, n_slab, n_slab];

textures = {n_incident_medium, {xb, L1}, {xb, L2}, n_transmitted_medium};
profile  = {[0, 500e-9, 500e-9, 0], [1, 2, 3, 4]};

POLS = {'TE', 'TM'};
DIRS = {'r', 't'};
NP = numel(POLS);
ND = numel(DIRS);
BLOCK_NAMES = {'TEinc_top_reflected', 'TEinc_top_transmitted', ...
               'TMinc_top_reflected', 'TMinc_top_transmitted'};

parm = res0;
parm.res1.champ = 1;

fprintf(['k_parallel = %g (EVANESCENT, > n_incident = %g), delta = %g deg, nn = %d, ' ...
    '%d wavelengths (%.6g-%.6g m)\n'], k_parallel, n_incident_medium, angle_delta0, ...
    nn, N, lambda_list(1), lambda_list(end));

%% ------------------------------------------------------------------ probe one wavelength

probe_wl = lambda_list(round(N/2));
fprintf('\nProbing one wavelength (%.6g m) before the sweep...\n', probe_wl);

aa = res1(probe_wl, period, textures, nn, k_parallel, angle_delta0, parm);
ef = res2(aa, profile);

n_orders_found = 0;
for ib = 1:numel(BLOCK_NAMES)
    b = get_block(ef, BLOCK_NAMES{ib});
    if ~isempty(b) && isfield(b, 'order') && ~isempty(b.order)
        n_orders_found = n_orders_found + numel(b.order(:, 1));
    end
end

if n_orders_found == 0
    fprintf(['\nNo diffracted orders came back. Dumping what res2 returned, next to a\n' ...
             'propagating incidence for contrast. Nothing is written; the sweep is skipped.\n']);

    for kp = [k_parallel_ref, k_parallel]
        fprintf('\n===============================================================\n');
        if kp <= n_incident_medium
            fprintf('k_parallel = %g  (PROPAGATING - known to work)\n', kp);
        else
            fprintf('k_parallel = %g  (EVANESCENT - the case that produced nothing)\n', kp);
        end
        fprintf('===============================================================\n');

        aa_d = res1(probe_wl, period, textures, nn, kp, angle_delta0, parm);
        ef_d = res2(aa_d, profile);

        fprintf('class(ef) = %s\n', class(ef_d));
        try
            fn = fieldnames(ef_d);
            fprintf('fieldnames(ef), %d total:\n', numel(fn));
            for i = 1:numel(fn)
                fprintf('    %s\n', fn{i});
            end
        catch
            fprintf('    (fieldnames unavailable for this class)\n');
        end

        for ib = 1:numel(BLOCK_NAMES)
            block_name = BLOCK_NAMES{ib};
            fprintf('\n--- ef.%s ---\n', block_name);
            b = get_block(ef_d, block_name);
            if isempty(b)
                fprintf('    MISSING or EMPTY\n');
                continue
            end
            bf = fieldnames(b);
            for j = 1:numel(bf)
                v = b.(bf{j});
                fprintf('      %-18s size %-12s %s\n', bf{j}, mat2str(size(v)), class(v));
            end
            if isfield(b, 'order') && ~isempty(b.order)
                fprintf('    order values:\n');
                disp(b.order.');
            end
            if isfield(b, 'efficiency') && ~isempty(b.efficiency)
                fprintf('    sum(efficiency) = %g\n', sum(b.efficiency));
            end
        end
    end

    fprintf(['\n\nHow to read this:\n' ...
             '  - Different fieldnames between the two runs  -> this script reads the wrong\n' ...
             '    names for evanescent incidence and can simply be pointed at the right ones.\n' ...
             '  - `order` non-empty at k_parallel = %g but empty at %g  -> RETICOLO declines\n' ...
             '    to define diffracted orders for an evanescent incident wave. That matches\n' ...
             '    the reason the Python side blocks the comparison: with no incident flux\n' ...
             '    there is nothing to normalize against. The total-internal-reflection setup\n' ...
             '    is then the way to validate the same physics.\n'], ...
             k_parallel_ref, k_parallel);
    retio;
    return
end

fprintf('  %d diffracted orders found - running the full sweep.\n', n_orders_found);

%% ------------------------------------------------------------------------- the full sweep

out_files = cell(NP, ND);
for ip = 1:NP
    for id = 1:ND
        out_files{ip, id} = sprintf( ...
            'RETICOLO_1D_isotropic_grating_evanescent_input_%s_%s.txt', POLS{ip}, DIRS{id});
        fid = fopen(out_files{ip, id}, 'w');
        fprintf(fid, ['wavelength_m, order_x, order_y, te_re, te_im, tm_re, tm_im, efficiency, ' ...
                      'Ex_re, Ex_im, Ey_re, Ey_im, Ez_re, Ez_im, ' ...
                      'Hx_re, Hx_im, Hy_re, Hy_im, Hz_re, Hz_im\n']);
        fclose(fid);
    end
end

n_ok = 0; n_fail = 0; n_rows = 0; n_empty = 0;

for i = 1:N
    wavelength = lambda_list(i);

    try
        aa = res1(wavelength, period, textures, nn, k_parallel, angle_delta0, parm);
        ef = res2(aa, profile);

        blocks = {get_block(ef, 'TEinc_top_reflected'), get_block(ef, 'TEinc_top_transmitted'); ...
                  get_block(ef, 'TMinc_top_reflected'), get_block(ef, 'TMinc_top_transmitted')};

        for ip = 1:NP
            for id = 1:ND
                b = blocks{ip, id};
                % Never skip silently - the first version of this script wrote zero rows
                % while still reporting "201 solved, 0 failed" because this branch said
                % nothing.
                if isempty(b) || ~isfield(b, 'order') || isempty(b.order)
                    if i == 1
                        fprintf(['  NO ORDERS: %s %s - res2 returned no diffracted orders ' ...
                                 'for this incidence\n'], POLS{ip}, DIRS{id});
                    end
                    n_empty = n_empty + 1;
                    continue
                end
                if ~isfield(b, 'amplitude_TE') || ~isfield(b, 'amplitude_TM')
                    error(['res2 returned no amplitude_TE/amplitude_TM - the conical ' ...
                           '(2D-form) output is required here.']);
                end
                if ~isfield(b, 'E') || ~isfield(b, 'H')
                    error(['res2 returned no E/H. parm.res1.champ = 1 is required so the ' ...
                           'fields are computed.']);
                end

                ord = b.order;
                if size(ord, 2) < 2
                    ord = [ord(:), zeros(size(ord, 1), 1)];
                end

                ampTE = b.amplitude_TE(:);
                ampTM = b.amplitude_TM(:);
                effic = b.efficiency(:);
                E = b.E;    % n x 3
                H = b.H;    % n x 3

                fid = fopen(out_files{ip, id}, 'a');
                for k = 1:numel(ampTE)
                    fprintf(fid, ['%.17g, %d, %d, %.17g, %.17g, %.17g, %.17g, %.17g, ' ...
                                  '%.17g, %.17g, %.17g, %.17g, %.17g, %.17g, ' ...
                                  '%.17g, %.17g, %.17g, %.17g, %.17g, %.17g\n'], ...
                        wavelength, round(ord(k, 1)), round(ord(k, 2)), ...
                        real(ampTE(k)), imag(ampTE(k)), ...
                        real(ampTM(k)), imag(ampTM(k)), effic(k), ...
                        real(E(k,1)), imag(E(k,1)), real(E(k,2)), imag(E(k,2)), ...
                        real(E(k,3)), imag(E(k,3)), ...
                        real(H(k,1)), imag(H(k,1)), real(H(k,2)), imag(H(k,2)), ...
                        real(H(k,3)), imag(H(k,3)));
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

retio;
fprintf('\n%d solved, %d failed, %d order-rows written, %d empty blocks\n', ...
    n_ok, n_fail, n_rows, n_empty);
for ip = 1:NP
    for id = 1:ND
        fprintf('Saved: %s\n', out_files{ip, id});
    end
end


%% -------------------------------------------------------------------------------- helper

function b = get_block(ef, name)
% Fetch one per-order block, tolerating both the struct and the class form res2 can return.
b = [];
try
    if isstruct(ef)
        if isfield(ef, name)
            b = ef.(name);
        end
    else
        b = ef.(name);          % reticolo class - property access
    end
catch
    b = [];
end
end
