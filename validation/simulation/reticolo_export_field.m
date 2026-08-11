function out_file = reticolo_export_field(case_name, wavelength_m, period, textures, ...
    nn, k_parallel, angle_delta0, parm, profile_field, res_x, res_y, res_z)
%RETICOLO_EXPORT_FIELD Export one-wavelength TE/TM fields for Meent comparison.
%
% profile_field contains four finite layers: top buffer, grating, slab, bottom
% buffer.  The buffers have the same refractive index as the adjacent half-spaces,
% so they expose the external fields without introducing a physical interface.

arguments
    case_name (1,:) char
    wavelength_m (1,1) double {mustBePositive}
    period double
    textures cell
    nn double
    k_parallel (1,1) double
    angle_delta0 (1,1) double
    parm struct
    profile_field cell
    res_x (1,1) double {mustBeInteger,mustBeGreaterThan(res_x,1)}
    res_y (1,1) double {mustBeInteger,mustBePositive}
    res_z (1,1) double {mustBeInteger,mustBeGreaterThan(res_z,1)}
end

parm.res1.champ = 1;
parm.res3.trace = 0;
% gauss = 0 puts the samples on linspace(0, h, npts) within each layer, endpoints
% included, which is exactly the grid Meent's calculate_field uses. Do not change it
% without changing _field_compare._z_coordinates with it.
parm.res3.gauss = 0;
% Scalar npts, applied to every layer of profile_field, so the returned z has
% numel(profile_field{1}) * res_z points. The Python side checks that count and the
% sample positions; see _field_compare.check_grid.
parm.res3.npts = res_z;

aa = res1(wavelength_m, period, textures, nn, k_parallel, angle_delta0, parm);
x = linspace(0, period(1), res_x);

if isscalar(period)
    y = 0;
    [field_TE, z_te] = res3(x, aa, profile_field, [0, 1], parm);
    [field_TM, z_tm] = res3(x, aa, profile_field, [1, 0], parm);
else
    % Meent stores its y grid from period_y down to zero.
    y = linspace(period(2), 0, res_y);
    [field_TE, z_te] = res3(x, y, aa, profile_field, [0, 1], parm);
    [field_TM, z_tm] = res3(x, y, aa, profile_field, [1, 0], parm);
end

if isempty(field_TE) || isempty(field_TM)
    error('RETICOLO res3 returned an empty field for %s at %.12g m.', ...
        case_name, wavelength_m);
end
if ~isequal(size(z_te), size(z_tm)) || any(abs(z_te(:) - z_tm(:)) > 1e-12)
    error('TE and TM res3 calls returned different z grids.');
end

z = z_te;
profile_heights = profile_field{1};
field_components = {'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'};
% These values are RETICOLO's own and are never altered downstream. The Python loader
% only reorders array axes; all convention conversion happens on the Meent side, inside
% RCWATorch.calculate_field. See _field_compare.load_reticolo_fields.
time_convention = 'RETICOLO raw; stored values are the reference and are not converted';
wl_tag = strrep(sprintf('%.12g', wavelength_m), '.', 'p');
out_file = fullfile(pwd, sprintf('RETICOLO_%s_field_%sm.mat', case_name, wl_tag));

save(out_file, 'case_name', 'wavelength_m', 'period', 'profile_heights', ...
    'res_x', 'res_y', 'res_z', 'x', 'y', 'z', 'field_TE', 'field_TM', ...
    'field_components', 'time_convention', '-v7');
fprintf('Saved %s\n', out_file);
fprintf('  TE size: %s, TM size: %s, z points: %d\n', ...
    mat2str(size(field_TE)), mat2str(size(field_TM)), numel(z));
end
