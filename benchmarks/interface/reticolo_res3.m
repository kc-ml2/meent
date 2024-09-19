%function [top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, e] = reticolo_res3(_pol,
function [top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm, bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, e] = reticolo_res3(_pol,
%function [res] = reticolo_res3(_pol,
    theta, phi, period, n_inc, nn, _textures, _profile, wavelength, grating_type, matlab_plot_field, res3_npts);
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

warning('off', 'Octave:possible-matlab-short-circuit-operator');
warning('off', 'Invalid UTF-8 byte sequences have been replaced.');
warning('off', 'findstr is obsolete; use strfind instead');

k_parallel = n_inc*sin(theta); % n_air, or whatever the refractive index of the medium where light is coming in.

if _pol == 0  # TE
    pol = 1;
else  # TM
    pol = -1;
end

if grating_type == 0
    parm = res0(pol); % TE pol. For TM : parm=res0(-1)
else
    parm = res0;
end
parm.not_io = 1;  % no write data on hard disk
parm.res1.champ = 1; % the electromagnetic field is calculated accurately
%parm.res1.trace = 1; % show the texture

textures = cell(1, size(_textures, 2));
for i = 1:length(_textures)
    textures(i) = _textures(i);
end

profile = cell(1, size(_profile, 1));
profile(1) = _profile(1, :);
profile(2) = _profile(2, :);

if grating_type == 0
    aa = res1(wavelength,period,textures,nn,k_parallel,parm);
    res = res2(aa, profile);
else
    aa = res1(wavelength,period,textures,nn,k_parallel, phi, parm);
    res = res2(aa, profile);
end

x = linspace(0, period(1), 11);

%parm.res3.sens=1;
%##parm.res3.gauss_x = 100
if matlab_plot_field == 1
    parm.res3.trace=1 ;%trace automatique % automatic trace
else
    parm.res3.trace=0;
end

if res3_npts ~= 0
    parm.res3.npts = res3_npts;
end

if grating_type == 0
    [e,z,o]=res3(x,aa,profile,1,parm);
else
%    y = linspace(period(1), 0, 50);
    if grating_type == 1
        y=0;
    else
        y = linspace(period(1), 0, 11);
    end

    if pol == 1
        einc = [0, 1];
    elseif pol == -1
        einc = [1, 0];
    else
        disp('only TE or TM is allowed.');
    end
    [e,z,o]=res3(x, y, aa,profile,einc, parm);
end

if grating_type == 0
%    top_refl_info = res.inc_top_reflected;
%    top_tran_info = res.inc_top_transmitted;
%    bottom_refl_info = res.inc_bottom_reflected;
%    bottom_tran_info = res.inc_bottom_transmitted;

    top_refl_info_te = res.inc_top_reflected;
    top_tran_info_te = res.inc_top_transmitted;
    top_refl_info_tm = res.inc_top_reflected;
    top_tran_info_tm = res.inc_top_transmitted;

    bottom_refl_info_te = res.inc_bottom_reflected;
    bottom_tran_info_te = res.inc_bottom_transmitted;
    bottom_refl_info_tm = res.inc_bottom_reflected;
    bottom_tran_info_tm = res.inc_bottom_transmitted;

else
    top_refl_info_te = res.TEinc_top_reflected;
    top_tran_info_te = res.TEinc_top_transmitted;
    top_refl_info_tm = res.TMinc_top_reflected;
    top_tran_info_tm = res.TMinc_top_transmitted;

    bottom_refl_info_te = res.TEinc_bottom_reflected;
    bottom_tran_info_te = res.TEinc_bottom_transmitted;
    bottom_refl_info_tm = res.TMinc_bottom_reflected;
    bottom_tran_info_tm = res.TMinc_bottom_transmitted;

end
