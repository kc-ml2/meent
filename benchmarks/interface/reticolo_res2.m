function [top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info] = run_reticolo(_pol, theta, phi, period, n_inc, nn, _textures, _profile, wavelength, grating_type, field)
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
    parm = res0(pol); % TE polarization. For TM : parm=res0(-1)
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
    aa = res1(wavelength,period,textures,nn,k_parallel,phi,parm);
    res = res2(aa, profile);
end

if grating_type == 0
    top_refl_info = res.inc_top_reflected;
    top_tran_info = res.inc_top_transmitted;
    bottom_refl_info = res.inc_bottom_reflected;
    bottom_tran_info = res.inc_bottom_transmitted;
else
    if pol == 1  % TE
        top_refl_info = res.TEinc_top_reflected;
        top_tran_info = res.TEinc_top_transmitted;
        bottom_refl_info = res.TEinc_bottom_reflected;
        bottom_tran_info = res.TEinc_bottom_transmitted;
    else  % TM
        top_refl_info = res.TMinc_top_reflected;
        top_tran_info = res.TMinc_top_transmitted;
        bottom_refl_info = res.TMinc_bottom_reflected;
        bottom_tran_info = res.TMinc_bottom_transmitted;
    end
end
