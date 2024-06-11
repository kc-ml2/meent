function [R, T] = run_reticolo(_pol, theta, period, n_inc, nn, _textures, _profile, wavelength)
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

parm = res0(pol); % TE polarization. For TM : parm=res0(-1)
parm.res1.champ = 1; % the electromagnetic field is calculated accurately
%parm.res1.trace = 1; % show the texture

textures = cell(1, length(_textures));
for i = 1:length(_textures)
    textures(i) = _textures(i);
end

profile = cell(1, 2);
for i = 1:size(_profile)(1)
    profile(i) = _profile(i,:);
end

aa = res1(wavelength,period,textures,nn,k_parallel,parm);
res = res2(aa, profile);

R = res.inc_top_reflected.efficiency;
T = res.inc_top_transmitted.efficiency;
