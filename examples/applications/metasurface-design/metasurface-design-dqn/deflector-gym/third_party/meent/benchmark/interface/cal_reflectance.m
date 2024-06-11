function [R, T] = cal_reflectance(_pol, angle_in, angle_out, n_air, n_glass, thickness, nn, _textures, _profile, wavelength)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

warning('off', 'Octave:possible-matlab-short-circuit-operator');
warning('off', 'Invalid UTF-8 byte sequences have been replaced');
warning('off', 'findstr is obsolete; use strfind instead');

angle_theta0 = 0; % Incidence angle in degrees
k_parallel = n_air*sin(angle_theta0*pi/180); % n_air, or whatever the refractive index of the medium where light is coming in.

if _pol == 0  # TE
    pol = 1;
else  # TM
    pol = -1;
end

parm = res0(pol); % TE polarization. For TM : parm=res0(-1)
parm.res1.champ = 1; % the electromagnetic field is calculated accurately
%parm.res1.trace = 1; % show the texture

period = abs(wavelength/sind(angle_out));

textures = cell(1, length(_textures));
for i = 1:length(_textures)
    textures(i) = _textures(i);
end

profile = cell(1, 2);
for i = 1:size(_profile)(1)
    profile(i) = _profile(i,:);
end

aa = res1(wavelength,period,textures,nn,k_parallel,parm);
AA = res2(aa, profile);
%theta_arr = one_D_TM.inc_bottom_transmitted.theta;
%idx = find(abs(theta_arr) == min(abs(theta_arr)));

%abseff = one_D_TM.inc_bottom_transmitted.efficiency(idx)
R = AA.inc_top_reflected.efficiency
T = AA.inc_top_transmitted.efficiency


