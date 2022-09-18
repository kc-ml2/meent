%function [R,T] = cal_reflectance(theta, phi, psi, deflector_angle, n_air, n_glass, thickness, nn, config, textures,
%    profile, wavelength)
function [abseff] = cal_reflectance(angle_in, angle_out, n_air, n_glass, thickness, nn, textures, profile, wavelength)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

warning('off', 'Octave:possible-matlab-short-circuit-operator');
warning('off', 'Invalid UTF-8 byte sequences have been replaced');
warning('off', 'findstr is obsolete; use strfind instead');

angle_theta0 = 0; % Incidence angle in degrees
k_parallel = n_air*sin(angle_theta0*pi/180); % n_air, or whatever the refractive index of the medium where light is coming in.

parm = res0(-1); % TE polarization. For TM : parm=res0(-1)
parm.res1.champ = 1; % the electromagnetic field is calculated accurately
%parm.res1.trace = 1; % show the texture

period = abs(wavelength/sind(angle_out));

len = length(textures);
aa = cell(1, len);

for i = 1:length(textures)
    aa(i) = textures(i);
end

textures = aa;

len = length(textures);
aa = cell(1, len);

for i = 1:length(profile)
    aa(i) = profile(i);
end
profile = aa;

aa = res1(wavelength,period,textures,nn,k_parallel,parm);
profile
profile = num2cell(profile)
profile = {[0, thickness, 0], [1, 2, 3]} %Thicknesses of the layers, and layers, from top to bottom.
one_D_TM = res2(aa, profile);
theta_arr = one_D_TM.inc_bottom_transmitted.theta - angle_out;
idx = find(abs(theta_arr) == min(abs(theta_arr)));
abseff = one_D_TM.inc_bottom_transmitted.efficiency(idx);
effi = one_D_TM.inc_bottom_transmitted.efficiency
