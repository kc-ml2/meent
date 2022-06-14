function [R,T] = cal_reflectance(config, textures, profile, wavelength)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

warning('off', 'Octave:possible-matlab-short-circuit-operator');
warning('off', 'Invalid UTF-8 byte sequences have been replaced');
warning('off', 'findstr is obsolete; use strfind instead');

incident_angle = config{1};
detector_angle = config{2};
nn = config{3};
period = config{4};
tetm = config{5};

n_air = 1;

k_parallel = n_air*sin(incident_angle*pi/180); % n_air, or whatever the refractive index of the medium where light is coming in.

parm = res0(tetm); % TE polarization. For TM : parm=res0(-1)
%parm.res1.champ = 1; % the electromagnetic field is calculated accurately

%parm.res1.trace = 1; % show the texture

aa = res1(wavelength,period,textures,nn,k_parallel,parm);
one_D = res2(aa, profile);

R = sum(one_D.inc_top_reflected.efficiency);
T = sum(one_D.inc_top_transmitted.efficiency);
