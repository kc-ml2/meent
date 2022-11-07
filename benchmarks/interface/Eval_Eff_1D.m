function [abseff, effi_r, effi_t] = Eval_Eff_1D(img, wavelength, def_angle, fourier_order, n_air, n_glass,
    thickness, inc_angle, n_Si)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

warning('off', 'Octave:possible-matlab-short-circuit-operator');
warning('off', 'Invalid UTF-8 byte sequences have been replaced');
warning('off', 'findstr is obsolete; use strfind instead');

img = img/2.0 + 0.5;

%n_air = n_air;  % 1
%n_glass = n_glass;  % 1.45
%%thickness  = thickness;  % 325
%thickness  = 325;  % 325

load('/Users/yongha/project/metasurface/reticolo/V9/p_Si.mat')
% 210.31 ~ 1688.90

%n_Si = interp1(WL, n, wavelength);
%clear k n WL

%n_Si = 3.48

angle_theta0 = inc_angle; % 0; Incidence angle in degrees
k_parallel = n_air*sin(angle_theta0*pi/180); % n_air, or whatever the refractive index of the medium where light is coming in.

parm = res0(-1); % TE polarization. For TM : parm=res0(-1)
parm.res1.champ = 1; % the electromagnetic field is calculated accurately
%parm.res1.trace = 1; % show the texture

nn = fourier_order; % Fourier harmonics sarun from [-40,40]

period = abs(wavelength/sind(def_angle));

N = length(img);
dx = period/N;
x = [1:N]*dx - 0.5*period;
nvec = img*(n_Si - n_air) + n_air;


% textures for all layers including the top and bottom layers
textures =cell(1,3);
textures{1}= n_air; % Uniform, top layer

% Span for mid-layer patterns is [-period/2, period/2]
textures{2}={x, nvec};
textures{3}= n_glass; % Uniform, bottom layer

aa = res1(wavelength,period,textures,nn,k_parallel,parm);
profile = {[0, thickness, 0], [1, 2, 3]}; %Thicknesses of the layers, and layers, from top to bottom.
one_D_TM = res2(aa, profile);
theta_arr = one_D_TM.inc_bottom_transmitted.theta-def_angle ;
idx = find(abs(theta_arr) == min(abs(theta_arr)));
abseff = one_D_TM.inc_bottom_transmitted.efficiency(idx);
effi_r = one_D_TM.inc_bottom_reflected.efficiency;
effi_t = one_D_TM.inc_bottom_transmitted.efficiency;

%disp(abseff)
%catch
%   abseff = -1
%end
