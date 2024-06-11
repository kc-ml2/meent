function [abseff] = Eval_slit_1D(nn,slitsize)

wavelength = 1e-6;
angle = 30;

n_air = 1;
n_glass = 1.4504;
n_Si = 3.5750;
thickness  = 325e-9;

angle_theta0 = 0; % Incidence angle in degrees
k_parallel = n_air*sin(angle_theta0*pi/180); % n_air, or whatever the refractive index of the medium where light is coming in.

parm = res0(-1); % TE polarization. For TM : parm=res0(-1)
parm.res1.champ = 1; % the electromagnetic field is calculated accurately

period = abs(wavelength/sind(angle));

x = period * [-slitsize/2,slitsize/2];
nvec = [n_air,n_Si];

textures =cell(1,3);
textures{1}= n_air; % Uniform, top layer
textures{2}={x, nvec};
textures{3}= n_glass; % Uniform, bottom layer

aa = res1(wavelength,period,textures,nn,k_parallel,parm);
profile = {[0, thickness, 0], [1, 2, 3]}; %Thicknesses of the layers, and layers, from top to bottom.
one_D_TM = res2(aa, profile);
theta_arr = one_D_TM.inc_bottom_transmitted.theta-angle ;
idx = find(abs(theta_arr) == min(abs(theta_arr)));
abseff = one_D_TM.inc_bottom_transmitted.efficiency(idx);

end