tic
angle = 70;
wavelength = 700;
 
n_air = 1;
n_glass = 1.45;
thickness  = 325;
 
%load('p_Si.mat')
n_Si = 3.7081;%interp1(WL, n, wavelength);
clear k n WL
angle_theta0 = 0; % Incidence angle in degrees
k_parallel = n_air*sin(angle_theta0*pi/180);
 
per = abs(wavelength/sind(angle));
period = [per, per];
%N = 100;%length(img);
%dx = period/N;
%dy = dx;
%x = [1:N]*dx - 0.5*period;
%y = [1:N]*dy - 0.5*period;
nns = [15 15];
angle_delta = 0;
 
% textures for all layers including the top and bottom layers
textures =cell(1,3);
textures{1}= n_air; % Uniform, top layer
textures{2} = {n_air,[0,0,1/2*per,1/2*per,n_Si,1]};
textures{3}= n_glass; % Uniform, bottom layer
profile = {[0, thickness, 0], [1, 2, 3]};
aa = res1(wavelength,period,textures,nns,k_parallel,angle_delta);
two_D_TM = res2(aa, profile);
theta_arr = two_D_TM.TMinc_bottom_transmitted.theta-angle ;
idx = find(abs(theta_arr) == min(abs(theta_arr)));
abseff = two_D_TM.TMinc_bottom_transmitted.efficiency_TM(idx);
toc