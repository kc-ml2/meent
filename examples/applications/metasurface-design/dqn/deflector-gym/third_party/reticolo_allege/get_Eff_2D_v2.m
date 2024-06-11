function [ abseff ] = get_Eff_2D_v2(pattern, wavelength, angle)
[prv,vmax] = retio([],inf*1i);

tic;
n_air = 1;
n_glass = 1.45;
thickness  = 325;
 
load('p_Si.mat')
n_Si = interp1(WL, n, wavelength);  %TBD
clear k n WL
angle_theta0 = 0; % Incidence angle in degrees
k_parallel = n_air*sind(angle_theta0);
 
per = abs(wavelength/sind(angle));
period = [per, 0.5*wavelength]; %%%%
 
nns = [13 10];
angle_delta = 0;
parm = res0;
 
[Lx, Ly] = size(pattern);
nmat = pattern*(n_Si - n_air) + n_air;
dx = period(1)/Lx;
dy = period(2)/Ly;
a = {1};


for ii=1:Lx
    for jj = 1:Ly
        a{end+1} = [(ii-Lx/2-0.5)*dx,(jj-Ly-0.5)*dy,dx,dy,nmat(ii,jj),1];
        %a{end+1} = [(ii-Lx/2-0.5)*dx,(jj-Ly-0.5)*dy,dx,dy,3.7,1];
    end
end


textures =cell(1,3);
textures{1}= n_air; % Uniform, top layer
textures{2} = a;
textures{3}= n_glass; % Uniform, bottom layer
profile = {[0, thickness, 0], [1, 2, 3]};
aa = res1(wavelength,period,textures,nns,k_parallel,angle_delta);
two_D_TE = res2(aa, profile);
%theta_arr = two_D_TE.TEinc_bottom_transmitted.theta-angle ;
%idx = find(abs(theta_arr) == min(abs(theta_arr)));
abseff = two_D_TE.TEinc_bottom_transmitted.efficiency_TE{1}; %% {n} : nth order directly 
end


