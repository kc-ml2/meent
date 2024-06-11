clear;
load('60_1050_0.9754.mat')
tic;

xx = 10:1:30;
yy = 10:1:20;
[nnslist1, nnslist2] = meshgrid(xx,yy);
nnslist1 = nnslist1(:);
nnslist2 = nnslist2(:);

outputs = zeros(length(nnslist1),1);
pertime = zeros(length(nnslist1),1);
parfor i= 1:length(nnslist1)
    outputs(i) = get_Eff_2D_v3(pattern,wavelength,angle,[nnslist1(i) nnslist2(i)]);
    i
end
fintime = toc
showError;
save('outputs_from_60_1050_0.9754_2.mat')
