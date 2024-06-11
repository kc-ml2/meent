clear
load('55_1250_0.9947.mat')

orders = 2:2:20;
results = zeros(length(orders),1);
times = zeros(length(orders),1);

for i = 1:length(orders)
    tic
    results(i) = get_Eff_2D_v1(orders(i),pattern,1250,55)
    times(i)=toc
end
