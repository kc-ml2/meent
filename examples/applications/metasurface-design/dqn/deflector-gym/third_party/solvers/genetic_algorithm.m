rng default % for reproducibility

gaAvailable = false;
nvar = 64;
wl = 1100;
ang = 50;

gaoptions = optimoptions('ga', 'UseParallel', false,'MaxGenerations',4000, 'PlotFcn',{'gaplotbestf','gaplotbestindiv'}, 'MaxStallGenerations', inf, 'PopulationSize', 500, 'UseVectorized', false, 'PopulationType', 'bitString' ,'Display','iter');
startTime = tic;
[x,fval,exitflag,output,population,scores] = ga(@(x) Eval_Eff_1D( x, wl, ang)*(-1),nvar,[],[],[],[],[],[],[],gaoptions);
time_ga_sequential = toc(startTime);
fprintf('Serial GA optimization takes %g seconds.\n',time_ga_sequential);
gaAvailable = true;
