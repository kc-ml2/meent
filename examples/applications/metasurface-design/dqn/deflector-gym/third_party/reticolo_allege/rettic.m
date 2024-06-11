function rettic
% comme tic mais donne le temps cpu
global CPUTICTOC TICTOC;
TICTOC=clock;CPUTICTOC=cputime;
