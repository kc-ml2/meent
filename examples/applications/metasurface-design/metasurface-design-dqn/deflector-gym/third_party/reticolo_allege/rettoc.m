function [t,cpu]=rettoc;
% function [t,cpu]=rettoc;
% comme toc mais donne le temps reelt et le temps cpu
global CPUTICTOC TICTOC;
if isempty(TICTOC);rettic;end;
cpu=cputime-CPUTICTOC;
t=etime(clock,TICTOC);