function s=rets1(n,sog);
% function s=rets1(n,sog) ou s=rets1(init)
% s unite associée à une base de dimension n 
%  sog: s  (1) ou g  (0)  par defaut 1

if iscell(n);s=rets1(n{1},n{end}.sog);return;end; %s=rets1(init)


if nargin<2;sog=1;end;

if sog==1; %matrices s
s={speye(2*n);n;n;1};
else;  %matrices g
s={speye(2*n);speye(2*n);n;n;n;n};
end;