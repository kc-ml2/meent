function [f1,f2]=retfind(a,g);
% function [f1,f2]=retfind(a,g);
% si g est absent: f1=find(a) et f2=find(~a)
% sinon  f1=g(find(a(g)) f2=g(find(~a(g)))
% Exemple [loin,pres]=retfind(D>4*R);: [central,pres]=retfind(D<1.e-6*R,pres);
%
% See also: FIND,RETELIMINE,RETCOMPARE,RETASSOCIE
if isempty(a);f1=[];f2=[];return;end;
if nargin<2;
f1=find(a);if nargout>1;f2=find(~a);end;
else;
f1=g(a(g));if nargout>1;f2=g(~a(g));end;
end;    




