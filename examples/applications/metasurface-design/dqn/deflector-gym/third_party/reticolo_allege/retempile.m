function [xx,ii,jj]=retsort(x,p);
% comme sort mais donne aussi l'ordre inverse
%% EXEMPLE
% x=rand(5,4,3);[xx,ii,jj]=retsort(x,2);
% retcompare(x(3,:,2),xx(3,jj(3,:,2),2))
% retcompare(xx(3,:,2),x(3,ii(3,:,2),2))
if nargin<2;
[xx,ii]=sort(x);
if nargout<3;return;end;
[prv,jj]=sort(ii);
else;
[xx,ii]=sort(x,p);
if nargout<3;return;end;
[prv,jj]=sort(ii,p);
end;
