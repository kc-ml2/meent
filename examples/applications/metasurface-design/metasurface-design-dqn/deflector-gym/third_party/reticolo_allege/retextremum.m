function [xe,ye,max_ou_min]=retextremum(x,y,n,methode);
%  function [xe,ye,max_ou_min]=retextremum(x,y,n,methode);
%  recherche des extremum d'une fonction definie par des points x,y
%  x , y couples de points
% n nombre de points pour l'interpolation (100*length(x) par defaut )
% ( si n=nan pas d'interpolation )
% methode='spline' par defaut
%% Exemple
% x=8*pi*rand(1,100);y=sin(x)./(1+x);
% [xe,ye,max_ou_min]=retextremum(x,y);
% figure;plot(x,y,'.',xe(max_ou_min>0),ye(max_ou_min>0),'or',xe(max_ou_min<0),ye(max_ou_min<0),'ob');


if isempty(x);xe=[];ye=[];max_ou_min=[];return;end;
if nargin<4;methode=[];end;if isempty(methode);methode='spline';end;
if nargin<3;n=[];end;if isempty(n);n=100*length(x);end;

x=x(:).';y=y(:).';

[x,ii]=retelimine(x,100*eps);y=y(:,ii);
if ~isnan(n);
xx=retelimine(sort([x,linspace(min(x),max(x),n)]));
yy=retinterp(x,y,xx,methode);
else;yy=y;xx=x;
end;
prv=diff(yy);prv(abs(prv)<1.e-10*mean(abs(yy)))=0;
prv=sign(prv);prv(prv==0)=1;
f=1+find(diff(prv)~=0);xe=xx(f);ye=yy(f);
max_ou_min=prv(f-1);






