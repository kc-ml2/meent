function xx=retcaoo(xc,lc,d,x,sens);
% function xx=retcaoo(xc,lc,d,x,sens);
% calcul explicite du changement de coordonnees en une dimension 
% xc:point ou la derivee du changement de variable est nulle(image de l'infini)
%  (0<xc<d)
% lc:largeur de la partie modifiee 
% d  pas
% 
% SI sens==1 (ou absent) : de R a [0,d] 
% x:tableau de points(-inf a inf) a transformer en xx(reseau periode d)   
% la partie xc+lc/2-d <x<xc-lc/2 est INCHANGEE
% xx est compris entre 0 et d
% d et lc etant donnes,si on veut que l'image de points x entre 0 et L parte de xx=0
% et soit invariante -a une translation pres- autour de L/2
% il faut d'abord determiner la demi largeur de la partie 'recouverte': x0=retcaoo(d/2,lc,d,L/2) 
% et ensuite transformer les points par xx=retcaoo(x0+d/2,lc,d,x-L/2+x0)
% la partie 'non utilisee' de la periode est alors le segment:[d-2*xL,d]
%
% SI sens==-1 transformation inverse:de [0,d] a R  
%
%
%   Exemple:
% d=3;lc=1;L=4;x=linspace(0,L,100);x0=retcaoo(d/2,lc,d,L/2);xx=retcaoo(x0+d/2,lc,d,x-L/2+x0);plot(x,xx)
%   et en sens inverse:
% z=linspace(0,d,100);zz=retcaoo(x0+d/2,lc,d,z,-1)+L/2-x0;plot(zz,z)
%
%

if nargin<5;sens=1;end;
xc=mod(xc,d);caoi=imag(lc);lc=real(lc);

if sens==1; % de R a [0,d] 
xx=x;
if lc==0;xx(find((x>xc)|(x<(xc-d))))=xc;xx=mod(xx,d);return;end;% cas limite lc=0;
fm=find(x>(xc-lc/2));xx(fm)=xc-lc/2+lc/pi*atan((x(fm)-xc+lc/2)*pi/lc);
fp=find(x<(xc+lc/2-d));xx(fp)=xc+lc/2-d+lc/pi*atan((x(fp)-xc-lc/2+d)*pi/lc);
xx=mod(xx,d);
else;   % de [0,d] a R
xx=mod(x,d);
y=mod(x-xc,d);
fm=find(y>(d-lc/2));xx(fm)=mod(xc-lc/2,d)+lc/pi*tancao((y(fm)-d+lc/2)*pi/lc,caoi);
fp=find(y<lc/2);xx(fp)=mod(xc+lc/2,d)+lc/pi*tancao((y(fp)-lc/2)*pi/lc,caoi);
end;

function t=tancao(x,caoi);
t=tan(x);
if caoi==0;return;end;
a=sqrt(1+i*caoi);
t=(1+i*caoi)*(t-(i*caoi/a)*atan(t/a));






