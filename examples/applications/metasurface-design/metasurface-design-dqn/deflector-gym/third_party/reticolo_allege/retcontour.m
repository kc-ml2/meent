function [xy,c]=retcontour(x,y,fonc,v,parm,varargin);
% function [xy,c]=retcontour(x,y,fonc,v,parm,varargin);
%  contours fonc(x,y)=v raffinés par retcadilhac
%  z=fonc(x,y):fonction réelle de 2 variables relles x et y 
%  fonc doit accepter des variables vectorielles x et y ,et retourner un vecteur z de meme taille
% x,y vecteurs ligne definissant le maillage (fonc est calcule sur meshgrid(x,y))
% parm structure de parametres 
% par defaut parm=struct('itermax',30,'parmcolor',[]); 
%  itermax:nombre max d'iterations de retcadilhac
% parmcolor parametres de retcolor si on veut un trace 
% 2i mesh,1:pcolor en noir,2 pcolor en couleurs (voir retcolor)
%  direction:direction de recherche des poles(vecteur 2)
%  xy(1,:) valeurs de x,xy(2,:) valeurs de y convergees
%  c:cell array contenant des structures:;
% c{ii}.  valeur: 4.0000e-001 <---- v
%         x: [1x26 double]<----- valeurs de x
%         y: [1x26 double]<----- valeurs de y
%      test: [1 1 1 1 1...]<----- test de convergence
%
%
% si fonc est un tableau reel de taille [length(y),length(x)], il n'y a pas affinage
%
% % EXEMPLE
% xx=linspace(-1,1,21);yy=linspace(-1,1,31);zz=.4;
% xy=retcontour(xx,yy,inline('x.^2-y.^2-2*x.*y','x','y'),zz,struct('parmcolor',2i));


if nargin<5;parm=struct([]);end;
pparm=struct('itermax',30,'parmcolor',[],'direction',[]);% par defaut
   
itermax=retoptimget(parm,'itermax',pparm);
parmcolor=retoptimget(parm,'parmcolor',pparm);
direction=retoptimget(parm,'direction',pparm);
if ~isempty(direction);direction=direction/norm(direction);end;



[x,y]=meshgrid(x,y);
if ~isreal(fonc);z=real(feval(fonc,x,y,varargin{:}));else;z=fonc;end;
%if retversion>6;cc=contour('v6',x,y,z,[v,v]);else;cc=contour(x,y,z,[v,v]);end;
cc=contours(x,y,z,[v,v]);
% decomposition de c
c={};
n0=1;
while n0<size(cc,2);
n=cc(2,n0);
c=[c,{struct('valeur',cc(1,n0),'x',cc(1,n0+1:n+n0),'y',cc(2,n0+1:n+n0),'test',ones(size(cc(1,n0+1:n+n0))))}];
n0=n+n0+1;
end;

% affinage des points
if ~isreal(fonc)
for ii=1:length(c);if length(c{ii}.x)<2;break;end;
xx=c{ii}.x;yy=c{ii}.y;vv=c{ii}.valeur;
xx=[2*xx(1)-xx(2),xx,2*xx(end)-xx(end-1)];
yy=[2*yy(1)-yy(2),yy,2*yy(end)-yy(end-1)];
for jj=2:length(xx)-1;
if isempty(direction);direction=[yy(jj+1)-yy(jj-1),xx(jj-1)-xx(jj+1)];direction=direction/norm(direction);end;% normale
[t,iter,er,erfonc,test]=retcadilhac(@err,i*itermax,0,xx(jj),yy(jj),direction,vv,fonc,varargin{:});
if all(test);
c{ii}.x(jj-1)=c{ii}.x(jj-1)+t*direction(1);c{ii}.y(jj-1)=c{ii}.y(jj-1)+t*direction(2);
else;c{ii}.test(jj-1)=0;end;
end;
end;  
end;  
% construction de xy;

xy=zeros(2,0);
for ii=1:length(c);f=find(c{ii}.test==1);xy=[xy,[c{ii}.x(f);c{ii}.y(f)]];end;
% trace
if ~isempty(parmcolor);

figure;
retcolor(x+0,y+0,z+0,parmcolor);xlabel(inputname(1));ylabel(inputname(2));
hold on;
if parmcolor==2i;zlabel(inputname(4));
for ii=1:length(c);plot3(c{ii}.x,c{ii}.y,v*ones(size(c{ii}.x)),'-k','linewidth',5);end;
else;for ii=1:length(c);plot(c{ii}.x,c{ii}.y,'-k');end;end;
title(['valeur cherchee:  ',num2str(v)]);

end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function er=err(t,x,y,n,v,fonc,varargin)
er=feval(fonc,x+t*n(1),y+t*n(2),varargin{:})-v;