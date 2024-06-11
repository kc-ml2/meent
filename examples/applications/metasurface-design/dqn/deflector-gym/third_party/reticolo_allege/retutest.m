function w=retu(init,x,epx,y,epy,ww);
%  function w=retu(init,x,epx,y,epy,ww);
%
%  init (obtenu par retinit) peut etre remplace par d le pas du reseau
%
%  CAS 1D 
%  w=retu(init,x,ep); 
%  x discontinuitees de x (croissant)  ,ep: valeurs de ep (calcules par retep) a gauche de x 
% (en cas de depassement prioritee a gauche)
%  remplissage de  w={x ep} utilisees par retcouche
%  
%  CAS 2D   
%  w=retu(init,x,epx,y,epy,ww)
%  remplissage de  w={x y u} utilisees par retcouche
%  si ww n'existe pas:on  remplit w  
%  si ww existe on  SUBSTITUE uu a u QUAND uu~=0
%   l'objet (ou la variation de l'objet)est le produit d'une fonction de x par une fonction de y
%   definies comme en 1D par x,epx y,epy 
%   epx(6,nombre de discontinuitees en x=size(x,2))   epy(6,nombre de discontinuitees en y=size(y,2))  
%
%   milieu homogene: w=retu(ep)
%
%  METAUX INFINIMENT CONDUCTEURS:
%  function w=retu(init,x,dx,eps,pol,mm);
%
%   calcul de la structure w pour les metaux infiniment conducteurs:
%
%   w={x,dx,eps,pol,mm} est une structure decrivant l'objet
%
%  trous de centre x(:,2),de largeur dx(:,2) >0 remplis d'un milieu homogene isotrope decrit par ep(:,6)
%  (attention ep est le transpose de celui fourni par ret2ep)
%  chaque trou est decrit par mm(:,2) modes (par defaut le nombre d'ordres de fourier)
%  pol  0 metal electrique  2 metal magnetique  (par defaut pol=0)
%
%   si une valeur de dx est > au pas du reseau,le trou est considere comme infini dans cette direction
%   si une valeur de dx est egale au pas du reseau,il reste un bord metallique
%  
%   pour le metal massif: w=retu(init,[],[],[],pol)  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FORME SIMPLIFIEE EN 2 D    w=retu(init,texture) 
%
%  texture={ n1, [cx1,cy1,dx1,dy1,ni1,k1],[cx2,cy2,dx2,dy2,ni2,k2],...[cxn,cyn,dxn,dyn,nin,kn],k0}
%    n1:indice de la base
%   [cx1,cy1,          dx1,dy1,             ni1     k1   ]: premiere inclusion              k0=2*pi/ld
%    centre    largeurs en x et y        indice                               facteur d'echelle facultatif pour ld complexe)
% abs(real(k1))=1  l'inclusion est un rectangle  de  cotes  dx1,dy1
% abs(real(k1)) >1  l'inclusion est une ellipse de grands axes  dx1,dy1,  approchee par abs(real(k1)) rectangles 
%                   la surface totale de l'inclusion etant celle de l'ellipse
%                    (si real(k1)<0:regulier en angle ,si real(k1)>0 regulier en surface)
%                    (si abs(k1) nom entier k1=fix(abs(k1) et l'ellipse est remplacee par un losange)
%           si imag(kl)~=0 option 'morbido' de lissage de ep 
%     si le rectangle ou l'ellipse a une dimension plus grande que le pas il y a chevauchement (indice nil dans la partie commune)
%       exemple: k1=1  rectangle
%                k1=5  ellipse formee de 5 rectangles 'regulier en surface'  pas lissage
%                k1=-5  ellipse formee de 5 rectangles 'regulier en angles'  pas lissage
%                k1=5+i  ellipse formee de 5 rectangles 'regulier en surface'  avec lissage
%                k1=-5+i  ellipse formee de 5 rectangles 'regulier en angles'  avec lissage
%                k1=5.1  losange formee de 5 rectangles   pas lissage
%                k1=5.1+i  losange formee de 5 rectangles  avec lissage
%
%  on peut aussi au lieu d'une base homogene avoir un reseau  1 D
%  decrit par un tableau de points de discontinuites suivi du tableau des indices a gauche des points
%  les points de discontinuitee doivent etre en ordre croissant sur un intervalle strictement inferieur au pas 
%  et etre au moins 2  
%
%      si ces tableaux sont des vecteurs ligne le reseau est invariant en y 
%      si ces tableaux sont des vecteurs colonnes le reseau est invariant en x 
%
%   ATTENTION: l'ordre des inclusions est important car elles s'ecrasent l'une l'autre ..
% 
%   on peut aussi definir des plaques de metal infiniment conducteur percees de trous rectangulaires NE SE CHEVAUCHANT PAS
%  texture= { inf, [cx1,cy1,dx1,dy1,ni1],[cx2,cy2,dx2,dy2,ni2],..} pour le metal electrique
%  texture= {-inf, [cx1,cy1,dx1,dy1,ni1],[cx2,cy2,dx2,dy2,ni2],..} pour le metal magnetique
%   [cx1,cy1,          dx1,dy1,         ni1       ]: premier trou
%    centre    largeurs en x et y      indice
%  par exemple  texture=  inf est le metal massif en haut ou en bas  
%

if ~iscell(init);if length(init)==2;d=init;init=cell(1,14);init{9}=d;end;if length(init)==1;d=init;init=cell(1,6);init{4}=d;end;end;
% si on entre d a la place de init ...

if nargin==2; %  FORME SIMPLIFIEE EN 2 D    w=retu(init,texture) 
d=init{9};texture=x;if ~iscell(texture);texture={texture};end;
if length(texture)>1&length(texture{end})==1;k0=texture{end};texture=texture{1:end-1};else k0=1;end;
epz=ret2ep(0);epun=ret2ep(1);

% base
if (length(texture)>1)&(length(texture{1})==length(texture{2}));  % la base est une texture 1D 
if size(texture{1},1)==1;w=retu(init,texture{1},ret2ep(texture{2},k0),0,epun); % invariant en y  
else;w=retu(init,0,epun,texture{1},ret2ep(texture{2},k0));end;                 % invariant en x 
minc=3;
else;  % la base est un milieu homogene d'indice texture{1}
if isfinite(texture{1}); % dielectrique 
w=retu(ret2ep(texture{1},k0));
minc=2;
else;  % metal infiniment conducteur
xm=[];dxm=[];epm=[];if texture{1}>0 pola=0;else pola=2;end;% metal electrique ou magnetique
for in=2:size(texture,2);
xm=[xm;[texture{in}(1),texture{in}(2)]];   % centre
dxm=[dxm;[abs(texture{in}(3)),abs(texture{in}(4))]];  % cotes
epm=[epm;ret2ep(texture{in}(5),k0).'];
end;
w=retu(init,xm,dxm,epm,pola);
return; % pas d'inclusions dielectriques
end; 
end;
% sur la base on ajoute maintenant les inclusions
for in=minc:size(texture,2);
nt=real(texture{in}(6));morbido=imag(texture{in}(6));angles=sign(nt);ellipse=(mod(nt,1)==0);nt=fix(abs(nt));% si nt=1 rectangle  sinon ellipse (on  normalise la surface)
cx=texture{in}(1);cy=texture{in}(2);% centre
rx=abs(texture{in}(3)/2);ry=abs(texture{in}(4)/2);% demi cotes (ou demi grands axes) 
if nt==1;ax=1;ay=1;  % rectangle
    
else;                % ellipse
if ellipse;  % ellipse
if angles==-1;
t=pi/(4*nt)*(1:2:2*nt-1);     %regulier en angles
else;
t=.5*acos(1-(1:2:2*nt-1)/nt); %regulier en surfaces
end;
ax=cos(t);ay=sin(t);t=sum(ax.*(ay-[0,ay(1:end-1)])); % normalisation de la surface
if morbido~=0;t=(t+sum([ax(1),ax(1:end-1)].*(ay-[0,ay(1:end-1)])))/2;end;
t=sqrt(pi/(4*t));ax=ax*t;ay=ay*t;

else;  % losange
if morbido==1;
ax=linspace(1-1.e-6,1.e-6,nt);ay=1-ax;
else;
aa=1/(2*sqrt(2)+nt-1);ax=linspace(1-aa,aa,nt);ay=1-ax;    
t=sum(ax.*(ay-[0,ay(1:end-1)]));t=sqrt(1/(2*t));ax=ax*t;ay=ay*t; % normalisation de la surface
end;    
    
end;   

end;
ep=ret2ep(texture{in}(5),k0); % indice de l'inclusion
for it=1:nt;
xx=cx+rx*[-ax(it),ax(it)];if xx(2)-xx(1)>d(1);xx=[0,d(1)];end; % en cas de debordement on remplit toute la periode
yy=cy+ry*[-ay(it),ay(it)];if yy(2)-yy(1)>d(2);yy=[0,d(2)];end;
w=retu(init,xx,[epz,ep],yy,[epz,epun],w);

if morbido~=0;
k0=[3,6];k1=[1,4];k2=[2,5];
if it>1;
aax=(xxx(1,1)-xx(1,1))^2;aay=(yyy(1,1)-yy(1,1))^2;aa=aax+aay;aax=aax/aa;aay=aay/aa;
ep0=rettestobjet(init,w,-1,[],{(xx+xxx)/2,(yy+yyy)/2},1:6);epm=(zeros(6,1));% ep exterieur   
ep00=reshape(ep0(1,1,:),6,1);
ep0m=sqrt(ep.*ep00);ep0mm=epom/(1./ep+1./ep00);epm(k0)=ep0m(k0);epm(k1)=aax*ep0m(k1)+aay*ep0mm(k1);epm(k2)=aay*ep0m(k2)+aax*ep0mm(k2);
w=retu(init,[xxx(1),xx(1)],[epz,epm],[yy(1),yyy(1)],[epz,epun],w);
ep00=reshape(ep0(1,2,:),6,1);
ep0m=(ep+ep00)/2;ep0mm=2./(1./ep+1./ep00);epm(k0)=ep0m(k0);epm(k1)=aax*ep0m(k1)+aay*ep0mm(k1);epm(k2)=aay*ep0m(k2)+aax*ep0mm(k2);
w=retu(init,[xxx(1),xx(1)],[epz,epm],[yyy(2),yy(2)],[epz,epun],w);
ep00=reshape(ep0(2,1,:),6,1);
ep0m=(ep+ep00)/2;ep0mm=2./(1./ep+1./ep00);epm(k0)=ep0m(k0);epm(k1)=aax*ep0m(k1)+aay*ep0mm(k1);epm(k2)=aay*ep0m(k2)+aax*ep0mm(k2);
w=retu(init,[xx(2),xxx(2)],[epz,epm],[yy(1),yyy(1)],[epz,epun],w);
ep00=reshape(ep0(2,2,:),6,1);
ep0m=(ep+ep00)/2;ep0mm=2./(1./ep+1./ep00);epm(k0)=ep0m(k0);epm(k1)=aax*ep0m(k1)+aay*ep0mm(k1);epm(k2)=aay*ep0m(k2)+aax*ep0mm(k2);
w=retu(init,[xx(2),xxx(2)],[epz,epm],[yyy(2),yy(2)],[epz,epun],w);
end;
xxx=xx;yyy=yy;
end;

end;
end;
return;
end;  % fin forme simplifiee

if nargin>2 &(size(x,1)==size(epx,1)); %metal 
if nargin<4;y=0;end;if nargin<5;epy=[];end;if nargin<6;ww=[];end;    
if isempty(epy);epy=0;end; % pol par defaut       
w={x,epx,y,epy,ww};

else;   % dielectrique
    
if size(init,1)==6;w={1,1,reshape(init,1,1,6)};return;end;  %milieu homogene 2D  (init est alors ep)   
if size(init,1)==3;w={1,init};return;end;                   %milieu homogene 1D  (init est alors ep)   


if nargin<4;   % cas 1D
d=init{end}.d;
[x,epx]=retordonne(x,epx,d(1));
w={x*d(1),epx};    
else;              % cas 2D 

d=init{end}.d;
[x,epx]=retordonne(x,epx,d(1));[y,epy]=retordonne(y,epy,d(2));

mx=size(x,2);my=size(y,2);
u=zeros(mx,my,6);for ii=1:6;u(:,:,ii)=epx(ii,:).'*epy(ii,:);end;%remplissage de u    
    
if nargin>=6;
xx=ww{1};yy=ww{2};uu=ww{3};    
%if xx(1)==0;xx=xx(2:end);uu=uu(2:end,:,:);end;if yy(1)==0;yy=yy(2:end);uu=uu(:,2:end,:);end;    
%xx=xx./xx(end);yy=yy./yy(end);    
xxxx=sort([x,xx]);xxx=xxxx(1);for ii=2:size(xxxx,2);if xxxx(ii)>xxxx(ii-1);xxx=[xxx,xxxx(ii)];end;end;
yyyy=sort([y,yy]);yyy=yyyy(1);for ii=2:size(yyyy,2);if yyyy(ii)>yyyy(ii-1);yyy=[yyy,yyyy(ii)];end;end;    
mmmx=size(xxx,2);mmmy=size(yyy,2);uuu=zeros(mmmx,mmmy,6);    
mmx=size(xx,2);mmy=size(yy,2);x=[0,x];y=[0,y];xx=[0,xx];yy=[0,yy];
for ix=2:mmx+1;for iy=2:mmy+1;%on installe uuu=u
iix=find(xxx>xx(ix-1)&xxx<=xx(ix));iiy=find(yyy>yy(iy-1)&yyy<=yy(iy));
for ii=1:6;uuu(iix,iiy,ii)=uu(ix-1,iy-1,ii);end;
end;end;
for ix=2:mx+1;for iy=2:my+1;%on substitue uu a u si uu~=0
if ~all(u(ix-1,iy-1,:)==0); 
iix=find(xxx>x(ix-1)&xxx<=x(ix));iiy=find(yyy>y(iy-1)&yyy<=y(iy));
for ii=1:6;uuu(iix,iiy,ii)=u(ix-1,iy-1,ii);end;
end;
end;end;
u=uuu;x=xxx;y=yyy;
end;

% eventuelles simplifications
nx=length(x);if nx>1;ix=[find(~all(all(u(2:nx,:,:)==u(1:nx-1,:,:),3),2)).',nx];else ix=1;end;
ny=length(y);if ny>1;iy=[find(~all(all(u(:,2:ny,:)==u(:,1:ny-1,:),3),1)),ny];else iy=1;end;
w={x(ix),y(iy),u(ix,iy,:)};
end;


end;   % dielectrique

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xx,eep]=retordonne(x,ep,d);
%  function [xx,eep]=retordonne(x,ep,d);
%  mise en forme de x ep utilises par ret2u
%  d pas
%  x,ep discontinuitees de x (croissant )  valeurs de ep a gauche  
%  xx eep:idem mais xx est ordonnee  xx(0)>0  xx(end)=1

m=length(x);
masque=ones(size(x)); %on ne garde que les x qui sont strictement plus grands que les precedents
for ii=1:m-1;f=find(x(ii+1:m)<=x(ii));x(f+ii)=x(ii);masque(f+ii)=0;end;
f=find(masque==1);x=x(f);ep=ep(:,f);
if abs(x(end)-x(1)-d)<10*eps;x=x(2:end);ep=ep(:,2:end);end;
[x,ii]=sort(mod(x/d,1));ep=ep(:,ii);
xx=1;eep=ep(:,1);
for jj=length(x):-1:1;
if x(jj)>0 & ~all(ep(:,jj)==eep(:,1));
xx=[x(jj),xx];eep=[ep(:,jj),eep];
end;
end;
