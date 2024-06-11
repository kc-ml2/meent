function [w,xdisc]=retu(init,x,epx,y,epy,ww);
%  function [w,xdisc]=retu(init,x,epx,y,epy,ww);
%
%  init (obtenu par retinit) peut etre remplace par d le pas du reseau
%
%  CAS 1D 
%  [w,xdisc]=retu(init,x,ep); 
%  x discontinuitees de x (croissant)  ,ep: valeurs de ep (calcules par retep) a gauche de x 
% (en cas de depassement prioritee a gauche)
%  remplissage de  w={x ep} utilisees par retcouche xdisc  points de discontinuite en x (modulo d)
%
%   milieu homogene: w=retu(ep)
%
%  CAS 2D   
%  [w,xdisc]=retu(init,x,epx,y,epy,ww)
%  remplissage de  w={x y u} utilisees par retcouche et de xdisc={xd,yd} points de discontinuite en x et y (modulo d)
%  si ww n'existe pas:on  remplit w  
%  si ww existe on  SUBSTITUE uu a u QUAND uu~=0
%   l'objet (ou la variation de l'objet) est le produit d'une fonction de x par une fonction de y
%   definies comme en 1D par x,epx y,epy 
%   epx(6,nombre de discontinuitees en x=size(x,2))   epy(6,nombre de discontinuitees en y=size(y,2))  
%
%   milieu homogene: w=retu(ep)
%
%
%  METAUX INFINIMENT CONDUCTEURS:
%  function w=retu(init,x,dx,eps,pol,mm);
%
%   calcul du maillage de texture w pour les metaux infiniment conducteurs:
%   w={x,dx,eps,pol,mm} est un cell array  decrivant l'objet
%  
%   pour le metal massif: w=retu(init,[],[],[],pol)  
%
%  CAS 1D 
%  trous de centre x(:),de largeur dx(:) >0 remplis d'un milieu homogene isotrope decrit par ep(:,3)
%  (attention ep est le transpose de celui fourni par retep)
%  chaque trou est decrit par mm(:) modes (par defaut le nombre d'ordres de fourier)
%  pol  0 metal electrique  2 metal magnetique  (par defaut pol=0)
%
%  CAS 2D   
%  trous de centre x(:,2),de largeur dx(:,2) >0 remplis d'un milieu homogene isotrope decrit par ep(:,6)
%  (attention ep est le transpose de celui fourni par ret2ep)
%  chaque trou est decrit par mm(:,2) modes (par defaut le nombre d'ordres de fourier)
%  pol  0 metal electrique  2 metal magnetique  (par defaut pol=0)
%
%   si une valeur de dx est > au pas du reseau,le trou est considere comme infini dans cette direction
%   si une valeur de dx est egale au pas du reseau,il reste un bord metallique
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FORME SIMPLIFIEE EN 2 D    w=retu(init,texture) 
%
%  texture={ n1, [cx1,cy1,dx1,dy1,ni1,k1],[cx2,cy2,dx2,dy2,ni2,k2],...[cxn,cyn,dxn,dyn,nin,kn],k0,un}
%    n1:indice de la base
%   [cx1,cy1,       dx1,dy1,             ni1     k1   ]:  inclusion  1         k0=2*pi/ld                       un  
%    centre    largeurs en x et y      indice                          facteur d'echelle en eps         facteur d'echelle metrique
%                                                                    (facultatif pour ld complexe)       (facultatif)
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
%  les points de discontinuitee doivent être en ordre croissant sur un intervalle de longueur STRICTEMENT inferieure au pas 
%  ET ETRE AU MOINS 2  
%      si ces tableaux sont des vecteurs lignes le reseau est invariant en y 
%                       |  |    |
%                       |  |    | {[x1,x2,x3],[n1,n2,n3],...
%                       |  |    |
%      si ces tableaux sont des vecteurs colonnes le reseau est invariant en x 
%                     _________________
%                     _________________ {[x1;x2;x3],[n1;n2;n3],.
%                     _________________
%
%  on peut aussi au lieu d'une base homogene avoir un maillage de texture auquel on ajoute des inclusions
%
%
%   ATTENTION: l'ordre des inclusions est important car elles s'ecrasent l'une l'autre ..
% 
%   on peut aussi definir des plaques de metal infiniment conducteur percees de trous rectangulaires NE SE CHEVAUCHANT PAS
%  texture= { inf, [cx1,cy1,dx1,dy1,n1,nmodesx1,nmodesy1],[cx2,cy2,dx2,dy2,n2,nmodesx2,nmodesy2],..} pour le metal 'electrique' (E//=0)
%  texture= {-inf, [cx1,cy1,dx1,dy1,n1,nmodesx1,nmodesy1],[cx2,cy2,dx2,dy2,n2,nmodesx2,nmodesy2],..} pour le metal 'magnetique'  (H//=0)
%   [cx1,cy1,          dx1,dy1,         n1        ,     nmodesx1 , nmodesy1 ]: premier trou
%    centre    largeurs en x et y      indice      nb de modes retenus en x et en y(par defaut nb d'ordres de fourier)
%
%   on peut aussi avoir: (commode pour le cas conique ...)
%  texture= { +-inf, [cx1,dx1,n1,nmodes1],..}    ou texture= { +-inf, [cy1;dyx1;n1;nmodes1]],..}  qui sont interpretees comme:
%           {+-inf, [cx1,0,dx1,inf,n1,nmodes1,1], ..}     ou  { +-inf, [0,cy1,inf,dy1,n1,1,nmodes1],...}
%
% attention: on doit mettre nmodes partout ou pas du tout !
%  par exemple  texture=  inf est le metal massif en haut ou en bas  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FORME SIMPLIFIEE EN 1 D    w=retu(init,texture) 
%  texture = {x1,n1,pol,k0,un} 
%
%  profil d'une tranche du reseau
%  décrite par un tableau de points de discontinuites x1 suivi du tableau n1 des indices a gauche des points
%  les points de discontinuitee doivent etre en ordre croissant sur un intervalle de longueur STRICTEMENT inferieure au pas 
%  ET ETRE AU MOINS 2  
%
%   milieu homogene: w=retu(d,{n,pol,k0})
% 
%   on peut aussi definir des tranches de metal infiniment conducteur percees de trous rectangulaires NE SE CHEVAUCHANT PAS
%  texture = { inf, [cx1,dx1,n1,nmodes1],[cx2,dx2,n2,nmodes2],..,pol,k0} pour le metal electrique
%   texture = {-inf, [cx1,dx1,n1,nmodes1],[cx2,dx2,n2,nmodes2],..,pol,k0} pour le metal magnetique
%                     [cx  dx1  n1  nmodes1 ]: centre largeur indice du premier trou ,nb de modes retenus (par defaut nb d'ordres de fourier)
% attention: on doit mettre nmodes partout ou pas du tout !
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SIMPLIFICATIONS D'ECRITURE (pour ces formes simplifiees)
%  si on a une seule texture il n'est pas necessaire de la mettre en tableau de cell array
%  pour les milieux homogenes on peut entrer   texture=n1  
%  par exemple  texture=  inf est le metal massif en haut ou en bas  
%
% See also: RETHELP_POPOV

if ~iscell(init);
if length(init)==2;d=init;init=cell(1,11);init{9}=d;init{end}=struct('dim',2,'d',d,'genre',0);end;
if length(init)==1;d=init;init=cell(1,6);init{4}=d;init{end}=struct('dim',1,'d',d,'genre',0);end;
end;


if nargin==2; %  FORME SIMPLIFIEE    w=retu(init,texture) 
	
if init{end}.genre==1; % cylindres Popov
r=x{1};n=x{2};if length(x)<3;k0=1;else;k0=x{3};end;	
cao=init{5};
Pml=init{10};
if ~isempty(Pml);% pml reelles
r=retinterp_popov(r,Pml,2);% r numerique;
end;	

if cao(2)~=1;if isempty(r)|(cao(1)>r(1));r=[cao(3),cao(1),r];n=[n(1),n(1),n];end;end;% si pml complexe on ajoute la couche externe avant la pml
w={r,ret2ep(n,k0),[]};xdisc=r;% w=[r,ep]; du haut en bas length(r)=length(n)-1
return;end;           % Fin cylindres Popov
	
if init{end}.dim==2; %     EN 2 D    
d=init{end}.d;texture=x;if ~iscell(texture);texture={texture};end;

k0=1;un=1; % par defaut
if length(texture)>2&length(texture{end-1})==1;k0=texture{end-1};un=texture{end};texture=texture(1:end-2);
elseif length(texture)>1&length(texture{end})==1;k0=texture{end};texture=texture(1:end-1);end;
epz=ret2ep(0);epun=ones(size(epz));

% base
if iscell(texture{1});w=texture{1};minc=2; % la base est deja un maillage de texture
else;    
if (length(texture)>1)&(length(texture{1})==length(texture{2}));  % la base est une texture 1D 
if size(texture{1},1)==1;w=retu(init,texture{1}/un,ret2ep(texture{2},k0),0,epun); % invariant en y  
else;w=retu(init,0,epun,texture{1}/un,ret2ep(texture{2},k0));end;                 % invariant en x 
minc=3;
else;  % la base est un milieu homogene d'indice texture{1}
if isfinite(texture{1}); % dielectrique 
w=retu(ret2ep(texture{1},k0));
minc=2;
else;  % metal infiniment conducteur
xm=[];dxm=[];epm=[];nmodes=[];if texture{1}>0 pola=0;else pola=2;end;% metal electrique ou magnetique
for in=2:size(texture,2);
if length(texture{in})<5; %  fentes infinies
if size(texture{in},1)==1; %  on complete cy=0  dy=inf    fente infinie // oy   ( cas conique )   
xm=[xm;[texture{in}(1),0]];   % centre
dxm=[dxm;[abs(texture{in}(2)),inf]];  % cotes
if length(texture{in})==4;nmodes=[nmodes;[texture{in}(4),1]];end;
else;                    %   on complete  cx=0  dx=inf    fente infinie // ox
xm=[xm;[0,texture{in}(1)]];   % centre
dxm=[dxm;[inf,abs(texture{in}(2))]];  % cotes
if length(texture{in})==4;nmodes=[nmodes;[1,texture{in}(4)]];end;
end;      %  ouvertures rectangulaires  
epm=[epm;ret2ep(texture{in}(3),k0).'];
else;    
xm=[xm;[texture{in}(1),texture{in}(2)]];   % centre
dxm=[dxm;[abs(texture{in}(3)),abs(texture{in}(4))]];  % cotes
if length(texture{in})==7;nmodes=[nmodes;[texture{in}(6:7)]];end;

epm=[epm;ret2ep(texture{in}(5),k0).'];
end;
end;
if size(nmodes,1)<size(xm,1),nmodes=[];end;% on doit mettre nmodes partout ou pas
w=retu(init,xm/un,dxm/un,epm,pola,nmodes);
return; % pas d'inclusions dielectriques
end; 
end;
end;


% sur la base on ajoute maintenant les inclusions
for in=minc:size(texture,2);
nt=real(texture{in}(6));morbido=imag(texture{in}(6));angles=sign(nt);ellipse=(mod(nt,1)==0);nt=fix(abs(nt));% si nt=1 rectangle  sinon ellipse (on  normalise la surface)
cx=texture{in}(1)/un;cy=texture{in}(2)/un;% centre
rx=abs(texture{in}(3)/2)/un;ry=abs(texture{in}(4)/2)/un;% demi cotes (ou demi grands axes) 
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

if morbido~=0;% morbido

if it>1;
ep0=rettestobjet(init,w,-1,[],{(xx+xxx)/2,(yy+yyy)/2},1:6);% ep exterieur 
%figure;plot(xx([1,2,2,1,1]),yy([1,1,2,2,1]),'-r',xxx([1,2,2,1,1]),yyy([1,1,2,2,1]),'--b');axis off
k6=ret2ep(struct([]));
w=retu(init,[xxx(1),xx(1)],[epz,moyenne(xx(1),xxx(1),yy(1),yyy(1),ep0(1,1,:),ep,k6)],[yy(1),yyy(1)],[epz,epun],w);
w=retu(init,[xx(2),xxx(2)],[epz,moyenne(xx(2),xxx(2),yy(1),yyy(1),ep0(2,1,:),ep,k6)],[yy(1),yyy(1)],[epz,epun],w);
w=retu(init,[xxx(1),xx(1)],[epz,moyenne(xxx(1),xx(1),yyy(2),yy(2),ep0(1,2,:),ep,k6)],[yyy(2),yy(2)],[epz,epun],w);
w=retu(init,[xx(2),xxx(2)],[epz,moyenne(xxx(2),xx(2),yyy(2),yy(2),ep0(2,2,:),ep,k6)],[yyy(2),yy(2)],[epz,epun],w);


% for ii=1:2;for jj=1:2;
% w=retu(init,sort([xxx(ii),xx(ii)]),[epz,moyenne(xx(ii),xxx(ii),yy(jj),yyy(jj),ep0(ii,jj,:),ep)],sort([yy(jj),yyy(jj)]),[epz,epun],w);
% end;end;		
end;
xxx=xx;yyy=yy;
end;           % morbido

end;
end;

else; %     EN 1 D    

d=init{end}.d;texture=x;if ~iscell(texture);texture={texture};end;
pol=0;k0=1;un=1; % par defaut
if length(texture)>3&length(texture{end-2})==1;pol=texture{end-2};k0=texture{end-1};un=texture{end};texture=texture(1:end-3);
elseif length(texture)>2&length(texture{end-1})==1;pol=texture{end-1};k0=texture{end};texture=texture(1:end-2);
elseif length(texture)>1&length(texture{end})==1;pol=texture{end};texture=texture(1:end-1);end;

if (length(texture)>1)&(length(texture{1})==length(texture{2}));  %  milieu non homogene 
	if isnan(pol); % cylindrique_radial
	w=retu(init,texture{1}/un,ret2ep(texture{2},k0));
	else;
    w=retu(init,texture{1}/un,retep(texture{2},pol,k0));
	end;
else;  %  milieu homogene d'indice texture{1}
if isfinite(texture{1}); % dielectrique 
	if isnan(pol); % cylindrique_radial
	w=retu(init,0,ret2ep(texture{1},k0));
	else;
    w=retu(init,0,retep(texture{1},pol,k0));
	end;
else;  % metal infiniment conducteur
xm=[];dxm=[];epm=[];nmodes=[];
if texture{1}*(1-pol)>0 pola=0;else pola=2;end;% metal electrique ou magnetique
for in=2:size(texture,2);
xm=[xm;texture{in}(1)];   % centre
dxm=[dxm;abs(texture{in}(2))];  % cotes
epm=[epm;retep(texture{in}(3),pol,k0).'];
if length(texture{in})==4;nmodes=[nmodes;[texture{in}(4)]];end;
end;
if size(nmodes,1)<size(xm,1),nmodes=[];end;% on doit mettre nmodes partout ou pas

w=retu(init,xm/un,dxm/un,epm,pola,nmodes);
return;
end; 
end;

end; % 1D 2D
if nargout>1;xdisc=retdisc(d,w);end;
return;
end;  % fin forme simplifiee

if nargin>2 &(size(x,1)==size(epx,1)); %metal 
if nargin<4;y=0;end;if nargin<5;epy=[];ww=[];end;if nargin<6;ww=[];end;    
if isempty(epy);epy=0;end; % pol par defaut       
w={x,epx,y,epy,ww};

else;   % dielectrique
if nargin<6;ww=[];end;
if ~iscell(init);
if size(init,1)>=6;w={1,1,reshape(init,1,1,length(init))};xdisc={[],[]};return;end;  % milieu homogene 2D  (init est alors ep)   
if size(init,1)<=3;w={1,init};xdisc=[];return;end;                        % milieu homogene 1D  (init est alors ep)   
end;

if nargin<4;   % cas 1D
d=init{end}.d;
[x,epx]=retordonne(x,epx,d(1));
w={x*d(1),epx};if nargout>1;xdisc=retdisc(d,w);end;
  
else;              % cas 2D 
d=init{end}.d;
[x,epx]=retordonne(x,epx,d(1));[y,epy]=retordonne(y,epy,d(2));

mx=size(x,2);my=size(y,2);k6=size(epx,1);
u=zeros(mx,my,k6);for ii=1:k6;u(:,:,ii)=epx(ii,:).'*epy(ii,:);end;%remplissage de u    

if ~isempty(ww);
xx=ww{1};yy=ww{2};uu=ww{3};    
% xxxx=sort([x,xx]);xxx=xxxx(1);for ii=2:size(xxxx,2);if xxxx(ii)>xxxx(ii-1);xxx=[xxx,xxxx(ii)];end;end;
% yyyy=sort([y,yy]);yyy=yyyy(1);for ii=2:size(yyyy,2);if yyyy(ii)>yyyy(ii-1);yyy=[yyy,yyyy(ii)];end;end;    

% xxx=retelimine(sort([x,xx]));
% yyy=retelimine(sort([y,yy]));
[xxx,prv,k]=retelimine([x,xx],100*eps);x=xxx(k(1:mx));xx=xxx(k(mx+1:end));% pour eviter les doublons
[yyy,prv,k]=retelimine([y,yy],100*eps);y=yyy(k(1:my));yy=yyy(k(my+1:end));
mmmx=size(xxx,2);mmmy=size(yyy,2);uuu=zeros(mmmx,mmmy,k6);    
mmx=size(xx,2);mmy=size(yy,2);x=[0,x];y=[0,y];xx=[0,xx];yy=[0,yy];
% 
% for ix=2:mmx+1;for iy=2:mmy+1;%on installe uuu=u
% iix=find(xxx>xx(ix-1)&xxx<=xx(ix));iiy=find(yyy>yy(iy-1)&yyy<=yy(iy));
% for ii=1:6;uuu(iix,iiy,ii)=uu(ix-1,iy-1,ii);end;
% end;end;
% for ix=2:mx+1;for iy=2:my+1;% on substitue u a uu si u~=0
% if ~all(u(ix-1,iy-1,:)==0); 
% iix=find(xxx>x(ix-1)&xxx<=x(ix));iiy=find(yyy>y(iy-1)&yyy<=y(iy));
% for ii=1:6;uuu(iix,iiy,ii)=u(ix-1,iy-1,ii);end;
% end;
% end;end;
% 

% attention à xx et x !
Ix=1:mmmx;for ix=1:mmx;Ix(xxx>xx(ix)&xxx<=xx(ix+1))=ix;end;
Iy=1:mmmy;for iy=1:mmy;Iy(yyy>yy(iy)&yyy<=yy(iy+1))=iy;end;
uuu=uu(Ix,Iy,:);
kx=cell(1,mx);for ix=1:mx;kx{ix}=find(xxx>x(ix)&xxx<=x(ix+1));end;
ky=cell(1,my);for iy=1:my;ky{iy}=find(yyy>y(iy)&yyy<=y(iy+1));end;
for ix=1:mx;for iy=1:my; % on substitue u a uu si u~=0
if ~all(u(ix,iy,:)==0);
uuu(kx{ix},ky{iy},:)=u(ix(ones(1,length(kx{ix}))),iy(ones(1,length(ky{iy}))),:);
end;
end;end;


u=uuu;x=xxx;y=yyy;
end;
% eventuelles simplifications
% nx=length(x);if nx>1;ix=[find(~all(all(u(2:nx,:,:)==u(1:nx-1,:,:),3),2)).',nx];else ix=1;end;
% ny=length(y);if ny>1;iy=[find(~all(all(u(:,2:ny,:)==u(:,1:ny-1,:),3),1)),ny];else iy=1;end;
% w={x(ix),y(iy),u(ix,iy,:)};

w={x,y,u};w=retsimplifie(w);


end;
if nargout>1;xdisc=retdisc(d,w);end;

end;   % dielectrique
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
function epm=moyenne1(xx,xxx,yy,yyy,ep0,ep);ep0=ep0(:);% pour morbido
kk0=[3,6];kk1=[1,4];kk2=[2,5];
aax=(xxx-xx)^2;aay=(yyy-yy)^2;aa=aax+aay;aax=aax/aa;aay=aay/aa;
epm=zeros(size(ep));if length(ep0)<length(ep);ep0=[ep0;zeros(length(ep)-length(ep0),1)];end;
ep0m=(ep+ep0)/2;
ep0mm=2./(1./ep+1./ep0);
epm(kk0)=ep0m(kk0);
epm(kk1)=aax*ep0m(kk1)+aay*ep0mm(kk1);
epm(kk2)=aay*ep0m(kk2)+aax*ep0mm(kk2);
% % 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
function epm=moyenne(xx,xxx,yy,yyy,ep0,ep,k6);ep0=ep0(:);% pour morbido
c=(xx-xxx);s=(yy-yyy);aa=sqrt(c^2+s^2);[c,s]=deal(c/aa,s/aa);
epm=zeros(12,1);if length(ep0)<length(ep);ep0=[ep0;zeros(length(ep)-length(ep0),1)];end;
epp=[ep,ep0];
% epm([4,5,6,10])=homogenise_inv(epp(4,:),epp(5,:),epp(6,:),c,s);
% epm([1,2,3,7])=homogenise_inv(epp(1,:),epp(2,:),epp(3,:),c,s);
epm([4,5,6,10])=homogenise(epp(4,:),epp(5,:),epp(6,:),c,s);
epm([1,2,3,7])=homogenise(epp(1,:),epp(2,:),epp(3,:),c,s);
epm=epm(1:k6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
function ep=homogenise(epx,epy,epz,c,s);
a11=mean(epx.*epy./(epx*s^2+epy*c^2));
a12=s*c*mean((epy-epx)./(epx*s^2+epy*c^2));
a22=mean(1./(epx*s^2+epy*c^2));
%epxx  epyy epzz epxy
exy=(c*s*(a11*a22+a12^2-1)+(c^2-s^2)*a12)/a22;if abs(exy)<100*eps;exy=0;end;
ep=[(c^2*(a11*a22+a12^2)-2*c*s*a12+s^2)/a22;(s^2*(a11*a22+a12^2)+2*c*s*a12+c^2)/a22;mean(epz);exy];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
function ep=homogenise_inv(epx,epy,epz,c,s);
a11=mean(1./(epx*c^2+epy*s^2));
a21=s*c*mean((epy-epx)./(epx*c^2+epy*s^2));
a22=mean(epx.*epy./(epx*c^2+epy*s^2));
%epxx  epyy epzz epxy
ep=[(s^2*(a11*a22+a21^2)-2*c*s*a21+c^2)/a11;(c^2*(a11*a22+a21^2)+2*c*s*a21+s^2)/a11;1./mean(1./epz);(c*s*(1-a11*a22-a21^2)+(c^2-s^2)*a21)/a11];





