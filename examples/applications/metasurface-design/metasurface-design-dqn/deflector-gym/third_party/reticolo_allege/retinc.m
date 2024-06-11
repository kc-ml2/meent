function s=retinc(init,e,h,inc,pml,varargin);
%   function s=retinc(init,e,h,inc,pml,varargin);
%
%  calcul de la matrice s associee a un champ incident considere comme source
% On retranche au champ total un champ constitue du champ produit par la reflexion
%  du champ incident sur un miroir plan parfait electrique (E=0) ou magnetique (H=0) 
%
% e  h: fonctions permettant le calcul de E ou H tangentiels en tout point de coordonnees complexes
%       E=feval(e,x,varargin{:})     H=feval(h,x,varargin{:})
%   en 1D E H et x sont des scalaires
%   en 2 D des vecteurs   E={EX(:,:),EY(:,:)} H={HX,HY} x=[x(:,:),y(:,:)] 
%
%   un des 2 e ou h est [] <***** ATTENTION
%  (il s'agit d'un masque ne tenant pas compte du terme d'inclinaison qui est calculé dans le programme grace à inc )
%  en cas de pml reelles ce sont les fonctions avant les pml,dans l'espace physique,le calcul des pml 
%   est fait grace au parametre pml )
%    Si ce sont des constantes on peut mettre ces constantes (scalaire en 1 D vecteur de longueur 2 en 2D)
%
%
%   inc: incidence (K//) les E et H calcules plus haut sont multiplies par exp(i*x*inc) en 1D
%   et exp(i*(x*inc(1)+y*inc(2))) en 2D
%  de plus ,suivant le signe de real(inc) les champs sont tronqués 'doucement' du bon coté 
%  pour ne pas 'exploser' dans le changement de coordonnees complexe.
%
%  pml={dpml,xpml,fpml},ou {dpml,xpml,fpmlx,ypml,fpmly}, ou [], ou absent
%
%% EXEMPLE 1D faiseau gaussien focalise sur un dioptre air_verre
%% (le champ au dessus n'est pas significatif)
% ld=1;k0=2*pi/ld;pol=0;nh=1;nb=1.5;
% lcao=ld;d=15*ld;w=ld;
% cao=[-d/2,lcao+i];sym=[];
% init=retinit(d,[-50,50],[],sym,cao);
% uh=retu(d,{nh,pol,k0});ah=retcouche(init,uh,1);
% ub=retu(d,{nb,pol,k0});ab=retcouche(init,ub,1);
% dd=5*ld;teta=20;
% X=@(x) (-x*sind(teta)-dd*cosd(teta));
% Y=@(x) (x*cosd(teta)-dd*sind(teta));
% Fac=@(X) w^2./(w^2+2i*X/(k0*nh));
% amp=@(x) (abs(x)<(d-lcao)/2).*sqrt(Fac(X(x))).*exp(i*k0*nh*X(x)).*exp(-Fac(X(x)).*(Y(x)/w).^2);
% % amp=@(x) (abs(x)<(d-lcao)/2).*sqrt(w^2./(w^2+2i*(-x*sind(teta)-dd*cosd(teta))/(k0*nh))).*exp(i*k0*nh*(x*cosd(teta)-dd*sind(teta))).*exp(-(x*cosd(teta)-dd*sind(teta)).^2./(w^2+2i*(-x*sind(teta)-dd*cosd(teta))/(k0*nh)));
% s_inc=retinc(init,amp,[],0);
% sh=retio(retb(init,ah,1.e-3,0,[],[]),1);sb=retio(retb(init,ab,-1.e-3,0,[],[]),1);
% tab=[[ld, 1, 20,1];[ 0, 1, 3i,1];[ dd, 1, 20,10];[dd*nb, 2, 20,10]];
% x=linspace(-d/2,d/2,201);y=0;
% [e,y,w,o]=retchamp(init,{ah,ab,s_inc},sh,sb,1,x,tab);rettchamp(e,o,x,y,pol,[1,15,i]);
%
%
%% EXEMPLE 2D faiseau gaussien incident sur une fibre carree
%% (le champ au dessus n'est pas significatif)
% ld=1;k0=2*pi/ld;
% teta=[-23,0];
% lcao=[1,1]*ld;d=[8,8]*ld;w=d./8;
% beta=k0*sind(teta);
% cao=[d/2,lcao+i];sym=[0,1,0,0];
% init=retinit(d,[-8,8,-6,6],[],sym,cao);
% uh=retu(d,{1,k0});ah=retcouche(init,uh,1);
% ub=retu(d,{1,[0,0,d/4,3,1],k0});ab=retcouche(init,ub,1);
% amp=@(x,y) ({exp(-(((x+ld*tand(teta(1)))/w(1)).^2+(y/w(2)).^2)),0});
% s_inc=retinc(init,amp,[],beta);
% sh=retio(retb(init,ah,1.e-3,0,[],[]),1);sb=retio(retb(init,ab,-1.e-3,0,[],[]),1);
% tab=[[ld, 1, 20,5];[ 0, 1, 3i,1];[ ld, 1, 20,5];[ 4*ld, 2, 20,10]];
% x=linspace(-d(1)/2,d(1)/2,201);y=0;
% [e,z,w,o]=retchamp(init,{ah,ab,s_inc},sh,sb,1,{x,y},tab);rettchamp(e,o,x,y,z,[1,3,5,i]);

if init{end}.genre==1;% <<<<  CYLINDRES POPOV   
if nargin<3;h=[];end;
if nargin<2;e=1;end;


n=init{1};N=n/2;L=init{2};k=init{3};wk=init{4};sog=init{end}.sog;sym=init{end}.sym;cao=init{end}.cao;Pml=init{10};
%if ~isempty(Pml);rr=retelimine(Pml{2}(1:end-1,1));else;rr=0;end;
if ~isempty(Pml);rr=retelimine(Pml{1}(:,1));else;rr=0;end;

rmax=2*max([init{6};rr]);
cc=10;[r,wr]=retgauss(0,rmax,cc,ceil(2*max(k)*rmax/cc),rr);nr=length(r);
if ~isempty(Pml);
pml=interp1(Pml{2}(:,1),Pml{2}(:,2),r,'nearest');
wr=wr.*pml;
r_phys=retinterp(Pml{1}(:,1),Pml{1}(:,2),r);% r physique
else;
r_phys=r;
end;

if isempty(h);% solution magnetique
if ~isnumeric(e);e=feval(e,r_phys);e=e(:);end;
e=2*besselj(0,k(:)*r)*retdiag(r)*(wr.'.*e);
if sym==1;e=i*e;end;
e=[zeros(N,1);e;zeros(2*N,1)];
else;         % solution electrique
if ~isnumeric(h);e=feval(h,r);e=e(:);end;
e=2*besselj(0,k(:)*r)*retdiag(r)*(wr.'.*e);
if sym==-1;e=i*e;end;
e=[zeros(2*N,1);zeros(N,1);e];
end;          % electrique ou magnetique	
	
	
if sog==1;  %  matrice S 
s={[speye(2*n),[e(1:n);-e(n+1:2*n)] ];n;n;1};    
else;       % matrice G
s={[speye(2*n)];[speye(2*n),-e];n;n;n;n+1};    
end;    
return
end;  %<<<<   FIN CYLINDRES POPOV


if iscell(e);% compatibilite avec une version ancienne
if nargin<5;s=retinc_ancien(init,e,h,inc);else;s=retinc_ancien(init,e,h,inc,pml,varargin{:});end;
return;end;


sens=sign(real(inc));

n=init{1};beta=init{2};
if nargin<5;pml=[];end;if isempty(pml);pml={[1,1],[],[],[],[]};end;

% calcul de e dans la base de Fourier non symetrique

if init{end}.dim==2; % 2D
xpml=pml{2};ypml=pml{4};mx=init{6};my=init{7};m=mx*my;
if ~isempty(xpml);[xpml,ypml]=retautomatique(pml{1},{'PML';xpml;ypml;[]},pml{2:end});end;

d=init{end}.d;cao=init{end}.cao;cao(1)=mod(cao(1),d(1));if cao(1)==0;cao(1)=d(1);end;cao(2)=mod(cao(2),d(2));if cao(2)==0;cao(2)=d(2);end;
[x,wx]=retgauss(0,d(1),50,max(ceil(5*mx/50),10),mod([cao(1)+real(cao(3))/2;cao(1)-real(cao(3))/2;xpml(:)],d(1)));
xx=retcaoo(0,cao(3),d(1),x,-1)-retcaoo(0,cao(3),d(1),d(1)/2,-1)+cao(1)-d(1)/2;x=x+cao(1)-d(1);
[y,wy]=retgauss(0,d(2),50,max(ceil(5*my/50),10),mod([cao(2)+real(cao(4))/2;cao(2)-real(cao(4))/2;ypml(:)],d(2)));
yy=retcaoo(0,cao(4),d(2),y,-1)-retcaoo(0,cao(4),d(2),d(2)/2,-1)+cao(2)-d(2)/2;y=y+cao(2)-d(2);
nx=length(x);ny=length(y);

if isempty(h);champ=champ_2D(xx,yy,d,cao,e,inc,pml{:},varargin{:});else;champ=champ_2D(xx,yy,d,cao,h,inc,pml{:},varargin{:});end;
champ_x=champ{1};champ_y=champ{2};
champ_x=(champ_x.*(wy.'*wx))/(d(1)*d(2));
champ_y=(champ_y.*(wy.'*wx))/(d(1)*d(2));
betax=exp(-i*x.'*beta(1,1:mx));betax=repmat(betax,1,my);
betay=exp(-i*y.'*beta(2,1:mx:m));betay=reshape(repmat(betay,mx,1),ny,m);

champ=zeros(2*m,1);
for ii=1:m;
champ(ii)=sum(sum(champ_x.*(betay(:,ii)*betax(:,ii).')));
champ(ii+m)=sum(sum(champ_y.*(betay(:,ii)*betax(:,ii).')));
end;

    
else; % 1 D
xpml=pml{2};pml=pml(1:3);
d=init{end}.d;cao=init{end}.cao;if isempty(cao);cao=[0,0];end;cao(1)=mod(cao(1),d);if cao(1)==0;cao(1)=d;end;
[x,w]=retgauss(0,d,50,max(ceil(5*init{end}.nfourier/50),10),mod([cao(1)+real(cao(2))/2;cao(1)-real(cao(2))/2;xpml(:)],d));
%[x,w]=retgauss(0,d,50,10,mod([cao(1)+real(cao(2))/2;cao(1)-real(cao(2))/2;xpml(:)],d));
xx=retcaoo(0,cao(2),d,x,-1)-retcaoo(0,cao(2),d,d/2,-1)+cao(1)-d/2;x=x+cao(1)-d;
%figure;plot(x,real(xx));title(rettexte(cao(1)-d,cao(1)));axis tight;stop
if isempty(h);champ=champ_1D(xx,d,cao,e,inc,pml{:},0,varargin{:});else;champ=champ_1D(xx,d,cao,h,inc,pml{:},2,varargin{:});end;
champ=champ.*(w/d);
champ=exp(-i*(beta.'*x))*champ.';
end;   % 1 D 2 D 


if isempty(h);ee=2*champ;hh=zeros(size(champ));
else;hh=2*champ;ee=zeros(size(champ));
end; 

% modification de e et h dans le cas d'utilisation de symetries  
if init{end}.dim==2; 
si=init{8};if ~isempty(si);ee=si{2}*ee;hh=si{4}*hh;end;
else;             
si=init{3};if ~isempty(si);ee=si{2}*ee;hh=si{2}*hh;end;
end;
d=init{end}.d;

sog=init{end}.sog;
if sog==1;  %  matrice S 
s={[speye(2*n),[ee;-hh] ];n;n;1};    
else;       % matrice G
s={[speye(2*n)];[speye(2*n),-[ee;hh]];n;n;n;n+1};    
end;    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=champ_1D(xx,d,cao,amplitude_incidente,inc,dpml,xpml,fpml,pol,varargin)
[x,masque]=calpml(xx,d,cao(1),real(cao(2)),dpml,xpml,fpml,sign(real(inc)));
if isnumeric(amplitude_incidente);a=amplitude_incidente*ones(size(x));else;a=feval(amplitude_incidente,x,varargin{:});end;
a=a.*exp(i*(inc*x)).*masque(1,:,1);
if pol==2;a=a./masque(1,:,2);end;
a(isnan(real(a))|isnan(imag(a)))=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=champ_2D(xx,yy,d,cao,amplitude_incidente,inc,dpml,xpml,fpmlx,ypml,fpmly,varargin);
[x,masquex]=calpml(xx,d(1),cao(1),real(cao(3)),dpml(1),xpml,fpmlx,sign(real(inc(1))));
[y,masquey]=calpml(yy,d(2),cao(2),real(cao(4)),dpml(2),ypml,fpmly,sign(real(inc(2))));

[x,y]=meshgrid(x,y);
if isnumeric(amplitude_incidente);a={amplitude_incidente(1)*ones(size(x)),amplitude_incidente(2)*ones(size(y))};
else;a=feval(amplitude_incidente,x,y,varargin{:});end;
prv=exp(i*(inc(1)*x+inc(2)*y));a{1}=a{1}.*prv;a{2}=a{2}.*prv;

a{1}=retdiag(masquey(1,:,1))*a{1}*retdiag(masquex(1,:,1)./masquex(1,:,2));
a{2}=retdiag(masquey(1,:,1)./masquey(1,:,2))*a{2}*retdiag(masquex(1,:,1));

a{1}(isnan(real(a{1}))|isnan(imag(a{1})))=0;a{2}(isnan(real(a{2}))|isnan(imag(a{2})))=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,masque]=calpml(xx,d,cao,lcao,dpml,xpml,fpml,sens);
masque=ones(1,length(xx),3);
if isempty(xpml);
x=xx;
else;
xbout=[cao-d+lcao/2,cao-lcao/2];
xxbout=retautomatique(dpml,{' ';xbout;[]},xpml,fpml);
xr=real(xx);xi=imag(xx);
f=find((xr>xbout(1))&(xr<xbout(2)));fg=find(xr<=xbout(1));fd=find(xr>=xbout(2));
[masque(1,f,:),xr(f)]=retautomatique(dpml,{masque(1,f,:);xr(f);0},xpml,fpml);
xr(fg)=xr(fg)-xbout(1)+xxbout(1);
xr(fd)=xr(fd)-xbout(2)+xxbout(2);
x=xr+i*xi;
masque=masque(1,:,1:2);
end;
if sens==1;f=find(real(xx)<(cao-d));masque(1,f,1)=masque(1,f,1).*(exp(-((real(xx(f))-cao+d)/(lcao/4)).^2));end;
if  sens==-1;f=find(real(xx)>cao);masque(1,f,1)=masque(1,f,1).*(exp(-((real(xx(f))-cao)/(lcao/4)).^2));end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s=retinc_ancien(init,a,e,sens,varargin);% version ancienne
%   function s=retinc(init,a,e,sens,varargin);
%
%  calcul de la matrice s associee a un champ incident considere comme source
% 
%  si imag(sens)~=0   e fonction permettant le calcul de E en tout point de la periode d
%       E=feval(e,x,varargin{:})
%   en 1D e=EZ et x sont des scalaires
%   en 2 D des vecteurs de dimension 2 e=[EX,EY] x=[x,y] 
%
%  si imag(sens)==0   e composantes de fourier de E dans la base non symetrique
%   en 1D :EZ si pol=0 HZ si pol=2
%   en 2D :EX,EY
%  a descripteur de la couche
%  sens 1 incident du haut,  -1 incident du bas
%  attention si on calcule le champ par e et si on a un changement de coordonnees 
%  dans cao xc->+inf dans l'espace reel ( et pas modulo d)
% 
%  si sens est de dimension 2  sens(2)=betamax pour tronquer les composantes de fourier
% 

n=init{1};beta=init{2};

if length(sens)==1;betamax=inf;else;betamax=sens(2);sens=sens(1);end;

% calcul de e dans la base de Fourier non symetrique
if imag(sens)~=0; % e permet le calcul de E
    
beta=init{2};    
if init{end}.dim==2; % 2D
d=init{end}.d;cao=init{end}.cao;mx=init{6};my=init{7};
[x,wx]=retgauss(0,d(1),5,20);
xx=retcaoo(0,cao(3),d(1),x,-1)-retcaoo(0,cao(3),d(1),d(1)/2,-1)+cao(1)-d(1)/2;x=x+cao(1)-d(1);

[y,wy]=retgauss(0,d(2),5,20);
yy=retcaoo(0,cao(4),d(2),y,-1)-retcaoo(0,cao(4),d(2),d(2)/2,-1)+cao(2)-d(2)/2;y=y+cao(2)-d(2);
nx=length(x);ny=length(y);m=mx*my;
[eex,eey]=feval(e,xx,yy,varargin{:});
eex=spdiags(wx.',0,nx,nx)*eex*spdiags(wy.',0,ny,ny)/(d(1)*d(2));
eey=spdiags(wx.',0,nx,nx)*eey*spdiags(wy.',0,ny,ny)/(d(1)*d(2));

betax=exp(-i*x.'*beta(1,1:mx));betax=repmat(betax,1,my);
betay=exp(-i*y.'*beta(2,1:mx:m));betay=reshape(repmat(betay,mx,1),ny,m);
e=zeros(2*m,1);
for ii=1:m;
if sqrt(abs(beta(1,ii))^2+abs(beta(2,ii))^2)>beamax;   
e(ii)=sum(sum(spdiags(betax(:,ii),0,nx,nx)*eex*spdiags(betay(:,ii),0,ny,ny)));
e(ii+m)=sum(sum(spdiags(betax(:,ii),0,nx,nx)*eey*spdiags(betay(:,ii),0,ny,ny)));
end;
end;

    
else; % 1 D
d=init{end}.d;cao=init{end}.cao;
[x,w]=retgauss(0,d,50,20,mod([cao(1)+real(cao(2))/2,cao(1)-real(cao(2))/2],d));
xx=retcaoo(0,cao(2),d,x,-1)-retcaoo(0,cao(2),d,d/2,-1)+cao(1)-d/2;x=x+cao(1)-d;
ee=feval(e,xx,varargin{:});

ee(isnan(real(ee))|isnan(imag(ee)))=0;
ee=ee.*(w/d);
e=exp(-i*(beta.'*x))*ee.';
f=find(abs(beta)>=betamax);e(f)=0;
end;    
    
    
end;
sens=real(sens);

% modification de e dans le cas d'utilisation de symetries  
if init{end}.dim==2; 
si=init{8};if ~isempty(si);e=si{2}*e;end;
else             
si=init{3};if ~isempty(si);e=si{2}*e;end;
end;
d=init{end}.d;
% calcul de H 
s=retb(init,a,-sens,0,[],[]);s=reteval(s);    
if sens==1;s=inv(s);end;
e=[e;s*e];



%e(1:n)=0;e(n+1:2*n)=2*e(n+1:2*n);
e(n+1:2*n)=2i*e(1:n);e(1:n)=0;
sog=init{end}.sog;pp=size(e,2);
if sog==1;  %  matrice S 
s={[speye(2*n),[e(1:n);-e(n+1:2*n)] ];n;n;1};    
else;       % matrice G
s={[speye(2*n)];[speye(2*n),-e];n;n;n;n+1};    
end;    

