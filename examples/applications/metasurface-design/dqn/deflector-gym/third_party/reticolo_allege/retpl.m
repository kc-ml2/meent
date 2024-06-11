function varargout=retpl(varargin);%varargout=cell(1,nargout);
%         1 D            
%**************************
%                                                % (ancienne version pour memoire    [pl_g,pl_d,eg,ed]=retpl(n,y_dioptre,e,x,y,wy,k0);)
% [pl_g,pl_d,eg,ed]=retpl(n,y_dioptre,e,x,y,wx,wy,k0,PML);
% 
%         y    /\
%              |
% 		**********************
% 		**********************
% 		******  n(1)  ********
% 		**********************   <---- y=y_dioptre  position du dioptre
%              |
%              |   n(2) 
%              |
%              |--------------------> x
% 
% e,x,y,wx,wy  calcules par  [e,y,wy]=retchamp(init,a,sh,sb,inc,x,...				
% wx poids pour integration de Gauss en x
% wy poids pour integration de Gauss en y
%
%        2 D           
% **************************
% function [pl_g,pl_d,eg,ed]=retpl(n,z_dioptre,e,x,y,z,wx,wy,wz,k0,teta,PML);
% 
%         z    /\
%              |
% 		**********************
% 		**********************
% 		******  n(1)  ********
% 		**********************   <---- z=z_dioptre  position du dioptre
%              |
%              |   n(2) 
%              |
%           y  |--------------------> x
% 
% e,x,y,z,wx,wy,wz  calcules par  [e,z,wz]=retchamp(init,a,sh,sb,inc,x,...				
% wx poids pour integration de Gauss en x
% wy poids pour integration de Gauss en y
% wz poids pour integration de Gauss en z
% ATTENTION 
% par defaut le dioptre metal_dielectrique est perpendiculaire à Oz.
%   Si ce n'est pas le cas  z_dioptre doit etre un vecteur de dimension 3 comportant 2 Nan
% ( par exemple: z_dioptre=[nan,2,nan] si le dioptre est y=2)
% Dans tous les cas on peut specifier la direction d'integration
% dioptre perpendiculaire à Oz wx=[] integration en z et y  pl_g,pl_d sont des fonctions de x
%                              wy=[] integration en z et x  pl_g,pl_d sont des fonctions de y
% 							 
% dioptre perpendiculaire à Ox wy=[] integration en x et z  pl_g,pl_d sont des fonctions de y
%                              wz=[] integration en x et y  pl_g,pl_d sont des fonctions de z
% 							 
% dioptre perpendiculaire à Oy wx=[] integration en y et z  pl_g,pl_d sont des fonctions de x
%                              wz=[] integration en y et x  pl_g,pl_d sont des fonctions de z
% le plasmon peut etre tourne d'un angle teta (rad)  facultatif (attention:il s'agit d'une rotation qui n'a rien à voir avec le plasmon oblique)
%
%    dans les 2 cas     
%**************************
%  k0=2*pi/ld
%  pl_g  pl_d  fonction de x(ou y) : amplitude complexe sur le plasmon à droite et à gauche
% obtenue par produit scalaire avec le champ du plasmon
% ed eg: champ des plasmons normalises vers la droite et vers la  gauche
%  le metal peut etre en haut ou en bas
%
% normalisation:le flux de poynting  de eg et ed en x=0 est 1
% ( calcul analytique du flux (real(betapl/nm^2)/imag(khiplm)+real(betapl/nd^2)/imag(khipl))/(4*k0)  )
%
% si e=[], on peut retourner seulement eg et ed aux points x,y :
% [eg,ed]=retpl(n,y_dioptre,e,x,y,wx,wy,k0) wx wy ne servent qu'à déterminer la direction des dioptres
%    en 1D un est [], en 2D 2 sont []. Le poids non vide peut etre remplacé par n'importe quel vecteur non vide ( par exemple 0)
%
% PML: possibilité de 'plonger' le champ dans une pml(complexe) dans la direction perpendiculaire au dioptre, pour eviter d'avoir à calculer les champs sur une hauteur trop grande.
% PML={[haut,points de discontinuité ...   ],[pml en dessous des points du tableau precedent]}
% le champ e doit avoir été calculé avec la même pml . 
%
% GENERALISATION A DES GUIDES QUELCONQUES
% n vecteur colonne :indices du haut en bas suivi de la valeur approximative de neff(non multiplié par k0) du mode puis de pol (même en 2D)
% y_dioptre vecteur colonne cotes des dioptres du haut en bas 
% la normalisation du mode par le vecteur du poynting est faite analytiquement independamment du domaine de calcul
%  (PML non fonctionnel dans ce cas)
%
% NORMALISATION
% par defaut, la normalisation de retpl est le flux du vecteur de poynting.
% Pour normaliser par Lorentz, faire avant le premier passage de retpl :retpl(1) 
% Pour revenir à la normalisation par poynting: retpl(0)
%  On peut demander l'etat retpl -> 0 poynting  ->1 lorentz
%
% See also:RETPLASMON,RETCHAMP,RETOP,RETAUTOMATIQUE

% z_dioptre vecteur colonne cotes des dioptres du haut en bas 
% ou matrice de 3 colonnes avec des nan dans les dimensions inutiles ??????????????????

if nargin<=1;[varargout{1:nargout}]=cal_pl(varargin{:});return;end;
if nargin<10;[varargout{1:nargout}]=retpl_1D(varargin{:});else;[varargout{1:nargout}]=retpl_2D(varargin{:});end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout=retpl_1D(n,y_dioptre,e,x,y,wx,wy,k0,PML);
if nargin==7;k0=wy;wy=wx;wx=[];end; % compatibilite avec une ancienne version  function [pl_g,pl_d,eg,ed]=retpl_1D(n,y_dioptre,e,x,y,wy,k0);
if nargin<9;PML=[];end;
if isempty(wy);% dioptre // oy 
if size(e,3)==3;dim=3;else;dim=-4;end;
if isempty(e);[varargout{1:nargout}]=retpl_1Dy(n,y_dioptre,e,y,x,wx,k0,dim,PML);% calcul uniquement du plasmon
for ii=1:nargout;varargout{ii}=permute(varargout{ii},[2,1,3]);varargout{ii}(:,:,[2,3])=varargout{ii}(:,:,[3,2]);varargout{ii}(:,:,1)=-varargout{ii}(:,:,1);end;
return;end;	
e=e(:,:,1:3);
e=permute(e,[2,1,3]);e=e(:,:,[1,3,2]);e(:,:,1)=-e(:,:,1);
[varargout{1:nargout}]=retpl_1Dy(n,y_dioptre,e,y,x,wx,k0,dim,PML);
for ii=3:nargout;varargout{ii}=permute(varargout{ii},[2,1,3]);varargout{ii}(:,:,[2,3])=varargout{ii}(:,:,[3,2]);varargout{ii}(:,:,1)=-varargout{ii}(:,:,1);end;
else;   % dioptre // ox 
if size(e,3)==3;dim=3;else;dim=4;end;
[varargout{1:nargout}]=retpl_1Dy(n,y_dioptre,e,x,y,wy,k0,dim,PML);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pl_g,pl_d,eg,ed]=retpl_1Dy(n,y_dioptre,e,x,y,wy,k0,dim,PML);
[eg,ed]=cal_pl(y,y_dioptre,x,n,k0,dim,PML);
if isempty(e);pl_g=eg;pl_d=ed;return;end; 
pl_g=scale_1D(e,ed,y,wy)/scale_1D(eg(:,1,:),ed(:,1,:),y,wy);   % amplitude du plasmon vers la gauche fonction de x
pl_d=scale_1D(e,eg,y,wy)/scale_1D(ed(:,1,:),eg(:,1,:),y,wy);   % amplitude du plasmon vers la droite fonction de x

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout=retpl_2D(n,z_dioptre,e,x,y,z,wx,wy,wz,k0,teta,PML);
if nargin<11;teta=0;end;if nargin<12;PML=[];end;
if size(z_dioptre,2)==1;sens=3;else;sens=find(~isnan(z_dioptre(1,:)));z_dioptre=z_dioptre(:,sens);end; 
switch(sens);
	
case 1;% le dioptre est perpendiculaire à  ox   [ x y z ] -->[ z x y]
if ~isempty(e);e=permute(e,[2,3,1,4]);e=e(:,:,:,[2,3,1,5,6,4]);N=3;else;N=1;end;
%[varargout{1:nargout}]=retpl_2Dz(n,z_dioptre,e,z,x,y,wz,wx,wy,k0,teta,PML);
[varargout{1:nargout}]=retpl_2Dz(n,z_dioptre,e,y,z,x,wy,wz,wx,k0,teta,PML);
for ii=N:nargout;varargout{ii}=permute(varargout{ii},[3,1,2,4]);varargout{ii}=varargout{ii}(:,:,:,[3,1,2,6,4,5]);end;

case 2;% le dioptre est perpendiculaire à oy    [ x y z ] -->[ y z x ]
if ~isempty(e);e=permute(e,[3,1,2,4]);e=e(:,:,:,[3,1,2,6,4,5]);N=3;else;N=1;end;
[varargout{1:nargout}]=retpl_2Dz(n,z_dioptre,e,z,x,y,wz,wx,wy,k0,teta,PML);
for ii=N:nargout;varargout{ii}=permute(varargout{ii},[2,3,1,4]);varargout{ii}=varargout{ii}(:,:,:,[2,3,1,5,6,4]);end;
	
case 3;% le dioptre est perpendiculaire à  oz
[varargout{1:nargout}]=retpl_2Dz(n,z_dioptre,e,x,y,z,wx,wy,wz,k0,teta,PML);
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pl_g,pl_d,eg,ed]=retpl_2Dz(n,z_dioptre,e,x,y,z,wx,wy,wz,k0,teta,PML);
if(length(n)>(length(z_dioptre)+1))&(n(end)==0);K=[3,5,6,1,2,4];else K=1:6;end;

if isempty(wx); % <<<<< integrale en y fonction de x
[X,Y]=ndgrid(x,y);X=X*cos(teta)+Y*sin(teta);clear Y;
sz=[length(z),length(x),length(y),6];ed=zeros(sz);eg=zeros(sz);
[egg,edd]=cal_pl(z,z_dioptre,X(:).',n,k0,3,PML);

ed(:,:,:,5)=reshape(edd(:,:,1),sz(1),sz(2),sz(3));% hy
eg(:,:,:,5)=reshape(egg(:,:,1),sz(1),sz(2),sz(3));
ed(:,:,:,1)=reshape(edd(:,:,2),sz(1),sz(2),sz(3));% ex
eg(:,:,:,1)=reshape(egg(:,:,2),sz(1),sz(2),sz(3));
ed(:,:,:,3)=reshape(edd(:,:,3),sz(1),sz(2),sz(3));% ez
eg(:,:,:,3)=reshape(egg(:,:,3),sz(1),sz(2),sz(3));

eg(:,:,:,2)=eg(:,:,:,1)*sin(teta);eg(:,:,:,1)=eg(:,:,:,1)*cos(teta);
eg(:,:,:,4)=-eg(:,:,:,5)*sin(teta);eg(:,:,:,5)=eg(:,:,:,5)*cos(teta);
ed(:,:,:,2)=ed(:,:,:,1)*sin(teta);ed(:,:,:,1)=ed(:,:,:,1)*cos(teta);
ed(:,:,:,4)=-ed(:,:,:,5)*sin(teta);ed(:,:,:,5)=ed(:,:,:,5)*cos(teta);

if isempty(e);pl_g=eg;pl_d=ed;return;end;% calcul des plasmons uniquement 
pl_g=scale_2Dx(e,ed,y,z,wy,wz)/scale_2Dx(eg(:,1,:,:),ed(:,1,:,:),y,z,wy,wz);   % amplitude du plasmon vers la gauche fonction de x
pl_d=scale_2Dx(e,eg,y,z,wy,wz)/scale_2Dx(ed(:,1,:,:),eg(:,1,:,:),y,z,wy,wz);   % amplitude du plasmon vers la droite fonction de x
else;      % <<<<<  integrale en x fonction de y
[X,Y]=ndgrid(x,y);Y=Y*cos(teta)-X*sin(teta);clear X;
sz=[length(z),length(x),length(y),6];ed=zeros(sz);eg=zeros(sz);

[egg,edd]=cal_pl(z,z_dioptre,Y(:).',n,k0,3,PML);

ed(:,:,:,4)=reshape(edd(:,:,1),sz(1),sz(2),sz(3));% hx
eg(:,:,:,4)=reshape(egg(:,:,1),sz(1),sz(2),sz(3));
ed(:,:,:,2)=-reshape(edd(:,:,2),sz(1),sz(2),sz(3));% ey
eg(:,:,:,2)=-reshape(egg(:,:,2),sz(1),sz(2),sz(3));
ed(:,:,:,3)=-reshape(edd(:,:,3),sz(1),sz(2),sz(3));% ez
eg(:,:,:,3)=-reshape(egg(:,:,3),sz(1),sz(2),sz(3));

eg(:,:,:,5)=eg(:,:,:,4)*sin(teta);eg(:,:,:,4)=eg(:,:,:,4)*cos(teta);
eg(:,:,:,2)=-eg(:,:,:,3)*sin(teta);eg(:,:,:,2)=eg(:,:,:,2)*cos(teta);

ed(:,:,:,5)=ed(:,:,:,4)*sin(teta);ed(:,:,:,4)=ed(:,:,:,4)*cos(teta);
ed(:,:,:,2)=-ed(:,:,:,3)*sin(teta);ed(:,:,:,2)=ed(:,:,:,2)*cos(teta);


if isempty(e);pl_g=eg;pl_d=ed;return;end;% calcul des plasmons uniquement  
pl_g=scale_2Dy(e,ed,x,z,wx,wz)/scale_2Dy(eg(:,:,1,:),ed(:,:,1,:),x,z,wx,wz);   % amplitude du plasmon vers la gauche fonction de y
pl_d=scale_2Dy(e,eg,x,z,wx,wz)/scale_2Dy(ed(:,:,1,:),eg(:,:,1,:),x,z,wx,wz);   % amplitude du plasmon vers la droite fonction de y

end;          % <<<<< 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [eg,ed]=cal_pl(y,y_dioptre,x,n,k0,dim,PML);
persistent normalisation;
if nargin==1;normalisation=y;eg=normalisation;return;end;if isempty(normalisation);normalisation=0;end;
if nargin==0;eg=normalisation;return;end;
yst=y;
if length(n)>2;[eg,ed]=cal_guide(y,y_dioptre,x,n,k0,dim,normalisation);return;end;
nm=n(1);nd=n(2);
plasmon=retplasmon(nm,nd,2*pi/k0);     
betapl=plasmon.constante_de_propagation;
khiplm=plasmon.khi_metal;
khipl=plasmon.khi_dielectrique;
betapl=betapl*k0;khiplm=khiplm*k0;khipl=khipl*k0;
ed=zeros(length(y),length(x),dim);eg=zeros(length(y),length(x),dim);

if ~isempty(PML);% modification de y avec la pml
ypml=[PML{1}(:);min(y)-1];
ylim=retelimine([max(y)+1;ypml;y_dioptre]);
yc=(ylim(1:end-1)+ylim(2:end))/2;
pmlc=zeros(size(yc));
for ii=1:length(ypml)-1;
f=find(yc<=ypml(ii) & yc>ypml(ii+1));pmlc(f)=PML{2}(ii);
end;	
Ylim=[0;cumsum(pmlc.*diff(ylim))];
[prv,f_dioptre]=min(abs(ylim-y_dioptre));
y=interp1(ylim,Ylim,y);
y=y_dioptre+y-Ylim(f_dioptre);
end;

[fh,fb]=retfind(yst>y_dioptre);
if isempty(fh);expmd=zeros(0,length(x));expmg=zeros(0,length(x));
else;
expmd=exp(i*khiplm*(y(fh(:)).'-y_dioptre))*exp(i*betapl*x);
expmg=exp(i*khiplm*(y(fh(:)).'-y_dioptre))*exp(-i*betapl*x);
end;
if isempty(fb);expd=zeros(0,length(x));expg=zeros(0,length(x));
else;
expd=exp(i*khipl*(y_dioptre-y(fb(:)).'))*exp(i*betapl*x);
expg=exp(i*khipl*(y_dioptre-y(fb(:)).'))*exp(-i*betapl*x);
end;
ed(fh,:,1)=expmd; % Hz  champ dans le métal  plasmon vers la droite
eg(fh,:,1)=expmg;
ed(fb,:,1)=expd;    % champ dans nd
eg(fb,:,1)=expg;

ed(fh,:,2)=1/(k0*(nm^2))*khiplm*expmd;% -ex
eg(fh,:,2)=1/(k0*(nm^2))*khiplm*expmg;
ed(fb,:,2)=-1/(k0*(nd^2))*khipl*expd;
eg(fb,:,2)=-1/(k0*(nd^2))*khipl*expg;

ed(fh,:,3)=-1/(k0*(nm^2))*betapl*expmd;% -ey
eg(fh,:,3)= 1/(k0*(nm^2))*betapl*expmg;
ed(fb,:,3)=-1/(k0*(nd^2))*betapl*expd;
eg(fb,:,3)=1/(k0*(nd^2))*betapl*expg;
if dim==4; % 1D dioptre//ox
ed(fh,:,4)=khiplm*expmd;% -Dx
eg(fh,:,4)=khiplm*expmg;
ed(fb,:,4)=-khipl*expd;
eg(fb,:,4)=-khipl*expg;
end;
if dim==-4;% 1D dioptre//oy
ed(fh,:,4)=-betapl*expmd;% -Dy
eg(fh,:,4)=betapl*expmg;
ed(fb,:,4)=-betapl*expd;
eg(fb,:,4)=betapl*expg;
end;

% normalisation:le flux de poynting =1 ou lorentz=-4i
if normalisation==1;a=sqrt(plasmon.int_lorentz/4);else;a=sqrt(plasmon.poynting);end;% modif 1 11 2011
%a=sqrt((real(betapl/nm^2)/imag(khiplm)+real(betapl/nd^2)/imag(khipl))/(4*k0));
ed=ed/a;eg=eg/a;
if mean(real(ed(:,1,1)))<0;ed=-ed;eg=-eg;end;

if ~isempty(PML); % champ dans la pml
ypml=[PML{1}(:);-inf];
for ii=1:length(ypml)-1;
f=find(yst<=ypml(ii) & yst>ypml(ii+1));
ed(f,:,3)=ed(f,:,3)*PML{2}(ii);
eg(f,:,3)=eg(f,:,3)*PML{2}(ii);
if dim==-4;
ed(f,:,4)=ed(f,:,4)*PML{2}(ii);
eg(f,:,4)=eg(f,:,4)*PML{2}(ii);
end;

end;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [eg,ed]=cal_guide(y,y_dioptre,x,n,k0,dim,normalisation);
pol=n(end);neff=n(end-1);
n=n(1:end-2);m=length(n);
y_dioptre=y_dioptre(:);n=n(:);y=y(:);x=x(:);
% recherche du mode
%beta=k0*retmode(pol,n,-diff(y_dioptre)*k0,neff);
beta=retmode(pol,n,-diff(y_dioptre),neff*k0,[],[],[],k0);
% amplitudes dans les couches
khi=retsqrt((k0*n).^2-beta^2,-1);mu=k0*n.^pol;
y_dioptre=[y_dioptre(1);y_dioptre;y_dioptre(end)];
% 
% a=zeros(m,2);a(1,1)=1;
% for ii=1:m-1;
% prv1=a(ii,1);if ii==1;prv2=0;else;prv2=a(ii,2)*exp(-i*khi(ii)*(y_dioptre(ii+1)-y_dioptre(ii)));end;
% [prv1,prv2]=deal((prv1+prv2),khi(ii)*mu(ii+1)/(mu(ii)*khi(ii+1))*(prv1-prv2));
% a(ii+1,1)=exp(-i*khi(ii+1)*(y_dioptre(ii+1)-y_dioptre(ii+2)))*(prv1+prv2)/2;
% a(ii+1,2)=(prv1-prv2)/2;
% end
% a(end,1)=0;
% % 


p=(1:m-1).';
prv1=exp(-i*khi(p).*(y_dioptre(p+1)-y_dioptre(p)));
prv2=exp(i*khi(p+1).*(y_dioptre(p+1)-y_dioptre(p+2)));
prv3=khi(p).*mu(p+1)./(mu(p).*khi(p+1));
ii=[1;m+1];jj=[1;m+1];M=[1;1];

ii=[ii;p+1];jj=[jj;p];M=[M;ones(m-1,1)];
ii=[ii;p+1];jj=[jj;p+m];M=[M;prv1];
ii=[ii;p+1];jj=[jj;p+m+1];M=[M;-ones(m-1,1)];
ii=[ii;p+1];jj=[jj;p+1];M=[M;-prv2];

ii=[ii;p+1+m];jj=[jj;p];M=[M;prv3];
ii=[ii;p+1+m];jj=[jj;p+m];M=[M;-prv1.*prv3];
ii=[ii;p+1+m];jj=[jj;p+m+1];M=[M;ones(m-1,1)];
ii=[ii;p+1+m];jj=[jj;p+1];M=[M;-prv2];
M=sparse(ii,jj,M,2*m,2*m);
a=M\[1;zeros(2*m-1,1)];
a=full(reshape(a,m,2));
a(end,1)=0;


[ed,eg]=deal(zeros(length(y),length(x),abs(dim)));
y_dioptre(1)=inf;y_dioptre(end)=-inf;
exp_d=exp(i*beta*x(:).');exp_g=exp(-i*beta*x(:).');Flux=0;
for ii=1:m; % ii <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
f=find((y<=y_dioptre(ii))&(y>y_dioptre(ii+1)));
if ~isempty(f); % <------------- f non vide
if ii==1;prv2=0;else;prv2=a(ii,2)*exp(-i*khi(ii)*(y(f)-y_dioptre(ii)));end;
if ii==m;prv1=0;else;prv1=a(ii,1)*exp(i*khi(ii)*(y(f)-y_dioptre(ii+1)));end;
[prv1,prv2]=deal(prv1+prv2,prv1-prv2);
ed(f,:,1)=prv1*exp_d;% Ez 
eg(f,:,1)=prv1*exp_g;
ed(f,:,2)=(i*khi(ii)/mu(ii))*prv2*exp_d;% Hx
eg(f,:,2)=(i*khi(ii)/mu(ii))*prv2*exp_g;
ed(f,:,3)=(-i*beta/mu(ii))*prv1*exp_d;% Hy
eg(f,:,3)=(i*beta/mu(ii))*prv1*exp_g;
if dim==4; % 1D dioptre//ox
ed(f,:,4)=(i*khi(ii))*prv2*exp_d;% Bx
eg(f,:,4)=(i*khi(ii))*prv2*exp_g;
end;
if dim==-4;% 1D dioptre//oy
% ed(f,:,4)=(i*beta)*prv1*exp_d;% -By
% eg(f,:,4)=(-i*beta)*prv1*exp_g;
ed(f,:,4)=(-i*beta)*prv1*exp_d;% By
eg(f,:,4)=(i*beta)*prv1*exp_g;
end;
end;          % <------------- f non vide

% normalisation
if normalisation==1    % Lorentz
	switch ii;
	case 1;Flux=Flux+a(ii,1)^2*calint(2i*khi(ii),0,y_dioptre(ii)-y_dioptre(ii+1))/mu(ii);	
	case m;Flux=Flux+a(ii,2)^2*calint(-2i*khi(ii),y_dioptre(ii+1)-y_dioptre(ii),0)/mu(ii);
	otherwise;Flux=Flux+a(ii,1)^2*calint(2i*khi(ii),0,y_dioptre(ii)-y_dioptre(ii+1))/mu(ii)+...
	a(ii,2)^2*calint(-2i*khi(ii),y_dioptre(ii+1)-y_dioptre(ii),0)/mu(ii)+...
	2*a(ii,1)*a(ii,2)*exp(i*khi(ii)*(y_dioptre(ii)-y_dioptre(ii+1)))*(y_dioptre(ii)-y_dioptre(ii+1))/mu(ii);
	end;

else;                     % flux de poynting    
	switch ii;
	case 1;Flux=Flux+abs(a(ii,1))^2*calint(-2*imag(khi(ii)),0,y_dioptre(ii)-y_dioptre(ii+1))/mu(ii);
	case m;Flux=Flux+abs(a(ii,2))^2*calint(2*imag(khi(ii)),y_dioptre(ii+1)-y_dioptre(ii),0)/mu(ii);
	otherwise; Flux=Flux+abs(a(ii,1))^2*calint(-2*imag(khi(ii)),0,y_dioptre(ii)-y_dioptre(ii+1))/mu(ii)+...
	abs(a(ii,2))^2*calint(2*imag(khi(ii)),y_dioptre(ii+1)-y_dioptre(ii),0)/mu(ii)+...
	2*real(a(ii,1)*conj(a(ii,2))*exp(-i*(khi(ii)*y_dioptre(ii+1)+conj(khi(ii))*y_dioptre(ii)))*calint(2i*real(khi(ii)),y_dioptre(ii+1),y_dioptre(ii)))/mu(ii);
	end;

end;                    % Lorentz ou flux de poynting  

end;% ii <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if normalisation==1;Flux=sqrt(beta*Flux/2);else;Flux=sqrt(real(beta*Flux/2));end;
% normalisation:le flux de poynting est 1 ou Lorentz=-4i
ed=ed/Flux;eg=eg/Flux;
if mean(real(ed(:,1,1)))<0;ed=-ed;eg=-eg;end;
ed(:,:,2:end)=-i*ed(:,:,2:end);eg(:,:,2:end)=-i*eg(:,:,2:end);% declonage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=calint(u,y1,y2);% somme de y1 à y2 de exp(uy)
if ~isfinite(y1);if abs(u)>100*eps;a=exp(u*y2)/u;else;a=0;end;return;end;
if ~isfinite(y2);if abs(u)>100*eps;a=-exp(u*y1)/u;else;a=0;end;return;end;
a=(y2-y1)*exp(u*(y1+y2)/2)*retsinc(u*(y2-y1)/(2i*pi));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s=scale_1D(e1,e2,y,wy); % produit scalaire de 2 champs fonction de x (vecteur ligne);
s=wy*(e2(:,:,1).*e1(:,:,3)-e1(:,:,1).*e2(:,:,3));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s=scale_2Dx(e1,e2,y,z,wy,wz); % produit scalaire de 2 champs fonction de x
nx=size(e1,2);
s=zeros(nx,1);
for ii=1:nx;
s(ii)=retscale(wz.'*wy,e1(:,ii,:,2),e2(:,ii,:,6))...% E1y * H2z
 -retscale(wz.'*wy,e1(:,ii,:,3),e2(:,ii,:,5))... .% -E1z * H2y
 -retscale(wz.'*wy,e2(:,ii,:,2),e1(:,ii,:,6))... .% -E2y * H1z
 +retscale(wz.'*wy,e2(:,ii,:,3),e1(:,ii,:,5));    % +E2z * H1y
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s=scale_2Dy(e1,e2,x,z,wx,wz); % produit scalaire de 2 champs fonction de y
ny=size(e1,3);
s=zeros(ny,1);
for ii=1:ny;
s(ii)=retscale(wz.'*wx,e1(:,:,ii,3),e2(:,:,ii,4))...% E1z * H2x
 -retscale(wz.'*wx,e1(:,:,ii,1),e2(:,:,ii,6))... .% -E1x * H2z
 -retscale(wz.'*wx,e2(:,:,ii,3),e1(:,:,ii,4))... .% -E2z * H1x
 +retscale(wz.'*wx,e2(:,:,ii,1),e1(:,:,ii,6));    % +E2x * H1z
end;
