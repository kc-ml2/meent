function varargout=retop(varargin)
%  developpement en ondes planes dans un milieu homogene d'indice n 
% 		%%%%%%%%%%%%%%%%
% 		%    2 D       %
% 		%%%%%%%%%%%%%%%%
%  function [Ep,Em,angles]=retop(n,u,v,e,x,y,z,wx,wy,wz,k0,parm);
%
% 
% n:indice  du milieu homogene k0=2*pi/ld
% u,v: incidences ou on veut calculer le developpement en ondes planes
%            E = somme_sur_u_v de (tf(E) *exp(i (u*x+v*y)) )
%    u et v sont 2 vecteurs.Le calcul est fait sur ndgrid(u,v)
% e(z,x,y,1:6) champ calcule par retchamp (6 composantes)
% wx,wy,wz:poids pour l'integration de Fourier
% l'un de ces vecteurs est vide :le developpement en ondes planes est fait dans la direction perpendiculaire
% parm=struct('apod',0,'delt',0,'uvmesh',0); (par defaut)
%     apod:pour l'integration de Fourier 
%     delt:pour la singularite de l'incidence normale 
% si uvmesh=1 u et v sont des vecteurs de meme dimension donnant les couples de valeurs de u et v
% dans les tableaux suivants u ,v sont alors 'soudés' et ces tableaux perdent une dimension
%
% si wz=[] Ep(u,v,1:6,z),Em(u,v,1:6,z),sont les composantes de Fourier de e(z,x,y,1:6) par rapport à x,y fonctions de z (x->u y->v)
% si wx=[] Ep(u,v,1:6,x),Em(u,v,1:6,x),sont les composantes de Fourier de e(z,x,y,1:6) par rapport à y,z fonctions de x (y->u z->v)
% si wy=[] Ep(u,v,1:6,y),Em(u,v,1:6,y),sont les composantes de Fourier de e(z,x,y,1:6) par rapport à z,x fonctions de y (z->u x->v)
% ces composantes sont definies dans le meme repere que e (Ep(u,v,1,z) est Ex etc ...)
%*******************************************************************************
%  mise en forme des ondes planes en 2D (si angles est en sortie)
%
% si length(u)=nu,length(v)=nv,length(z)=nz:
% size(Ep)= size(Em)= [nu,nv,6,nz]
%          teta: [ nu nv ] angle de oz avec kp (0 a pi/2)
%         delta: [ nu nv ] angle de oy avec vp oriente par oz  (-pi a pi)
%          psip: [ nu nv nz] angle(oriente par kp) entre up et la direction principale de polarisation de E
%            kp: [ nu nv 3]  vecteur d'onde UNITAIRE dirige vers le haut
%            vp: [ nu nv 3]  unitaire perpendiculaire au plan d'incidence  (le triedre 0z, kp ,vp etant direct)  
%            up: [ nu nv 3]  up=vp /\ kp (le triedre up vp kp est direct)
%           EEp: [ nu nv 2 nz] composantes de E sur up et vp
%           HHp: [ nu nv 2 nz] composantes de H sur up et vp
%          EEEp: [ nu nv 2 nz] amplitude de E dans le repere deduit de up vp par rotation d'angle psip
%          HHHp: [ nu nv 2 nz] amplitude de E dans le repere deduit de up vp par rotation d'angle psip
%  angles.   Fp: [ nu nv nz]diagramme de diffraction vers le haut en energie (*1/r^2 ...)
%          psim: [ nu nv nz]:angle(oriente par km) entre um et la direction principale de polarisation de E
%            km: [ nu nv 3] vecteur d'onde UNITAIRE dirige vers le bas
%            vm: [ nu nv 3] =vm unitaire perpendiculaire au plan d'incidence  (le triedre 0z, km ,vm est direct)  
%            um: [ nu nv 3] = vm /\ km (le triedre um vm km est direct) 
%           EEm: [ nu nv 2 nz] composantes de E sur um et vm
%           HHm: [ nu nv 2 nz] composantes de H sur um et vm
%          EEEm: [ nu nv 2 nz] amplitude de E dans le repere deduit de um vm par rotation d'angle psim
%          HHHm: [ nu nv 2 nz] amplitude de H dans le repere deduit de um vm par rotation d'angle psim
%            Fm: [ nu nv nz]diagramme de diffraction vers le bas en energie (*1/r^2 ...)
% pour l'incidence normale   (cas degenere) vp=vm=[-sin(delt),cos(delt)] 
% (l'angle delt peut etre introduit en parametre et vaut 0 par defaut)
% 
% 		%%%%%%%%%%%%%%%%
% 		%    1 D       %
% 		%%%%%%%%%%%%%%%%
%  function [Ep,Em,angles]=retop(n,u,e,x,y,wx,wy,pol,k0,parm);
%
% n:indice  du milieu homogene k0=2*pi/ld pol:0 E//  2 H//
% u: incidences où on veut calculer le developpement en ondes planes ( k0*n*sin(teta) )
%            E = somme_sur_u de (tf(E) *exp(i (u*x) )
% e(y,x,:) champ calcule par retchamp (3 ou 4 composantes)
% wx,wy:poids pour l'integration de Fourier
% l'un de ces vecteurs est vide :le developpement en ondes planes est fait dans la direction perpendiculaire
% parm=struct('apod',0); (par defaut)
%     apod:pour l'intégration de Fourier 
% 
% si wx=[] Ep(u,1:3,x),Em(u,1:3,x),sont les composantes de Fourier de e(y,x,1:3) par rapport à y fonctions de x
% si wy=[] Ep(u,1:3,y),Em(u,1:3,y),sont les composantes de Fourier de e(y,x,1:3) par rapport à x fonctions de y
%
%*******************************************************************************
%  mise en forme des ondes planes  en 1D (si angles est en sortie)
% si length(u)=nu,length(z)=ny:
% size(Ep)= size(Em)= [nu,3,nz]
%         teta: [ nu 1 ] angle de kp avec Oy orienté par Oz (-pi/2 a pi/2)
%           kp: [ nu 2 ] vecteur d'onde UNITAIRE dirige vers le haut
%           up: [ nu 2 ] perpendiculaire a kp  (Oz up kp direct) 
%          EEp: [ nu ny] composante de E sur Oz
%          HHp: [ nu ny] composante de H sur up
% angles.   Fp: [ nu ny] diagramme de diffraction vers le haut en energie (*1/r ...)
%           km: [ nu 2 ] vecteur d'onde UNITAIRE dirige vers le bas
%           um: [ nu 2 ] perpendiculaire a km  (Oz um km direct) 
%          EEm: [ nu ny] composante de E sur Oz
%          HHm: [ nu ny] composante de H sur um
%           Fm: [ nu ny] diagramme de diffraction vers le bas en energie (*1/r ...)
%*******************************************************************************
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% correction de Haitao en 1D %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% on ajoute des plasmons sur les cotés.Ces plasmons doivent avoir été calculés auparavent par retpl (ou autrement)
% e doit etre remplacé par: {e,Plg,Pld} où : Pl={pl,x0,[nh,nb],y_dioptre}
% 	pl: amplitude du plasmon comme donné par retpl : donc normalisé par le flux du vecteur de poynting en x=0
%                pl est un scalaire (moyenne ou valeur en un point)
% 	x0: point à partir duquel on ajoute le plasmon
% 	nh: en haut
% 	nb: en bas
% 	y_dioptre: dioptre métal 
% Dans ce cas l'apodisation n'agit que sur le champ moins le plasmon
%
%
% 
% 		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 		%      METHODE DE L'INTEGRALE        %
% 		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  On integre Lorentz sur une boite à l'exterieur de laquelle le milieu est stratifié
%      un des champs est celui calculé ( qui doit satisfaire des COS)
%      l'autre est la réponse du milieu stratifié à une onde plane incidente
%       %%%%%%%%%%%%%%
% 		%     1 D    %
% 		%%%%%%%%%%%%%%
%   angles=retop(n,y,u,  e_x,x_x,y_x,w_x,   e_y,x_y,y_y,w_y, pol,k0,sens);
%
%  n: indices de haut en bas
%  y: cotes des dioptres de haut en bas ( méme origine que dans le calcul des champs)
% u: incidences où on veut calculer le developpement en ondes planes ( k0*n*sin(teta) )
% e_x,x_x,y_x,w_x champ calculé par retchamp sur 2 coupes en y ( y_x) w-x poids pour l'integration en x
% e_y,x_y,y_y,w_y champ calculé par retchamp sur 2 coupes en y ( y_x) w-x poids pour l'integration en x
% real(sens)  1 developpement au dessus   en y(1)    imag(sens)  0 les strates sont perpendiculaires à oy 
%            -1 developpement au dessous  en y(end)                 1 les strates sont perpendiculaires à ox 
%
%  angles structure à champs: 'teta', 'k', 'u', 'EE', 'HH', 'F' 
%
% Modes 
%-------
% angles=retop([n,neff],z,  e_x,x_x,y_x,w_x,   e_y,x_y,y_y,w_y, pol,k0,sens);
%        attention ne pas mettre pol après neff
% angles structure à champs de champs : amp_p, Fp=abs(amp_p)^2 (mode à droite)
%                                       amp_m, Fm=abs(amp_m)^2 (mode à gauche)
%    et aussi les champs 0D(non clonés) ayant servi à engendrer les modes 
%		
%       %%%%%%%%%%%%%%
% 		%     2 D    %
% 		%%%%%%%%%%%%%%
% angles=retop(n,z,u,v, e_xy,x_xy,y_xy,z_xy,wx_xy,wy_xy,   e_zx,x_zx,y_zx,z_zx,wz_zx,wx_zx, e_yz,x_yz,y_yz,z_yz,wy_yz,wz_yz,k0,sens);  
% e_xy,x_xy,y_xy,z_xy,wx_xy,wy_xy, ...% champ sur 2 coupes perpendiculaires à oz avec les points et les poids
% e_zx,x_zx,y_zx,z_zx,wz_zx,wx_zx, ...% champ sur 2 coupes perpendiculaires à oy avec les points et les poids
% e_yz,x_yz,y_yz,z_yz,wy_yz,wz_yz,... % champ sur 2 coupes perpendiculaires à ox avec les points et les poids
% k0,sens,parm);
% real(sens)= 1 developpement au dessus   en z(1)      imag(sens)=  0 les strates sont perpendiculaires à oz 
% real(sens)=-1 developpement au dessous  en z(end)    imag(sens)=  1 les strates sont perpendiculaires à ox 
%                                                      imag(sens)= -1 les strates sont perpendiculaires à oy 
% 
%  parm structure     defaultopt=struct('uvmesh',0,'clone',0);
% si uvmesh=1 u et v sont des vecteurs de meme dimension donnant les couples de valeurs de u et v
%  angles structure à champs: 'teta', 'delta', 'psi', 'k', 'v', 'u' ,'EE' ,'HH' ,'EEE' ,'HHH' ,'F' 
% Modes cylindriques
%-----------------------
% angles=retop([n,neff,pol],z,LL,teta, e_xy,x_xy,y_xy,z_xy,wx_xy,wy_xy,   e_zx,x_zx,y_zx,z_zx,wz_zx,wx_zx, e_yz,x_yz,y_yz,z_yz,wy_yz,wz_yz,k0,sens); 
% sens comme pour les ondes planes
% si LL=[] calcul direct pour les teta
% si LL est non vide  si LL=L1 LLL=L1,  si LL=[L1,L2] LLL=L1:L2,  si LL=[L1,L2,L3], LLL=L1:L2:L3, sinon LL=retelimine de LL
%      calcul de coefficients de Fourier LLL, puis synthese de Fourier en teta
% angles structure à champs de champs f, amp, F f coefficients de Fourier, amp amplitude(teta) F intensite(teta), LLL
%    et aussi les champs 0D(non clonés)ayant servi à engendrer les modes. 
% si [teta,wteta]=retgauss(0,2*pi,...), l'energie sur le mode est sum(angles.F.*wteta(:)) ou 2*pi*sum(abs(f).^2)
%
%       %%%%%%%%%%%%%%%%%%%%%%%%%%%
% 		%    cylindres Popov      %
% 		%%%%%%%%%%%%%%%%%%%%%%%%%%%
% angles=retop(n,z,u,v,  e_x,x_x,z_x,wx_x,    e_z,x_z,z_z,wz_z,   k0,sens,L,sym,parm);    
% e_x,x_x,z_x,wx_x      champ sur 2 coupes perpendiculaires à oz x_x variant de 0 à r avec les points et les poids
% e_z,x_z,y_z,z_z,wz_z  champ pour x_z=r 
% TRES IMPORTANT:
%  Le calcul des champs doit etre IMPERATIVEMENT effectué avec l'option sym =0 meme si on calcule le developpement avec sym=1 ou -1
% ( recalculer init ou faire init{end}.sym=0 avant retchamp )
%
%  sens  1 developpement au dessus   en z(1)
%       -1 developpement au dessous  en z(end)
%  L sym parametres de popov
%
%  parm,u,v,angles:  comme en 2D
%
% Modes cylindriques
%-----------------------
% angles=retop([n,neff,pol],z,L,teta,   e_x,x_x,z_x,wx_x,    e_z,x_z,z_z,wz_z, k0,sym);
% angles structure à champs de champs f, amp, F f coefficient de Fourier sur L, amp amplitude(teta) tenant compte du parametre sym 
% F intensite(teta)  tenant compte du parametre sym et aussi les champs 0D(non clonés)ayant servi à engendrer les modes   
%
%
% % EXEMPLE 2 D
% n=1.5;k0=2*pi/1.2;% Fourier en x z
% y=[-2,2]/(k0*n);wy=[];[z,wz]=retgauss(-100/(k0*n),100/(k0*n),10,25,[-6,6]*max(abs(y)));[x,wx]=retgauss(-100/(k0*n),100/(k0*n),10,25,[-6,6]*max(abs(y)));
% pols=rand(1,6)+i*rand(1,6);e=retpoint(k0*n^2,k0,pols,[0,0,0],x,y,z);% source ponctuelle
% [teta,wt]=retgauss(0,pi/2,25);[delta,wd]=retgauss(0,2*pi,-50);nt=length(teta);nd=length(delta);
% [Teta,Delta]=ndgrid(teta,delta);[wt,wd]=ndgrid(wt,wd);w=wt(:).*wd(:).*sin(Teta(:));
% u=k0*n*sin(Teta).*cos(Delta);v=k0*n*sin(Teta).*sin(Delta);
% [ep,em,angles]=retop(n,u(:),v(:),e,x,y,z,wx,wy,wz,k0,struct('apod',7.25,'uvmesh',1));
%  energie_vers_le_haut=w.'*angles.Fp(:,2)
%  energie_vers_le_bas=w.'*angles.Fm(:,1)
%
% %  EXEMPLE 1 D
% %Fourier en x
% n=1.5;k0=2*pi/1.2;pol=2;
% y=[-2,2]/(k0*n);wy=[];[x,wx]=retgauss(-300/(k0*n),300/(k0*n),20,20,[-6,6]*max(abs(y)));
% pols=randn(1,3)+i*randn(1,3);
% e=retpoint(k0*n^2,k0,pols,[0,0],x,y,pol);
% [teta,wt]=retgauss(-pi/2,pi/2,20,20);u=k0*n*sin(teta);
% [ep,em,angles]=retop(n,u,e,x,y,wx,wy,pol,k0,struct('apod',7.25));%figure;plot(u,squeeze(abs(ep)).^2,u,squeeze(abs(em)).^2,'.')
% Ap=angles.Fp.'*wt.';
% Am=angles.Fm.'*wt.';
% pt=retpoynting(e,[0,1]);Pt=diag([-1,1])*pt*wx.';
% retcompare(Pt,[Am(1),Ap(2)])
%
%
% See also:RETB,RETCHAMP,RETAPOD,RETGAUSS,RETPL,RETHELP_POPOV
%

if iscell(varargin{1});[varargout{1:nargout}]=ondes_planes_popov(varargin{:});return;end;% cylindres Popov

if nargin>23;[varargout{1:nargout}]=caldop_2D(varargin{:});return;end;    % 2 D integrale (version 2010)
if nargin>16;[varargout{1:nargout}]=caldop_popov(varargin{:});return;end; % popov integrale (version 2010)
if nargin>12;
if length(varargin{1})>length(varargin{2})+2;[varargout{1:nargout}]=caldop_modes_popov(varargin{:});% modes circulaires symetrie popov
else;[varargout{1:nargout}]=caldop_1D(varargin{:}); % 1 D integrale (version 2010)
end;
return;end;
if nargin>10;[varargout{1:nargout}]=ondes_planes_2D(varargin{:});return;end;% 2 D
if nargin>5;[varargout{1:nargout}]=ondes_planes_1D(varargin{:});return;end;% 1 D
[varargout{1:nargout}]=ondes_planes_1D_ancien(varargin{:}); % compatibilitee avec une ancienne version 1 D

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout=ondes_planes_1D(n,u,e,x,y,wx,wy,pol,k0,parm);
if nargin<10;parm=[];end;
if nargin<9;k0=1;end;
if nargin<8;pol=0;end;
if nargin<7;wy=[];end;
ep=retep(n,pol,k0);

defaultopt=struct('apod',0);
apod=retoptimget(parm,'apod',defaultopt,'fast');
cal_angles=nargout>2;
if iscell(e);e{1}=e{1}(:,:,1:3);else;e=e(:,:,1:3);end;% cas de la correction de Haitao


if isempty(wx);%[ x y ] -->[ y x]
if iscell(e);e{1}=permute(e{1},[2,1,3]);e{1}=e{1}(:,:,[1,3,2]);e{1}(:,:,1)=-e{1}(:,:,1); % cas de la correction de Haitao
else;e=permute(e,[2,1,3]);e=e(:,:,[1,3,2]);e(:,:,1)=-e(:,:,1);end;
[varargout{1:nargout}]=ondes_planes_1Dy(ep,u,e,y,x,wy,apod,cal_angles,k0);
for ii=1:min(2,nargout);varargout{ii}(:,1,:)=-varargout{ii}(:,1,:);varargout{ii}=varargout{ii}(:,[1,3,2],:);end;
return;end;
if isempty(wy);
[varargout{1:nargout}]=ondes_planes_1Dy(ep,u,e,x,y,wx,apod,cal_angles,k0);
return;end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Ep,Em,angles]=ondes_planes_1Dy(ep,u,e,x,y,wx,apod,cal_angles,k0);
if iscell(e);% cas de la correction de Haitao
Plg=e{2};Pld=e{3};Haitao=1;
e=e{1};	
else;Haitao=0;end;
e(:,:,2:3)=i*e(:,:,2:3);% clonnage
nx=length(x);ny=length(y);nu=length(u);ne=3;
Ep=zeros(ny,nu,ne);Em=zeros(ny,nu,ne);

if Haitao;% on ajoute la Tf des plasmons sur les cotès
if apod~=0;epl=zeros(size(e));x1=min(x);x2=max(x);[fg,fd]=retfind(x<(x1+x2)/2);end;
if ~isempty(Pld);% droite
[pl_g,pl_d,eg,ed]=retpl(Pld{3},Pld{4},zeros(length(y),1,3),0,y,[],1,k0);	
ed(:,:,2:3)=i*ed(:,:,2:3);% clonnage
if length(Pld{3})==2;beta_pl=retplasmon(Pld{3}(1),Pld{3}(2),2*pi/k0);beta_pl=beta_pl.constante_de_propagation;else;beta_pl=retmode(Pld{3}(end),Pld{3}(1:end-2),-diff(Pld{4})*k0,Pld{3}(end-1));end;
prv=i*(k0*beta_pl-u(:).');
for ie=1:2;
Ep(:,:,ie)=Ep(:,:,ie)-(Pld{1}/(2*pi))*(ed(:,1,ie)*(exp(prv*Pld{2})./prv));
if apod~=0;epl(:,fd,ie)=epl(:,fd,ie)+Pld{1}*ed(:,1,ie)*exp(i*k0*beta_pl*x(fd));end;
end;
end;	
if ~isempty(Plg); % gauche
[pl_g,pl_d,eg,ed]=retpl(Plg{3},Plg{4},zeros(length(y),1,3),0,y,[],1,k0);	
eg(:,:,2:3)=i*eg(:,:,2:3); % clonnage
if length(Plg{3})==2;beta_pl=retplasmon(Plg{3}(1),Plg{3}(2),2*pi/k0);beta_pl=beta_pl.constante_de_propagation;else;beta_pl=retmode(Plg{3}(end),Plg{3}(1:end-2),-diff(Plg{4})*k0,Plg{3}(end-1));end;
prv=-i*(k0*beta_pl+u(:).');
for ie=1:2;
Ep(:,:,ie)=Ep(:,:,ie)+(Plg{1}/(2*pi))*(eg(:,1,ie)*(exp(prv*Plg{2})./prv));
if apod~=0;epl(:,fg,ie)=epl(:,fg,ie)+Plg{1}*eg(:,1,ie)*exp(-i*k0*beta_pl*x(fg));end;
end;
end;
end;% Haitao

%ax=retdiag(wx.*retchamp([apod,nx]))*exp(-i*x(:)*u(:).')/(2*pi);
if apod~=0;apodise=interp1(linspace(min(x),max(x),100),retchamp([apod,100]),x);
if Haitao;for ie=1:2;e(:,:,ie)=epl(:,:,ie)+(e(:,:,ie)-epl(:,:,ie))*retdiag(apodise);end;% on n'apodise que la partie sans plasmons
else;for ie=1:2;e(:,:,ie)=e(:,:,ie)*retdiag(apodise);end;end;
end;

ax=retdiag(wx)*exp(-i*x(:)*u(:).')/(2*pi);
for ie=1:2;Ep(:,:,ie)=Ep(:,:,ie)+e(:,:,ie)*ax;end;

khi=retsqrt(ep(1)*ep(2)-u.^2,-1);khi=repmat(khi,ny,1);
[Em(:,:,1),Em(:,:,2)]=deal(.5*(Ep(:,:,1)+(i/ep(3))*Ep(:,:,2)./khi),.5*(-(i*ep(3))*Ep(:,:,1).*khi+Ep(:,:,2)));
Em(:,:,3)=-(i/ep(2))*repmat(u,ny,1).*Em(:,:,1);
[Ep(:,:,1),Ep(:,:,2)]=deal(.5*(Ep(:,:,1)-(i/ep(3))*Ep(:,:,2)./khi),.5*((i*ep(3))*Ep(:,:,1).*khi+Ep(:,:,2)));
Ep(:,:,3)=-(i/ep(2))*repmat(u,ny,1).*Ep(:,:,1);
for ie=1:ne;
Ep(:,:,ie)=exp(-i*khi.*repmat(y(:),1,nu)).*Ep(:,:,ie);
Em(:,:,ie)=exp(i*khi.*repmat(y(:),1,nu)).*Em(:,:,ie);
end;
Ep(:,:,2:3)=-i*Ep(:,:,2:3);Em(:,:,2:3)=-i*Em(:,:,2:3);% de_clonnage


if cal_angles;
%   mise en forme des ondes planes  en 1D
km=[u(:),-khi(1,:).'];km=km./repmat(sqrt(sum((km.^2),2)),1,size(km,2));% vecteurs d'onde normalises
um=[km(:,2),-km(:,1)];% um= -ez vect.  km
HHm=repmat(um(:,1).',ny,1).*Em(:,:,2)+repmat(um(:,2).',ny,1).*Em(:,:,3);
kp=km;kp(:,2)=-kp(:,2);
up=um;up(:,1)=-up(:,1);
HHp=repmat(up(:,1).',ny,1).*Ep(:,:,2)+repmat(up(:,2).',ny,1).*Ep(:,:,3);
teta=atan2(real(kp(:,1)),real(kp(:,2)));
EEp=Ep(:,:,1).';EEm=Em(:,:,1).';% on met y en dernier
HHp=HHp.';HHm=HHm.';
Fp=pi*sqrt(ep(1)*ep(2))*retdiag(cos(teta).^2)*real(EEp.*conj(HHp));Fm=pi*sqrt(ep(1)*ep(2))*retdiag(cos(teta).^2)*real(EEm.*conj(HHm));
angles=struct('teta',teta,'kp',kp,'up',up,'EEp',EEp,'HHp',HHp,'Fp',full(Fp),'km',km,'um',um,'EEm',EEm,'HHm',HHm,'Fm',full(Fm));

end;
Ep=permute(Ep,[2,3,1]);Em=permute(Em,[2,3,1]);% on met y en dernier

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout=ondes_planes_2D(n,u,v,e,x,y,z,wx,wy,wz,k0,parm);
if nargin<12;parm=[];end;
if nargin<11;k0=1;end;
if nargin<10;wz=[];end;
ep=ret2ep(n,k0);

if iscell(ep);ep=ret2ep(ep{:});end;

defaultopt=struct('apod',0,'delt',0,'uvmesh',0,'clone',0);
if nargin<11;parm=[];end;
if isempty(parm);parm=defaultopt;end;
apod=retoptimget(parm,'apod',defaultopt,'fast');
cal_angles=nargout>2;
delt=retoptimget(parm,'delt',defaultopt,'fast');
uvmesh=retoptimget(parm,'uvmesh',defaultopt,'fast');
clone=retoptimget(parm,'clone',defaultopt,'fast');
if clone==0;e(:,:,:,4:6)=i*e(:,:,:,4:6);end% clonage


if isempty(wx);% [ x y z ] -->[ y z x ]
e=permute(e,[2,3,1,4]);e=e(:,:,:,[2,3,1,5,6,4]);
[varargout{1:nargout}]=ondes_planes_2Dz(ep([2,3,1,5,6,4]),u,v,e,y,z,x,wy,wz,apod,cal_angles,delt,uvmesh);
if uvmesh==1;for ii=1:min(2,nargout);varargout{ii}=varargout{ii}(:,[3,1,2,6,4,5],:);end;% remise en ordre des composantes
else;for ii=1:min(2,nargout);varargout{ii}=varargout{ii}(:,:,[3,1,2,6,4,5],:);end;end;

return;end;
if isempty(wy);% [ x y z ] -->[ z x y]
e=permute(e,[3,1,2,4]);e=e(:,:,:,[3,1,2,6,4,5]);
[varargout{1:nargout}]=ondes_planes_2Dz(ep([3,1,2,6,4,5]),u,v,e,z,x,y,wz,wx,apod,cal_angles,delt,uvmesh);
if uvmesh==1;for ii=1:min(2,nargout);varargout{ii}=varargout{ii}(:,[2,3,1,5,6,4],:);end;% remise en ordre des composantes
else;for ii=1:min(2,nargout);varargout{ii}=varargout{ii}(:,:,[2,3,1,5,6,4],:);end;end;

return;end;
if isempty(wz);
[varargout{1:nargout}]=ondes_planes_2Dz(ep,u,v,e,x,y,z,wx,wy,apod,cal_angles,delt,uvmesh);
return;end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Ep,Em,angles]=ondes_planes_2Dz(ep,u,v,e,x,y,z,wx,wy,apod,cal_angles,delt,uvmesh);
  
nx=length(x);ny=length(y);nz=length(z);ne=size(e,4);
if imag(uvmesh)==0;% ****** pas cylindres popov
if apod~=0;
apodise_x=interp1(linspace(min(x),max(x),100),retchamp([apod,100]),x);
apodise_y=interp1(linspace(min(y),max(y),100),retchamp([apod,100]),y);
else;apodise_x=1;apodise_y=1;end;	
	
if uvmesh==1; % <----u v meshed
nuv=length(u);
Ep=zeros(nz,nuv,ne);Em=zeros(nz,nuv,ne);
e=reshape(e,[nz,nx*ny,ne]);
wx=wx.*apodise_x/(2*pi);wy=wy.*apodise_y/(2*pi);
%wx=wx.*retchamp([apod,nx])/(2*pi);wy=wy.*retchamp([apod,ny])/(2*pi);
for iuv=1:nuv;%prv=ax(:,iuv)*ay(iuv,:);
prv=(exp(-i*u(iuv)*x(:)).*wx(:))*(exp(-i*v(iuv)*y(:)).*wy(:)).';
for ie=[1,2,4,5];
Ep(:,iuv,ie)=e(:,:,ie)*prv(:);
end;end;

clear e;
else;         % <----u v non  meshed
% ax=(exp(-i*u(:)*x(:).')*retdiag(wx.*retchamp([apod,nx]))/(2*pi)).';
% ay=exp(-i*v(:)*y(:).')*retdiag(wy.*retchamp([apod,ny]))/(2*pi);
ax=(exp(-i*u(:)*x(:).')*retdiag(wx.*apodise_x)/(2*pi)).';
ay=exp(-i*v(:)*y(:).')*retdiag(wy.*apodise_y)/(2*pi);
nu=length(u);nv=length(v);nuv=nu*nv;
Ep=zeros(nz,nu,nv,ne);Em=zeros(nz,nu,nv,ne);
for iu=1:nu;for iv=1:nv;
prv=ax(:,iu)*ay(iv,:);
prv=reshape(repmat(full(prv(:).'),nz,1),[nz,nx,ny,1]);% full sinon bug si nx=1
for ie=[1,2,4,5];
Ep(:,iu,iv,ie)=sum(sum(e(:,:,:,ie).*prv,2),3);
end;
end;end;
clear e;
[u,v]=ndgrid(u,v);
Ep=reshape(Ep,[nz,nuv,ne]);Em=reshape(Em,[nz,nuv,ne]);
end;         % <----u v meshed ? 
else;                % ********** cylindres popov
uvmesh=real(uvmesh);
Ep=e;clear e;
if uvmesh==1; % <----u v meshed
nuv=length(u);
else;         % <----u v non  meshed
nu=length(u);nv=length(v);nuv=nu*nv;
[u,v]=ndgrid(u,v);
end;
end;

khi=retbidouille(retsqrt(ep(6)*ep(3)-u.^2-v.^2,-1));
khi=repmat(khi(:).',nz,1);
u=repmat(u(:).',nz,1);
v=repmat(v(:).',nz,1);

Ep(:,:,3)=(i/ep(6))*(u.*Ep(:,:,5)-v.*Ep(:,:,4));
Ep(:,:,6)=(i/ep(3))*(u.*Ep(:,:,2)-v.*Ep(:,:,1));

[Em(:,:,1),Em(:,:,2),Em(:,:,4),Em(:,:,5)]=deal(...
.5*( Ep(:,:,1)+i*(ep(2)*Ep(:,:,5)+i*u.*Ep(:,:,3))./khi),...
.5*( Ep(:,:,2)+i*(-ep(1)*Ep(:,:,4)+i*v.*Ep(:,:,3))./khi),...
.5*( Ep(:,:,4)+i*(ep(5)*Ep(:,:,2)+i*u.*Ep(:,:,6))./khi),...
.5*( Ep(:,:,5)+i*(-ep(4)*Ep(:,:,1)+i*v.*Ep(:,:,6))./khi));
Em(:,:,3)=(i/ep(6))*(u.*Em(:,:,5)-v.*Em(:,:,4));
Em(:,:,6)=(i/ep(3))*(u.*Em(:,:,2)-v.*Em(:,:,1));

[Ep(:,:,1),Ep(:,:,2),Ep(:,:,4),Ep(:,:,5)]=deal(...
.5*( Ep(:,:,1)-i*(ep(2)*Ep(:,:,5)+i*u.*Ep(:,:,3))./khi),...
.5*( Ep(:,:,2)-i*(-ep(1)*Ep(:,:,4)+i*v.*Ep(:,:,3))./khi),...
.5*( Ep(:,:,4)-i*(ep(5)*Ep(:,:,2)+i*u.*Ep(:,:,6))./khi),...
.5*( Ep(:,:,5)-i*(-ep(4)*Ep(:,:,1)+i*v.*Ep(:,:,6))./khi));
Ep(:,:,3)=(i/ep(6))*(u.*Ep(:,:,5)-v.*Ep(:,:,4));
Ep(:,:,6)=(i/ep(3))*(u.*Ep(:,:,2)-v.*Ep(:,:,1));

Ep(:,:,4:6)=-i*Ep(:,:,4:6);Em(:,:,4:6)=-i*Em(:,:,4:6);% de_clonnage
for ie=1:ne;
Ep(:,:,ie)=exp(-i*khi.*repmat(z(:),[1,nuv])).*Ep(:,:,ie);
Em(:,:,ie)=exp(i*khi.*repmat(z(:),[1,nuv])).*Em(:,:,ie);
end;
if cal_angles==1;
khi=khi(1,:);u=u(1,:);v=v(1,:);

km=[u(:),v(:),-khi(:)];km=km./repmat(sqrt(sum((km.^2),2)),1,size(km,2));%vecteurs d'onde normalises
vm=[-km(:,2),km(:,1),zeros(size(km,1),1)];
ii=find(abs(km(:,1).^2+km(:,2).^2)<100*eps);vm(ii,1)=-sin(delt);vm(ii,2)=cos(delt);%incidence normale: delt valeur par defaut de delt
vm=vm./repmat(sqrt(sum((vm.^2),2)),1,size(vm,2));
um=[km(:,3).*vm(:,2),-km(:,3).*vm(:,1),km(:,2).*vm(:,1)-km(:,1).*vm(:,2)];%um= vm vect.  km
kp=km;kp(:,3)=-kp(:,3);
vp=vm;up=um;up(:,1:2)=-up(:,1:2);

psip=zeros(nz,nuv);psim=zeros(nz,nuv);
EEp=zeros(nz,nuv,2);EEEp=zeros(nz,nuv,2);HHp=zeros(nz,nuv,2);HHHp=zeros(nz,nuv,2);
EEm=zeros(nz,nuv,2);EEEm=zeros(nz,nuv,2);HHm=zeros(nz,nuv,2);HHHm=zeros(nz,nuv,2);

%EEEEm=zeros(nz,nuv);HHHHm=zeros(nz,nuv);
vp=vp.';up=up.';kp=kp.';vm=vm.';um=um.';km=km.';

for iz=1:nz;

EEp(iz,:,1)=Ep(iz,:,1).*up(1,:)+Ep(iz,:,2).*up(2,:)+Ep(iz,:,3).*up(3,:);
HHp(iz,:,1)=Ep(iz,:,4).*up(1,:)+Ep(iz,:,5).*up(2,:)+Ep(iz,:,6).*up(3,:);
EEm(iz,:,1)=Em(iz,:,1).*um(1,:)+Em(iz,:,2).*um(2,:)+Em(iz,:,3).*um(3,:);
HHm(iz,:,1)=Em(iz,:,4).*um(1,:)+Em(iz,:,5).*um(2,:)+Em(iz,:,6).*um(3,:);

EEp(iz,:,2)=Ep(iz,:,1).*vp(1,:)+Ep(iz,:,2).*vp(2,:)+Ep(iz,:,3).*vp(3,:);
HHp(iz,:,2)=Ep(iz,:,4).*vp(1,:)+Ep(iz,:,5).*vp(2,:)+Ep(iz,:,6).*vp(3,:);
EEm(iz,:,2)=Em(iz,:,1).*vm(1,:)+Em(iz,:,2).*vm(2,:)+Em(iz,:,3).*vm(3,:);
HHm(iz,:,2)=Em(iz,:,4).*vm(1,:)+Em(iz,:,5).*vm(2,:)+Em(iz,:,6).*vm(3,:);

% analyse des polarisations
psip(iz,:)=.5*atan2(real(EEp(iz,:,1)).*real(EEp(iz,:,2))+imag(EEp(iz,:,1)).*imag(EEp(iz,:,2)),.5*(abs(EEp(iz,:,1)).^2-abs(EEp(iz,:,2)).^2));
EEEp(iz,:,1)=EEp(iz,:,1).*cos(psip(iz,:))+EEp(iz,:,2).*sin(psip(iz,:));EEEp(iz,:,2)=-EEp(iz,:,1).*sin(psip(iz,:))+EEp(iz,:,2).*cos(psip(iz,:));
ii=find(abs(EEEp(iz,:,2))>abs(EEEp(iz,:,1)));psip(iz,ii)=psip(iz,ii)+pi/2;[EEEp(iz,ii,1),EEEp(iz,ii,2)]=deal(EEEp(iz,ii,2),-EEEp(iz,ii,1));
ii=find(psip(iz,:)<=-pi/2+100*eps);psip(iz,ii)=psip(iz,ii)+pi;EEEp(iz,ii,:)=-EEEp(iz,ii,:);
HHHp(iz,:,1)=HHp(iz,:,1).*cos(psip(iz,:))+HHp(iz,:,2).*sin(psip(iz,:));HHHp(iz,:,2)=-HHp(iz,:,1).*sin(psip(iz,:))+HHp(iz,:,2).*cos(psip(iz,:));
ii=find(abs(psip(iz,:)-pi/2)<100*eps);psip(iz,ii)=pi/2;ii=find(abs(psip(iz,:))<100*eps);psip(iz,ii)=0;

psim(iz,:)=.5*atan2(real(EEm(iz,:,1)).*real(EEm(iz,:,2))+imag(EEm(iz,:,1)).*imag(EEm(iz,:,2)),.5*(abs(EEm(iz,:,1)).^2-abs(EEm(iz,:,2)).^2));
EEEm(iz,:,1)=EEm(iz,:,1).*cos(psim(iz,:))+EEm(iz,:,2).*sin(psim(iz,:));EEEm(iz,:,2)=-EEm(iz,:,1).*sin(psim(iz,:))+EEm(iz,:,2).*cos(psim(iz,:));
ii=find(abs(EEEm(iz,:,2))>abs(EEEm(iz,:,1)));psim(iz,ii)=psim(iz,ii)+pi/2;[EEEm(iz,ii,1),EEEm(iz,ii,2)]=deal(EEEm(iz,ii,2),-EEEm(iz,ii,1));
ii=find(psim(iz,:)<=-pi/2+100*eps);psim(iz,ii)=psim(iz,ii)+pi;EEEm(iz,ii,:)=-EEEm(iz,ii,:);
HHHm(iz,:,1)=HHm(iz,:,1).*cos(psim(iz,:))+HHm(iz,:,2).*sin(psim(iz,:));HHHm(iz,:,2)=-HHm(iz,:,1).*sin(psim(iz,:))+HHm(iz,:,2).*cos(psim(iz,:));
ii=find(abs(psim(iz,:)-pi/2)<100*eps);psim(iz,ii)=pi/2;ii=find(abs(psim(iz,:))<100*eps);psim(iz,ii)=0;

end; %iz

vp=vp.';up=up.';kp=kp.';vm=vm.';um=um.';km=km.';
teta=acos(kp(:,3));delta=atan2(-real(vm(:,1)),real(vm(:,2)));

if uvmesh~=1; % <----u v pas meshed
Ep=reshape(Ep,nz,nu,nv,6);Em=reshape(Em,nz,nu,nv,6);
EEp=reshape(EEp,nz,nu,nv,2);EEEp=reshape(EEEp,nz,nu,nv,2);HHp=reshape(HHp,nz,nu,nv,2);HHHp=reshape(HHHp,nz,nu,nv,2);
EEm=reshape(EEm,nz,nu,nv,2);EEEm=reshape(EEEm,nz,nu,nv,2);HHm=reshape(HHm,nz,nu,nv,2);HHHm=reshape(HHHm,nz,nu,nv,2);
psip=reshape(psip,nz,nu,nv);kp=reshape(kp,nu,nv,3);up=reshape(up,nu,nv,3);vp=reshape(vp,nu,nv,3);
psim=reshape(psim,nz,nu,nv);km=reshape(km,nu,nv,3);um=reshape(um,nu,nv,3);vm=reshape(vm,nu,nv,3);
% on met z en dernier
EEp=permute(EEp,[2,3,4,1]);EEm=permute(EEm,[2,3,4,1]);EEEp=permute(EEEp,[2,3,4,1]);EEEm=permute(EEEm,[2,3,4,1]);
HHp=permute(HHp,[2,3,4,1]);HHm=permute(HHm,[2,3,4,1]);HHHp=permute(HHHp,[2,3,4,1]);HHHm=permute(HHHm,[2,3,4,1]);
psip=permute(psip,[2,3,1]);psim=permute(psim,[2,3,1]);
Fp=2*pi^2*ep(1)*ep(4)*real(EEp(:,:,1,:).*conj(HHp(:,:,2,:))-EEp(:,:,2,:).*conj(HHp(:,:,1,:)));
Fm=2*pi^2*ep(1)*ep(4)*real(EEm(:,:,1,:).*conj(HHm(:,:,2,:))-EEm(:,:,2,:).*conj(HHm(:,:,1,:)));
Fp=retdiag(cos(teta).^2)*reshape(Fp,[nuv,nz]);Fp=reshape(Fp,[nu,nv,nz]);
Fm=retdiag(cos(teta).^2)*reshape(Fm,[nuv,nz]);Fm=reshape(Fm,[nu,nv,nz]);
teta=reshape(teta,nu,nv);delta=reshape(delta,nu,nv);


else;%   <----u v meshed
% on met z en dernier
EEp=permute(EEp,[2,3,1]);EEm=permute(EEm,[2,3,1]);EEEp=permute(EEEp,[2,3,1]);EEEm=permute(EEEm,[2,3,1]);
HHp=permute(HHp,[2,3,1]);HHm=permute(HHm,[2,3,1]);HHHp=permute(HHHp,[2,3,1]);HHHm=permute(HHHm,[2,3,1]);
psip=permute(psip,[2,1]);psim=permute(psim,[2,1]);

Fp=2*pi^2*ep(1)*ep(4)*real(EEp(:,1,:).*conj(HHp(:,2,:))-EEp(:,2,:).*conj(HHp(:,1,:)));Fp=retdiag(cos(teta).^2)*reshape(Fp,[nuv,nz]);
Fm=2*pi^2*ep(1)*ep(4)*real(EEm(:,1,:).*conj(HHm(:,2,:))-EEm(:,2,:).*conj(HHm(:,1,:)));Fm=retdiag(cos(teta).^2)*reshape(Fm,[nuv,nz]);

end;% <----u v meshed %

%angles={teta,delta,psip,kp,up,vp,EEp,HHp,EEEp,HHHp,psim,km,um,vm,EEm,HHm,EEEm,HHHm};
angles=struct('teta',teta,'delta',delta,'psip',psip,'kp',kp,'vp',vp,'up',up,'EEp',EEp,'HHp',HHp,'EEEp',EEEp,'HHHp',HHHp,'Fp',full(Fp),'psim',psim,'km',km,'vm',vm,'um',um,'EEm',EEm,'HHm',HHm,'EEEm',EEEm,'HHHm',HHHm,'Fm',full(Fm));
end;  % cal_angles
if uvmesh==1; % <----u v meshed
Ep=permute(Ep,[2,3,1]);Em=permute(Em,[2,3,1]);% on met z en dernier
else;
Ep=reshape(Ep,nz,nu,nv,6);Em=reshape(Em,nz,nu,nv,6);
Ep=permute(Ep,[2,3,4,1]);Em=permute(Em,[2,3,4,1]);% on met z en dernier
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [eep,eem]=ondes_planes_1D_ancien(u,ep,e,yy,ww);% ancienne version 1 D
%developpement en ondes planes dans un milieu homogene ep=[epsilon;mux;1/muy] 
% constante de propagation sur y:u 
%eep: x croissant eem x decroissant
%ez=e(:,1) hy=e(:,2) champs donnes aux points yy (poids d'integration ww) vecteurs ligne
eez=ww*((e(:,1)*ones(size(u))).*exp(-i.*yy.'*u));
hhy=ww*((e(:,3)*ones(size(u))).*exp(-i.*yy.'*u));
hhy=-ep(2)*hhy./sqrt(ep(1)*ep(2)-u.^2);
eep=(eez+hhy)./(4*pi);eem=(eez-hhy)./(4*pi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout=ondes_planes_popov(init,sh,sb,a,n,u,v,z,inc,k0,parm);
%  function [Ep,Em,angles]=retop(init,sh,sb,a,n,u,v,z,inc,k0,parm);
% modif Alex: ajout de inc


% if nargin < 11;
% 	% assert(nargin == 10);% Pas valable en Matlab 7.3
% 	parm = k0;
% 	k0 = inc;
% 	inc = 1;
% end
if nargin==9;[inc,k0,parm]=deal(1,inc,[]);end;% modif 3 2011
if nargin==10;if isstruct(k0)|isempty(k0);[inc,k0,parm]=deal(1,inc,k0);else;parm=[];end;end;


sh=retio(sh);sb=retio(sb);a=retio(a);

N=init{end}.nhanckel;sym=init{end}.sym;L=init{2};k=init{3};wk=init{4};cao=init{5};
ep_mu=ret2ep(n,k0);
if iscell(ep_mu);ep_mu=ret2ep(ep_mu{:});end;
defaultopt=struct('uvmesh',0,'delt',0);
%if nargin<10;parm=[];end;
if isempty(parm);parm=defaultopt;end;

uvmesh=retoptimget(parm,'uvmesh',defaultopt,'fast');
delt=retoptimget(parm,'delt',defaultopt,'fast');
if uvmesh==0;[uu,vv]=ndgrid(u,v);uu=uu(:);vv=vv(:);else;uu=u(:);vv=v(:);end;

kk=sqrt(uu.^2+vv.^2);phi_p=atan2(vv,uu)-pi/2;

Ep=zeros(length(z),numel(uu),6);
if sym==0;
expp=exp(i*(L+1)*phi_p);expm=exp(i*(L-1)*phi_p);
else;
cosp=cos((L+1)*phi_p);cosm=cos((L-1)*phi_p);sinp=sin((L+1)*phi_p);sinm=sin((L-1)*phi_p);
end;

if isfield(parm,'rmax');% calcul de champ sur un rayon puis transformation de Hankel 
Pml=init{10};
r=real(retinterp_popov(a{7}{1},Pml,1));
if ~isempty(Pml);xx=retelimine([parm.rmax,real(retinterp_popov(r,Pml,1)),0]);else;xx=retelimine([rmax,r,0]);end;% on ajoute les pml reelles aux discontinuites
xx=xx(xx<=parm.rmax*(1+10*eps));
[x,wx]=retgauss(0,max(xx),50,2,xx);
if z(1)==0;tab=[0,1,1];else;tab=[[0,1,1];[z(1),1,0]];end;for ii=1:length(z)-1;tab=[[0,1,1];[z(ii+1)-z(ii),1,0];tab];end;
% modif Alex: ajout de inc
iinit=init;iinit{end}.sym=0;e=retchamp(iinit,{a},sh,sb,inc,{x,0},tab);
eep=.5*(e(:,:,:,2)-i*e(:,:,:,1));eem=.5*(e(:,:,:,2)+i*e(:,:,:,1));
hhp=.5i*(e(:,:,:,5)-i*e(:,:,:,4));hhm=.5i*(e(:,:,:,5)+i*e(:,:,:,4));clear e;% clonage
JLp1=besselj(L+1,kk(:)*x(:).')*retdiag(x.*wx);
JLm1=besselj(L-1,kk(:)*x(:).')*retdiag(x.*wx);
end;

for kz=1:length(z);% <************** kz
if isfield(parm,'rmax');
ep=JLp1*retcolonne(eep(kz,:,:));em=JLm1*retcolonne(eem(kz,:,:));	
hp=JLp1*retcolonne(hhp(kz,:,:));hm=JLm1*retcolonne(hhm(kz,:,:));	
else;
% modif Alex: ajout de inc
uv=retsc(sh,retss(retc(a,z(kz)),sb),inc,2*N);
ep=retinterp(k,uv(1:N),kk,'cubic');em=retinterp(k,uv(N+1:2*N),kk,'cubic');
hp=retinterp(k,uv(2*N+1:3*N),kk,'cubic');hm=retinterp(k,uv(3*N+1:4*N),kk,'cubic');
end;
switch sym;% sym
case 0
Ep(kz,:,1)=(i/(2*pi))*(expp.*ep-expm.*em).';
Ep(kz,:,2)=(1/(2*pi))*(expp.*ep+expm.*em).';
Ep(kz,:,4)=(i/(2*pi))*(expp.*hp-expm.*hm).';
Ep(kz,:,5)=(1/(2*pi))*(expp.*hp+expm.*hm).';
case 1
Ep(kz,:,1)=(i/(2*pi))*(cosp.*ep-cosm.*em).';
Ep(kz,:,2)=(i/(2*pi))*(sinp.*ep+sinm.*em).';
Ep(kz,:,4)=(1/(2*pi))*(-sinp.*hp+sinm.*hm).';
Ep(kz,:,5)=(1/(2*pi))*(cosp.*hp+cosm.*hm).';
case -1
Ep(kz,:,1)=(1/(2*pi))*(-sinp.*ep+sinm.*em).';
Ep(kz,:,2)=(1/(2*pi))*(cosp.*ep+cosm.*em).';
Ep(kz,:,4)=(i/(2*pi))*(cosp.*hp-cosm.*hm).';
Ep(kz,:,5)=(i/(2*pi))*(sinp.*hp+sinm.*hm).';
end;% sym
end;% <************** kz
cal_angles=nargout>2;
[varargout{1:nargout}]=ondes_planes_2Dz(ep_mu,u,v,Ep,0,0,z,0,0,0,cal_angles,delt,uvmesh+i);






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             VERSIONS 2010                      %
%         integration locale                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%
%%%%%%  1D  %%%%%%%
%%%%%%%%%%%%%%%%%%%
function angles=caldop_1D(n,y,u,  e_x,x_x,y_x,w_x,   e_y,x_y,y_y,w_y, pol,k0,sens);
if nargin<14;[e_x,x_x,y_x,w_x,   e_y,x_y,y_y,w_y, pol,k0,sens]=deal(u,  e_x,x_x,y_x,w_x,   e_y,x_y,y_y,w_y, pol,k0);end;% si pas u (modes en 1D)
if imag(sens)~=0;%[ x y ] -->[ y x]
e_x=permute(e_x,[2,1,3]);e_x=e_x(:,:,[1,3,2]);e_x(:,:,1)=-e_x(:,:,1);
e_y=permute(e_y,[2,1,3]);e_y=e_y(:,:,[1,3,2]);e_y(:,:,1)=-e_y(:,:,1);
angles=caldop_1Dy(n,y,u,  e_y,y_y,x_y,w_y,  e_x,y_x,x_x,w_x,    pol,k0,real(sens));
else;
angles=caldop_1Dy(n,y,u,  e_x,x_x,y_x,w_x,   e_y,x_y,y_y,w_y, pol,k0,sens);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function angles=caldop_1Dy(n,y,u,  e_x,x_x,y_x,w_x,   e_y,x_y,y_y,w_y, pol,k0,sens);
if length(n)>length(y)+1;
nb_couches=length(y)+1;
angles=caldop_modes_1D(n(nb_couches+1),n(1:nb_couches),y,u,  e_x,x_x,y_x,w_x,   e_y,x_y,y_y,w_y, pol,k0,sens);
return;
end;

% clonage des champs
e_x(:,:,2:3)=1i*e_x(:,:,2:3);
e_y(:,:,2:3)=1i*e_y(:,:,2:3);

init={pol,-u,k0};
sh=retb(init,n(1),1);sb=retb(init,n(end),-1);

Y=[y_x(:);y_y(:)];% tous les Y où on veut le champ
[Y_reduit,prv,kkY]=retelimine(Y,-1);
hh=max(Y_reduit)-y(1)*10*eps;hb=y(end)-min(Y_reduit)+10*eps;
tab=[[hh;-diff(y(:));hb],n(:)];
if sens==1;inc=[0,1];n_op=n(1);y_op=y(1);else;inc=[1,0];n_op=n(end);y_op=y(end);end;

tab0=tab;tab0(:,2)=n_op;sh0=retb(init,n_op,1);sb0=retb(init,n_op,-1);
einc=retchamp(init,tab0,sh0,sb0,inc,{y_op-y(end)+hb});einc(:,:,2:3,:,:)=1i*einc(:,:,2:3,:,:);% clonage

e=retchamp(init,tab,sh,sb,inc,{Y_reduit-y(end)+hb});e(:,:,2:3,:,:)=1i*e(:,:,2:3,:,:);% clonage
for ii=1:size(e,5);e(:,:,:,:,ii)=e(:,:,:,:,ii)/einc(:,:,1,:,ii);end;% normalisation E=1

e=e(kkY,:,:,:,:);
ee_x=e(1:length(y_x),:,:,:,:);
ee_y=e(1+length(y_x):end,:,:,:,:);
EE=zeros(length(u),1);
mu=k0*n_op^pol;khi=sqrt(max(eps,(k0*n_op)^2-u.^2));
for ii=1:length(u);%<<<<<<<< ii
eee_x=zeros(size(e_x));
prv=exp(-i*u(ii)*retcolonne(x_x,1));
for jj=1:3;
eee_x(:,:,jj)=ee_x(:,:,jj,1,ii)*prv;
end
eee_y=zeros(size(e_y));
prv=exp(-i*u(ii)*retcolonne(x_y,1));
for jj=1:3;
eee_y(:,:,jj)=ee_y(:,:,jj,1,ii)*prv;
end
EE(ii)=-mu/(4i*pi*khi(ii))*(diff(ret_int_lorentz(e_x,eee_x,w_x,[]))+diff(ret_int_lorentz(e_y,eee_y,[],w_y)));
end;              %<<<<<<<< ii

HH=(k0*n_op/mu)*EE;% decloné
%   mise en forme des ondes planes  en 1D

if sens==1;
K=[u(:),khi(:)];K=K./repmat(sqrt(sum((K.^2),2)),1,size(K,2));% vecteurs d'onde normalises
U=[K(:,2),-K(:,1)];
else;
K=[u(:),-khi(:)];K=K./repmat(sqrt(sum((K.^2),2)),1,size(K,2));% vecteurs d'onde normalises
U=[K(:,2),-K(:,1)];
end
teta=atan2(u(:),khi(:));

F=pi*(k0*n_op)*(cos(teta).^2).*real(EE.*conj(HH));
angles=struct('teta',teta,'k',K,'u',U,'EE',EE,'HH',HH,'F',full(F));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%
%%%%%%  2D  %%%%%%%
%%%%%%%%%%%%%%%%%%%


function angles=caldop_2D(n,z,u,v,  e_xy,x_xy,y_xy,z_xy,wx_xy,wy_xy,    e_zx,x_zx,y_zx,z_zx,wz_zx,wx_zx,    e_yz,x_yz,y_yz,z_yz,wy_yz,wz_yz, k0,sens,parm);
if nargin<25;parm=[];end;
if imag(sens)>0;% [ x y z ] -->[ y z x ]  strates // Ox
e_xy=permute(e_xy,[2,3,1,4]);e_xy=e_xy(:,:,:,[2,3,1,5,6,4]);
e_zx=permute(e_zx,[2,3,1,4]);e_zx=e_zx(:,:,:,[2,3,1,5,6,4]);
e_yz=permute(e_yz,[2,3,1,4]);e_yz=e_yz(:,:,:,[2,3,1,5,6,4]);
angles=caldop_2Dz(n,z,u,v,  e_yz,y_yz,z_yz,x_yz,wy_yz,wz_yz,    e_xy,y_xy,z_xy,x_xy,wx_xy,wy_xy,    e_zx,y_zx,z_zx,x_zx,wz_zx,wx_zx, k0,real(sens),parm);

return
end;

if imag(sens)<0;% [ x y z ] -->[ z x y]  strates // Oy
e_xy=permute(e_xy,[3,1,2,4]);e_xy=e_xy(:,:,:,[3,1,2,6,4,5]);
e_zx=permute(e_zx,[3,1,2,4]);e_zx=e_zx(:,:,:,[3,1,2,6,4,5]);
e_yz=permute(e_yz,[3,1,2,4]);e_yz=e_yz(:,:,:,[3,1,2,6,4,5]);
angles=caldop_2Dz(n,z,u,v,  e_zx,z_zx,x_zx,y_zx,wz_zx,wx_zx,    e_yz,z_yz,x_yz,y_yz,wy_yz,wz_yz,    e_xy,z_xy,x_xy,y_xy,wx_xy,wy_xy, k0,real(sens),parm);
return
end;

angles=caldop_2Dz(n,z,u,v,  e_xy,x_xy,y_xy,z_xy,wx_xy,wy_xy,    e_zx,x_zx,y_zx,z_zx,wz_zx,wx_zx,    e_yz,x_yz,y_yz,z_yz,wy_yz,wz_yz, k0,sens,parm);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function angles=caldop_2Dz(n,z,u,v,  e_xy,x_xy,y_xy,z_xy,wx_xy,wy_xy,    e_zx,x_zx,y_zx,z_zx,wz_zx,wx_zx,    e_yz,x_yz,y_yz,z_yz,wy_yz,wz_yz, k0,sens,parm);
if nargin<25;parm=[];end;
if length(n)>length(z)+1;
nb_couches=length(z)+1;
angles=caldop_modes_2D(n(nb_couches+1),n(nb_couches+2),n(1:nb_couches),z,u,v,  e_xy,x_xy,y_xy,z_xy,wx_xy,wy_xy,    e_zx,x_zx,y_zx,z_zx,wz_zx,wx_zx,    e_yz,x_yz,y_yz,z_yz,wy_yz,wz_yz, k0,sens,parm);
return;
end;
defaultopt=struct('uvmesh',0);
if isempty(parm);parm=defaultopt;end;
uvmesh=retoptimget(parm,'uvmesh',defaultopt,'fast');
if ~uvmesh;
nu=length(u);nv=length(v);nuv=nu*nv;
[u,v]=ndgrid(u,v);
end;

u=u(:);v=v(:);
% clonage des champs
e_xy(:,:,:,4:6)=1i*e_xy(:,:,:,4:6);
e_zx(:,:,:,4:6)=1i*e_zx(:,:,:,4:6);
e_yz(:,:,:,4:6)=1i*e_yz(:,:,:,4:6);

if sens==1;inc=[0,1];n_op=n(1);z_op=z(1);else;inc=[1,0];n_op=n(end);z_op=z(end);end;
khi=sqrt(max(eps,(k0*n_op)^2-u.^2-v.^2));
delt=0;
	if sens==1;
	K=[u,v,khi];K=K./repmat(sqrt(sum((K.^2),2)),1,size(K,2));%vecteurs d'onde normalises
	V=[-K(:,2),K(:,1),zeros(size(K,1),1)];
	ii=find(abs(K(:,1).^2+K(:,2).^2)<100*eps);V(ii,1)=-sin(delt);V(ii,2)=cos(delt);%incidence normale: delt valeur par defaut de delt
	V=V./repmat(sqrt(sum((V.^2),2)),1,size(V,2));
	U=[K(:,3).*V(:,2),-K(:,3).*V(:,1),K(:,2).*V(:,1)-K(:,1).*V(:,2)];%U= V vect.  K
	else;
	K=[u,v,-khi];K=K./repmat(sqrt(sum((K.^2),2)),1,size(K,2));%vecteurs d'onde normalises
	V=[-K(:,2),K(:,1),zeros(size(K,1),1)];
	ii=find(abs(K(:,1).^2+K(:,2).^2)<100*eps);V(ii,1)=-sin(delt);V(ii,2)=cos(delt);%incidence normale: delt valeur par defaut de delt
	V=V./repmat(sqrt(sum((V.^2),2)),1,size(V,2));
	U=[K(:,3).*V(:,2),-K(:,3).*V(:,1),K(:,2).*V(:,1)-K(:,1).*V(:,2)];%U= V vect.  K
	end;
delta=atan2(-V(:,1),V(:,2));cos_delta=cos(delta);sin_delta=sin(delta);
[uv_reduit,prv,kkuv]=retelimine(sqrt(u.^2+v.^2),-1);

Z=[z_xy(:);z_zx(:);z_yz(:)];% tous les Y où on veut le champ
[Z_reduit,prv,kkZ]=retelimine(Z,-1);
hh=max(Z_reduit)-z(1);hb=z(end)-min(Z_reduit);
tab=[[hh;-diff(z(:));hb],n(:)];
tab0=tab;tab0(:,2)=n_op;

e=cell(1,2);
for kpol=1:2;
init={2*kpol-2,-uv_reduit,k0};
sh=retb(init,n(1),1);sb=retb(init,n(end),-1);
sh_inc=retb(init,n_op,1);sb_inc=retb(init,n_op,-1);
e_inc=retchamp(init,tab0,sh_inc,sb_inc,inc,{z_op-z(end)+hb});e_inc(:,:,2:3,:,:)=1i*e_inc(:,:,2:3,:,:);% clonage
e{kpol}=retchamp(init,tab,sh,sb,inc,{Z_reduit-z(end)+hb});e{kpol}(:,:,2:3,:,:)=1i*e{kpol}(:,:,2:3,:,:);% clonage
for ii=1:size(e{kpol},5);e{kpol}(:,:,:,:,ii)=e{kpol}(:,:,:,:,ii)/e_inc(:,:,1,:,ii);end;% normalisation E=1
end;  % kpol
[X_xy,Y_xy]=ndgrid(x_xy,y_xy);
[X_zx,Y_zx]=ndgrid(x_zx,y_zx);
[X_yz,Y_yz]=ndgrid(x_yz,y_yz);
EE=zeros(length(u),2);
HH=zeros(length(u),2);

for kpol=1:2;
for ii=1:length(u);
eee_xy=cale(u(ii),v(ii),X_xy,Y_xy,e{kpol}(kkZ(1:length(z_xy)),:,:,:,kkuv(ii)),size(e_xy),cos_delta(ii),sin_delta(ii),kpol);
eee_zx=cale(u(ii),v(ii),X_zx,Y_zx,e{kpol}(kkZ(1+length(z_xy):length(z_xy)+length(z_zx)),:,:,:,kkuv(ii)),size(e_zx),cos_delta(ii),sin_delta(ii),kpol);
eee_yz=cale(u(ii),v(ii),X_yz,Y_yz,e{kpol}(kkZ(1+length(z_xy)+length(z_zx):length(z_xy)+length(z_zx)+length(z_yz)),:,:,:,kkuv(ii)),size(e_yz),cos_delta(ii),sin_delta(ii),kpol);


if kpol==1;
EE(ii,2)=k0/(8i*pi^2*khi(ii))*(diff(ret_int_lorentz(e_xy,eee_xy,wx_xy,wy_xy,[]))+diff(ret_int_lorentz(e_yz,eee_yz,[],wy_yz,wz_yz))+diff(ret_int_lorentz(e_zx,eee_zx,wx_zx,[],wz_zx)));
else;
HH(ii,2)=k0*n_op^2/(8i*pi^2*khi(ii))*(diff(ret_int_lorentz(e_xy,eee_xy,wx_xy,wy_xy,[]))+diff(ret_int_lorentz(e_yz,eee_yz,[],wy_yz,wz_yz))+diff(ret_int_lorentz(e_zx,eee_zx,wx_zx,[],wz_zx)));
end;	
end;%ii
end;%kpol

mu=k0;ep=k0*n_op^2;
HH(:,1)=-i*(k0*n_op/mu)*EE(:,2);% -1 à cause de l'orientation
EE(:,1)=-i*(k0*n_op/ep)*HH(:,2);

HH=-i*HH;% declonage
% analyse des polarisations
psi=.5*atan2(real(EE(:,1)).*real(EE(:,2))+imag(EE(:,1)).*imag(EE(:,2)),.5*(abs(EE(:,1)).^2-abs(EE(:,2)).^2));
EEE(:,1)=EE(:,1).*cos(psi)+EE(:,2).*sin(psi);EEE(:,2)=-EE(:,1).*sin(psi)+EE(:,2).*cos(psi);
ii=find(abs(EEE(:,2))>abs(EEE(:,1)));psi(ii)=psi(ii)+pi/2;[EEE(ii,1),EEE(ii,2)]=deal(EEE(ii,2),-EEE(ii,1));
ii=find(psi<=-pi/2+100*eps);psi(ii)=psi(ii)+pi;EEE(ii,:)=-EEE(ii,:);
HHH(:,1)=HH(:,1).*cos(psi)+HH(:,2).*sin(psi);HHH(:,2)=-HH(:,1).*sin(psi(:))+HH(:,2).*cos(psi);
ii=find(abs(psi-pi/2)<100*eps);psi(ii)=pi/2;ii=find(abs(psi)<100*eps);psi(ii)=0;
teta=acos(K(:,3));

F=2*pi^2*(k0*n_op)^2*retdiag(cos(teta).^2)*real(EE(:,1).*conj(HH(:,2))-EE(:,2).*conj(HH(:,1)));
if ~uvmesh;
EE=reshape(EE,nu,nv,[]);
HH=reshape(HH,nu,nv,[]);
EEE=reshape(EEE,nu,nv,[]);
HHH=reshape(HHH,nu,nv,[]);
F=reshape(F,nu,nv,[]);
end;
angles=struct('teta',teta,'delta',delta,'psi',psi,'k',K,'v',V,'u',U,'EE',EE,'HH',HH,'EEE',EEE,'HHH',HHH,'F',full(F));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function eee=cale(u,v,X,Y,e,sz,cos_delta,sin_delta,kpol);
eee=zeros(sz);
prv=exp(-i*(u*retcolonne(X,1)+v*retcolonne(Y,1)));% champ se propageant en -u -v
if kpol==1;
eee(:,:,:,2)=-reshape(e(:,1,1)*retcolonne(prv,1),sz(1:3));
eee(:,:,:,4)=reshape(e(:,1,2)*retcolonne(prv,1),sz(1:3));
eee(:,:,:,6)=reshape(e(:,1,3)*retcolonne(prv,1),sz(1:3));
eee(:,:,:,1)=-eee(:,:,:,2)*sin_delta;eee(:,:,:,2)=eee(:,:,:,2)*cos_delta;
eee(:,:,:,5)=eee(:,:,:,4)*sin_delta;eee(:,:,:,4)=eee(:,:,:,4)*cos_delta;
else;
eee(:,:,:,5)=-reshape(e(:,1,1)*retcolonne(prv,1),sz(1:3));
eee(:,:,:,1)=reshape(e(:,1,2)*retcolonne(prv,1),sz(1:3));
eee(:,:,:,3)=reshape(e(:,1,3)*retcolonne(prv,1),sz(1:3));
eee(:,:,:,2)=eee(:,:,:,1)*sin_delta;eee(:,:,:,1)=eee(:,:,:,1)*cos_delta;
eee(:,:,:,4)=-eee(:,:,:,5)*sin_delta;eee(:,:,:,5)=eee(:,:,:,5)*cos_delta;
end;
% plus rapide de le faire plus haut
% [eee(:,:,:,1),eee(:,:,:,2)]=deal(eee(:,:,:,1)*cos_delta-eee(:,:,:,2)*sin_delta,eee(:,:,:,1)*sin_delta+eee(:,:,:,2)*cos_delta);
% [eee(:,:,:,4),eee(:,:,:,5)]=deal(eee(:,:,:,4)*cos_delta-eee(:,:,:,5)*sin_delta,eee(:,:,:,4)*sin_delta+eee(:,:,:,5)*cos_delta);


%%%%%%%%%%%%%%%
%   MODES     %
%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function angles=caldop_modes_1D(neff,n,y,u,  e_x,x_x,y_x,w_x,   e_y,x_y,y_y,w_y, pol,k0,sens);
% clonage des champs
e_x(:,:,2:3)=1i*e_x(:,:,2:3);
e_y(:,:,2:3)=1i*e_y(:,:,2:3);
[gama,ex,ey]=calmode(n,y,neff,pol,k0,y_x,y_y);
ee_x=expand_1D(ex,x_x,gama,-1);
ee_y=expand_1D(ey,x_y,gama,-1);
amp_p=.25i*(diff(ret_int_lorentz(e_x,ee_x,w_x,[]))+diff(ret_int_lorentz(e_y,ee_y,[],w_y)));
ee_x=expand_1D(ex,x_x,gama,1);
ee_y=expand_1D(ey,x_y,gama,1);
amp_m=.25i*(diff(ret_int_lorentz(e_x,ee_x,w_x,[]))+diff(ret_int_lorentz(e_y,ee_y,[],w_y)));

ex0=expand_1D(ex,0,gama,-1);ey0=expand_1D(ey,0,gama,-1);% mode à l'origine
ex0(:,:,2:3)=-i*ex0(:,:,2:3);ey0(:,:,2:3)=-i*ey0(:,:,2:3);% declonage
angles=struct('amp_p',amp_p,'Fp',abs(amp_p)^2,'amp_m',amp_m,'Fm',abs(amp_m)^2,'ex',ex0,'ey',ey0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function angles=caldop_modes_2D(neff,pol,n,z,L,teta,  e_xy,x_xy,y_xy,z_xy,wx_xy,wy_xy,    e_zx,x_zx,y_zx,z_zx,wz_zx,wx_zx,    e_yz,x_yz,y_yz,z_yz,wy_yz,wz_yz, k0,sens,parm);
% clonage des champs
e_xy(:,:,:,4:6)=1i*e_xy(:,:,:,4:6);
e_zx(:,:,:,4:6)=1i*e_zx(:,:,:,4:6);
e_yz(:,:,:,4:6)=1i*e_yz(:,:,:,4:6);

[gama,exy,ezx,eyz]=calmode(n,z,neff,pol,k0,z_xy,z_zx,z_yz);
exy(:,:,1)=-exy(:,:,1);ezx(:,:,1)=-ezx(:,:,1);eyz(:,:,1)=-eyz(:,:,1);% orientation

if ~isempty(L);% bessel
%LLL=L(1):L(2);
switch length(L);% modif 4 2011
case 1;LLL=L;
case 2;LLL=L(1):L(2);
case 3;LLL=L(1):L(2):L(3);
otherwise;LLL=retelimine(L); 
end;
f=zeros(1,length(LLL));
ee_xy=enroule(exy,x_xy,y_xy,z_xy,-LLL,gama,pol);
ee_zx=enroule(ezx,x_zx,y_zx,z_zx,-LLL,gama,pol);
ee_yz=enroule(eyz,x_yz,y_yz,z_yz,-LLL,gama,pol);
for iL=1:length(LLL);L=LLL(iL);
f(iL)=-.25i*sqrt(gama/(2*pi))*exp(.5i*(L-.5)*pi)*(diff(ret_int_lorentz(e_xy,ee_xy{iL},wx_xy,wy_xy,[]))+diff(ret_int_lorentz(e_zx,ee_zx{iL},wx_zx,[],wz_zx))+diff(ret_int_lorentz(e_yz,ee_yz{iL},[],wy_yz,wz_yz)));
end;
amp=f*exp(i*LLL.'*retcolonne(teta,1));

else;% cartesien
amp=zeros(size(teta));
for ii=1:length(teta);
ee_xy=expand_2D(exy,x_xy,y_xy,z_xy,pi+teta(ii),gama,pol);
ee_zx=expand_2D(ezx,x_zx,y_zx,z_zx,pi+teta(ii),gama,pol);
ee_yz=expand_2D(eyz,x_yz,y_yz,z_yz,pi+teta(ii),gama,pol);
amp(ii)=(diff(ret_int_lorentz(e_xy,ee_xy,wx_xy,wy_xy,[]))+diff(ret_int_lorentz(e_zx,ee_zx,wx_zx,[],wz_zx))+diff(ret_int_lorentz(e_yz,ee_yz,[],wy_yz,wz_yz)));
end;
amp=-sqrt(gama/(32*pi))*exp(.25i*pi)*amp;
f=[];
end;% Bessel ou cartesien
% modes à l'origine et declonage
exy0=expand_2D(exy,0,0,z_xy,0,gama,pol);exy0(:,:,:,4:6)=-1i*exy0(:,:,:,4:6);
ezx0=expand_2D(ezx,0,0,z_zx,0,gama,pol);ezx0(:,:,:,4:6)=-1i*ezx0(:,:,:,4:6);
eyz0=expand_2D(eyz,0,0,z_yz,0,gama,pol);eyz0(:,:,:,4:6)=-1i*eyz0(:,:,:,4:6);

angles=struct('f',f,'amp',amp,'F',abs(amp).^2,'L',LLL,'exy',exy0,'ezx',ezx0,'eyz',eyz0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e_cartesien=expand_1D(e,x,gama,sens);
e_cartesien=zeros(size(e,1),length(x),3);
if sens==1;
prv=retcolonne(exp(1i*gama*x),1);
else;
prv=retcolonne(exp(-1i*gama*x),1);e(:,:,3)=-e(:,:,3);
end;
for ii=1:3;e_cartesien(:,:,ii)=e(:,:,ii)*prv;end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e_cartesien=expand_2D(e,x,y,z,teta,gama,pol);
e_cartesien=zeros(length(z),length(x)*length(y),6);
cost=cos(teta);sint=sin(teta);
[X,Y]=ndgrid(x,y);X=X(:);Y=Y(:);
prv=exp(1i*gama*(cost*X+sint*Y)).';
if pol==0;
e_cartesien(:,:,1)=(-e(:,:,1)*sint)*prv;
e_cartesien(:,:,2)=(e(:,:,1)*cost)*prv;
e_cartesien(:,:,4)=(e(:,:,2)*cost)*prv;
e_cartesien(:,:,5)=(e(:,:,2)*sint)*prv;
e_cartesien(:,:,6)=e(:,:,3)*prv;
else;
e_cartesien(:,:,1)=(e(:,:,2)*cost)*prv;
e_cartesien(:,:,2)=(e(:,:,2)*sint)*prv;
e_cartesien(:,:,3)=e(:,:,3)*prv;
e_cartesien(:,:,4)=(-e(:,:,1)*sint)*prv;
e_cartesien(:,:,5)=(e(:,:,1)*cost)*prv;
end;
e_cartesien=reshape(e_cartesien,length(z),length(x),length(y),6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e_bessel=enroule(e,x,y,z,L,gama,pol);
e_bessel=cell(1,length(L));
[e_bessel{:}]=deal(zeros(length(z),length(x)*length(y),6));
[X,Y]=ndgrid(x,y);X=X(:);Y=Y(:);
R=sqrt(X.^2+Y.^2);
Teta=atan2(Y,X);cosTeta=retdiag(cos(Teta));sinTeta=retdiag(sin(Teta));
[LL,k,kL]=retelimine([L-1,L,L+1]);
[RR,k,kR]=retelimine(R);
J=retbessel('j',LL,gama*RR);
kL=reshape(kL,[],3);
clear R X Y
for iL=1:length(L);
if pol==0;
e_bessel{iL}(:,:,1)=-.5*e(:,:,1)*(J(kR,kL(iL,3))+J(kR,kL(iL,1))).';
e_bessel{iL}(:,:,2)=.5i*e(:,:,1)*(J(kR,kL(iL,3))-J(kR,kL(iL,1))).';
e_bessel{iL}(:,:,4)=.5i*e(:,:,2)*(J(kR,kL(iL,3))-J(kR,kL(iL,1))).';
e_bessel{iL}(:,:,5)=.5*e(:,:,2)*(J(kR,kL(iL,3))+J(kR,kL(iL,1))).';
e_bessel{iL}(:,:,6)=e(:,:,3)*J(kR,kL(iL,2)).';
else;
e_bessel{iL}(:,:,1)=.5i*e(:,:,2)*(J(kR,kL(iL,3))-J(kR,kL(iL,1))).';
e_bessel{iL}(:,:,2)=.5*e(:,:,2)*(J(kR,kL(iL,3))+J(kR,kL(iL,1))).';
e_bessel{iL}(:,:,3)=e(:,:,3)*J(kR,kL(iL,2)).';
e_bessel{iL}(:,:,4)=-.5*e(:,:,1)*(J(kR,kL(iL,3))+J(kR,kL(iL,1))).';
e_bessel{iL}(:,:,5)=.5i*e(:,:,1)*(J(kR,kL(iL,3))-J(kR,kL(iL,1))).';
end;
for ii=1:6;e_bessel{iL}(:,:,ii)=e_bessel{iL}(:,:,ii)*retdiag(exp(1i*L(iL)*Teta));end;
[e_bessel{iL}(:,:,1),e_bessel{iL}(:,:,2)]=deal(e_bessel{iL}(:,:,1)*cosTeta-e_bessel{iL}(:,:,2)*sinTeta,e_bessel{iL}(:,:,1)*sinTeta+e_bessel{iL}(:,:,2)*cosTeta);% Ex Ey
[e_bessel{iL}(:,:,4),e_bessel{iL}(:,:,5)]=deal(e_bessel{iL}(:,:,4)*cosTeta-e_bessel{iL}(:,:,5)*sinTeta,e_bessel{iL}(:,:,4)*sinTeta+e_bessel{iL}(:,:,5)*cosTeta);% Hx Hy

e_bessel{iL}=reshape(e_bessel{iL},length(z),length(x),length(y),6);
end; % iL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gama,e1,e2,e3]=calmode(n,y,neff,pol,k0,y1,y2,y3);
% calcul du mode 0D et du champ normalisé pour divers cotes
y=y(:);n=n(:);
hrabh=(y1(2)-y(1))*1.1;hrabb=(y(end)-y1(1))*1.1;
[gama,kmax,vm,er,emode,o,ymode]=retmode(pol,n,-diff(y),k0*neff,[],[hrabh,hrabb,0],[],k0);
tab=[[hrabh;-diff(y);hrabb],n];
init={pol,gama,k0};
sh=retb(init,n(1),1);sb=retb(init,n(end),-1);
e1=retchamp(init,tab,sh,sb,[1,0],{ymode{1}(2:end-1)});
fac=retcolonne(e1(:,:,1:2))\retcolonne(emode{1}(2:end-1,:,1:2)); %retcompare(e1*fac,emode{1}(2:end-1,:,:)),% pour normalisation
%figure;plot(ymode{1}(2:end-1),real(squeeze(fac*e1(:,1,:))),ymode{1},real(squeeze(emode{1}(:,1,:))),ymode{1}(2:end-1),imag(squeeze(fac*e1(:,1,:))),ymode{1},imag(squeeze(emode{1}(:,1,:))))
e1=retchamp(init,tab,sh,sb,[fac,0],{y1-y(end)+hrabb});e1(:,:,2:3)=i*e1(:,:,2:3);% clonage
e2=retchamp(init,tab,sh,sb,[fac,0],{y2-y(end)+hrabb});e2(:,:,2:3)=i*e2(:,:,2:3);% clonage
if nargin>7;
e3=retchamp(init,tab,sh,sb,[fac,0],{y3-y(end)+hrabb});e3(:,:,2:3)=i*e3(:,:,2:3);% clonage
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function angles=caldop_modes_popov(n,z,L,teta, e_x,x_x,z_x,wx_x,    e_z,x_z,z_z,wz_z, k0,sym);
[neff,pol,n]=deal(n(end-1),n(end),n(1:end-2));
% clonage des champs
e_x(:,:,:,4:6)=1i*e_x(:,:,:,4:6);
e_z(:,:,:,4:6)=1i*e_z(:,:,:,4:6);
[gama,ex,ez]=calmode(n,z,neff,pol,k0,z_x,z_z);
ex(:,:,1)=-ex(:,:,1);ez(:,:,1)=-ez(:,:,1);% orientation
ee_x=enroule(ex,x_x,0,z_x,-L,gama,pol);
ee_z=enroule(ez,x_z,0,z_z,-L,gama,pol);
f=-.25i*sqrt(gama/(2*pi))*exp(.5i*(L-.5)*pi)*(diff(ret_int_lorentz(e_x,ee_x{1},(2*pi*x_x).*wx_x,1,[]))+ret_int_lorentz(e_z,ee_z{1},[],2*pi*x_z,wz_z));
teta=teta(:);
switch sym
case 1;amp=f*cos(L*teta);
case -1;amp=f*sin(L*teta);
otherwise;amp=f*exp(i*L*teta);
end;

ex(:,:,2:3)=-i*ex(:,:,2:3);ez(:,:,2:3)=-i*ez(:,:,2:3);% declonage
angles=struct('f',f,'amp',amp,'F',abs(amp).^2,'ex',ex,'ez',ez);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  POPOV  %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

function angles=caldop_popov(n,z,u,v,  e_x,x_x,z_x,wx_x,    e_z,x_z,z_z,wz_z, k0,sens,L,sym,parm);
if nargin<16;sym=0;end;
if nargin<17;parm=[];end;


defaultopt=struct('uvmesh',0,'clone',0);
if isempty(parm);parm=defaultopt;end;
uvmesh=retoptimget(parm,'uvmesh',defaultopt,'fast');
clone=retoptimget(parm,'clone',defaultopt,'fast');
	
if ~uvmesh;
nu=length(u);nv=length(v);nuv=nu*nv;
[u,v]=ndgrid(u,v);

end;
if isempty(z);z=0;n=[n,n];end;
u=u(:);v=v(:);
% clonage des champs
if clone==0;
e_x(:,:,:,4:6)=1i*e_x(:,:,:,4:6);
e_z(:,:,:,4:6)=1i*e_z(:,:,:,4:6);
end;

if sens==1;inc=[0,1];n_op=n(1);z_op=z(1);else;inc=[1,0];n_op=n(end);z_op=z(end);end;
khi=sqrt(max(eps,(k0*n_op)^2-u.^2-v.^2));
delt=0;
	if sens==1;
	K=[u,v,khi];K=K./repmat(sqrt(sum((K.^2),2)),1,size(K,2));%vecteurs d'onde normalises
	V=[-K(:,2),K(:,1),zeros(size(K,1),1)];
	ii=find(abs(K(:,1).^2+K(:,2).^2)<100*eps);V(ii,1)=-sin(delt);V(ii,2)=cos(delt);%incidence normale: delt valeur par defaut de delt
	V=V./repmat(sqrt(sum((V.^2),2)),1,size(V,2));
	U=[K(:,3).*V(:,2),-K(:,3).*V(:,1),K(:,2).*V(:,1)-K(:,1).*V(:,2)];%U= V vect.  K
	else;
	K=[u,v,-khi];K=K./repmat(sqrt(sum((K.^2),2)),1,size(K,2));%vecteurs d'onde normalises
	V=[-K(:,2),K(:,1),zeros(size(K,1),1)];
	ii=find(abs(K(:,1).^2+K(:,2).^2)<100*eps);V(ii,1)=-sin(delt);V(ii,2)=cos(delt);%incidence normale: delt valeur par defaut de delt
	V=V./repmat(sqrt(sum((V.^2),2)),1,size(V,2));
	U=[K(:,3).*V(:,2),-K(:,3).*V(:,1),K(:,2).*V(:,1)-K(:,1).*V(:,2)];%U= V vect.  K
	end
delta=atan2(-V(:,1),V(:,2));cos_delta=cos(delta);sin_delta=sin(delta);
[uv_reduit,prv,kkuv]=retelimine(sqrt(u.^2+v.^2),-1);

Z=[z_x(:);z_z(:)];% tous les Z où on veut le champ
[Z_reduit,prv,kkZ]=retelimine(Z,-1);
% calcul des Bessels (le 'r' de l'integration intervient ici)
if sym==0;
JL=retbessel('j',[L-1,L,L+1],retcolonne(uv_reduit*([x_x(:);x_z(1)].')));% on met en dernier le Z_x qui est constant
JLM1=reshape(JL(:,1),length(uv_reduit),length(x_x)+1)*retdiag([x_x(:);x_z(1)]);
JLP1=reshape(JL(:,3),length(uv_reduit),length(x_x)+1)*retdiag([x_x(:);x_z(1)]);
JL=reshape(JL(:,2),length(uv_reduit),length(x_x)+1)*retdiag([x_x(:);x_z(1)]);
else;
JML=retbessel('j',[L-1,L,L+1,-L-1,-L,-L+1],retcolonne(uv_reduit*([x_x(:);x_z(1)].')));% on met en dernier le Z_x qui est constant
JLM1=reshape(JML(:,1),length(uv_reduit),length(x_x)+1)*retdiag([x_x(:);x_z(1)]);
JLP1=reshape(JML(:,3),length(uv_reduit),length(x_x)+1)*retdiag([x_x(:);x_z(1)]);
JL=reshape(JML(:,2),length(uv_reduit),length(x_x)+1)*retdiag([x_x(:);x_z(1)]);
JMLM1=reshape(JML(:,4),length(uv_reduit),length(x_x)+1)*retdiag([x_x(:);x_z(1)]);
JMLP1=reshape(JML(:,6),length(uv_reduit),length(x_x)+1)*retdiag([x_x(:);x_z(1)]);
JML=reshape(JML(:,5),length(uv_reduit),length(x_x)+1)*retdiag([x_x(:);x_z(1)]);
end;

hh=max(Z_reduit)-z(1)+1.e-6;hb=z(end)-min(Z_reduit)+1.e-6;% modif a tester
tab=[[hh;-diff(z(:));hb],n(:)];
tab0=tab;tab0(:,2)=n_op;

e=cell(1,2);
for kpol=1:2;
init={2*kpol-2,-uv_reduit,k0};
sh=retb(init,n(1),1);sb=retb(init,n(end),-1);
sh_inc=retb(init,n_op,1);sb_inc=retb(init,n_op,-1);
e_inc=retchamp(init,tab0,sh_inc,sb_inc,inc,{z_op-z(end)+hb});e_inc(:,:,2:3,:,:)=1i*e_inc(:,:,2:3,:,:);% clonage
e{kpol}=retchamp(init,tab,sh,sb,inc,{Z_reduit-z(end)+hb});e{kpol}(:,:,2:3,:,:)=1i*e{kpol}(:,:,2:3,:,:);% clonage


for ii=1:size(e{kpol},5);e{kpol}(:,:,:,:,ii)=e{kpol}(:,:,:,:,ii)/e_inc(:,:,1,:,ii);end;% normalisation E=1
end;  % kpol

EE=zeros(length(u),2);
HH=zeros(length(u),2);

for kpol=1:2;
for ii=1:length(u);
eee_x=cale_popov(x_x,e{kpol}(kkZ(1:length(z_x)),:,:,:,kkuv(ii)),size(e_x),cos_delta(ii),sin_delta(ii),kpol,L,delta(ii),JLM1(kkuv(ii),1:end-1),JL(kkuv(ii),1:end-1),JLP1(kkuv(ii),1:end-1));
eee_z=cale_popov(x_z,e{kpol}(kkZ(1+length(z_x):length(z_x)+length(z_z)),:,:,:,kkuv(ii)),size(e_z),cos_delta(ii),sin_delta(ii),kpol,L,delta(ii),JLM1(kkuv(ii),end),JL(kkuv(ii),end),JLP1(kkuv(ii),end));
if sym~=0;
e_xx=e_x;e_zz=e_z;e_xx(:,:,:,[2,4,6])=-e_xx(:,:,:,[2,4,6]);e_zz(:,:,:,[2,4,6])=-e_zz(:,:,:,[2,4,6]);
eee_xx=cale_popov(x_x,e{kpol}(kkZ(1:length(z_x)),:,:,:,kkuv(ii)),size(e_x),cos_delta(ii),sin_delta(ii),kpol,-L,delta(ii),JMLM1(kkuv(ii),1:end-1),JML(kkuv(ii),1:end-1),JMLP1(kkuv(ii),1:end-1));
eee_zz=cale_popov(x_z,e{kpol}(kkZ(1+length(z_x):length(z_x)+length(z_z)),:,:,:,kkuv(ii)),size(e_z),cos_delta(ii),sin_delta(ii),kpol,-L,delta(ii),JMLM1(kkuv(ii),end),JML(kkuv(ii),end),(-1)^(L+1)*JMLP1(kkuv(ii),end));
end;
switch sym;
case 0;EH=diff(ret_int_lorentz(e_x,eee_x,wx_x,1,[]))+ret_int_lorentz(e_z,eee_z,[],1,wz_z);
case 1;EH=.5*(diff(ret_int_lorentz(e_x,eee_x,wx_x,1,[]))+ret_int_lorentz(e_z,eee_z,[],1,wz_z)+diff(ret_int_lorentz(e_xx,eee_xx,wx_x,1,[]))+ret_int_lorentz(e_zz,eee_zz,[],1,wz_z));
case -1;EH=.5*(diff(ret_int_lorentz(e_x,eee_x,wx_x,1,[]))+ret_int_lorentz(e_z,eee_z,[],1,wz_z)-diff(ret_int_lorentz(e_xx,eee_xx,wx_x,1,[]))-ret_int_lorentz(e_zz,eee_zz,[],1,wz_z)); 
end;

if kpol==1;EE(ii,2)=k0/(8i*pi^2*khi(ii))*EH;else;HH(ii,2)=k0*n_op^2/(8i*pi^2*khi(ii))*EH;end;
end;%ii
end;%kpol

mu=k0;ep=k0*n_op^2;
HH(:,1)=-i*(k0*n_op/mu)*EE(:,2);% -1 à cause de l'orientation
EE(:,1)=-i*(k0*n_op/ep)*HH(:,2);

HH=-i*HH;% declonage
% analyse des polarisations
psi=.5*atan2(real(EE(:,1)).*real(EE(:,2))+imag(EE(:,1)).*imag(EE(:,2)),.5*(abs(EE(:,1)).^2-abs(EE(:,2)).^2));
EEE(:,1)=EE(:,1).*cos(psi)+EE(:,2).*sin(psi);EEE(:,2)=-EE(:,1).*sin(psi)+EE(:,2).*cos(psi);
ii=find(abs(EEE(:,2))>abs(EEE(:,1)));psi(ii)=psi(ii)+pi/2;[EEE(ii,1),EEE(ii,2)]=deal(EEE(ii,2),-EEE(ii,1));
ii=find(psi<=-pi/2+100*eps);psi(ii)=psi(ii)+pi;EEE(ii,:)=-EEE(ii,:);
HHH(:,1)=HH(:,1).*cos(psi)+HH(:,2).*sin(psi);HHH(:,2)=-HH(:,1).*sin(psi(:))+HH(:,2).*cos(psi);
ii=find(abs(psi-pi/2)<100*eps);psi(ii)=pi/2;ii=find(abs(psi)<100*eps);psi(ii)=0;
teta=acos(K(:,3));

F=2*pi^2*(k0*n_op)^2*retdiag(cos(teta).^2)*real(EE(:,1).*conj(HH(:,2))-EE(:,2).*conj(HH(:,1)));
if ~uvmesh;
EE=reshape(EE,nu,nv,[]);
HH=reshape(HH,nu,nv,[]);
EEE=reshape(EEE,nu,nv,[]);
HHH=reshape(HHH,nu,nv,[]);
F=reshape(F,nu,nv,[]);
end;
angles=struct('teta',teta,'delta',delta,'psi',psi,'k',K,'v',V,'u',U,'EE',EE,'HH',HH,'EEE',EEE,'HHH',HHH,'F',full(F));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function eee=cale_popov(X,e,sz,cos_delta,sin_delta,kpol,L,delta,JLM1,JL,JLP1);
delta=delta+pi;
eee=zeros(sz);
if kpol==1;
eee(:,:,:,2)=-reshape(repmat(e(:,1,1),1,length(X)),sz(1:3));
eee(:,:,:,4)=reshape(repmat(e(:,1,2),1,length(X)),sz(1:3));
eee(:,:,:,6)=reshape(repmat(e(:,1,3),1,length(X)),sz(1:3));
else;
eee(:,:,:,5)=-reshape(repmat(e(:,1,1),1,length(X)),sz(1:3));
eee(:,:,:,1)=reshape(repmat(e(:,1,2),1,length(X)),sz(1:3));
eee(:,:,:,3)=reshape(repmat(e(:,1,3),1,length(X)),sz(1:3));
end;

[eee(:,:,:,1),eee(:,:,:,2)]=deal(eee(:,:,:,1)*cos_delta-eee(:,:,:,2)*sin_delta,eee(:,:,:,1)*sin_delta+eee(:,:,:,2)*cos_delta);
[eee(:,:,:,4),eee(:,:,:,5)]=deal(eee(:,:,:,4)*cos_delta-eee(:,:,:,5)*sin_delta,eee(:,:,:,4)*sin_delta+eee(:,:,:,5)*cos_delta);

JLP1=(pi*exp(1i*((L+1)*delta+L*pi/2)))*JLP1;
JL=(pi*exp(1i*L*(delta+pi/2)))*JL;
JLM1=(pi*exp(1i*((L-1)*delta+L*pi/2)))*JLM1;
if size(eee,1)==2;JL=[JL(:).';JL(:).'];JLP1=[JLP1(:).';JLP1(:).'];JLM1=[JLM1(:).';JLM1(:).'];% pour e_x
else;JL=repmat(JL,1,size(eee,1));JLP1=repmat(JLP1,1,size(eee,1));JLM1=repmat(JLM1,1,size(eee,1)); % pour e_z
end;
JL=reshape(JL,size(eee(:,:,:,1)));JLP1=reshape(JLP1,size(eee(:,:,:,1)));JLM1=reshape(JLM1,size(eee(:,:,:,1)));
[eee(:,:,:,1),eee(:,:,:,2),eee(:,:,:,3)]=deal(...
(1i*eee(:,:,:,1)+eee(:,:,:,2)).*JLP1-(1i*eee(:,:,:,1)-eee(:,:,:,2)).*JLM1,...
(-eee(:,:,:,1)+1i*eee(:,:,:,2)).*JLP1-(eee(:,:,:,1)+1i*eee(:,:,:,2)).*JLM1,...
2*eee(:,:,:,3).*JL);
[eee(:,:,:,4),eee(:,:,:,5),eee(:,:,:,6)]=deal(...
(1i*eee(:,:,:,4)+eee(:,:,:,5)).*JLP1-(1i*eee(:,:,:,4)-eee(:,:,:,5)).*JLM1,...
(-eee(:,:,:,4)+1i*eee(:,:,:,5)).*JLP1-(eee(:,:,:,4)+1i*eee(:,:,:,5)).*JLM1,...
2*eee(:,:,:,6).*JL);

