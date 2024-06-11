function p=retpoynting(e,n,wx,wy,wz);
%  function p=retpoynting(e,n);
%
% p=retpoynting(e);
%  Calcul du vecteur de poynting associe à un champ e calculé par retchamp
%  e doit avoir toutes ses composantes (6 en 2D au moins 3 en 1D)
%  p est calculé aux mêmes points que e: si e=e(z,x,y,1:6) p=p(z,x,y,1:3) 
%
% p=retpoynting(e,n);
%  Calcul du produit scalaire du vecteur de Poynting par le vecteur n (n dimension 3 en 2 D  2 en 1 D)
%  si e=e(z,x,y,1:6)  p=p(z,x,y)
% 
% les dimensions égales à 1 sont ensuite éliminées par l'ordre 'squeeze'
%     Attention : si size(p)= [1,n]  size(squeeze(p))= [1,n] 
%                 si size(p)= [n,1]  size(squeeze(p))= [n,1]
% Il est aussi possible de calculer le vecteur de poynting d'un champ non meshé
% e=e(nb_pts,1:6) en 2D  (ou e=e(nb_pts,1:3) en 1D ) alors p=p(nb_pts,1:3) (ou p=p(nb_pts,1:2))
% si n est precise, il doit avoir pour dimension [nb_pts,3] (ou [nb_pts,2]) n=[nx(:),ny(:),nz(:)] (ou  n=[nx(:),ny(:)])
% 
% p=retpoynting(e,n,wx,wy); en 1D  p=retpoynting(e,n,wx,wy,wz); en 2D
% calcul du flux du produit scalaire par n du vecteur de poynting avec les poids w
% un de ces poids doit être vide ( en general celui qui correspond à la composante non nulle de n ) 
% Le resultat est un vecteur ligne dont la longueur est la dimension de e qui correspond à la composante non nulle de n
%
% si e est un cell_array, e est considére comme clone et est declone avant le calcul du vecteur de Poynting
%
%%% Exemples: 
%% en 1D
% ld=.9;pol=0;[x,wx]=retgauss(-ld,ld,10);e=retpoint(2*pi/ld,2*pi/ld,[1,0,0],[0,0],x,[1,1.1],pol);
%Flux_1D_y = retpoynting(e,[0,1],wx,[])
%
% ld=.9;pol=0;[y,wy]=retgauss(-ld,ld,11);e=retpoint(2*pi/ld,2*pi/ld,[1,0,0],[0,0],[1,1.1],y,pol);
% Flux_1D_x = retpoynting(e,[1,0],[],wy)
%
%% en 2D
% ld=.9;[y,wy]=retgauss(-ld,ld,11);[z,wz]=retgauss(-ld,ld,10);e=retpoint(2*pi/ld,2*pi/ld,[1,0,0,0,0,0],[0,0,0],[1,1.1],y,z);
% Flux_2D_y=retpoynting(e,[1,0,0],[],wy,wz)
%
% ld=.9;[x,wx]=retgauss(-ld,ld,10);[z,wz]=retgauss(-ld,ld,11);e=retpoint(2*pi/ld,2*pi/ld,[1,0,0,0,0,0],[0,0,0],x,[1,1.1],z);
% Flux_2D_z=retpoynting(e,[0,1,0],wx,[],wz)
%
% ld=.9;[x,wx]=retgauss(-ld,ld,10);[y,wy]=retgauss(-ld,ld,11);e=retpoint(2*pi/ld,2*pi/ld,[1,0,0,0,0,0],[0,0,0],x,y,[1,1.1]);
% Flux_2D_z=retpoynting(e,[0,0,1],wx,wy,[])
%
% See also RET_INT_LORENTZ

if iscell(e);clone=1;e=e{1};else clone=0;end;% declonage d'un champ clone

sz=size(e);
if length(sz)==2;  % <<<<<<<<< champ non meshé 
	if sz(end)>4; % 2 D
	if clone;e(:,4:6)=-i*e(:,4:6);end;% declonage
	p=e(:,1:3);
	p(:,1)=real(e(:,2).*conj(e(:,6))-e(:,3).*conj(e(:,5)))/2;
	p(:,2)=real(e(:,3).*conj(e(:,4))-e(:,1).*conj(e(:,6)))/2;
	p(:,3)=real(e(:,1).*conj(e(:,5))-e(:,2).*conj(e(:,4)))/2;
	if nargin>1;p=sum(p.*n,2);end;
	else;   % 1 D
	if clone;e(:,2:end)=-i*e(:,2:end);end;% declonage
	p=e(:,1:2);
	p(:,1)=-real(e(:,1).*conj(e(:,3)))/2;
	p(:,2)=real(e(:,1).*conj(e(:,2)))/2;
	if nargin>1;p=sum(p.*n,2);end;
	end;
else;	             % <<<<<<<<< champ  meshé 
	if sz(end)>4; % 2 D
	if clone;e(:,:,:,4:6)=-i*e(:,:,:,4:6);end;% declonage	
	p=e(:,:,:,1:3);
	p(:,:,:,1)=real(e(:,:,:,2).*conj(e(:,:,:,6))-e(:,:,:,3).*conj(e(:,:,:,5)))/2;
	p(:,:,:,2)=real(e(:,:,:,3).*conj(e(:,:,:,4))-e(:,:,:,1).*conj(e(:,:,:,6)))/2;
	p(:,:,:,3)=real(e(:,:,:,1).*conj(e(:,:,:,5))-e(:,:,:,2).*conj(e(:,:,:,4)))/2;
	if nargin>1;pp=p(:,:,:,1)*n(1)+p(:,:,:,2)*n(2)+p(:,:,:,3)*n(3);p=pp;end;% bug de Matlab 6 il faut passer par une variable intermediaire
	
	else;   % 1 D
	if clone;e(:,:,2:end)=-i*e(:,:,2:end);end;% declonage	
	p=e(:,:,1:2);
	p(:,:,1)=-real(e(:,:,1).*conj(e(:,:,3)))/2;
	p(:,:,2)=real(e(:,:,1).*conj(e(:,:,2)))/2;
	if nargin>1;pp=p(:,:,1)*n(1)+p(:,:,2)*n(2);p=pp;end;
	end;
end;                  % <<<<<<<<< champ  meshé ?
if nargin<3;p=squeeze(p);return;end;
% Calcul du flux



if length(size(e))==4; % 2 D
if isempty(wx);n=size(p,2);pp=zeros(1,n);
for ii=1:n;pp(ii)=wz(:).'*reshape(p(:,ii,:),length(wz),length(wy))*wy(:);end;p=pp;
end;
if isempty(wy);n=size(p,3);pp=zeros(1,n);
for ii=1:n;pp(ii)=wz(:).'*reshape(p(:,:,ii),length(wz),length(wx))*wx(:);end;p=pp;
end;
if isempty(wz);n=size(p,1);pp=zeros(1,n);
for ii=1:n;pp(ii)=wx(:).'*reshape(p(ii,:,:),length(wx),length(wy))*wy(:);end;p=pp;
end;
else;   % 1 D
if isempty(wx);p=wy(:).'*p;end;
if isempty(wy);p=(p*wx(:)).';end;
end;



