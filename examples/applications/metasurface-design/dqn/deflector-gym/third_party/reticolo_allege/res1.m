function [a,nef]=res1(LD,D,TEXTURES,nn,ro,delta0,parm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%               %%%%%%%%%%%%%%%%%
%               %     2 D       %
%               %%%%%%%%%%%%%%%%%
%    function a=res1(LD,D,TEXTURES,nn,ro,delta0,parm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% construction de a  cell array contenant des donnees sur les textures
% et nef cell array des 'indices efficaces' des textures (pour les metaux infiniment conducteurs nef{ii}=[])
%  ces 'indices efficaces' sont classes par attenuation decroissante
%
%  LD,D  longueur d'onde ,pas en unites metriques
%  TEXTURES ={ { n1,[cx1,cy1,dx1,dy1,ni1,k1],[cx2,cy2,dx2,dy2,ni2 k2],...[cxn,cyn,dxn,dyn,nin kn]}  % premiere texture
%      ,   {n2,...   }  % deuxieme texture
%      ,   {nm,...    }  } % derniere texture
%
%  TEXTURES{2} ={ n1, [cx1,cy1,dx1,dy1,ni1,k1],[cx2,cy2,dx2,dy2,ni2,k2],...[cxn,cyn,dxn,dyn,nin,kn]}
%    n1:indice de la base
%   [cx1,cy1,          dx1,dy1,             ni1     k1   ]: premiere inclusion
%    centre    largeurs en x et y      indice
%  k1=1  l'inclusion est un rectangle  de  cotes  dx1,dy1
%  k1 >1  l'inclusion est une ellipse de grands axes  dx1,dy1,  approchee par k1 rectangles 
%                   la surface totale de l'inclusion etant celle de l'ellipse
%     si le rectangle ou l'ellipse a une dimension plus grande que le pas il y a chevauchement (indice ni1 dans la partie commune)
%
%
%  on peut aussi au lieu d'une base homogene avoir un reseau  1 D
%  decrit par un tableau de points de discontinuites suivi du tableau des indices a gauche des points
%  les points de discontinuitee doivent etre en ordre croissant sur un intervalle strictement inferieur au pas 
%   et etre au moins 2  
%      si ces tableaux sont des vecteurs ligne le reseau est invariant en y (cas conique)
%      si ces tableaux sont des vecteurs colonnes le reseau est invariant en x 
%
%   ATTENTION: l'ordre des inclusions est important car elles s'ecrasent l'une l'autre ..
% 
%   on peut aussi definir des plaques de metal infiniment conducteur percees de trous rectangulaires NE SE CHEVAUCHANT PAS
%  TEXTURES{ }= { inf, [cx1,cy1,dx1,dy1,ni1],[cx2,cy2,dx2,dy2,ni2],..} pour le metal electrique
%  TEXTURES{ }= {-inf, [cx1,cy1,dx1,dy1,ni1],[cx2,cy2,dx2,dy2,ni2],..} pour le metal magnetique
%   [cx1,cy1,          dx1,dy1,         ni1       ]: premier trou
%    centre    largeurs en x et y      indice
%  par exemple  TEXTURES{ }= { inf} est le metal massif en haut ou en bas  
%
% SIMPLIFICATIONS D'ECRITURE
%  si on a une seule texture il n'est pas necessaire de la mettre en tableau de cell
%   TEXTURES ={ n1, [cx1,cy1,dx1,dy1,ni1,k1],[cx2,cy2,dx2,dy2,ni2 k2],...[cxn,cyn,dxn,dyn,nin kn]}
%  pour les milieux homogenes on peut entrer  TEXTURES{k} =n1  
%
%
%
%  nn nombre de termes de fourier en x et y
%           si size(nn)=[1,2]    -nn(1) a nn(1) en x -nn(2) a nn(2) en y
%           si size(nn)=[2,2]     nn(1,1) a nn(2,1) en x  nn(1,2) a nn(2,2) en y
%           si size(nn)=[1,1]    -nn(1) a nn(1) en x  0  en y (cas conique)
%           si size(nn)=[2,1]    nn(1,1) a nn(2,1) en x  0  en y (cas conique) 
%
%  ro,deltao: incidence beta0=beta0=ro*[cos(delta0*pi/180),sin(delta0*pi/180)]   par defaut ro=1, delta0=0 
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  symetries
%  parm.sym.x=x0   x=x0 :plan de symetrie seulement si beta0(1)=0     par defaut parm.sym.x=[]  pas de symetrie en x 
%  parm.sym.y=y0   y=y0 :plan de symetrie seulement si beta0(2)=0     par defaut parm.sym.y=[]  pas de symetrie en y
%  parm.sym.pol=1;   1 TE  -1:TM  par defaut 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  parm.res1.trace:  1  trace des textures     (defaut 0)
%
%  parm.res1.nx    parm.res1.xlimite (valeurs de x et nombre de points pour le trace)
%  parm.res1.ny    parm.res1.ylimite (valeurs de y et nombre de points pour le trace)
%     (par defaut la maille du reseau centree avec 100*100 points)
%  exemple:  parm.res1.xlimite=[-D(1),D(1)]; parm.res1.nx=500; % parametres pour le trace des textures en x
%
%  parm.res1.angles: pour les ellipses   1 :regulier en angles  0 meilleure repartition reguliers en surface (defaut 1)
%  parm.res1.calcul: 1   calcul  (defaut 1)  (si on ne veut que visualiser les textures prendre parm.res1.calcul=0 )
%  parm.res1.champ:  options pour le trace futur des champs  1: les champs sont bien calcules( gourmand en temps et memoire)
%        0 :les champs non continus sont calcules de facon approchee  et Ez et Hz ne sont pas calcules et remplaces par 0
%                              (defaut 0)
%  parm.res1.ftemp: 1   fichiers temporaires  (defaut 1) 
%  parm.res1.fperm: 'aa'   le resultat est mis sur un fichier permanent  de nom 'aa123..'  (defaut [] donc pas ecriture0) 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               %%%%%%%%%%%%%%%%%
%               %     1 D       %
%               %%%%%%%%%%%%%%%%%
%    function [a,nef]=res1D(LD,D,TEXTURES,nn,pol,beta0,parm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% construction de a  cell array contenant des donnees sur les textures  cas 1 D
% et nef cell array des 'indices efficaces' des textures (pour les metaux infiniment conducteurs nef{ii}=[])
%  ces 'indices efficaces' sont classes par attenuation decroissante
%
%  LD,D  longueur d'onde ,pas en unites metriques
%  TEXTURES ={ {x1,n1},{x2,n2},...{xn,nn}} 
%
%
%  profil des tranches du reseau
%  decrites par un tableau de points de discontinuites suivi du tableau des indices a gauche des points
%  les points de discontinuitee doivent etre en ordre croissant sur un intervalle strictement inferieur au pas 
%   et etre au moins 2  
%
% 
%   on peut aussi definir des tranches de metal infiniment conducteur percees de trous rectangulaires NE SE CHEVAUCHANT PAS
%  TEXTURES{ }= { inf, [cx1,dx1,ni1],[cx2,dx2,ni2],..} pour le metal electrique
%  TEXTURES{ }= {-inf, [cx1,dx1,n1],[cx2,dx2,ni2],..} pour le metal magnetique
%   [cx1,          dx1,         ni1       ]: premier trou
%    centre    largeurs      indice
%  par exemple  TEXTURES{ }= { inf} est le metal massif en haut ou en bas  
%
% SIMPLIFICATIONS D'ECRITURE
%  si on a une seule texture il n'est pas necessaire de la mettre en tableau de cell

%  pour les milieux homogenes on peut entrer  TEXTURES{k} =n1  
%
%
%
%  nn nombre de termes de fourier
%
%           si length(nn)=1    -nn a nn 
%           si length(nn)=2    nn(1) a nn(2)
%
%   beta0=n*sin(teta*pi/180)  par defaut beta0=0 
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  symetries
%  parm.sym.x=x0   x=x0 :plan de symetrie    par defaut parm.sym.x=[]  pas de symetrie 
%     possible uniquement si beta0=0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% polarisation  pol 1:TE  -1:TM  par defaut TE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù%



if nargin>=6&isstruct(delta0);[a,nef]=res1D(LD,D,TEXTURES,nn,ro,delta0);return;end; % cas 1 D   
%     function [a,nef]=res1D(LD,D,TEXTURES,nn,beta0,parm);
    
if nargin<7;parm=[];end;if isempty(parm);parm=res0;end;
    
if nargin<5;ro=0;end;
if nargin<6;delta0=0;end;beta0=ro*[cos(delta0*pi/180),sin(delta0*pi/180)];
UN=LD/(2*pi);
if (length(D)<2);D=[D,D];parm.sym.y=0;end;if size(nn,2)==1;nn=[nn,zeros(size(nn))];end; % cas conique( symetrie en y)
if size(nn,1)==1;nn=[-nn;nn];end; % en general les termes de fourier vont de -nn a nn mais on se reserve la possibilitee generale

sym=[0,0,0,0]; % symetries
if abs(sin(delta0*pi/180))<10*eps;sym(1:2)=-parm.sym.pol;end;
if abs(cos(delta0*pi/180))<10*eps;sym(1:2)=parm.sym.pol;end;
if ~isempty(parm.sym.x)&(abs(beta0(1))<10*eps)&(nn(2,1)==-nn(1,1));sym(3)=parm.sym.x/UN;else;sym(1)=0;end;   % symetrie par rapport au plan x=x0 
if ~isempty(parm.sym.y)&(abs(beta0(2))<10*eps)&(nn(2,2)==-nn(1,2));sym(4)=parm.sym.y/UN;else;sym(2)=0;end;   % symetrie par rapport au plan y=y0 

if parm.res1.trace==1;    % trace 
if isempty(parm.res1.xlimite);x=linspace(-D(1)/2,D(1)/2,parm.res1.nx)/UN;else;x=linspace(parm.res1.xlimite(1),parm.res1.xlimite(2),parm.res1.nx)/UN;end;    
if isempty(parm.res1.ylimite);y=linspace(-D(2)/2,D(2)/2,parm.res1.ny)/UN;else;y=linspace(parm.res1.ylimite(1),parm.res1.ylimite(2),parm.res1.ny)/UN;end;    
figure;
end;

% mise en forme de TEXTURE
if ~iscell(TEXTURES);TEXTURES={TEXTURES};end;%                              TEXTURES = 1.5
if ~any(cellfun('isclass',TEXTURES,'cell'))&~all(cellfun('length',TEXTURES)==1);TEXTURES={TEXTURES};end;
%       TEXTURES={1.5,[..]}  ou    TEXTURES={[-.5,1,3],[2,1.3,1.5] , [0,0,6,6,  1,  5] ,  [0,0,2,2,  1.5,  1]    };
%   mais pas TEXTURES={1, 1.5 } qui signifie 2 milieux homogenes


NTEXTURES=size(TEXTURES,2);
nf=ceil(sqrt(NTEXTURES)); % pour le trace
a=cell(NTEXTURES,1);nef=cell(NTEXTURES,1);
sog=parm.res1.sog;
for ii=1:NTEXTURES;if ~iscell(TEXTURES{ii});TEXTURES{ii}={TEXTURES{ii}};end;
if ~isfinite(TEXTURES{ii}{1}(1,1));sog=0;end;
end;

[init,n]=retinit(D/UN,[nn(1,1)+i*(1-sog),nn(2,1),nn(1,2),nn(2,2)],[beta0,delta0*pi/180],sym);

for ii=1:NTEXTURES; %construction des a
minc=2;
if (length(TEXTURES{ii})>1)&(length(TEXTURES{ii}{1})==length(TEXTURES{ii}{2}));  % la base est une texture 1D 
minc=3; 
end;
for in=minc:size(TEXTURES{ii},2);
if parm.res1.angles==1&length(TEXTURES{ii}{in})>5;TEXTURES{ii}{in}(6)=-TEXTURES{ii}{in}(6);end;
end;
u=retu(init,[TEXTURES{ii},{1,UN}]); % k0=1 UN mise a l'echelle
if length(u)==3&length(u{1})==1&length(u{2})==1;champ=1;else;champ=parm.res1.champ;end; % pour les milieux homogenes
    
if parm.res1.trace==1; % trace
[ee,xy]=rettestobjet(init,u,-1,[],{x,y});
subplot(nf,nf,ii);retcolor(xy{1}*UN,xy{2}*UN,real(ee.'));axis equal;
title(['texture  ',num2str(ii)],'fontsize',7);xlabel('X','fontsize',7);ylabel('Y','fontsize',7);pause(eps);
end;
if parm.res1.calcul==1;       %  diagonalisation
[a{ii},neff]=retcouche(init,u,champ+i*parm.res1.ftemp,parm.res1.li);
if parm.res1.ftemp==1;a{ii}=retio(a{ii},1);end;
if nargout>1;nef{ii}=i*neff;  % indices effectifs
[prv,iii]=sort(imag(nef{ii}));nef{ii}=nef{ii}(iii);
f=find(abs(imag(nef{ii}))<100*eps);[prv,iii]=sort(real(nef{ii}(f)));nef{ii}(f)=nef{ii}(f(iii));
end;
end;
end;  % boucle sur ii
a={a,init,n,UN,D,beta0,sym};
if ~isempty(parm.res1.fperm);a=retio(a,parm.res1.fperm,0);end; % ecriture eventuelle sur fichier permanent


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [a,nef]=res1D(LD,D,TEXTURES,nn,beta0,parm);

pol=1-parm.dim;
UN=LD/(2*pi);
if length(nn)==1;nn=[-nn,nn];end; 

if abs(beta0)<10*eps&~isempty(parm.sym.x);sym=[1,parm.sym.x];else sym=[];end;

if parm.res1.trace==1;    % trace 
if isempty(parm.res1.xlimite);x=linspace(-D/2,D/2,parm.res1.nx)/UN;else;x=linspace(parm.res1.xlimite(1),parm.res1.xlimite(2),parm.res1.nx)/UN;end;    
figure;
end;


% mise en forme de TEXTURE
if ~iscell(TEXTURES);TEXTURES={TEXTURES};end;%                              TEXTURES = 1.5
if ~any(cellfun('isclass',TEXTURES,'cell'))&~all(cellfun('length',TEXTURES)==1);TEXTURES={TEXTURES};end;
%       TEXTURES={1.5,[..]}  
%       TEXTURES={1, 1.5 }  signifie 2 milieux homogenes


NTEXTURES=size(TEXTURES,2);
nf=ceil(sqrt(NTEXTURES)); % pour le trace


a=cell(NTEXTURES,1);nef=cell(NTEXTURES,1);
sog=parm.res1.sog;
for ii=1:NTEXTURES;if ~iscell(TEXTURES{ii});TEXTURES{ii}={TEXTURES{ii}};end;
if ~isfinite(TEXTURES{ii}{1}(1));sog=0;end;
end;

[init,n]=retinit(D/UN,[nn(1)+i*(1-sog),nn(2)],beta0,sym);

for ii=1:NTEXTURES; % construction des a
u=retu(init,[TEXTURES{ii},{pol,1,UN}]);

if parm.res1.trace==1; % trace
[ee,x]=rettestobjet(init,u,-1,[],x);
subplot(nf,nf,ii);plot(x*UN,real(ee));axis([min(x*UN),max(x*UN),0,max(abs(real(ee)))*1.1]);
title(['texture  ',num2str(ii)],'fontsize',7);xlabel('X','fontsize',7);ylabel('indice','fontsize',7);pause(eps);
end;

if length(u{1})==1;champ=1;else;champ=parm.res1.champ;end; % pour les milieux homogenes

[a{ii},neff]=retcouche(init,u,champ);
if nargout>1;nef{ii}=i*neff;  % indices effectifs
[prv,iii]=sort(imag(nef{ii}));nef{ii}=nef{ii}(iii);
f=find(abs(imag(nef{ii}))<100*eps);[prv,iii]=sort(real(nef{ii}(f)));nef{ii}(f)=nef{ii}(f(iii));
end;
end;  % boucle sur ii    
a={a,init,n,UN,D,beta0,sym,pol};
if ~isempty(parm.res1.fperm);a=retio(a,parm.res1.fperm,0);end; % ecriture eventuelle sur fichier permanent


