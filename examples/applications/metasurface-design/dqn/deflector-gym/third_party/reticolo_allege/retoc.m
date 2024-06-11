function varargout=retoc(varargin);
% 	¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
% 	¤   VERSION 2D ( reseau 1D E// ou H//) ¤
% 	¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
% 
% [e,e_oc,e_res]=retoc(n,S,P,x,y,pol,k0,parm);
% 
% Calcul du champ émis par une source sur un dioptre
% Intégration par la methode du col , Formes approchees asympotiques, Separation  Plasmon <--> onde cylindrique
% 
%            y   phi ( >0 dans cet exemple) -pi <phi <pi                y
%            ^       *                                        .  .  .  .^
%            |      /                                          .  .  .  |
%            |     /                                          .  .  .  .|
%            |    / rho                                        .  .  .  |
%            |   /                                            .  n(2)  .|   n(1) 
%            |  /                                              . n(3) S P----------------> x
%           y0                                                .  .  .  .|
%            |                                                 .  .  .  |
%    n(1)    |                                                .  .  .  .|
% ---------- P ----------------->  x                           .  .  .  |  
%    n(2)    S  .  .  .  .  .  .  .                           .  .  .  .|
%  .  .  .  n(3) .  .  .  .  .  .                              .  .  .  |
%         
% la source au point de coordonnees [P(1),P(2)]        la source au point de coordonnees [P(1);P(2)] 
%                   est sur le dioptre dans un milieu d'indice n(3) 
% 
% S vecteur source ( vecteur ligne pour les sources en dirac) 
%
%    EZ,HX,HY                <-- 3 composantes du vecteur source  ( en H// c'est HZ,-EX,-EY ) 
%       dirac(M-P) dans  rot(E):  HX,HY,0
%       dirac(M-P) dans  rot(H):  0,0,EZ
% au retour: e champ total,  e_oc 'onde cylindrique', e_res plasmon,  (e=e_oc+e_res)
% ( si le vecteur S a plus de 3 colonnes ,seuls les 3 premiéres comptent et la source est considérée comme clonée)
% si S a plusieurs lignes: 
% 	la premiere ligne correspond à des sources en dirac,
% 	la seconde: dérivées de diracs par rapport à x etc...
%
% 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	% Forme exacte ou approchee 'precise' %
% 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [e,e_oc,e_res]=retoc(n,S,P,x,y,pol,k0,parm);
% parm=struct('xymesh',0,'y0_varie',1,'approche',0)
% parm.xymesh= 0: x et y ne sont pas 'meshed'. On calcule les champs sur la grille 
%               les champs: e e_oc et e_res ont alors pour dimension [length(y) length(x) 3]
% 
% parm.xymesh=1:x et y  ont  même dimension. On calcule les champs aux points de coordonnees x(:),y(:) 
%                                     option par defaut si length(x)=length(y) parm.xymesh=1 sinon parm.xymesh=0
%               les champs: e e_oc et e_res ont alors pour dimension [length(x) 3]
%               le programme trie les points ayant la même valeur de phi , on a donc interêt à avoir des points sur des droites passant par l'origine
% parm.y0_varie= 1  le programme cherche au maximum à travailler à phi=pi/2 en utilisant un y0 non nul (gain de temps)			  
% parm.test= 1      calcul direct de l'integrale (pour tests)	(valeur par defaut 0 )
% parm.approche= 1   calcul approché pour tous les angles(y0=0 donc methode 'precise' )	(valeur par defaut 0 )
%  P coordonnees de la source. Si P est un vecteur ligne, le dioptre est paralléle à Ox
%                              Si P est un vecteur colonne, le dioptre est paralléle à Oy
%
% 	%%%%%%%%%%%%%%%%%%%
% 	% Forme approchée %
% 	%%%%%%%%%%%%%%%%%%%
% [e,e_oc,e_res,approche]=retoc(n,S,y0,phi,rho,pol,k0);
% phi y0 ont une seule valeur 
% si rho est non vide, calcul des champs pour ces valeurs
% les champs: e e_oc et e_res ont alors pour dimension [length(rho) 3]
% approche=struct('pas_residu',pas_residu,'A0',A0,'F0',F0,'A1',A1,'B1',B1,'F1',F1,'F2',F2,'A2',A2,'B2',B2,'F3',F3);
% structure à champs contenant les parametres de l'approximation
% 				pas_residu: 0
% 				A0: -0.0063 + 7.5265i
% 				F0: [-1.4767 - 0.0297i 0.0123 - 0.3012i 1.5071 + 0.0316i]
% 				A1: 0 + 7.3746i
% 				B1: 0.2813 - 0.2699i
% 	 approche   F1: [0 0 0]
% 				F2: [2.5998 + 2.3923i -0.4987 + 0.5199i -2.5998 - 2.3923i]
% 				A2: -36.8732 + 0.7375i
% 				B2: 0.5567 - 6.0973i
% 				F3: [-3.4545e-005 +5.0386e-004i -0.0001 + 0.0026i 0.0025 + 0.0001i]
% calcul de la forme approchee à partir de la structure 'approche' :
% 	rho=rho(:); 
% 	e_res=exp(approche.A0*rho)*approche.F0;
% 	e=(exp(rho*approche.A1)./sqrt(rho))*approche.F1+(exp(rho*approche.A1).*retoc(sqrt(rho)*approche.B1,1)./(rho.*sqrt(rho)))*approche.F2;
% 	if isfinite(approche.A2);e=e+(exp(rho*approche.A2).*retoc(sqrt(rho)*approche.B2,1)./(rho.*sqrt(rho)))*approche.F3;end;
% 	e_oc=e-approche.pas_residu*e_res;e=e_oc+e_res;
%
% Forme approchée 'rudimentaire' phi=pi/2 tres rapide mais limitée à y petit
% [e_oc,e_oc,e_res]=retoc(n,S,nan,x,y,pol,k0,parm);	 % idem à la forme approchee avec y0=nan ou inf		  
% si parm=1 : version 'amelioree' is parm=i  version simple sans le coupure du cote metal (pour tests et article)
% %
%
% 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	% calcul du developpement en ondes planes correspondant à des sources  %
% 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Ep,Em,angles]=retoc(n,{S1,..Sn},u,[xs1,..xsn],pol,k0,sens);% developpement en ondes planes dans le dielectrique
% sens=1 :dielectrique au dessus, sens=-1 :dielectrique au dessous, 
% u,Ep,Em,angles: voir retop
%
% 	%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	% calcul des integrales  %
%   %%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% I_0(x)=somme de -inf à inf de:  exp(-t^2)/(t-x) dt:                                          I_0 = retoc(x)
% I_1(x)=somme de -inf à inf de: (-2*x^2/sqrt(pi)) * t * exp(-t^2)/(t-x) dt:                   I_1 = retoc(x,1)
% L_2(x,a)=somme de -inf à inf de: (-2*x/sqrt(pi)) * t^2 * exp(2*a*t)* exp(-t^2)/(t-x) dt:     L_2 = retoc(x,a,2)
% W=@(x) retoc(sqrt(-i*x),1); % fonction de l'article
%
%%% EXEMPLES
% nb=sqrt(-33.22+1.17i);nh=1;ns=1;S=[.7i,.2,.5];pol=2;ld=.852;k0=2*pi/ld;
%%% coupe x=Cte  y varie
% x=2*ld;y=(-.2:.05:2)*ld;
% [e,e_oc]=retoc([nh,nb,ns],S,[0,0],x,y,pol,k0,struct('xymesh',0,'y0_varie',0)); % calcul exact
% [e,e_oc_a]=retoc([nh,nb,ns],S,[0,0],x,y,pol,k0,struct('xymesh',0,'y0_varie',0,'approche',1)); % calcul approché phi varie
% [e_a,e_oc_a90]=retoc([nh,nb,ns],S,nan,x,y,pol,k0);                      % calcul approché phi=pi/2
% figure;plot(y/ld,abs(e_oc(:,:,1)).^2,'-k',y/ld,abs(e_oc_a(:,:,1)).^2,'--r',y/ld,abs(e_oc_a90(:,:,1)).^2,'-.g','linewidth',2);xlabel('y/\lambda');legend('exact','approché','approché 90',2);title(rettexte('abs(H)^2  x/\lambda =',x/ld));grid;drawnow;
%
%%% coupe y=Cte  x varie
% y=0;x=logspace(-1,5,100)*ld;
% [e,e_oc]=retoc([nh,nb,ns],S,[0,0],x,y,pol,k0,struct('xymesh',0,'y0_varie',0)); % calcul exact
% [e_a,e_oc_a,e_res]=retoc([nh,nb,ns],S,nan,x,y,pol,k0);                               % calcul approché 'rudimentaire' 
% 
% figure;loglog(x/ld,abs(e_oc(:,:,1)),'-k',x/ld,abs(e_oc_a(:,:,1)),'--r',x/ld,abs(e_res(:,:,1)),'-.b','linewidth',2);xlabel('x/\lambda');legend('exact','approché','plasmon');title(rettexte('abs(H)  y/\lambda =',y/ld));grid;
%
% y=0;x=linspace(.1,25,1000)*ld;
% [e,e_oc]=retoc([nh,nb,ns],S,[0,0],x,y,pol,k0,struct('xymesh',0,'y0_varie',0)); % calcul exact
% [e_a,e_oc_a]=retoc([nh,nb,ns],S,nan,x,y,pol,k0);                               % calcul approché 'rudimentaire' 
%
% figure;plot(x/ld,real(e_oc(:,:,1)),'-k',x/ld,real(e_oc_a(:,:,1)),'--r','linewidth',2);xlabel('x/\lambda');legend('exact','approché');title(rettexte('real(H)  y/\lambda =',y/ld));grid;
%
% 	¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
% 	¤  VERSION 3D ( reseau 2D )  ¤
% 	¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
% 
% [e,e_oc,e_res]=retoc(n,S,P,x,y,z,k0,parm)
%
% S tableau de dimension 3 decrivant les sources:
% 
% le nombre de lignes est 1 ou 3
% 	premiere  ligne: sources directes
% 	seconde   ligne: sources dérivées en x
% 	troisieme ligne: sources dérivées en y
% le nombre de colonnes est 6: Ex Ey Ez Hx Hy Hz ( si 7 les sources sont considérées comme clonées )	
% la troisieme dimension est le nombre de sources	
%   Par exemple pour une source directe Ex: S=[1,0,0,0,0,0];	
%               pour explorer toutes les sources directes: S=reshape(eye(6),1,6,6);	
%               pour explorer toutes les sources directes et dérivées: S=permute(reshape(eye(18),6,3,18),[2,1,3]);	
%
% P: coordonnees du centre de la source (3 elements)
% si size(P)=[1,3,1], dioptre perpendiculaire à Oz ( exemple: P=[0,1,.5])
% si size(P)=[3,1,1], dioptre perpendiculaire à Ox ( exemple: P=[0;1;.5])
% si size(P)=[1,1,3], dioptre perpendiculaire à Oy ( exemple: P=zeros(1,1,3);P(1,1,:)=[0,1,.5])
%
%  si plus d'une source, les sorties e, e_oc e_res sont des cell_arrays de longueur egale au nombre de sources
%  si une seule source, les sorties sont des tableaux de dimension  [length(z),length(x),length(y),6]
% parm.xyzmesh=1:x  y et z ont  même dimension. On calcule les champs aux points de coordonnees x(:),y(:),z(:)
%                       option par defaut:  si length(x)=length(y)=length(z) parm.xyzmesh=1, sinon parm.xyzmesh=0   
%               les champs: e e_oc et e_res ont alors pour dimension [length(x) 6]
% parm.z0_varie= 1  le programme cherche au maximum à travailler à phi=pi/2 en utilisant un z0 non nul (gain de temps)			  
%
%% Exemple 3D 
% ld=.8;k0=2*pi/ld;nh=1;nb=retindice(ld,2);ns=nb;S=[0,0,0,0,1,0];
% x=linspace(-1,20,301)*ld;y=0;z=0;[e,e_oc,e_res]=retoc([nh,nb,ns],S,[0,0,0],x,y,z,k0,struct('z0_varie',0));		
% figure;font=retfont;pol=5;plot(x/ld,imag(e(:,:,:,pol)),'-k',x/ld,imag(e_oc(:,:,:,pol)),'--r',x/ld,imag(e_res(:,:,:,pol)),'--b','linewidth',2);
% set(legend('total','Onde spherique','plasmon'),font{3:end});title(rettexte(pol,'z/\lambda =',z/ld,'y/\lambda =',y/ld),font{3:end});
% xlabel('x/\lambda',font{3:end});axis tight;grid;ylim([-5,5]);set(gca,font{3:end});
%
%
%%% See also: RETPARTICULE,RETCHAMP,RETS,RETPL,RETTCHAMP,RETPOINT,RETSOMMERFELD,RETBAHAR,RETOP



if nargin<4;
if nargin<3;[varargout{1:nargout}]=cal_I0(varargin{:});else;[varargout{1:nargout}]=cal_L2(varargin{:});end    % cal_I0
elseif iscell(varargin{2});if nargin<7;[varargout{1:nargout}]=calpl(varargin{:});else;[varargout{1:nargout}]=calop(varargin{:});end;return;  % oc
else;
	
if length(varargin{3})>2;[varargout{1:nargout}]=retoc3D(varargin{:});else;[varargout{1:nargout}]=retoc1(varargin{:});end;
return;
end;






% 	¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
% 	¤   VERSION 2D ( reseau 1D E// ou H//) ¤
% 	¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤

function varargout=retoc1(n,S,xs,x,y,pol,k0,parm);
if nargin<8;parm=[];end;
if nargin<7;k0=1;end;
if nargin<6;pol=0;end;
if size(S,2)==3;S(:,1)=i*S(:,1);else;S=S(:,1:3);end;% clonage de la source si moins de 4 elements
if length(xs)==2; % ****  forme exacte
x=x-xs(1);y=y-xs(2);
if size(xs,2)==2;[varargout{1:nargout}]=onde_cylindrique_general(x,y,S,n(2),n(1),n(3),pol,k0,parm);% dioptre // ox
else;% dioptre // oy
[varargout{1:nargout}]=onde_cylindrique_general(y,x,[-S(:,1),S(:,3),S(:,2)],n(2),n(1),n(3),pol,k0,parm);
for ii=1:nargout;% echange des composantes
if size(varargout{ii},3)>1;[varargout{ii}(:,:,1),varargout{ii}(:,:,2),varargout{ii}(:,:,3)]=deal(-varargout{ii}(:,:,1),varargout{ii}(:,:,3),varargout{ii}(:,:,2));
varargout{ii}=permute(varargout{ii},[2,1,3]);
else;[varargout{ii}(:,1),varargout{ii}(:,2),varargout{ii}(:,3)]=deal(-varargout{ii}(:,1),varargout{ii}(:,3),varargout{ii}(:,2));end;
end;
end;
else;	          % ****  forme approchee
if isfinite(xs);  % <<< forme approchee generale	
[varargout{1:nargout}]=onde_cylindrique_approchee(x,  xs,S ,n(2),n(1),n(3),pol, k0, y,parm);
                     %onde_cylindrique_approchee(phi,y0,S, nb,  nh,   ns, pol, k0,rho,parm);
else;             % <<< forme approchee 90
[varargout{1:nargout}]=retoc_90(x,y,S,n(2),n(1),n(3),pol,k0,parm); 
                    % retoc_90(x,y,S,nb,nh,ns,pol,k0,parm);
end;               % <<< 
end;             % ****  forme exacte ?


%%%%%%%%%%%%%%%%%%%%%
%  CALCUL EXACT     %
%%%%%%%%%%%%%%%%%%%%%
%***************************************************************************************************************
function [e,e_oc,e_res]=onde_cylindrique_general(x,y,S,nb,nh,ns,pol,k0,parm);
defaultopt=struct('xymesh',1,'y0_varie',0,'test',0,'approche',0,'periode',[],'beta',0);
xymesh=retoptimget(parm,'xymesh',defaultopt,'fast');
if length(x)~=length(y) xymesh=0;end;
if isfield(parm,'periode');periode=parm.periode*k0;else;periode=[];end;
if isfield(parm,'beta');beta=parm.beta/k0;else;beta=0;end;
y0_varie=retoptimget(parm,'y0_varie',defaultopt,'fast');
test=retoptimget(parm,'test',defaultopt,'fast');
approche=retoptimget(parm,'approche',defaultopt,'fast');
if xymesh==0; % <----u v non meshed
nx=length(x);ny=length(y);
[y,x]=ndgrid(y,x);x=x(:).';y=y(:).';
end
parm=struct('xymesh',1,'y0_varie',y0_varie,'test',test,'approche',approche,'periode',periode,'beta',beta);
[e,e_oc,e_res]=deal(zeros(length(x),3));if isempty(x);return;end;

[f,ff]=retfind(x>=0);[e(ff,:),e_oc(ff,:),e_res(ff,:)]=onde_cylindrique_general(-x(ff),y(ff),diag((-1).^(0:size(S,1)-1))*S*diag([1,1,-1]),nb,nh,ns,pol,k0,parm);e(ff,3)=-e(ff,3);e_oc(ff,3)=-e_oc(ff,3);e_res(ff,3)=-e_res(ff,3);
[f,ff]=retfind(y>=0,f);[e(ff,:),e_oc(ff,:),e_res(ff,:)]=onde_cylindrique_general(x(ff),-y(ff),S*diag([1,-1,1]),nh,nb,ns,pol,k0,parm);e(ff,2)=-e(ff,2);e_oc(ff,2)=-e_oc(ff,2);e_res(ff,2)=-e_res(ff,2);

[e(f,:),e_oc(f,:),e_res(f,:)]=onde_cylindrique_haut(x(f)*k0,y(f)*k0,diag(k0.^(1:size(S,1)))*S,nb,nh,ns,pol,y0_varie,parm);

if xymesh==0; % <----u v non meshed
e=reshape(e,ny,nx,[]);
e_oc=reshape(e_oc,ny,nx,[]);
e_res=reshape(e_res,ny,nx,[]);
end;

%***************************************************************************************************************
function [e,e_oc,e_res]=onde_cylindrique_haut(x,y,S,nb,nh,ns,pol,y0_varie,parm);
[e,e_oc,e_res]=deal(zeros(length(x),3));if isempty(x);return;end;

	if parm.test==1;% calcul direct brutal pour tests
	plasmon=retplasmon(nb,nh,2*pi);%if ~plasmon.test_d_existence;plasmon.constante_de_propagation=nan;end;
	fin=[-1,-.1,-.01,0,.01,.1,1];
	A=5*max(real(nh),real(nb));B=5*30*max(real(nh),real(nb));
	[u1,wu1]=retgauss(-B,-A,5,1000);
	[u2,wu2]=retgauss(-A,A,20,1500,[real(nh)+.01*fin,-real(nh)+.01*fin,real(nb)+.01*fin,-real(nb)+.01*fin,real(plasmon.constante_de_propagation)+imag(plasmon.constante_de_propagation)*fin,-real(plasmon.constante_de_propagation)+imag(plasmon.constante_de_propagation)*fin]);
	[u3,wu3]=retgauss(A,B,5,1000);
	u=[u1,u2,u3];wu=[wu1,wu2,wu3];
	for ii=1:length(x);	
	hh=calhh(S,u.',nh,nb,ns,0,x(ii),y(ii),pol);
	e(ii,:)=hh*(wu.');
	end;
	% figure;plot(u,real(hh(1,:)),'-k','linewidth',2);
	return;
	end;
	
	if parm.approche==1;% calcul approché avec la methode precise
	phi=atan2(x,y);
	rho=sqrt(x.^2+y.^2);
	[phi,k,kk]=retelimine(phi); % tri des phi egaux (gain de temps)
	for ii=1:length(k);
	jj=find(kk==ii);
	[e(jj,:),e_oc(jj,:),e_res(jj,:)]=onde_cylindrique_approchee(phi(ii),0,S,nb,nh,ns,pol,1,rho(jj),parm);
	end;
	return;
	end;

if y0_varie;% *************
[f,ff]=retfind(y<(pi/abs(real(sqrt(-2i*nh))))*sqrt(x));  
[yy,k,kk]=retelimine(y(f));     % tri des y egaux (gain de temps)
for ii=1:length(k);
jj=f(kk==ii);
[e(jj,:),e_oc(jj,:),e_res(jj,:)]=onde_cylindrique(x(jj),pi/2,S,nb,nh,ns,pol,yy(ii),parm);
end;

% pour les petits phi on fait le calcul avec y0=0;
phi=atan2(x(ff),y(ff));
rho=sqrt(x(ff).^2+y(ff).^2);
[phi,k,kk]=retelimine(phi);  % tri des phi egaux (gain de temps)
for ii=1:length(k);
jj=find(kk==ii);
[e(ff(jj),:),e_oc(ff(jj),:),e_res(ff(jj),:)]=onde_cylindrique(rho(jj),phi(ii),S,nb,nh,ns,pol,0,parm);
end;

else;    % *************
phi=atan2(x,y);
rho=sqrt(x.^2+y.^2);
[phi,k,kk]=retelimine(phi); % tri des phi egaux (gain de temps)
for ii=1:length(k);
jj=find(kk==ii);
[e(jj,:),e_oc(jj,:),e_res(jj,:)]=onde_cylindrique(rho(jj),phi(ii),S,nb,nh,ns,pol,0,parm);
end;
	
end;     % *************

% declonage
e(:,2:3)=-i*e(:,2:3);
e_oc(:,2:3)=-i*e_oc(:,2:3);
e_res(:,2:3)=-i*e_res(:,2:3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  CALCUL APPROCHEE (METHODE DU COL)   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%***************************************************************************************************************
function [e,e_oc,e_res,approche]=onde_cylindrique_approchee(phi,y0,S,nb,nh,ns,pol,k0,rho,parm);
if (phi<0);
[e,e_oc,e_res,approche]=onde_cylindrique_approchee(-phi,y0,S,nb,nh,ns,pol,k0,rho,parm);
approche.F1(3)=-approche.F1(3);approche.F2(3)=-approche.F2(3);
approche.F3(3)=-approche.F3(3);
approche.F0(3)=-approche.F0(3);
e(:,3)=-e(:,3);
e_oc(:,3)=-e_oc(:,3);
e_res(:,3)=-e_res(:,3);

return;
end
if (phi>pi/2);
[e,e_oc,e_res,approche]=onde_cylindrique_approchee(pi-phi,-y0,S*diag([1,-1,1]),nh,nb,ns,pol,k0,rho,parm);
approche.F1(2)=-approche.F1(2);approche.F2(2)=-approche.F2(2);
approche.F3(2)=-approche.F3(2);
approche.F0(2)=-approche.F0(2);
e(:,2)=-e(:,2);
e_oc(:,2)=-e_oc(:,2);
e_res(:,2)=-e_res(:,2);
return;
end

[e,e_oc,e_res,approche]=onde_cylindrique([],phi,diag(k0.^(1:size(S,1)))*S,nb,nh,ns,pol,y0*k0,parm);
% mise à l'echelle en k0 et declonage
approche.A0=approche.A0*k0;
approche.A1=approche.A1*k0;approche.B1=approche.B1*sqrt(k0);approche.F1=approche.F1/sqrt(k0);approche.F2=approche.F2/(k0*sqrt(k0));
approche.A2=approche.A2*k0;approche.B2=approche.B2*sqrt(k0);approche.F3=approche.F3/(k0*sqrt(k0));
approche.F0(2:3)=-i*approche.F0(2:3);
approche.F1(2:3)=-i*approche.F1(2:3);
approche.F2(2:3)=-i*approche.F2(2:3);
approche.F3(2:3)=-i*approche.F3(2:3);


% calcul de la forme approchee mise à l'echelle en k0 et declonée
rho=rho(:);rho(rho==0)=1/realmax;
e_res=exp(approche.A0*rho)*approche.F0;
e=(exp(rho*approche.A1)./sqrt(rho))*approche.F1+(exp(rho*approche.A1).*retoc(sqrt(rho)*approche.B1,1)./(rho.*sqrt(rho)))*approche.F2;
if isfinite(approche.A2);e=e+(exp(rho*approche.A2).*retoc(sqrt(rho)*approche.B2,1)./(rho.*sqrt(rho)))*approche.F3;end;
e_oc=e-approche.pas_residu*e_res;
e=e_oc+e_res;
%***************************************************************************************************************
function [e,e_oc,e_res,approche]=onde_cylindrique(rho,phi,S,nb,nh,ns,pol,y0,parm);if nargin<8;y0=0;end;
persistent lorentz;if isempty(lorentz);lorentz=retlorentz(i,51,10);end;% gain de temps

rho=rho(:).';rho(rho==0)=1/realmax;
if abs(phi-pi/2)<100*eps;[e,e_oc,e_res,approche]=onde_cylindrique_90(rho,S,nb,nh,ns,pol,y0,parm);return;end;% forme simplifiée
[e,e_oc,e_res]=deal(zeros(length(rho),3));[F0,A1,B1,F1,F2,A2,B2]=deal(zeros(1,3));
cphi=cos(phi);sphi=sin(phi);
fin=[-10,-1,1,10];

beta=sqrt(1./(nh.^-2+nb.^-2));C=[];

khid=coupure_droite(beta,nh);khim=coupure_droite(beta,nb);
khihb=coupure_droite(nb,nh);
test=abs((khim+(nb./nh).^2.*khid)./(2*khim))<1.e-10;

if (real(nb)>real(nh))&(phi<atan(-real(khid-khihb)/real(beta-nb)));test=0;end;% existance du plasmon

%C=init_coupure(nh,nb,cphi,sphi);[khid,h,khim]=calkhi(beta,nh,nb,cphi,sphi,C);test=abs((khim+(nb./nh).^2.*khid)./(2*khim))<1.e-10;


plasmon=struct('constante_de_propagation',beta,'khi_metal',khim,'khi_dielectrique',khid,'test_d_existence',test);
if pol==0;plasmon.test_d_existence=0;end;
if plasmon.test_d_existence;
tho_pl=sqrt(-i*(plasmon.constante_de_propagation*sphi+plasmon.khi_dielectrique*cphi-nh));
else;plasmon.constante_de_propagation=nan;tho_pl=nan+i*nan;
end
tho_nb=sqrt(-i*(nb*sphi+coupure_droite(nb,nh)*cphi-nh));
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal(tho_pl.',cphi,sphi,nh,nb,ns,S,pol,C,y0);resonance=max(abs(f1))>1.e10;
super_resonance=resonance & abs(imag(tho_pl))<.01;

if resonance;tho=[real(tho_pl)+lorentz*imag(tho_pl),real(tho_nb)+lorentz*imag(tho_nb)];else;tho=real(tho_nb)+lorentz*imag(tho_nb);end;
[u1,u2]=lowenthal(tho.',cphi,sphi,nh,nb,ns,S,pol,C);
u=[u1,u2];[x,ii]=retelimine(real(u));y=imag(u(ii));
if length(x)>2;% cas phi=pi/2
test_nb=retinterp(x,y,real(nb))>imag(nb); % il faut ajouter la coupure ( garder le retinterp pour l'extrapolation )
if plasmon.test_d_existence;test_pl=retinterp(x,y,real(plasmon.constante_de_propagation))<=imag(plasmon.constante_de_propagation);else;test_pl=0;end;% il faut retrancher le plasmon
%figure;plot(real(nb),imag(nb),'ob',real(nh),imag(nh),'or',real(plasmon.constante_de_propagation),imag(plasmon.constante_de_propagation),'dc',x,y,'.-k','markerfacecolor','k');legend('nb','nh','pl','Talweg');grid;title(rettexte(test_nb,test_pl));
else;test_nb=1;test_pl=0;end;

if isempty(rho);  % forme approchee
dtho=1.e-4;tho=retgauss(0,dtho,4);
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal(tho.',cphi,sphi,nh,nb,ns,S,pol,C,y0);
k1=retdiag(du1)*squeeze(f1);k2=retdiag(du2)*squeeze(f2);kk=[flipud(k2);k1];
Tho=[-fliplr(tho),tho].';
[F1,F2]=col(Tho,kk);
A1=i*nh;
if resonance;B1=tho_pl;else;B1=nan+i*nan;end;
else;%  <----- calcul de l'integrale pour tous les rho
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal(0,cphi,sphi,nh,nb,ns,S,pol,C,y0);% pour gain de temps
f=find(real(h1*rho)>-36);
if ~isempty(f);%<<<<<<<<<<<<<<

[tho0,tho_max]=choisi_tho(rho(f));
	if super_resonance;% on retranche la resonance pour l'integrer à la main
	tho=retgauss(real(tho_pl)-imag(tho_pl),real(tho_pl)+imag(tho_pl),5);	
	[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal(tho.',cphi,sphi,nh,nb,ns,S,pol,C,y0);
	f0=zeros(1,3);
	for ii=1:3;
	f0(ii)=calder(tho.'-tho_pl,du1.*(tho.'-tho_pl).*f1(:,:,ii));
	end
	else;f0=zeros(1,3);
	end;

[tho,wtho]=retgauss(0,tho_max,20,10,[tho0,abs(real(tho_nb))+imag(tho_nb)*fin]);
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal(tho.',cphi,sphi,nh,nb,ns,S,pol,C,y0);
	%if super_resonance;for ii=1:3;f1(:,:,ii)=f1(:,:,ii)-(1./((tho.'-tho_pl).*du1))*f0(ii);f2(:,:,ii)=f2(:,:,ii)-(1./((-tho.'-tho_pl).*du2))*f0(ii);end;end;
	if super_resonance;prv1=1./((tho.'-tho_pl).*du1);prv2=1./((-tho.'-tho_pl).*du2);for ii=1:3;f1(:,:,ii)=f1(:,:,ii)-prv1*f0(ii);f2(:,:,ii)=f2(:,:,ii)-prv2*f0(ii);end;end;

g=zeros(length(h1),length(f));prv=h1*rho(f);% gain de temps
ff=find(  real(prv)> (max(real(prv(:))) -36));
g(ff)=exp(prv(ff));
%g=exp(h1*rho(f));
for ii=1:3;	e_oc(f,ii)=(wtho.*du1.'.*f1(:,:,ii).')*g+(wtho.*du2.'.*f2(:,:,ii).')*g;end;
%%ii=1;figure;plot([-fliplr(tho),tho],real([flipud(f2(:,:,ii).*du2.*g(:,1));f1(:,:,ii).*du1.*g(:,1)]),'.r',[-fliplr(tho),tho],imag([flipud(f2(:,:,ii).*du2.*g(:,1));f1(:,:,ii).*du1.*g(:,1)]),'.b');
%ii=1;figure;plot([-fliplr(tho),tho],real([flipud(f2(:,:,ii).*du2.*g(:,1));f1(:,:,ii).*du1.*g(:,1)]),'-k','linewidth',2);

   if super_resonance;e_oc(f,:)=e_oc(f,:)+(exp(i*nh*rho(f)).*retoc(tho_pl*sqrt(rho(f)))).'*f0;end;
	
end;         %<<<<<<<<<<<<<<
end;            %  <-----
[e_res,F0,A0]=onde_cylindrique_residu(rho,cphi,sphi,S,nb,nh,ns,plasmon,pol,y0);

if test_nb;% on ajoute la coupure
[ec,A2,B2,F3,uu1]=coupure(cphi,sphi,rho,S,nb,nh,ns,plasmon,pol,C,y0);e_oc=e_oc+ec;
else;F3=deal(zeros(1,3));B2=nan+i*nan;A2=nan+i*nan;uu1=[];
end;
if test_pl;e=e_oc;e_oc=e-e_res;else;e=e_oc+e_res;end;
pas_residu=double(test_pl);
approche=struct('pas_residu',pas_residu,'A0',A0,'F0',F0,'A1',A1,'B1',B1,'F1',F1,'F2',F2,'A2',A2,'B2',B2,'F3',F3);

%figure;font=retfont;plot(real(nb),imag(nb),'ob',real(nh),imag(nh),'or',real(plasmon.constante_de_propagation),imag(plasmon.constante_de_propagation),'dc',x,y,'.-k',real(uu1),imag(uu1),'.-g','markerfacecolor','k');set(legend('nb','nh','pl','Talweg','coupure'),font{3:end});grid;title(rettexte(test_nb,test_pl),font{:});set(gca,font{3:end});xlabel('Real( u )',font{3:end});ylabel('Imag( u )',font{3:end});
%***************************************************************************************************************
function [e_res,F0,A0]=onde_cylindrique_residu(rho,cphi,sphi,S,nb,nh,ns,plasmon,pol,y0);% residu
e_res=zeros(length(rho),3);F0=zeros(1,3);A0=0;if ~plasmon.test_d_existence;return;end;
Sh=calSh(S,plasmon.constante_de_propagation,plasmon.khi_dielectrique,plasmon.khi_metal,nh,nb,ns,pol,y0);
res=-Sh/(plasmon.constante_de_propagation*(1/(nh^pol*plasmon.khi_dielectrique)+1/(nb^pol*plasmon.khi_metal)));
%res=-((nh^3*nb^3)/(nh^4-nb^4))*Sh;% mauvaise formule car depend du sens
F0=2i*pi*reshape(res,1,3);A0=calh(plasmon.constante_de_propagation,plasmon.khi_dielectrique,plasmon.khi_metal,cphi,sphi);
if ~isempty(rho);e_res=exp(A0*rho.')*F0;end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [e,A2,B2,F3,u1]=coupure(cphi,sphi,rho,S,nb,nh,ns,plasmon,pol,C,y0);
e=zeros(length(rho),3);[A2,B2,F3]=deal([]);
fin=[-10,-1,1,10];

tho_pl=sqrt(i*((nb-plasmon.constante_de_propagation)*sphi+(coupure_droite(nb,nh)-plasmon.khi_dielectrique)*cphi));
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_bis(tho_pl.',cphi,sphi,nh,nb,ns,S,pol,C,y0);
resonance=max(abs(f1))>1.e10;
super_resonance=resonance & abs(imag(tho_pl))<.01;

if isempty(rho); %  <----- forme approchee
dtho=1.e-2;tho=retgauss(0,dtho,4);
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_bis(tho.',cphi,sphi,nh,nb,ns,S,pol,C,y0);
k1=retdiag(du1)*squeeze(f1);k2=retdiag(du2)*squeeze(f2);kk=[flipud(k2);k1];
Tho=[-fliplr(tho),tho].';
A2=i*nb*sphi+i*coupure_droite(nb,nh)*cphi;if resonance;B2=tho_pl;else;B2=nan+i*nan;end;
[prv,F3]=col(Tho,kk);

else;            %  <---- calcul de l'integrale pour tous les rho
[u1,u2,h0,h2,f1,f2,du1,du2]=lowenthal_bis(0,cphi,sphi,nh,nb,ns,S,pol,C,y0);% pour gain de temps
                                     % h0 est utilise pour  traiter la 'super resonance'
f=find(real(h0*rho)>-36);if isempty(f);return;end;	
[tho0,tho_max]=choisi_tho(rho(f));

	if super_resonance;% on retranche la resonance pour l'integrer à la main
	tho=retgauss(real(tho_pl)-imag(tho_pl),real(tho_pl)+imag(tho_pl),5);
	[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_bis(tho.',cphi,sphi,nh,nb,ns,S,pol,C,y0);
	f0=zeros(1,3);
	for ii=1:3;
	f0(ii)=calder(tho.'-tho_pl,du1.*(tho.'-tho_pl).*f1(:,:,ii));
	end
	else;f0=zeros(1,3);
	end;

if resonance;
[tho,wtho]=retgauss(0,tho_max,20,10,[tho0,abs(real(tho_pl))+imag(tho_pl)*fin]);
else;
[tho,wtho]=retgauss(0,tho_max,20,10,tho0);
end;
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_bis(tho.',cphi,sphi,nh,nb,ns,S,pol,C,y0);
	if super_resonance;prv1=1./((tho.'-tho_pl).*du1);prv2=1./((-tho.'-tho_pl).*du2);for ii=1:3;f1(:,:,ii)=f1(:,:,ii)-prv1*f0(ii);f2(:,:,ii)=f2(:,:,ii)-prv2*f0(ii);end;end;

g=zeros(length(h1),length(f));prv=h1*rho(f);% gain de temps
ff=find(  real(prv)> (max(real(prv(:))) -36));
g(ff)=exp(prv(ff));

%g=exp(h1*rho(f));
for ii=1:3;	e(f,ii)=(wtho.*du1.'.*f1(:,:,ii).')*g+(wtho.*du2.'.*f2(:,:,ii).')*g;end;
%  %ii=1;figure;plot([-fliplr(tho),tho],real([flipud(f2(:,:,ii).*du2.*g(:,1));f1(:,:,ii).*du1.*g(:,1)]),'.r',[-fliplr(tho),tho],imag([flipud(f2(:,:,ii).*du2.*g(:,1));f1(:,:,ii).*du1.*g(:,1)]),'.b');
% ii=1;figure;plot([-fliplr(tho),tho],real([flipud(f2(:,:,ii).*du2.*g(:,1));f1(:,:,ii).*du1.*g(:,1)]),'-k','linewidth',2);
   if super_resonance; e(f,:)=e(f,:)+(exp(h0*rho(f)).*retoc(tho_pl*sqrt(rho(f)))).'*f0;end;

end;              %  <-----

%***************************************************************************************************************
function hh=calhh(S,u,nh,nb,ns,y0,x,y,pol);
khih=retsqrt(nh^2-u.^2,-1);
khib=retsqrt(nb^2-u.^2,-1);
Sh=calSh(S,u,khih,khib,nh,nb,ns,pol,y0);
for ii=1:3;
hh(ii,:)=(Sh(:,:,ii)./(khih/nh.^pol+khib/nb.^pol)).*exp(i*(u*x+khih*y));
end;
%***************************************************************************************************************
function h=calh(u,khih,khib,cphi,sphi);
h=i*(u*sphi+khih*cphi);
%***************************************************************************************************************
function Sh=calSh(S,u,khih,khib,nh,nb,ns,pol,y0);
Sh=zeros(length(u),length(y0),3);
Sh(:,:,1)=retdiag((i*S(1,1)+khib/nb^pol*S(1,2)-(u./ns^pol)*S(1,3))/(2*pi))*exp(i*khih*y0(:).');  % E
for ii=2:size(S,1);Sh(:,:,1)=Sh(:,:,1)+retdiag(((i*u).^(ii-1)).*(i*S(ii,1)+khib/nb^pol*S(ii,2)-(u./ns^pol)*S(ii,3))/(2*pi))*exp(i*khih*y0(:).');end;  % H
Sh(:,:,2)=retdiag(khih*(i/nh^pol))*Sh(:,:,1);                                              % Hx
Sh(:,:,3)=retdiag(-u*(i/nh^pol))*Sh(:,:,1);                                                % Hy
%***************************************************************************************************************
function [u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal(tho,cphi,sphi,nh,nb,ns,S,pol,C,y0);
h1=i*nh-tho.^2;h2=h1;
v=retsqrt(tho.^2-2*i*nh,-1,pi);retsqrt(0,0,pi/2);
u1=sphi*(nh+i*tho.^2)+tho.*cphi.*v;
u2=sphi*(nh+i*tho.^2)-tho.*cphi.*v;
if nargout<3;return;end;
du1=2*(i*tho*sphi+cphi*(tho.^2-i*nh)./v);
du2=2*(-i*tho*sphi+cphi*(tho.^2-i*nh)./v);
khih1=retsqrt(nh^2-u1.^2,-1,0);
khih2=retsqrt(nh^2-u2.^2,-1,pi);
retsqrt(0,0,pi/2);

u0=nh*sphi;khib0=retsqrt(nb^2-u0^2,-1);% la coupure de Maystre suffit ?
khib1=Sqrt(nb^2-[u0;u1].^2,pi/2);if abs(khib1(1)-khib0)>abs(khib1(1)+khib0);khib1=-khib1;end;khib1=khib1(2:end);
khib2=Sqrt(nb^2-[u0;u2].^2,pi/2);if abs(khib2(1)-khib0)>abs(khib2(1)+khib0);khib2=-khib2;end;khib2=khib2(2:end);



Sh1=calSh(S,u1,khih1,khib1,nh,nb,ns,pol,y0);
Sh2=calSh(S,u2,khih2,khib2,nh,nb,ns,pol,y0);
f1=zeros(size(Sh1)); f2=zeros(size(Sh2));
st_warning=warning;warning off;
for ii=1:3;
f1(:,:,ii)=Sh1(:,:,ii)./(khih1/nh^pol+khib1/nb^pol);
f2(:,:,ii)=Sh2(:,:,ii)./(khih2/nh^pol+khib2/nb^pol);
end;
warning(st_warning);

%***************************************************************************************************************
function [u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_bis(tho,cphi,sphi,nh,nb,ns,S,pol,C,y0);
khib=coupure_droite(nb,nh);
a1=nb*sphi+khib*cphi;
a2=nb*cphi-khib*sphi;
h1=i*a1-tho.^2;h2=h1;
v=Sqrt(1+tho.^2.*(tho.^2-2i*a1)/a2^2,pi);
u1=sphi.*(a1+i*tho.^2)+(a2*cphi)*v;u2=u1;
khih1=coupure_droite(u1,nh);
khih2=khih1;

du1=2*tho.*(i*sphi+(cphi/a2)*(tho.^2-i*a1)./v);du2=-du1;

prv=(.5/a2)*(1-(1/a2^2)*((tho.^2-2i*a1)./(v+1)).^2);
z=-2*nb*cphi*prv-(-i*khib/a2+cphi*tho.^2.*prv).^2;
aa=2i*nb*khib/a2;
khib1=retsqrt(aa,0)*tho.*Sqrt(1+(tho.^2.*z)/aa,pi);

khib2=-khib1;

Sh1=calSh(S,u1,khih1,khib1,nh,nb,ns,pol,y0);
Sh2=calSh(S,u2,khih2,khib2,nh,nb,ns,pol,y0);
f1=zeros(size(Sh1)); f2=zeros(size(Sh2));
st_warning=warning;warning off;
for ii=1:3;
f1(:,:,ii)=Sh1(:,:,ii)./(khih1/nh^pol+khib1/nb^pol);
f2(:,:,ii)=Sh2(:,:,ii)./(khih2/nh^pol+khib2/nb^pol);
end;
warning(st_warning);
%***************************************************************************************************************
function a=Acos(t);
a=acos(t);prv=angle(a);
f=find(abs(unwrap(2*prv)-2*prv)>.1);
a(f)=-a(f);
%***************************************************************************************************************
function a=Sqrt(t,tet);
a=retsqrt(t,-1,tet);retsqrt(0,0,pi/2);prv=angle(a);
f=find(abs(unwrap(2*prv)-2*prv)>.1);
a(f)=-a(f);
%***************************************************************************************************************
function C=init_coupure(nh,nb,cphi,sphi);
persistent nh_store khih1 khih2 u % pour gain de temps
if isempty(nh_store) nh_store=nan;end;;
if nh~=nh_store;
t=[ 0:.01:9.99 , 10:.1:99.9 , 100:1:10000 ].^2;
u=nb+i*t;
khih1=coupure_droite(u,nh);
khih2=coupure_droite(-u,nh);
nh_store=nh;
end;
prv=u*sphi;
h1_sur_i=prv+khih1*cphi;
h2_sur_i=-prv+khih2*cphi;

xc1=-imag(h1_sur_i);yc1=real(h1_sur_i);
xc2=-imag(h2_sur_i);yc2=real(h2_sur_i);

C={{xc1,yc1},{xc2,yc2}};

%***************************************************************************************************************
function khi=coupure_droite(u,n);
% khi=zeros(size(u));
% [f,ff]=retfind((real(u)>=real(nh))|(real(u)<=-real(nh)));
% khi(ff)=retsqrt(n^2-u(ff).^2,-1,1.5*pi/2);
% khi(f)=retsqrt(n^2-u(f).^2,-1,0);retsqrt(0,0,pi/2);
khi=i*sqrt(i*(u-n)).*sqrt(-i*(u+n));

%***************************************************************************************************************
function [khih,h,khib]=calkhi(u,nh,nb,cphi,sphi,C);
% calcul de khih
khih=coupure_droite(u,nh);
if nargout<2;return;end;
% calcul de h
h=i*(u*sphi+khih*cphi);
if nargout<3;return;end;
khib=coupure_droite(u,nb);

	if real(nb)>real(nh);
	f1=find(imag(u)>imag(nb)  & real(u)>=real(nh)  );f1=f1(change_coupure(h(f1),C{1},-1));khib(f1)=-khib(f1); 
	f2=find(imag(u)<imag(-nb) & real(u)<=real(-nh) );f2=f2(change_coupure(h(f2),C{2},-1));khib(f2)=-khib(f2); 
	else;
	f1=find(imag(u)>imag(nb)  & real(u)<=real(nh)    );f1=f1(change_coupure(h(f1),C{1},1));khib(f1)=-khib(f1); 
	f2=find(imag(u)<imag(-nb) & real(u)>=real(-nh)    );f2=f2(change_coupure(h(f2),C{2},1));khib(f2)=-khib(f2); 
	end;

% F1=find(imag(u)>0 &(real(u)-real(nh))*(real(nb)-real(nh))>=0);F1=F1(change_coupure(h(F1),C{1},1));khib(F1)=-khib(F1); 
% F2=find(imag(u)<0 &(real(u)-real(-nh))*(real(-nb)-real(-nh))>=0);F2=F2(change_coupure(h(F2),C{2},-1));khib(F2)=-khib(F2); 
%***************************************************************************************************************
function f=change_coupure(h,C,sens);
%C={xc1,yc1}  limites ouvertes
sens=sens*sign(C{1}(1)-C{1}(2));

f=find(imag(h)>C{2}(1)+10*eps&imag(h)<C{2}(end)-10*eps);
b=interp1(C{2},C{1},imag(h(f)));
if sens>0;f=f( real(h(f))<b-10*eps );else;f=f( real(h(f))>b+10*eps );end;
%***************************************************************************************************************
function [F1,F2,der0,der1,der2]=col(Tho,kk); % approximation de la methode du col
[der0,der1,der2]=deal(zeros(1,3));
for ii=1:3;
[der0(ii),der1(ii),der2(ii)]=calder(Tho,kk(:,ii));
end;
F1=sqrt(pi)*der0;
F2=(sqrt(pi)/4)*der2;
%***************************************************************************************************************
function [g0,g1,g2]=calder(Tho,gg);            % derivèes à l'origine
[p,prv,mu]=polyfit(Tho,gg(:,1),4);
pp=polyder(p);ppp=polyder(pp);
g0=polyval(p,-mu(1)/mu(2));
g1=polyval(pp,-mu(1)/mu(2))/mu(2);
g2=polyval(ppp,-mu(1)/mu(2))/mu(2)^2;



	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% %  TRAITEMENT SPECIAL DU CAS phi=pi/2 % %
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%***************************************************************************************************************
function [e,e_oc,e_res,approche]=onde_cylindrique_90(rho,S,nb,nh,ns,pol,y0,parm);if nargin<8;parm=[];end;if nargin<7;y0=0;end;
persistent lorentz;if isempty(lorentz);lorentz=retlorentz(i,51,10);end;% gain de temps
[e,e_oc,e_res]=deal(zeros(length(rho),3));[F0,A1,B1,F1,F2,F3,A2,B2]=deal(zeros(1,3));
fin=[-10,-1,1,10];
beta=sqrt(1./(nh.^-2+nb.^-2));khid=coupure_droite(beta,nh);khim=coupure_droite(beta,nb);
test=abs((khim+(nb./nh).^2.*khid)./(2*khim))<1.e-10;
plasmon=struct('constante_de_propagation',beta,'khi_metal',khim,'khi_dielectrique',khid,'test_d_existence',test);
if pol==0;plasmon.test_d_existence=0;end;

tho_pl=sqrt(-i*(plasmon.constante_de_propagation-nh));
tho_nb=sqrt(-i*(nb-nh));
tho=[real(tho_pl)+lorentz*imag(tho_pl),real(tho_nb)+lorentz*imag(tho_nb)];
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_90(tho_pl.',nh,nb,ns,S,pol,y0);resonance=max(abs(f1))>1.e10;
super_resonance=resonance & abs(imag(tho_pl))<.01;
if isempty(rho); %  <----- forme approchee  calcul analytique des dérivéés
khib0=coupure_droite(nh,nb);khih0=coupure_droite(nb,nh);
muh=nh^pol;mub=nb^pol;mus=ns^pol;
f_col=i*sqrt(pi)*((i-1)*sqrt(nh)/(2*pi))*(mub/khib0)*[i*y0-mub/(muh*khib0);i/muh;-(i*nh/muh)*(i*y0-mub/(muh*khib0))]*[i,khib0/mub,-nh/mus];
f_b=i*sqrt(pi)*((i-1)*sqrt(nb)/(2*pi))*(muh/(mub*khih0))*exp(i*khih0*y0)*[1;i*khih0/muh;-i*nb/muh]*[-i*muh/khih0,1,nb*muh/(mus*khih0)];
if size(S,1)>1;S0=(i*nh).^(0:size(S,1)-1)*S;Sb=(i*nb).^(0:size(S,1)-1)*S;else;S0=S;Sb=S;end;
F1=zeros(1,3);
F2=(f_col*S0.').';
F3=(f_b*Sb.').';
if resonance B1=tho_pl;else;B1=nan+i*nan;end;
A1=i*nh;A2=i*nb;B2=sqrt(i*(nb-plasmon.constante_de_propagation));
else;            %  <----- calcul de l'integrale pour tous les rho
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_90(0,nh,nb,ns,S,pol,y0);% pour gain de temps
f=find(real(h1*rho)>-36);
if ~isempty(f);%<<<<<<<<<<<<<<
	if super_resonance;% on retranche la resonance pour l'integrer à la main
	tho=retgauss(real(tho_pl)-imag(tho_pl),real(tho_pl)+imag(tho_pl),5);	
	[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_90(tho.',nh,nb,ns,S,pol,y0);
	f0=zeros(1,3);
	for ii=1:3;
	f0(ii)=calder(tho.'-tho_pl,du1.*(tho.'-tho_pl).*f1(:,:,ii));
	end
	else;f0=zeros(1,3);
	end;
[tho0,tho_max]=choisi_tho(rho(f));
[tho,wtho]=retgauss(0,tho_max,20,10,[tho0,abs(real(tho_nb))+imag(tho_nb)*fin,abs(real(tho_pl))+imag(tho_pl)*fin]);
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_90(tho.',nh,nb,ns,S,pol,y0);
	if super_resonance;prv1=1./((tho.'-tho_pl).*du1);prv2=1./((-tho.'-tho_pl).*du2);for ii=1:3;f1(:,:,ii)=f1(:,:,ii)-prv1*f0(ii);f2(:,:,ii)=f2(:,:,ii)-prv2*f0(ii);end;end;
g=exp(h1*rho(f));if ~isempty(parm.periode);g=retdiag(1./(1-exp((h1-i*parm.beta)*parm.periode)))*g;end;% Methode Haitao
for ii=1:3;	e_oc(f,ii)=(wtho.*du1.'.*f1(:,:,ii).')*g+(wtho.*du2.'.*f2(:,:,ii).')*g;end;
	
%%  ii=1;figure;plot([-fliplr(tho),tho],real([flipud(f2(:,:,ii).*g2(:,1));f1(:,:,ii).*g1(:,1)]),'-r',[-fliplr(tho),tho],imag([flipud(f2(:,:,ii).*g2(:,1));f1(:,:,ii).*g1(:,1)]),'--b');
%ii=1;figure;plot([-fliplr(tho),tho],real([flipud(f2(:,:,ii).*du2.*g(:,1));f1(:,:,ii).*du1.*g(:,1)]),'-k','linewidth',2);
   if super_resonance;e_oc(f,:)=e_oc(f,:)+(exp(i*nh*rho(f)).*retoc(tho_pl*sqrt(rho(f)))).'*f0;end;
end;  %<<<<<<<<<<<<<<
ec=coupure_90(rho,S,nb,nh,ns,plasmon,pol,y0);e_oc=e_oc+ec;
end;            %  <-----
[e_res,F0,A0]=onde_cylindrique_residu(rho,0,1,S,nb,nh,ns,plasmon,pol,y0);


pas_residu=0;e=e_oc+e_res;
approche=struct('pas_residu',pas_residu,'A0',A0,'F0',F0,'A1',A1,'B1',B1,'F1',F1,'F2',F2,'A2',A2,'B2',B2,'F3',F3);
%***************************************************************************************************************
function e=coupure_90(rho,S,nb,nh,ns,plasmon,pol,y0);
e=zeros(length(rho),3);
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_bis_90(0,nh,nb,ns,S,pol,y0);% pour gain de temps
f=find(real(h1*rho)>-36);if isempty(f);return;end;	
fin=[-10,-1,1,10];
tho_pl=sqrt(i*(nb-plasmon.constante_de_propagation));
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_bis_90(tho_pl.',nh,nb,ns,S,pol,y0);
resonance=max(abs(f1))>1.e10;
super_resonance=resonance & abs(imag(tho_pl))<.01;
	if super_resonance;% on retranche la resonance pour l'integrer à la main
	tho=retgauss(real(tho_pl)-imag(tho_pl),real(tho_pl)+imag(tho_pl),5);
	[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_bis_90(tho.',nh,nb,ns,S,pol,y0);
	f0=zeros(1,3);
	for ii=1:3;
	f0(ii)=calder(tho.'-tho_pl,du1.*(tho.'-tho_pl).*f1(:,:,ii));
	end
	else;f0=zeros(1,3);
	end;
[tho0,tho_max]=choisi_tho(rho(f));
[tho,wtho]=retgauss(0,tho_max,20,10,[tho0,abs(real(tho_pl))+imag(tho_pl)*fin]);
[u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_bis_90(tho.',nh,nb,ns,S,pol,y0);
	if super_resonance;prv1=1./((tho.'-tho_pl).*du1);prv2=1./((-tho.'-tho_pl).*du2);for ii=1:3;f1(:,:,ii)=f1(:,:,ii)-prv1*f0(ii);f2(:,:,ii)=f2(:,:,ii)-prv2*f0(ii);end;end;
g=exp(h1*rho(f));
for ii=1:3;	e(f,ii)=(wtho.*du1.'.*f1(:,:,ii).')*g+(wtho.*du2.'.*f2(:,:,ii).')*g;end;
% ii=1;figure;plot([-fliplr(tho),tho],real([flipud(f2(:,:,ii).*du2.*g(:,1));f1(:,:,ii).*du1.*g(:,1)]),'-k','linewidth',2);
   if super_resonance;e(f,:)=e(f,:)+(exp(i*nb*rho(f)).*retoc(tho_pl*sqrt(rho(f)))).'*f0;end;

%***************************************************************************************************************
function [u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_90(tho,nh,nb,ns,S,pol,y0);
h1=i*nh-tho.^2;h2=h1;
u1=nh+i*tho.^2;u2=u1;
if nargout<3;return;end;
du1=2*i*tho;du2=-du1;
khih1=(i-1)*sqrt(nh)*tho.*sqrt(1+(.5*i/nh)*tho.^2);
khih2=-khih1;
khib1=coupure_droite(u1,nb);
khib2=khib1;
Sh1=calSh(S,u1,khih1,khib1,nh,nb,ns,pol,y0);
Sh2=calSh(S,u2,khih2,khib2,nh,nb,ns,pol,y0);
f1=zeros(size(Sh1)); f2=zeros(size(Sh2));
st_warning=warning;warning off;
for ii=1:3;
f1(:,:,ii)=Sh1(:,:,ii)./(khih1/nh^pol+khib1/nb^pol);
f2(:,:,ii)=Sh2(:,:,ii)./(khih2/nh^pol+khib2/nb^pol);
end;
warning(st_warning);
%***************************************************************************************************************
function [u1,u2,h1,h2,f1,f2,du1,du2]=lowenthal_bis_90(tho,nh,nb,ns,S,pol,y0);
h1=i*nb-tho.^2;h2=h1;
u1=nb+i*tho.^2;u2=u1;
if nargout<3;return;end;
du1=2*i*tho;du2=-du1;
khib1=(i-1)*sqrt(nb)*tho.*sqrt(1+(.5*i/nb)*tho.^2);
khib2=-khib1;
khih1=coupure_droite(u1,nh);
khih2=khih1;
Sh1=calSh(S,u1,khih1,khib1,nh,nb,ns,pol,y0);
Sh2=calSh(S,u2,khih2,khib2,nh,nb,ns,pol,y0);
f1=zeros(size(Sh1)); f2=zeros(size(Sh2));
st_warning=warning;warning off;
for ii=1:3;
f1(:,:,ii)=Sh1(:,:,ii)./(khih1/nh^pol+khib1/nb^pol);
f2(:,:,ii)=Sh2(:,:,ii)./(khih2/nh^pol+khib2/nb^pol);
end;
warning(st_warning);


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%  forme approchée rudimentaire  phi=pi/2  %%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%***************************************************************************************************************
function [e,e_oc,e_res]=retoc_90(x,y,S,nb,nh,ns,pol,k0,parm);
[e,e_oc,e_res]=deal(zeros(length(y),length(x),3));if isempty(e);return;end;
[f,ff]=retfind(x>=0);
[e(:,ff,:),e_oc(:,ff,:),e_res(:,ff,:)]=retoc_90(-x(ff),y,diag((-1).^(0:size(S,1)-1))*S*diag([1,1,-1]),nb,nh,ns,pol,k0,parm);e(:,ff,3)=-e(:,ff,3);e_oc(:,ff,3)=-e_oc(:,ff,3);e_res(:,ff,3)=-e_res(:,ff,3);
[g,gg]=retfind(y>=0);
[e(gg,f,:),e_oc(gg,f,:),e_res(gg,f,:)]=retoc_90(x(f),-y(gg),S*diag([1,-1,1]),nh,nb,ns,pol,k0,parm);e(gg,f,2)=-e(gg,f,2);e_oc(gg,f,2)=-e_oc(gg,f,2);e_res(gg,f,2)=-e_res(gg,f,2);
[e(g,f,:),e_oc(g,f,:),e_res(g,f,:)]=oc_90(x(f)*k0,y(g)*k0,diag(k0.^(1:size(S,1)))*S,nb,nh,ns,pol,k0,parm);
return;
%***************************************************************************************************************
function [e,e_oc,e_res]=oc_90(x,y,S,nb,nh,ns,pol,k0,parm);
pparm=0;if ~isempty(parm);if parm==i;parm=[];pparm=1;end;end;
[e,e_oc,e_res]=deal(zeros(length(y),length(x),3));if isempty(e);return;end;
khib0=coupure_droite(nh,nb);khih0=coupure_droite(nb,nh);
muh=nh^pol;mub=nb^pol;mus=ns^pol;
beta=sqrt(1./(nh.^-2+nb.^-2));khid=coupure_droite(beta,nh);khim=coupure_droite(beta,nb);
test=abs((khim+(nb./nh).^2.*khid)./(2*khim))<1.e-10;
plasmon=struct('constante_de_propagation',beta,'khi_metal',khim,'khi_dielectrique',khid,'test_d_existence',test);
if pol==0;plasmon.test_d_existence=0;end;
if plasmon.test_d_existence

Sh=calSh(S,plasmon.constante_de_propagation,plasmon.khi_dielectrique,plasmon.khi_metal,nh,nb,ns,pol,y);
res=-Sh/(plasmon.constante_de_propagation*(1/(nh^pol*plasmon.khi_dielectrique)+1/(nb^pol*plasmon.khi_metal)));
A0=i*plasmon.constante_de_propagation;
B1=sqrt(-i*(plasmon.constante_de_propagation-nh));
B2=sqrt(i*(nb-plasmon.constante_de_propagation));
else;
A0=nan+i*nan;B1=nan+i*nan;B2=nan+i*nan;
end;
A1=i*nh;A2=i*nb;
rho=x(:).';rho(rho==0)=1/realmax; 

if ~isempty(parm);% <<<<<<<<<<<<<<   version 'ameliorée'
bb1=(exp(rho*A1)./(rho.*sqrt(rho)));
b2=(exp(rho*A2).*retoc(sqrt(rho)*B2,1)./(rho.*sqrt(rho)));
if isfinite(A0);F0=2i*pi*reshape(res,[],3);a0=exp(rho*A0);end;
for ii=1:length(y);
y0=y(ii);
c=.5*i*(i-1)*sqrt(nh);
f_col0=(i*sqrt(pi)/(2*pi))*(mub/khib0)*[1;0;-(i*nh/muh)]*[i,khib0/mub,-nh/mus];
f_col1=i*sqrt(pi)*((i-1)*sqrt(nh)/(2*pi))*(mub/khib0)*[-mub/(muh*khib0);i/muh;-(i*nh/muh)*(-mub/(muh*khib0))]*[i,khib0/mub,-nh/mus];
f_b=i*sqrt(pi)*((i-1)*sqrt(nb)/(2*pi))*(muh/(mub*khih0))*exp(i*khih0*y0)*[1;i*khih0/muh;-i*nb/muh]*[-i*muh/khih0,1,nb*muh/(mus*khih0)];
if size(S,1)>1;S0=(i*nh).^(0:size(S,1)-1)*S;Sb=(i*nb).^(0:size(S,1)-1)*S;else;S0=S;Sb=S;end;
F20=(f_col0*S0.').';
F21=(f_col1*S0.').';
F3=(f_b*Sb.').';
if isfinite(A0);e_res(ii,:,:)=reshape(a0.'*F0(ii,:),1,[],3);end;
b11=bb1.*retoc(sqrt(rho)*B1,c*y0./sqrt(rho),2);
b10=(2*c*y0)*bb1.*exp((c*y0)^2./rho);
e_oc(ii,:,:)=reshape(b10.'*F20,1,[],3)+reshape(b11.'*F21,1,[],3);
e_oc(ii,:,:)=e_oc(ii,:,:)+reshape(b2.'*F3,1,[],3);
end;
else;    % <<<<<<<<<<<<<< version 'brute'
rho=x(:).';rho(rho==0)=1/realmax; 
b1=(exp(rho*A1).*retoc(sqrt(rho)*B1,1)./(rho.*sqrt(rho)));
b2=(exp(rho*A2).*retoc(sqrt(rho)*B2,1)./(rho.*sqrt(rho)));
if isfinite(A0);F0=2i*pi*reshape(res,[],3);a0=exp(rho*A0);end;
for ii=1:length(y);
y0=y(ii);
f_col=i*sqrt(pi)*((i-1)*sqrt(nh)/(2*pi))*(mub/khib0)*[i*y0-mub/(muh*khib0);i/muh;-(i*nh/muh)*(i*y0-mub/(muh*khib0))]*[i,khib0/mub,-nh/mus];
f_b=i*sqrt(pi)*((i-1)*sqrt(nb)/(2*pi))*(muh/(mub*khih0))*exp(i*khih0*y0)*[1;i*khih0/muh;-i*nb/muh]*[-i*muh/khih0,1,nb*muh/(mus*khih0)];
if size(S,1)>1;S0=(i*nh).^(0:size(S,1)-1)*S;Sb=(i*nb).^(0:size(S,1)-1)*S;else;S0=S;Sb=S;end;
F2=(f_col*S0.').';
F3=(f_b*Sb.').';
if isfinite(A0);e_res(ii,:,:)=reshape(a0.'*F0(ii,:),1,[],3);end;
if pparm==1;% seulement la forme où l'indice est reel;
if abs(imag(plasmon.khi_dielectrique))<abs(imag(plasmon.khi_metal));% metal dessous
e_oc(ii,:,:)=reshape(b1.'*F2,1,[],3);	
else
e_oc(ii,:,:)=reshape(b2.'*F3,1,[],3);	
end;
else;	
e_oc(ii,:,:)=reshape(b1.'*F2,1,[],3);
e_oc(ii,:,:)=e_oc(ii,:,:)+reshape(b2.'*F3,1,[],3);
end;
end;
end;     % <<<<<<<<<<<<<<
e=e_oc+e_res;

% declonage
e(:,:,2:3)=-i*e(:,:,2:3);
e_oc(:,:,2:3)=-i*e_oc(:,:,2:3);
e_res(:,:,2:3)=-i*e_res(:,:,2:3);



	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%     Calcul des integrales I0 I1 L2 intervenant dans la forme approchee     %%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%***************************************************************************************************************
function I=cal_L2(x0,a,n);
I=ones(size(x0));
f=retfind(isfinite(x0));
if n==2;I(f)=-2*exp(a(f).^2).*x0(f).*(x0(f)+a(f)+(x0(f).^2/sqrt(pi)).*retoc(x0(f)-a(f)));end;
%***************************************************************************************************************
function I=cal_I0(x0,n);
if nargin<2;n=0;end;
I=zeros(size(x0));
[fff,ffff]=retfind(isfinite(x0));
[f,ff]=retfind(real(-(x0.^2))<4,fff);
I(f)=cal_I1(x0(f));
I(ff)=cal_I2(x0(ff));
if n==1;I(fff)=-2*(x0(fff).^2).*(1+x0(fff).*I(fff)/sqrt(pi));I(ffff)=1;end;% I1 normalise 1 à l'infini
%***************************************************************************************************************
function I=cal_I2(x0);% calcul de l'integrale
if isempty(x0);I=x0;return;end;
[x,wx]=retgauss(-10,10,20,10);
[X0,X]=ndgrid(x0(:),x);
I=sum((1./(X-X0))*retdiag(wx.*exp(-x.^2)),2);
%***************************************************************************************************************
function I=cal_I1(x0);% forme analytique (fonction de fresnel)
if isempty(x0);I=x0;return;end;
X=x0*sqrt(2/pi)*exp(-.25i*pi);
[y,yy,yyy]=retfresnel(X);
[f,ff]=retfind(abs(x0.^2)<2);
I(f)=exp(-x0(f).^2).*(-pi*sqrt(2)*exp(.25i*pi).*y(f)+i*pi*sign(imag(x0(f))));
I(ff)=exp(-x0(ff).^2).*(-pi*sqrt(2)*exp(.25i*pi)*yyy(ff)+i*pi*sign(imag(x0(ff))))+exp(.25i*pi)*i*sqrt(2)*(1+.5*pi*yy(ff))./X(ff);
%***************************************************************************************************************
function [Ep,Em,angles]=calop(n,S,u,xs,pol,k0,sens);% developpement en ondes planes
Ep=zeros(length(u),3);
if sens<0;n([2,1])=n([1,2]);end;

u=u(:)/k0;xs=k0*xs;
khih=retsqrt(n(1)^2-u.^2,-1);
khib=retsqrt(n(2)^2-u.^2,-1);

for ii=1:length(xs);
if size(S{ii},2)==3;S{ii}(:,1)=i*S{ii}(:,1);else;S{ii}=S{ii}(:,1:3);end;% clonage de la source si moins de 4 elements
S{ii}=diag(k0.^(1:size(S{ii},1)))*S{ii};
if sens<0;S{ii}=S{ii}*diag([1,-1,1]);end;
Sh=calSh(S{ii},u,khih,khib,n(1),n(2),n(3),pol,0);Sh=reshape(Sh,size(Ep));	
Ep=Ep+retdiag(exp(-i*xs(ii)*u)./(khih/n(1)^pol+khib/n(2)^pol))*Sh;	
end;	
Ep=Ep/k0;
Ep(:,2:3)=-i*Ep(:,2:3);% declonage
if sens<0;Em=Ep;Em(:,2)=-Em(:,2);Ep=zeros(length(u),3);else;Em=zeros(length(u),3);end;
%   mise en forme des ondes planes 
if nargout>2;
khi=retsqrt(n(1)^2-u.^2,-1);
ny=1;Ep=reshape(Ep,1,[],3);Em=reshape(Em,1,[],3);
km=[u(:),-khi(:)];km=km./repmat(sqrt(sum((km.^2),2)),1,size(km,2));% vecteurs d'onde normalises
um=[km(:,2),-km(:,1)];% um= -ez vect.  km
HHm=repmat(um(:,1).',ny,1).*Em(:,:,2)+repmat(um(:,2).',ny,1).*Em(:,:,3);
kp=km;kp(:,2)=-kp(:,2);
up=um;up(:,1)=-up(:,1);
HHp=repmat(up(:,1).',ny,1).*Ep(:,:,2)+repmat(up(:,2).',ny,1).*Ep(:,:,3);
teta=atan2(real(kp(:,1)),real(kp(:,2)));
EEp=Ep(:,:,1).';EEm=Em(:,:,1).';% on met y en dernier
HHp=HHp.';HHm=HHm.';
Fp=pi*k0*n(1)*retdiag(cos(teta).^2)*real(EEp.*conj(HHp));Fm=pi*k0*n(1)*retdiag(cos(teta).^2)*real(EEm.*conj(HHm));
Ep=reshape(Ep,[],3);Em=reshape(Em,[],3);
end;
angles=struct('teta',teta,'kp',kp,'up',up,'EEp',EEp,'HHp',HHp,'Fp',full(Fp),'km',km,'um',um,'EEm',EEm,'HHm',HHm,'Fm',full(Fm));
% 
%***************************************************************************************************************
function [plg,pld,Plg,Pld]=calpl(n,S,x,XS,k0,sens);% amplitude des plasmons aux points x
if sens<0;n([2,1])=n([2,1]);end;
n_der=size(S{1},1);

plasmon=retplasmon(n(2),n(1),2*pi/k0);
prv=(n(1)*n(2))^3/(n(1)^4-n(2)^4);
Ef_pld=prv*[1,-(i*plasmon.khi_metal/n(2)^2),(i*plasmon.constante_de_propagation/n(3)^2)];
Ef_pld=((i*plasmon.constante_de_propagation).^(0:n_der-1)).'*Ef_pld;

Ef_plg=prv*[1,-(i*plasmon.khi_metal/n(2)^2),-(i*plasmon.constante_de_propagation/n(3)^2)];
Ef_plg=((i*plasmon.constante_de_propagation).^(0:n_der-1)).'*Ef_plg;

plg=zeros(size(x));pld=zeros(size(x));
for ii=1:length(S);[fg,fd]=retfind(x<XS(ii));
if size(S{ii},2)==3;S{ii}(:,1)=i*S{ii}(:,1);else;S{ii}=S{ii}(:,1:3);end;% clonage de la source si moins de 4 elements
S{ii}=diag(k0.^(1:n_der))*S{ii};
if sens<0;S{ii}=S{ii}*diag([1,-1,1]);end;

for jj=1:n_der;
plg(fg)=plg(fg)+sum(Ef_plg(jj,:).*S{ii}(jj,:),2)*exp(i*k0*plasmon.constante_de_propagation*XS(ii));	
pld(fd)=pld(fd)+sum(Ef_pld(jj,:).*S{ii}(jj,:),2)*exp(-i*k0*plasmon.constante_de_propagation*XS(ii));	
end;
end;% ii

Pld=plasmon.poynting*abs(pld).^2;Plg=plasmon.poynting*abs(plg).^2;





%
% 	¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
% 	¤  VERSION 3D ( reseau 2D )  ¤
% 	¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
% 
function  [e,e_oc,e_res]=retoc3D(n,S,P,x,y,z,k0,parm);if nargin<8;parm=[];end;

defaultopt=struct('xyzmesh',1,'z0_varie',0,'test',0,'calder3D',0);
parm.xyzmesh=retoptimget(parm,'xyzmesh',defaultopt,'fast');
if (length(x)~=length(y))|(length(x)~=length(z)) parm.xyzmesh=0;end;

parm.z0_varie=retoptimget(parm,'z0_varie',defaultopt,'fast');
parm.test=retoptimget(parm,'test',defaultopt,'fast');
parm.calder3D=size(S,1)>1;
if size(P,2)==3;parm.Dim=3;end;% dioptre perpendiculaire à Oz
if size(P,1)==3;parm.Dim=1;end;% dioptre perpendiculaire à Ox
if size(P,3)==3;parm.Dim=2;end;% dioptre perpendiculaire à Oy

% clonage et mise à l'échelle
if size(S,2)==6;S(:,1:3,:)=i*S(:,1:3,:);else;S=S(:,1:6,:);end;% clonage de la source si moins de 4 elements
S(1,:,:)=k0^2*S(1,:,:);S(2:end,:,:)=k0^3*S(2:end,:,:);
switch parm.Dim;
case 1;S=S(:,[2,3,1,5,6,4],:);P=P([2,3,1]);[x,y,z]=deal(y,z,x);% dioptre perpendiculaire à Ox
case 2;S=S(:,[3,1,2,6,4,5],:);P=P([3,1,2]);[x,y,z]=deal(z,x,y);% dioptre perpendiculaire à Oy
end;

% S=reshape(permute(S,[3,2,1]),[],size(S,3));% Modif 14 5 2010
S=reshape(permute(S,[2,1,3]),[],size(S,3));
 % S, reshape(permute(S,[2,1,3]),[],size(S,3)), stop

P=P*k0;x=x*k0;y=y*k0;z=z*k0; 
[nh,nb,ns]=deal(n(1),n(2),n(3));
if parm.xyzmesh;[Z,X,Y]=deal(z,x,y);sz=[length(z),6];
else;[Z,X,Y]=ndgrid(z,x,y);sz=[length(z),length(x),length(y),6];end;
clear x y z;
X=X-P(1);Y=Y-P(2);Z=Z-P(3);
R=sqrt(X.^2+Y.^2);R=R(:);Z=Z(:);f=find(R==0);R(f)=eps;

cT=X(:)./R(:);sT=Y(:)./R(:);cT(f)=1;sT(f)=0;

[E,E_oc,E_res]=onde_cylindrique3D(R,Z,nb,nh,ns,parm);
[e,e_oc,e_res]=deal(cell(1,size(S,2)));
for ii=1:length(e);
e{ii}=cylindrique_2_cartesien(S(:,ii),E,cT,sT,sz,parm);
if nargout>1;e_oc{ii}=cylindrique_2_cartesien(S(:,ii),E_oc,cT,sT,sz,parm);end;
if nargout>2;e_res{ii}=cylindrique_2_cartesien(S(:,ii),E_res,cT,sT,sz,parm);end;
end;
if length(e)==1;e=e{1};e_oc=e_oc{1};e_res=e_res{1};end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=cylindrique_2_cartesien(S,E,cT,sT,sz,parm);% passage en coordonnees cartesiennes et déclonage
if parm.calder3D==1;dim=12;else;dim=4;end;
E=reshape(E,[],6,dim);
e=-i*S(3)*E(:,:,1)-i*S(6)*E(:,:,2);
if S(1)~=0;e=e-S(1)*pair_impair(E(:,:,3),cT,sT,1);end;
if S(2)~=0;e=e+i*S(2)*pair_impair(E(:,:,3),cT,sT,-1);end;
if S(5)~=0;e=e+i*S(5)*pair_impair(E(:,:,4),cT,sT,1);end;
if S(4)~=0;e=e-S(4)*pair_impair(E(:,:,4),cT,sT,-1);end;

if parm.calder3D==1;
c2T=2*cT.^2-1;s2T=2*sT.*cT;

e=e+(S(7)+S(14))*E(:,:,5)+i*(S(8)-S(13))*E(:,:,6)+(S(10)+S(17))*E(:,:,7)+i*(S(11)-S(16))*E(:,:,8);	
if S(9)~=0;e=e+i*S(9)*pair_impair(E(:,:,9),cT,sT,1);end;
if S(15)~=0;e=e+S(15)*pair_impair(E(:,:,9),cT,sT,-1);end;
if (S(8)+S(13))~=0;e=e-(i*(S(8)+S(13)))*pair_impair(E(:,:,11),c2T,s2T,-1);end; 
if (S(7)-S(14))~=0;e=e+(S(7)-S(14))*pair_impair(E(:,:,11),c2T,s2T,1);end;

if S(12)~=0;e=e+i*S(12)*pair_impair(E(:,:,10),cT,sT,-1);end;
if S(18)~=0;e=e+S(18)*pair_impair(E(:,:,10),cT,sT,1);end;
if (S(11)+S(16))~=0;e=e-(i*(S(11)+S(16)))*pair_impair(E(:,:,12),c2T,s2T,1);end; 
if (S(10)-S(17))~=0;e=e+(S(10)-S(17))*pair_impair(E(:,:,12),c2T,s2T,-1);end;
end;	
% passage en Ex Ey, Hx Hy
[e(:,1),e(:,2)]=deal(e(:,1).*cT-e(:,2).*sT,e(:,1).*sT+e(:,2).*cT);
[e(:,4),e(:,5)]=deal(e(:,4).*cT-e(:,5).*sT,e(:,4).*sT+e(:,5).*cT);


e(:,4:6)=-i*e(:,4:6);% declonage
e=reshape(e,sz);

% eventuel changement de repere

if parm.xyzmesh;
switch parm.Dim;
case 1;e=e(:,[3,1,2,6,4,5]);% dioptre perpendiculaire à Ox
case 2;e=e(:,[2,3,1,5,6,4]);% dioptre perpendiculaire à Oy
end;

else;
switch parm.Dim;
case 1;e=permute(e(:,:,:,[3,1,2,6,4,5]),[3,1,2,4]);% dioptre perpendiculaire à Ox
case 2;e=permute(e(:,:,:,[2,3,1,5,6,4]),[2,3,1,4]);% dioptre perpendiculaire à Oy
end;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=pair_impair(e,cT,sT,sens);
if sens==1; % pair
e(:,1)=e(:,1).*cT;
e(:,2)=e(:,2).*i.*sT;
e(:,3)=e(:,3).*cT;
e(:,4)=e(:,4).*i.*sT;
e(:,5)=e(:,5).*cT;
e(:,6)=e(:,6).*i.*sT;
else;       % impair
e(:,1)=e(:,1).*i.*sT;
e(:,2)=e(:,2).*cT;
e(:,3)=e(:,3).*i.*sT;
e(:,4)=e(:,4).*cT;
e(:,5)=e(:,5).*i.*sT;
e(:,6)=e(:,6).*cT;
end;	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [e,e_oc,e_res]=onde_cylindrique3D(r,z,nb,nh,ns,parm);if nargin<6;parm=[];end;
defaultopt=struct('z0_varie',0,'test',0);
parm.z0_varie=retoptimget(parm,'z0_varie',defaultopt,'fast');
parm.test=retoptimget(parm,'test',defaultopt,'fast');
if parm.calder3D;[e,e_oc,e_res]=deal(zeros(length(r),6,12));else;[e,e_oc,e_res]=deal(zeros(length(r),6,4));end;
if isempty(r);return;end;

[f,ff]=retfind(z>=0);  % tri z>= 0  z<0
if ~isempty(ff);
[e(ff,:,:),e_oc(ff,:,:),e_res(ff,:,:)]=onde_cylindrique3D(r(ff),-z(ff),nh,nb,ns,parm);

if parm.calder3D;ind=[1,4,7,8,9,12];else;ind=[1,4];end;

e(ff,3:5,:)=-e(ff,3:5,:);e(ff,:,ind)=-e(ff,:,ind);
e_oc(ff,3:5,:)=-e_oc(ff,3:5,:);e_oc(ff,:,ind)=-e_oc(ff,:,ind);
e_res(ff,3:5,:)=-e_res(ff,3:5,:);e_res(ff,:,ind)=-e_res(ff,:,ind);

end
if ~isempty(f);
[e(f,:,:),e_oc(f,:,:),e_res(f,:,:)]=onde_cylindrique3D_haut(r(f),z(f),nb,nh,ns,parm);
end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
function [e,e_oc,e_res]=onde_cylindrique3D_haut(r,z,nb,nh,ns,parm);
if parm.calder3D;[e,e_oc,e_res]=deal(zeros(length(r),6,12));else;[e,e_oc,e_res]=deal(zeros(length(r),6,4));end;
%[f,ff]=retfind( r>3 |(r./(z+eps))>1 );%length(ff)
r=max(r,100*realmin);
[f,ff]=retfind(r>.3);% calcul direct pour r petit
if ~isempty(f);[e(f,:,:),e_oc(f,:,:),e_res(f,:,:)]=onde_cylindrique3D_haut1(r(f),z(f),nb,nh,ns,parm);end;
if ~isempty(ff);pparm=parm;pparm.z0_varie=0;pparm.test=3;[e(ff,:,:),e_oc(ff,:,:),e_res(ff,:,:)]=onde_cylindrique3D_haut1(r(ff),z(ff),nb,nh,ns,pparm);end;
%if ~isempty(ff);pparm=parm;pparm.test=3;[e(ff,:,:),e_oc(ff,:,:),e_res(ff,:,:)]=retoc3D_haut(r(ff),pi/2,nb,nh,ns,z(ff),pparm);end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
function [e,e_oc,e_res]=onde_cylindrique3D_haut1(r,z,nb,nh,ns,parm);
if parm.calder3D;[e,e_oc,e_res]=deal(zeros(length(r),6,12));else;[e,e_oc,e_res]=deal(zeros(length(r),6,4));end;

if parm.z0_varie;% *************
[f,ff]=retfind(z<(pi/abs(real(sqrt(-2i*nh))))*sqrt(r));
if ~isempty(f);[e(f,:,:),e_oc(f,:,:),e_res(f,:,:)]=retoc3D_haut(r(f),pi/2,nb,nh,ns,z(f),parm);end;
if ~isempty(ff);pparm=parm;pparm.z0_varie=0;[e(ff,:,:),e_oc(ff,:,:),e_res(ff,:,:)]=onde_cylindrique3D_haut(r(ff),z(ff),nb,nh,ns,pparm);end;

else;      % parm.z0_varie = 0 *************
phi=atan2(r,z);rho=sqrt(r.^2+z.^2);
[phi,k,kk]=retelimine(phi); % tri des phi egaux (gain de temps)
for ii=1:length(k);
jj=find(kk==ii);
if ~isempty(jj);[e(jj,:,:),e_oc(jj,:,:),e_res(jj,:,:)]=retoc3D_haut(rho(jj),phi(ii),nb,nh,ns,zeros(size(jj)),parm);end;
end;

end;       % z0_varie ? *************

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
function [e,e_oc,e_res]=retoc3D_haut(rho,phi,nb,nh,ns,z0,parm);
persistent lorentz;if isempty(lorentz);lorentz=retlorentz(i,51,10);end;% gain de temps
if parm.calder3D;[e,e_oc,e_res]=deal(zeros(length(rho),6,12));else;[e,e_oc,e_res]=deal(zeros(length(rho),6,4));end;
rho=rho(:).';rho(rho==0)=1/realmax;z0=z0(:).';

if parm.test==3;% calcul direct pour r petit avec J 
cphi=cos(phi);sphi=sin(phi);
plasmon=retplasmon(nb,nh,2*pi);%if ~plasmon.test_d_existence;plasmon.constante_de_propagation=nan;end;
%fin=[-10,-1,1,10];

% recherche des points irreguliers
%A=2*max(real(nh),real(nb));B=30*max(real(nh),real(nb));
for kk=1:length(rho);% on travaille point par point
A=2*max(real(nh),real(nb));B=min(100,max(30/rho(kk)*(sphi+cphi),5))*max(real(nh),real(nb));%min(1000,max(30/min(rho(kk)*(sphi+cphi)),5))
u=linspace(0,A,1000);tol=.2;
if plasmon.test_d_existence;u=u(abs(u-nh)<tol | abs(u-nb)<tol | abs(u-plasmon.constante_de_propagation)<tol);
else;u=u(abs(u-nh)<tol | abs(u-nb)<tol); end;
if ~isempty(u);
c0=(max(u)+min(u))/2;r0=(max(u)-min(u))/2;
[u1,wu1]=retgauss(0,c0-r0,10,5);
[u2,wu2]=retgauss(-pi,0,10,5);u2=r0*exp(i*u2);wu2=i*wu2.*u2;u2=u2+c0;
[u3,wu3]=retgauss(c0+r0,A,10,5);
[u4,wu4]=retgauss(A,B,10,5);
u=[u1,u2,u3,u4];wu=[wu1,wu2,wu3,wu4];
else;
[u,wu]=retgauss(0,B,10,5);
end;	
% figure;plot(u);return
khib=retsqrt(nb^2-u.^2,-1);khih=retsqrt(nh^2-u.^2,-1);
Seh=calSeh(u.',khih.',khib.',nh,nb,ns,parm,1);

J=cal_bessel(0,u.',rho(kk),sphi,exp(i*khih.'*(rho(kk).*cphi+z0(kk))),parm);
e(kk,:,:)=2*cale(wu,J,u,Seh,nh);
% complement avec Hanckel
if sphi>100*eps
[t1,wt1]=retgauss(0,1,10,15);[t2,wt2]=retgauss(1,20,10,15);t=[t1,t2];wt=[wt1,wt2];
%[t1,wt1]=retgauss(0,1,10,15);[t2,wt2]=retgauss(1,20,10,15);t=[t1,t2];wt=[wt1,wt2];
pml=1+.9i;%min(1000,2/(rho(kk)*(sphi+cphi)*imag(pml)))
pml=(2/(rho(kk)*(sphi+cphi)*imag(pml)))*pml;
wu=pml*wt;u=B+pml*t;khib=retsqrt(nb^2-u.^2,-1);khih=retsqrt(nh^2-u.^2,-1);
Seh=calSeh(u.',khih.',khib.',nh,nb,ns,parm,1);
H=cal_bessel(1,u.',rho(kk),sphi,exp(i*khih.'*(rho(kk).*cphi+z0(kk))+i*u.'*rho(kk).*sphi),parm);
e1=cale(wu,H,u,Seh,nh);

pml=conj(pml);
wu=pml*wt;u=-B-pml*t;khib=retsqrt(nb^2-u.^2,-1);khih=retsqrt(nh^2-u.^2,-1);
Seh=calSeh(u.',khih.',khib.',nh,nb,ns,parm,1);
H=cal_bessel(1,u.',rho(kk),sphi,exp(i*khih.'*(rho(kk).*cphi+z0(kk))+i*u.'*rho(kk).*sphi),parm);
e2=cale(wu,H,u,Seh,nh);
e(kk,:,:)=e(kk,:,:)+e1+e2;%[max(abs(e2(:))),max(abs(e1(:)))],
end;
end;% kk


if plasmon.test_d_existence;e_res=onde_cylindrique_3D_residu(rho,cphi,sphi,nb,nh,ns,plasmon,z0,parm);end;
e_oc=e-e_res;
return;
end;

    if parm.test==1;% calcul direct brutal avec J 
	cphi=cos(phi);sphi=sin(phi);
	plasmon=retplasmon(nb,nh,2*pi);%if ~plasmon.test_d_existence;plasmon.constante_de_propagation=nan;end;
	fin=[-10,-1,1,10];
	A=5*max(real(nh),real(nb));B=5*30*max(real(nh),real(nb));
	[u2,wu2]=retgauss(0,A,20,1+floor(max((rho.*sphi))*A/(2*pi)),[real(nh)+.01*fin,real(nb)+.01*fin,real(plasmon.constante_de_propagation)+imag(plasmon.constante_de_propagation)*fin]);
	[u3,wu3]=retgauss(A,B,20,1+floor(max((rho.*sphi))*(B-A)/(2*pi)));
	u=[u2,u3];wu=[wu2,wu3];khib=retsqrt(nb^2-u.^2,-1);khih=retsqrt(nh^2-u.^2,-1);
	J=reshape(retbessel('j',[0,1,1],retcolonne(u.'*(rho.*sphi))),length(u),length(rho),3);

	Seh=calSeh(u.',khih.',khib.',nh,nb,ns,parm,1);
		
	prv=exp(i*khih.'*(rho.*cphi+z0));
    for ii=1:2;J(:,:,ii)=J(:,:,ii).*prv;end;J(:,:,3)=J(:,:,2)*retdiag(1./(rho*sphi));
	e=2*cale(wu,J,u,Seh,nh);
	if plasmon.test_d_existence;e_res=onde_cylindrique_3D_residu(rho,cphi,sphi,nb,nh,ns,plasmon,z0,parm);end;
	e_oc=e-e_res;
	return;
	end;
	

    if parm.test==2;% calcul direct brutal pour tests avec H
	cphi=cos(phi);sphi=sin(phi);
	plasmon=retplasmon(nb,nh,2*pi);%if ~plasmon.test_d_existence;plasmon.constante_de_propagation=nan;end;
	fin=[-10,-1,1,10];
	A=5*max(real(nh),real(nb));B=10*30*max(real(nh),real(nb));
	[u1,wu1]=retgauss(-B,-A,5,1000);
	[u2,wu2]=retgauss(-A,A,20,1500,[real(nh)+.01*fin,-real(nh)+.01*fin,real(nb)+.01*fin,-real(nb)+.01*fin,real(plasmon.constante_de_propagation)+imag(plasmon.constante_de_propagation)*fin,-real(plasmon.constante_de_propagation)+imag(plasmon.constante_de_propagation)*fin]);
	[u3,wu3]=retgauss(A,B,5,1000);
	u=[u1,u2,u3];wu=[wu1,wu2,wu3];khib=retsqrt(nb^2-u.^2,-1);khih=retsqrt(nh^2-u.^2,-1);
% 	H=reshape(retbessel('h',[0,1,1],1,retcolonne(u.'*(rho.*sphi)),1),length(u),length(rho),3);
% 	prv=exp(i*khih.'*(rho.*cphi+z0)+i*u.'*rho.*sphi); 
%     for ii=1:2;H(:,:,ii)=H(:,:,ii).*prv;end;H(:,:,3)=H(:,:,2)*retdiag(1./(rho.*sphi));

    H=cal_bessel(1,u.',rho,sphi,exp(i*khih.'*(rho.*cphi+z0)+i*u.'*rho.*sphi),parm);	
	
	Seh=calSeh(u.',khih.',khib.',nh,nb,ns,parm,1);
	e=cale(wu,H,u,Seh,nh);
	e_res=onde_cylindrique_3D_residu(rho,cphi,sphi,nb,nh,ns,plasmon,z0,parm);
	e_oc=e-e_res;
	return;
	end;

if abs(phi-pi/2)<100*eps;[e,e_oc,e_res]=onde_cylindrique3D_90(rho,nb,nh,ns,z0,parm);return;end;% forme simplifiée
cphi=cos(phi);sphi=sin(phi);


beta=sqrt(1./(nh.^-2+nb.^-2));
khid=coupure_droite(beta,nh);khim=coupure_droite(beta,nb);
khihb=coupure_droite(nb,nh);
test=abs((khim+(nb./nh).^2.*khid)./(2*khim))<1.e-10;
plasmon=struct('constante_de_propagation',beta,'khi_metal',khim,'khi_dielectrique',khid,'test_d_existence',test);
	
fin=[-10,-1,1,10];
[tho_pl0,tho_nb0]=cal_tho_pl_tho_nb(1,0,plasmon,nh,nb);
[tho_pl,tho_nb]=cal_tho_pl_tho_nb(cphi,sphi,plasmon,nh,nb);

if ~plasmon.test_d_existence;plasmon.constante_de_propagation=nan;tho_pl=nan+i*nan;end;
%test_nb=retinterp(x,y,real(nb))>imag(nb); % il faut ajouter la coupure3D ( garder le retinterp pour l'extrapolation )
test_nb=imag(tho_nb0)*imag(tho_nb)<0;test_pl=imag(tho_pl0)*imag(tho_pl)>=0;
[u1,u2,h1,h2]=lowenthal3D(0,cphi,sphi,nh,nb,ns,parm);% calcul de h1 pour gain de temps

f=find(real(h1*rho)>-36);
if ~isempty(f);%<<<<<<<<<<<<<<
[tho0,tho_max]=choisi_tho(rho(f));

[tho,wtho]=retgauss(0,tho_max,20,10,[tho0,abs(real(tho_nb))+imag(tho_nb)*fin,abs(real(tho_pl))+imag(tho_pl)*fin]);
[u1,u2,h1,h2,Seh1,Seh2,du1,du2,khih1,khih2]=lowenthal3D(tho.',cphi,sphi,nh,nb,ns,parm);
 %figure;hold on;plot(u1,'.r');plot(u2,'.b');plot(real(nb),imag(nb),'ok');plot(plasmon.constante_de_propagation,'*c');% test

g=zeros(length(h1),length(f));prv=h1*rho(f);% gain de temps
ff=find(  real(prv)> (max(real(prv(:))) -36));
g(ff)=exp(prv(ff));

% H1=reshape(retbessel('h',[0,1,1],1,retcolonne(u1*(rho(f)*sphi)),1),length(u1),length(f),3);
% H2=reshape(retbessel('h',[0,1,1],1,retcolonne(u2*(rho(f)*sphi)),1),length(u2),length(f),3);
% for ii=1:3;H1(:,:,ii)=H1(:,:,ii).*g.*exp(i*khih1*z0(f));H2(:,:,ii)=H2(:,:,ii).*g.*exp(i*khih2*z0(f));end;
% prv=retdiag(1./(rho(f)*sphi));H1(:,:,3)=H1(:,:,2)*prv;H2(:,:,3)=H2(:,:,2)*prv;

H1=cal_bessel(1,u1,rho(f),sphi,g.*exp(i*khih1*z0(f)),parm);H2=cal_bessel(1,u2,rho(f),sphi,g.*exp(i*khih2*z0(f)),parm);

e_oc(f,:,:)=calee(wtho,H1,H2,u1.',u2.',Seh1,Seh2,du1.',du2.',nh);
end;         %<<<<<<<<<<<<<<


e_res=onde_cylindrique_3D_residu(rho,cphi,sphi,nb,nh,ns,plasmon,z0,parm);
if test_nb;% on ajoute la coupure3D du bas
ec=coupure3D(cphi,sphi,rho,nb,nh,ns,plasmon,z0,parm);e_oc=e_oc+ec;
end;
if test_pl;e=e_oc;e_oc=e-e_res;else;e=e_oc+e_res;end;
% [test_nb,test_pl]
%***************************************************************************************************************
function e_res=onde_cylindrique_3D_residu(rho,cphi,sphi,nb,nh,ns,plasmon,z0,parm);% residu
if parm.calder3D;e_res=zeros(length(rho),6,12);else;e_res=zeros(length(rho),6,4);end;
if ~plasmon.test_d_existence;return;end;
% e_res(:,:,1): phi_0e , e_res(:,:,2): phi_0h , e_res(:,:,3): phi_1e , e_res(:,:,4): phi_1h
res=-2i*pi/(plasmon.constante_de_propagation*(1/(nh^2*plasmon.khi_dielectrique)+1/(nb^2*plasmon.khi_metal)));
prv=res*exp(i*(plasmon.khi_dielectrique*(rho.*cphi+z0)+plasmon.constante_de_propagation*rho*sphi));
B=cal_bessel(1,plasmon.constante_de_propagation,rho,sphi,prv,parm);
% w=1;wu=plasmon.constante_de_propagation;wuu=wu.*plasmon.constante_de_propagation;
Seh=calSeh(plasmon.constante_de_propagation,plasmon.khi_dielectrique,plasmon.khi_metal,nh,nb,ns,parm);

for ii=1:length(Seh);Seh{ii}(:,[2,3])=0;end;
e_res=cale(1,B,plasmon.constante_de_propagation,Seh,nh);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function e=coupure3D(cphi,sphi,rho,nb,nh,ns,plasmon,z0,parm);% coupure3D nb
if parm.calder3D;[e,e_oc,e_res]=deal(zeros(length(rho),6,12));else;[e,e_oc,e_res]=deal(zeros(length(rho),6,4));end;
fin=[-10,-1,1,10];

tho_pl=sqrt(i*((nb-plasmon.constante_de_propagation)*sphi+(coupure_droite(nb,nh)-plasmon.khi_dielectrique)*cphi));
[u1,u2,h1]=lowenthal3D_bis(tho_pl.',cphi,sphi,nh,nb,ns);

f=find(real(h1*rho)>-36);if isempty(f);return;end;	
[tho0,tho_max]=choisi_tho(rho(f));
% 
[tho,wtho]=retgauss(0,tho_max,20,10,[tho0,abs(real(tho_pl))+imag(tho_pl)*fin]);
[u1,u2,h1,h2,Seh1,Seh2,du1,du2,khih1,khih2]=lowenthal3D_bis(tho.',cphi,sphi,nh,nb,ns,parm);


g=zeros(length(h1),length(f));prv=h1*rho(f);% gain de temps
ff=find(  real(prv)> (max(real(prv(:))) -36));
g(ff)=exp(prv(ff));

H1=cal_bessel(1,u1,rho(f),sphi,g.*exp(i*khih1*z0(f)),parm);H2=cal_bessel(1,u2,rho(f),sphi,g.*exp(i*khih2*z0(f)),parm);
e(f,:,:)=calee(wtho,H1,H2,u1.',u2.',Seh1,Seh2,du1.',du2.',nh);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [u1,u2,h1,h2,Seh1,Seh2,du1,du2,khih1,khih2]=lowenthal3D(tho,cphi,sphi,nh,nb,ns,parm);
h1=i*nh-tho.^2;h2=h1;
v=retsqrt(tho.^2-2*i*nh,-1,pi);retsqrt(0,0,pi/2);
u1=sphi*(nh+i*tho.^2)+tho.*cphi.*v;
u2=sphi*(nh+i*tho.^2)-tho.*cphi.*v;
if nargout<3;return;end;
du1=2*(i*tho*sphi+cphi*(tho.^2-i*nh)./v);
du2=2*(-i*tho*sphi+cphi*(tho.^2-i*nh)./v);
%if length(tho)>10;figure;plot(real(u1),imag(u1),'-r',real(u2),imag(u2),'-b');grid;end;

khih1=retsqrt(nh^2-u1.^2,-1,0);
khih2=retsqrt(nh^2-u2.^2,-1,pi);
retsqrt(0,0,pi/2);

u0=nh*sphi;khib0=retsqrt(nb^2-u0^2,-1);% la coupure3D de Maystre suffit ?
khib1=Sqrt(nb^2-[u0;u1].^2,pi/2);if abs(khib1(1)-khib0)>abs(khib1(1)+khib0);khib1=-khib1;end;khib1=khib1(2:end);
khib2=Sqrt(nb^2-[u0;u2].^2,pi/2);if abs(khib2(1)-khib0)>abs(khib2(1)+khib0);khib2=-khib2;end;khib2=khib2(2:end);
Seh1=calSeh(u1,khih1,khib1,nh,nb,ns,parm,1);
Seh2=calSeh(u2,khih2,khib2,nh,nb,ns,parm,1);

%**************************************************************************************************************
function [u1,u2,h1,h2,Seh1,Seh2,du1,du2,khih1,khih2]=lowenthal3D_bis(tho,cphi,sphi,nh,nb,ns,parm);
khib=coupure_droite(nb,nh);
a1=nb*sphi+khib*cphi;
a2=nb*cphi-khib*sphi;
h1=i*a1-tho.^2;h2=h1;
v=Sqrt(1+tho.^2.*(tho.^2-2i*a1)/a2^2,pi);
u1=sphi.*(a1+i*tho.^2)+(a2*cphi)*v;u2=u1;
if nargout<5;return;end;
khih1=coupure_droite(u1,nh);
khih2=khih1;

du1=2*tho.*(i*sphi+(cphi/a2)*(tho.^2-i*a1)./v);du2=-du1;

prv=(.5/a2)*(1-(1/a2^2)*((tho.^2-2i*a1)./(v+1)).^2);
z=-2*nb*cphi*prv-(-i*khib/a2+cphi*tho.^2.*prv).^2;
aa=2i*nb*khib/a2;
khib1=retsqrt(aa,0)*tho.*Sqrt(1+(tho.^2.*z)/aa,pi);

khib2=-khib1;
Seh1=calSeh(u1,khih1,khib1,nh,nb,ns,parm,1);
Seh2=calSeh(u2,khih2,khib2,nh,nb,ns,parm,1);





    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% %  TRAITEMENT SPECIAL DU CAS phi=pi/2 % %
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%***************************************************************************************************************
function [e,e_oc,e_res]=onde_cylindrique3D_90(rho,nb,nh,ns,z0,parm);
% persistent lorentz;if isempty(lorentz);lorentz=retlorentz(i,51,10);end;% gain de temps
if parm.calder3D;[e,e_oc]=deal(zeros(length(rho),6,12));else;[e,e_oc]=deal(zeros(length(rho),6,4));end;

fin=[-10,-1,1,10];
beta=sqrt(1./(nh.^-2+nb.^-2));khid=coupure_droite(beta,nh);khim=coupure_droite(beta,nb);
test=abs((khim+(nb./nh).^2.*khid)./(2*khim))<1.e-10;
plasmon=struct('constante_de_propagation',beta,'khi_metal',khim,'khi_dielectrique',khid,'test_d_existence',test);
[u1,u2,h1]=lowenthal3D_90(0,nh,nb,ns);

tho_pl=sqrt(-i*(plasmon.constante_de_propagation-nh));
tho_nb=sqrt(-i*(nb-nh));
% tho=[real(tho_pl)+lorentz*imag(tho_pl),real(tho_nb)+lorentz*imag(tho_nb)];
% f=find(real(h1*rho)>-36);
f=1:length(rho);

% if ~isempty(f);%<<<<<<<<<<<<<<
[tho0,tho_max]=choisi_tho(rho(f));
[tho,wtho]=retgauss(0,2*tho_max,40,10,[tho0,abs(real(tho_nb))+imag(tho_nb)*fin,abs(real(tho_pl))+imag(tho_pl)*fin]);
[u1,u2,h1,h2,Seh1,Seh2,du1,du2,khih1,khih2]=lowenthal3D_90(tho.',nh,nb,ns,parm);

g=zeros(length(h1),length(f));prv=h1*rho(f);% gain de temps
ff=find(  real(prv)> (max(real(prv(:))) -36));
g(ff)=exp(prv(ff));

% B1=reshape(retbessel('h',[0,1,1],1,retcolonne(u1*rho(f)),1),length(u1),length(f),3);
% B2=reshape(retbessel('h',[0,1,1],1,retcolonne(u2*rho(f)),1),length(u2),length(f),3);
% for ii=1:2;B1(:,:,ii)=B1(:,:,ii).*g.*exp(i*khih1*z0(f));B2(:,:,ii)=B2(:,:,ii).*g.*exp(i*khih2*z0(f));end;
% prv=retdiag(1./rho(f));B1(:,:,3)=B1(:,:,2)*prv;B2(:,:,3)=B2(:,:,2)*prv;

B1=cal_bessel(1,u1,rho(f),1,g.*exp(i*khih1*z0(f)),parm);B2=cal_bessel(1,u2,rho(f),1,g.*exp(i*khih2*z0(f)),parm);

e_oc(f,:,:)=calee(wtho,B1,B2,u1.',u2.',Seh1,Seh2,du1.',du2.',nh);

% end;         %<<<<<<<<<<<<<<
e_res=onde_cylindrique_3D_residu(rho,0,1,nb,nh,ns,plasmon,z0,parm);
ec=coupure3D_90(rho,nb,nh,ns,plasmon,z0,parm);e_oc=e_oc+ec;

e=e_oc+e_res;
%***************************************************************************************************************
function e=coupure3D_90(rho,nb,nh,ns,plasmon,z0,parm);
if parm.calder3D;e=zeros(length(rho),6,12);else;e=zeros(length(rho),6,4);end;

[u1,u2,h1,h2]=lowenthal3D_bis_90(0,nh,nb,ns);% pour gain de temps
f=find(real(h1*rho)>-36);if isempty(f);return;end;	
fin=[-10,-1,1,10];
tho_pl=sqrt(i*(nb-plasmon.constante_de_propagation));
[tho0,tho_max]=choisi_tho(rho(f));
[tho,wtho]=retgauss(0,tho_max,20,10,[tho0,abs(real(tho_pl))+imag(tho_pl)*fin]);
[u1,u2,h1,h2,Seh1,Seh2,du1,du2,khih1,khih2]=lowenthal3D_bis_90(tho.',nh,nb,ns,parm);
g=zeros(length(h1),length(f));prv=h1*rho(f);% gain de temps
ff=find(  real(prv)> (max(real(prv(:))) -36));
g(ff)=exp(prv(ff));

B1=cal_bessel(1,u1,rho(f),1,g.*exp(i*khih1*z0(f)),parm);B2=cal_bessel(1,u2,rho(f),1,g.*exp(i*khih2*z0(f)),parm);
e(f,:,:)=calee(wtho,B1,B2,u1.',u2.',Seh1,Seh2,du1.',du2.',nh);

%***************************************************************************************************************
function [u1,u2,h1,h2,Seh1,Seh2,du1,du2,khih1,khih2]=lowenthal3D_90(tho,nh,nb,ns,parm);
h1=i*nh-tho.^2;h2=h1;
u1=nh+i*tho.^2;u2=u1;
if nargout<5;return;end;
du1=2*i*tho;du2=-du1;
khih1=(i-1)*sqrt(nh)*tho.*sqrt(1+(.5*i/nh)*tho.^2);
khih2=-khih1;
khib1=coupure_droite(u1,nb);
khib2=khib1;
Seh1=calSeh(u1,khih1,khib1,nh,nb,ns,parm,1);
Seh2=calSeh(u2,khih2,khib2,nh,nb,ns,parm,1);

%***************************************************************************************************************
function [u1,u2,h1,h2,Seh1,Seh2,du1,du2,khih1,khih2]=lowenthal3D_bis_90(tho,nh,nb,ns,parm);
h1=i*nb-tho.^2;h2=h1;
u1=nb+i*tho.^2;u2=u1;
if nargout<5;return;end;
du1=2*i*tho;du2=-du1;
khib1=(i-1)*sqrt(nb)*tho.*sqrt(1+(.5*i/nb)*tho.^2);
khib2=-khib1;
khih1=coupure_droite(u1,nh);
khih2=khih1;

Seh1=calSeh(u1,khih1,khib1,nh,nb,ns,parm,1);
Seh2=calSeh(u2,khih2,khib2,nh,nb,ns,parm,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=calee(wtho,B1,B2,u1,u2,Seh1,Seh2,du1,du2,nh,test);if nargin<11;test=0;end;
e=cale(wtho.*du1,B1,u1,Seh1,nh,test)+cale(wtho.*du2,B2,u2,Seh2,nh,test);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=cale(w,B,u,Se0,nh,test);if nargin<6;test=0;end;
wu=w.*u;wuu=wu.*u;
if length(Se0)<5;% <------sources simples
[Se0,Sh0,Se1,Sh1]=deal(Se0{:});
e=zeros(size(B,2),6,4);
e(:,:,1)=calee0(wu,wuu,B,Se0.',nh,u,test);
e(:,:,2)=calee0(wu,wuu,B,Sh0.',nh,u,test);
e(:,:,3)=caleeL(w,wu,wuu,B,Se1.',nh,u,test);
e(:,:,4)=caleeL(w,wu,wuu,B,Sh1.',nh,u,test);
else;            % <------sources et dérivees
[Se0,Sh0,Se1,Sh1,Se0p_d,Se0m_d,Sh0p_d,Sh0m_d,Se1_d,Sh1_d,Se2_d,Sh2_d]=deal(Se0{:});
e=zeros(size(B,2),6,12);
e(:,:,1)=calee0(wu,wuu,B,Se0.',nh,u,test);
e(:,:,2)=calee0(wu,wuu,B,Sh0.',nh,u,test);
e(:,:,3)=caleeL(w,wu,wuu,B,Se1.',nh,u,test);
e(:,:,4)=caleeL(w,wu,wuu,B,Sh1.',nh,u,test);

e(:,:,5)=calee0(wu,wuu,B,Se0p_d.',nh,u,test);
e(:,:,6)=calee0(wu,wuu,B,Se0m_d.',nh,u,test);
e(:,:,7)=calee0(wu,wuu,B,Sh0p_d.',nh,u,test);
e(:,:,8)=calee0(wu,wuu,B,Sh0m_d.',nh,u,test);

e(:,:,9)=caleeL(w,wu,wuu,B,Se1_d.',nh,u,test);
e(:,:,10)=caleeL(w,wu,wuu,B,Sh1_d.',nh,u,test);

e(:,:,11)=caleeL(w,wu,wuu,B(:,:,[2,4,5]),Se2_d.',nh,u,test);
e(:,:,12)=caleeL(w,wu,wuu,B(:,:,[2,4,5]),Sh2_d.',nh,u,test);
end;             % <------sources et dérivees ?	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=calee0(wu,wuu,B,S,nh,u,test);
e=zeros(size(B,2),6);
if ~all(S(1,:)==0);e(:,1)=(i*(wu.*S(1,:))*B(:,:,2)).';end;
if ~all(S(2,:)==0);e(:,2)=((wu.*S(2,:))*B(:,:,2)).';e(:,6)=((wuu.*S(2,:))*B(:,:,1)).';end;
if ~all(S(4,:)==0);e(:,3)=((wuu.*S(4,:))*B(:,:,1)/nh^2).';e(:,5)=((wu.*S(4,:))*B(:,:,2)).';end;
if ~all(S(3,:)==0);e(:,4)=(i*(wu.*S(3,:))*B(:,:,2)).';end;

if test==0;return;end;%test
k=size(B,2);

figure;%[min(real(u)),max(real(u))]
x=real(retreal(u));
subplot(3,2,1);prv=u.'.*S(1,:).'.*B(:,k,2);plot(x,real(prv),'.-r',x,imag(prv),'.-b');
subplot(3,2,2);prv=u.'.*S(2,:).'.*B(:,k,2);;plot(x,real(prv),'.-r',x,imag(prv),'.-b');
subplot(3,2,3);prv=u.'.*u.'.*S(2,:).'.*B(:,k,2);plot(x,real(prv),'.-r',x,imag(prv),'.-b');
subplot(3,2,4);prv=u.'.*u.'.*S(4,:).'.*B(:,k,1);plot(x,real(prv),'.-r',x,imag(prv),'.-b');
subplot(3,2,5);prv=u.'.*S(4,:).'.*B(:,k,2);plot(x,real(prv),'.-r',x,imag(prv),'.-b');
subplot(3,2,6);prv=u.'.*S(3,:).'.*B(:,k,2);plot(x,real(prv),'.-r',x,imag(prv),'.-b');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=caleeL(w,wu,wuu,B,S,nh,u,test);
e=zeros(size(B,2),6);
% B(:,:,1) = H_L_1(u*rho*sphi) B(:,:,2) = H_L(u*rho*sphi)  B(:,:,3) = L*H_L(u*rho*sphi)/(rho*sphi) 
e(:,2)=((w.*(S(1,:)+S(2,:)))*B(:,:,3)).';e(:,1)=i*e(:,2);
e(:,5)=((w.*(S(3,:)+S(4,:)))*B(:,:,3)).';e(:,4)=i*e(:,5);
if ~all(S(1,:)==0);e(:,1)=e(:,1)-((i*wu.*S(1,:))*B(:,:,1)).';end;
if ~all(S(3,:)==0);e(:,4)=e(:,4)-((i*wu.*S(3,:))*B(:,:,1)).';end;
if ~all(S(2,:)==0);e(:,2)=e(:,2)-((wu.*S(2,:))*B(:,:,1)).';e(:,6)=((wuu.*S(2,:))*B(:,:,2)).';end;
if ~all(S(4,:)==0);e(:,3)=((wuu.*S(4,:))*B(:,:,2)/nh^2).';e(:,5)=e(:,5)-((wu.*S(4,:))*B(:,:,1)).';end;

if test==0;return;end;%test
k=size(B,2);
figure;%[min(real(u)),max(real(u))]
x=real(retreal(u));

subplot(3,2,1);prv=(S(1,:)+S(2,:)).'.*B(:,k,3)-u'.*S(1,:).'.*B(:,k,1);plot(real(u),real(prv),'.-r',real(u),imag(prv),'.-b');
subplot(3,2,2);prv=(S(1,:)+S(2,:)).'.*B(:,k,3)-u'.*S(2,:).'.*B(:,k,1);plot(real(u),real(prv),'.-r',real(u),imag(prv),'.-b');
subplot(3,2,3);prv=u.'.*u.'.*S(4,:).'.*B(:,k,2);plot(real(u),real(prv),'.-r',real(u),imag(prv),'.-b');
subplot(3,2,4);prv=(S(3,:)+S(4,:)).'.*B(:,k,3)-u'.*S(3,:).'.*B(:,k,1);plot(real(u),real(prv),'.-r',real(u),imag(prv),'.-b');
subplot(3,2,5);prv=(S(3,:)+S(4,:)).'.*B(:,k,3)-u'.*S(4,:).'.*B(:,k,1);plot(real(u),real(prv),'.-r',real(u),imag(prv),'.-b');
subplot(3,2,6);prv=u.'.*u.'.*S(2,:).'.*B(:,k,2);plot(real(u),real(prv),'.-r',real(u),imag(prv),'.-b');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
function Seh=calSeh(u,khih,khib,nh,nb,ns,parm,divise);
u_khih=u.*khih;khih_khib=khih.*khib;
% composantes 1 et 4: (eh+ + eh-)/2 (hh+ - hh-)/2   à diviser par:  (khih/nh^2+khib/nb^2)
% composantes 2 et 3: (eh+ - eh-)/2 (hh+ + hh-)/2   à diviser par:   (khih+khib)
[Se0,Sh0,Se1,Sh1]=deal(zeros(length(u),4));

Se0(:,1)=(.25/(pi*(nh*ns)^2))*u_khih;Se0(:,4)=(-.25/(pi*ns^2))*u;         % pour Phi0e
Sh0(:,2)=(-.25/pi)*u;Sh0(:,3)=(.25/pi)*u_khih;                            % pour Phi0h

Se1(:,2)=(-.25/pi);Se1(:,3)=(.25/pi)*khih;                               % pour Phi1e
Se1(:,1)=(.25/(pi*(nh*nb)^2))*khih_khib;Se1(:,4)=(-.25/(pi*nb^2))*khib; 

Sh1(:,2)=(-.25/pi)*khib;Sh1(:,3)=(.25/pi)*khih_khib;                     % pour Phi1h
Sh1(:,1)=(.25/(pi*nh^2))*khih;Sh1(:,4)=(-.25/pi);                        

if parm.calder3D==1;% <------sources et dérivees
u_khih_khib=u.*khih_khib;u_khib=u.*khib;
[Se0p_d,Se0m_d,Sh0p_d,Sh0m_d,Se1_d,Sh1_d,Se2_d,Sh2_d]=deal(zeros(length(u),4));
%[Se2_d,Sh2_d]=deal(zeros(length(u),4,2));

Se0p_d(:,1)=(-.125/(pi*(nh*nb)^2))*u_khih_khib;Se0p_d(:,4)=(.125/(pi*nb^2))*u_khib;% pour Phi0e'+
Se0m_d(:,2)=(-.125/pi)*u;Se0m_d(:,3)=(.125/pi)*u_khih;                             % pour Phi0e'-

Sh0p_d(:,2)=(.125/pi)*u_khib;Sh0p_d(:,3)=(-.125/pi)*u_khih.*khib;                  % pour Phi0h'+
Sh0m_d(:,1)=(.125/(pi*nh^2))*u_khih;Sh0m_d(:,4)=(-.125/pi)*u;                      % pour Phi0h'-

Se1_d(:,1)=(.25/(pi*(ns.*nh)^2))*u.*u_khih;Se1_d(:,4)=(-.25/(pi*ns^2))*u.^2;       % pour Phi1e'
Sh1_d(:,2)=(-.25/pi)*u.^2;Sh1_d(:,3)=(.25/pi)*u.*u_khih;                           % pour Phi1h'

Se2_d(:,2)=(-.125/pi)*u;Se2_d(:,3)=(.125/pi)*u_khih;                               % pour Phi2e'
Se2_d(:,1)=(.125/(pi*(nh*nb)^2))*u.*khih_khib;Se2_d(:,4)=(-.125/(pi*nb^2))*u_khib;

Sh2_d(:,2)=(-.125/pi)*u_khib;Sh2_d(:,3)=(.125/pi)*u.*khih_khib;                    % pour Phi2h'
Sh2_d(:,1)=(.125/(pi*nh^2))*u_khih;Sh2_d(:,4)=(-.125/pi)*u;

end

if nargin>7; % <<<<<


for ii=[1,4];prv=1./(khih/nh^2+khib/nb^2);
Se0(:,ii)=prv.*Se0(:,ii);
Se1(:,ii)=prv.*Se1(:,ii);
Sh1(:,ii)=prv.*Sh1(:,ii);
end;
for ii=[2,3];prv=1./(khih+khib);
Sh0(:,ii)=prv.*Sh0(:,ii);
Se1(:,ii)=prv.*Se1(:,ii);
Sh1(:,ii)=prv.*Sh1(:,ii);
end;

if parm.calder3D==1;% ------
for ii=[1,4];prv=1./(khih/nh^2+khib/nb^2);
Se0p_d(:,ii)=prv.*Se0p_d(:,ii);Sh0m_d(:,ii)=prv.*Sh0m_d(:,ii);Se1_d(:,ii)=prv.*Se1_d(:,ii);
Se2_d(:,ii)=prv.*Se2_d(:,ii);Sh2_d(:,ii)=prv.*Sh2_d(:,ii);
end;
for ii=[2,3];prv=1./(khih+khib);
Se0m_d(:,ii)=prv.*Se0m_d(:,ii);Sh0p_d(:,ii)=prv.*Sh0p_d(:,ii);Sh1_d(:,ii)=prv.*Sh1_d(:,ii);
Se2_d(:,ii)=prv.*Se2_d(:,ii);Sh2_d(:,ii)=prv.*Sh2_d(:,ii);
end;

end;              % ------
end;       % <<<<<

if parm.calder3D==1;
Seh={Se0,Sh0,Se1,Sh1,Se0p_d,Se0m_d,Sh0p_d,Sh0m_d,Se1_d,Sh1_d,Se2_d,Sh2_d};
else;Seh={Se0,Sh0,Se1,Sh1};end;
% %************************************************************************************************************************
function [tho_pl,tho_nb]=cal_tho_pl_tho_nb(cphi,sphi,plasmon,nh,nb);
tho_pl=sqrt(-i*(plasmon.constante_de_propagation*sphi+plasmon.khi_dielectrique*cphi-nh));
tho_nb=sqrt(-i*(nb*sphi+coupure_droite(nb,nh)*cphi-nh));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function B=cal_bessel(cas,u,rho,sphi,g,parm)
if parm.calder3D==0;% <------sources simples
	switch cas;
	case 1;	
	B=reshape(retbessel('h',[0,1,1],1,retcolonne(u*(rho*sphi)),1),length(u),length(rho),3);
	for ii=1:2;B(:,:,ii)=B(:,:,ii).*g;end;
	B(:,:,3)=B(:,:,2)*retdiag(1./(rho*sphi));
	case 0;	
	prv=retcolonne(u*(rho.*sphi));
	[f,ff]=retfind(abs(prv)>.5);
	B=retbessel('j',[0,1,1],prv);
	B(f,3)=B(f,3)./prv(f);   % J1(x)/x
	B(ff,3)=polyval([-1/176947200,1/1474560,-1/18432,1/384,-1/16,1/2],prv(ff).^2);
	B=reshape(B,length(u),length(rho),3);
	for ii=1:3;B(:,:,ii)=B(:,:,ii).*g;end;
	B(:,:,3)=retdiag(u)*B(:,:,3);
	end;
else;% <------sources et dérivees
	switch cas;
	case 1;	
	B=reshape(retbessel('h',[0,1,1,2,2],1,retcolonne(u*(rho*sphi)),1),length(u),length(rho),5);
	for ii=[1,2,4];B(:,:,ii)=B(:,:,ii).*g;end;
	prv=retdiag(1./(rho*sphi));B(:,:,3)=B(:,:,2)*prv;B(:,:,5)=2*B(:,:,4)*prv;
	case 0;	
	prv=retcolonne(u*(rho.*sphi));
	B=retbessel('j',[0,1,1,2,2],prv);
	[f,ff]=retfind(abs(prv)>.5);
	B(f,3)=B(f,3)./prv(f);% J1(x)/x
	B(ff,3)=polyval([-1/176947200,1/1474560,-1/18432,1/384,-1/16,1/2],prv(ff).^2);
	B(f,5)=2*B(f,5)./prv(f);% 2*J2(x)/x
	B(ff,5)=2*prv(ff).*polyval([-1/2477260800,1/17694720,-1/184320,1/3072,-1/96,1/8],prv(ff).^2);

	B=reshape(B,length(u),length(rho),5);
	for ii=1:5;B(:,:,ii)=B(:,:,ii).*g;end;
	B(:,:,3)=retdiag(u)*B(:,:,3);B(:,:,5)=retdiag(u)*B(:,:,5);
	end;

end
% %***************************************************************************************************************
function [tho,tho_max]=choisi_tho(r);
tho_max=sqrt(36/max(1.e-8,min(r)));

prv=ceil(log(5./sqrt(r))/log(2));tho=exp(retelimine(prv,-1)*log(2));% modif
% %***************************************************************************************************************
% function [e,e_oc,e_res]=cartesien(x,y,z,nb,nh,ns,parm);
% plasmon=retplasmon(nh,nb,2*pi);
% if plasmon.test_d_existence;
% fin=[-10,-1,1,10];
% [r,wr]=retgauss(0,20*max(real(nh),real(nb)),10,50,[real(nh)+.01*fin,real(nb)+.01*fin,real(plasmon.constante_de_propagation)+imag(plasmon.constante_de_propagation)*fin]);
% else;
% [r,wr]=retgauss(0,20*max(real(nh),real(nb)),10,50);
% end;
% [teta,wteta]=retgauss(0,2*pi,-50);
% [R,Teta]=ndgrid(r,teta);R=R(:);Teta=Teta(:);
% u=R.*cos(Teta);v=R.*sin(Teta);
% [Wr,Wteta]=ndgrid(wr,wteta);W=R.*Wr(:).*Wteta(:);
% r2=u.^2+v.^2;
% khih=retsqrt(nh^2-r2,-1);
% khib=retsqrt(nb^2-r2,-1);
% 
% prv=retdiag(W)*exp(i*(u*X+v*Y+khih*Z));
% e=cell(1,6);
% for K=1:6;
% e{K}=zeros(length(z),length(x),length(y),6);
% S=zeros(1,6);S(K)=1;
% sigma=[S(5)-i*u*S(3)/ns^2,-S(4)+i*v*S(3)/ns^2,S(2)-i*u*S(6),-S(1)+i*v*S(6)]/(4*pi^2);
% eh=((-nb^2)*(u.*sigma(:,1)+v.*sigma(:,2))+i*khib.*(-v.*sigma(:,3)+u.*sigma(:,4)))./(nb^2*khih+nh^2*khib);
% hh=(-(u.*sigma(:,3)+v.*sigma(:,4))+i*khib.*(-v.*sigma(:,1)+u.*sigma(:,2)))./(khih+khib);
% e{K}(:,:,:,1)=reshape(((-u.*khih.*eh+i*v.*hh)./r2).'*prv,length(z),length(x),length(y),1);
% e{K}(:,:,:,2)=reshape(((-v.*khih.*eh-i*u.*hh)./r2).'*prv,length(z),length(x),length(y),1);
% e{K}(:,:,:,3)=reshape(eh.'*prv,length(z),length(x),length(y),1);
% e{K}(:,:,:,4)=reshape(((-u.*khih.*hh+i*v.*eh)./r2).'*prv,length(z),length(x),length(y),1);
% e{K}(:,:,:,5)=reshape(((-v.*khih.*hh-i*u.*eh)./r2).'*prv,length(z),length(x),length(y),1);
% e{K}(:,:,:,6)=reshape(hh.'*prv,length(z),length(x),length(y),1);
% end
