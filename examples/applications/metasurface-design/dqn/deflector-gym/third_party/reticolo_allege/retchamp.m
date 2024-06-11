function [e,yy,ww,o,s,sssh]=retchamp(init,a,sh,sb,inc,x,y,h,cale,calyy,calww,calo,cals,met);
%  [e,yy,ww,o,s]=retchamp(init,a,sh,sb,inc,x,y,h,cale,calyy,calww,calo,cals); pour une seule couche
%  [e,yy,ww,o,s,sssh]=retchamp(init,a,sh,sb,inc,x,tab,sssh,cale,calyy,calww,calo,cals,met); forme generale conseillee
%   e=retchamp(init,a,inc,sens,x);  forme simplifiee une seule couche un seul plan
%   e=retchamp(init,sh,sb,inc);  autre forme simplifiee donnant le champ dans un plan
%
% CALCUL DES CHAMPS  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% en 1D:
%
% calcul des champs:
%   dans le cas de polarisation E// (pol=0)  
%  ez=e(:,:,1),hx=e(:,:,2),hy=e(:,:,3),bx=e(:,:,4) 
%  en yy, x (vecteurs ligne ou colonne) 
%
%   dans le cas de polarisation H// (pol=2)  
%   hz=e(:,:,1),ex=-e(:,:,2),ey=-e(:,:,3),dx=-e(:,:,4) 
% en 2D:
%
% calcul des champs ex=e(:,:,:,1),ey=e(:,:,:,2),ez=e(:,:,:,3),hx=e(:,:,:,4),hy=e(:,:,:,5),hz=e(:,:,:,6)
%  en yy,x,y  {x,y} regroupes en  cell array  dans les donnees pour compatibilitee avec le cas 1D
%  si  a  a  eté calculé sans parametre (ou c=0) le calcul est fait de maniére approchée pour les champs discontinus
%  alors ez et hz ne sont pas calcules (ez=hz=0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SI PLUSIEURS COUCHES:
%
%  [e,yy,ww,o,s,sssh]=retchamp(init,a,sh,sb,inc,x,tab,sssh,cale,calyy,calww,calo,cals);
%
% sh :matrice s du haut des couches au haut du système
% sb :matrice s du bas du système au bas des couches                                      
% inc:vecteur incident du système total (ligne ou colonne) 
% 
% dispositions pour économiser la mémoire:
%   a est un cell array  contenant soit des descripteurs de textures soit des matrices s de sources
%   les éléments de a  ainsi que a ,ainsi que sh et sb peuvent être des noms de fichiers
%                                                  créés par retio( ,1) ou retio( ,'fich')
%
% des matrices S sont tabulées dans un cell array: sssh 
%  on peut les réutiliser dans un autre passage avec d'autres x pourvu que tab ne soit pas changé
%  (ceci peut être utile pour faire divers coupes sagittales)
%
%
%      Y
%      |             inc h
%      |
%      |                         Sh
%      |          ------------------------------
%      |
%      |                 hauteur  tab(1,1)   descripteur de la couche:a{tab(1,2)} 
%      |           nombre de points:si gauss:        tab(1,3)  répète tab(nn,4) fois ( quatrieme colonne facultative)
%      |                            si trapèzes    - tab(1,3) points ( points aux extrémités)
%      |         si imag( tab(1,3))~=0  :source de matrice S: a{imag( tab(1,3)} alors pas calcul du champ sur la hauteur tab(1,1)
%      |            
%      |          ------------------------------
%      |
%      |            . . . . . . . . .
%      |
%      |          ------------------------------
%      |
%      |                 hauteur   tab(nn-1,1)   descripteur de la couche:a{tab(2,2)}
%      |           nombre de points:si gauss: tab(nn-1,3)  répète tab(nn-1,4) fois si trapèzes - tab(nn-1,3) points
%      |         si imag(tab(n-1,3))~=0  :source de matrice S: a{imag(tab(nn-1,3)} alors pas calcul du champ sur la hauteur tab(nn-1,1)
%      |
%      |          ------------------------------
%      |
%      |                 hauteur  tab(nn,1)   descripteur de la couche:a{tab(nn,2)}
%      |           nombre de points:si gauss:tab(nn,3)  répète tab(nn,4) fois si trapèzes -tab(nn,3) points
%      |         si imag(tab(1,3))~=0  :source de matrice S: a{imag(tab(nn,3)} alors pas calcul du champ sur la hauteur tab(nn,1)
%      |
%      |          ------------------------------
%      |                          Sb
%      |             inc b  
%      +--------------------------------------------------->  X 
%
%   si tab(1,1)<0   :échantillonnage régulier en y sur toute la hauteur
%   la hauteur de la première couche est abs(tab(1,1)) le nombre de points est tab(1,3)  
%    points calcules:     linspace(0,sum(tab(:,1)),abs(tab(1,3)))
%
%   ELEMENTS FINIS
%   certains éléments de a peuvent être des descripteurs de tronçons obtenus par les elements finis
%   Il est alors impératif que en dessus et en dessous il y ait des textures rcwa
%   si on veut calculer les champ en des hauteurs précises ,on peut prendre par exemple:
%   tab=[ .....;[hh,10];[h1,5,0];[0,5,1];[h2,5,0];[0,5,1];[h3,5,0];[hb,2,15];..
%   où a{1} a{2} sont les descripteurs des textures dessus et dessous (rcwa),
%   et a{5}  un descripteur de tronçons obtenus par les éléments finis   
%   (la hauteur totale de la partie 'éléments finis' h1+h2+h3 doit être exacte)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%   AU RETOUR:
%
%   ee:champs
%   yy:cotes des points ou le champ est calculé ,L' ORIGINE EST LE BAS de la structure ( donc yy>0 )
%   ww: poids associés ( si on veut intégrer)
%   o:eps de l'objet aux mêmes points que ee (pour tracer l'objet)
%   s:matrice s totale
%   le calcul de ces quantités est contrôlé par des paramètres:
%     cale   calcul de e
%     si une valeur de cale est i, elle est ignorée et le champ est clone
%     si imag(cale)~=0 apodisation avec le paramètre imag(cale)
%         en 2D les 2 valeurs des paramètres d'apodisation (en x et en y)
%         sont la première et la dernière valeur non nulle de imag(cale)
%         en 1D la valeur du paramètre d'apodisation est la première valeur non nulle de imag(cale)
%             ( la fonction d' apodisation 'masque' peut être visualisée et calculée par appel à 
%             masque=retchamp(init,apod) qui utilise le bon nombre de termes de Fourier
%             ou, masque=retchamp(apod)) qui utilise 100 termes de Fourier en 1D ,
%             ou, masque=retchamp([apod,m]) qui utilise m termes de Fourier en 1D )
%                0 pas d'apodisation  1 hanning  2  hamming  3  flat top  4  blackman   5  blackman harris
%                6.12..  trapèze (les décimales représentent un paramètre)   7.32..  trapèze arrondi
%                8  Schwartz exp(-1/(1-x^2))  9  exp(-1/(1-x^4)) 9.12 exp(-1/(1-x^(1/.12)))
%          si apod a une partie imaginaire ,troncature dans la proportion de imag(apod) (<1) puis apodisation avec real(apod)
%     real(cale)  tableau des composantes de ee désirées (1 a 6) défaut :[1:6]
%                si real(cale(1))<0      abs(e)^2 
%                si real(cale)=  0 ou cale=[] pas calcul de ee
%     exemples: cale=[1,2] calcul des composantes 1 et 2 sans apodisation
%               cale=[1+3i,2i] calcul de la composante 1 avec apodisation 3 en x 2 en y si on est en 2 D  3 en 1 D
%               cale=[1:6]+i calcul des composantes 1,2,3,4,5,6 avec apodisation 1 en x et 1 en y
%               cale=[-1,2,3] calcul des amplitudes au carré  ,des composantes 1,2,3 sans apodisation
%               cale=[1:6,i] calcul des composantes 1,2,3,4,5,6  sans apodisation et clonage
%
%     calyy  1 calcul si  0 ou [] pas calcul de yy                              défaut:  1
%         (  si calyy a une partie imaginaire les e yy ww o sont mis en cell array par couche puis sur fichiers temporaires
%          et calyy=real(calyy) ne pas oublier d'effacer les fichiers  )
%     calww  1 calcul si  0 ou [] pas calcul de ww                              défaut:  1  
%     calo   tableau des composantes de o désirées   
%                                 en 2D (1 a 6):mux,muy,muz,epx,epy,epz        
%                                 en 1D (1 a 3):eps,mux,muy (au sens du 1 D  donc epz muy mux du 2 D)       
%             ou bien si calo=i:     o=indice                                  defaut:  i
%             si calo=-i version 'economique en memoire' :o=real(indice)
%             si  0 ou [] pas calcul de o
% pour les metaux: eps=10i*rand  mu=1 pour les metaux electriques(si met==0 sinon eps=met*i)
%                  eps=1  mu=10i*rand pour les metaux magnetiques( si met==0 sinon mu=met*i)
%
%     cals   1 calcul  0 pas calcul de s                          défaut:1 si s en sortie 0 sinon  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% SI UNE SEULE COUCHE:
%
% [e,yy,ww,o,s]=retchamp(init,a,sh,sb,inc,x,y,h,cale,calyy,calww,calo,cals)
%
%  y:vecteur des hauteurs dans la couche considérée(y=0:bas de la couche  y=h:haut de la couche)
% le descripteur de texture a décrivant la couche a été calculée auparavant avec  retcouche  (paramètre c=1 conseille)
% sh :matrice s du haut de la couche au haut du système
% sb :matrice s du bas du système au bas de la couche                                      
% inc:vecteur incident du système total (ligne ou colonne)
% cale,calyy,calww,calo,cals : même signification que dans la forme générale
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%  FORME SIMPLIFIEE pour une seule couche en un seul plan 
%
%  e=retchamp(init,a,inc,sens,x);  forme simplifiée
% inc tableau de p COLONNES 2*n lignes (n=init{1}=init{end}.nsym).Chaque colonne de inc définit un champ
%
% sens=0   inc:composantes du champ dans la base de Fourier symétrique 
% sens=1   inc:composantes du champ dans la  base des modes:(de bas en haut puis de haut en bas)
%
% au retour:
% si x existe:       en 1D:  e(p,length(x),4)          en 2D: x={x,y}   e(p,length(x),length(y),6)
% si x =[]   e composantes de Fourier NON SYMETRIQUE en 1D: e(m,p,4)    en 2D: e(m,p,6)
%     m dimension de la base de Fourier non symétrique ( m=size(beta,2)=init{end}.nfourier       beta=init{2} )
% en 1D:
%   dans le cas de polarisation E// (pol=0)  ez=e(:,:,1),hx=e(:,:,2),hy=e(:,:,3),bx=e(:,:,4) 
%   dans le cas de polarisation H// (pol=2)  hz=e(:,:,1),ex=-e(:,:,2),ey=-e(:,:,3),dx=-e(:,:,4) 
%    ( si c'est possible  hx ou ex sont obtenus en divisant bx par mux et dx par epx)
% en 2D:
%  ex=e( ,1),ey=e( ,2),ez=e( ,3),hx=e( ,4),hy=e( ,5),hz=e( ,6)
%
%  ATTENTION: dans ce cas pas de cale ( donc pas d'apodisation ni de clonage )
%             ni de sh sb ( donc les modes propagatifs ne sont pas normalises ) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AUTRE FORME SIMPLIFIEE: calcul dans un seul plan des composantes de Fourier ou sur les modes du champ 
%
%  e=retchamp(init,sh,sb,inc);
% calcul dans un plan des composantes de           Fourier    (ou     sur les  modes)  du champ
% sb:matrice s du bas du système au plan    modes->Fourier    (ou       modes->modes)
% sh:matrice s du plan au haut du système   modes->Fourier    (ou       modes->modes)
% inc:champ incident [Ib,Ih] 
%  e est un vecteur colonne forme par les composantes de Fourier 
%                    (ou sur les modes  dans l'ordre ['vers le haut';'vers le bas'])
%
%  ATTENTION: dans ce cas pas de cale ( donc pas d'apodisation ni de clonage )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%   forme dègènerèe :ajout d'une couche dans tab avec 1 point à la hauteur h depuis le bas 
%  [tab,k]=retchamp(tab,h);
%  le numéro (dans le nouveau tab ) de la couche est k
%  si h<0 (ou h> sum(tab(:,1))) tab est prolonge par le dernier (ou premier) milieu 
% dans ce cas il n'y a pas de point dans les parties rajoutees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% autre forme dègènerèe :SOURCES MULTIPLES
% [Tab,Sh,Sb,Lh,Lb]=retchamp(tab,zs,sh,sb,ah,ab,source,ns);
% modification de tab sh sb pour utiliser les sources multiples
% ENTREES
% tab ancien tab
% zs coordonnees des sources l'origine etant le bas de tab ( z=0 de retchamp)
% sh sb sans les eventuelles sources exterieures au tab
% ah,ab descripteur de texture des milieux invariants en haut et en bas 
% source matrice S de la source à répéter
% ns numero mis dans tab pour designer la source
% SORTIES
% Tab,Sh,Sb :les nouveaux
% Lh,Lb hauteurs supplementaires en haut et en bas ( eventuelles sources exterieures au tab)
% 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%            RESUME
% pour adapter tab aux hauteurs, on peut le definir d'abord avec 3 colonnes, et le completer ensuite par:
%     tab(:,4)=ceil(8*tab(:,1)/ld);
%
%  [e,y,wy,o,s,sssh]=retchamp(init,a,sh,sb,inc,x,tab,   sssh,cale,calyy,calww,calo,cals);% forme generale conseillee 1D
%  [e,z,wz,o,s,sssh]=retchamp(init,a,sh,sb,inc,{x,y},tab,   sssh,cale,calyy,calww,calo,cals);% forme generale conseillee 2D
%   e=retchamp(init,a,inc,sens,x); % forme simplifiee une seule couche un seul plan
%   e=retchamp(init,sh,sb,inc); % autre forme simplifiee donnant le champ dans un plan
%
% pour memoire :rettchamp permet un trace automatique
%  en 1D  rettchamp(e,o,x,y,pol,[1:3,i]   ,parm,fig,Text); 
%  en 2D  rettchamp(e,o,x,y,z,[1:6,i]   ,parm,fig,Text); 
%
% See also: RETTCHAMP,RETPOYNTING,RETVM,RETPOINT,RETAPOD,RETHELP_POPOV,RETABELES


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if nargin>3&~iscell(init);
% [tab,sh,sb,Lh,Lb]=retchamp(tab,xs,sh,sb,ah,ab,source,ns);
[e,yy,ww,o,s]=sources_multiples(init,a,sh,sb,inc,x,y,h);
return;end;	



if nargin<3;  % affichage et trace de la fonction d'apodisation  y=retchamp(init,apod) ou y=retchamp(apod) 
              % ou ajout d'une couche dans tab
              
if size(init,2)>=3;% ajout d'une couche dans tab     [tab,k]=retchamp(tab,h); ------------------------
tab=init;hh=cumsum(tab(:,1));
h=hh(end)-a;% hauteur depuis le haut
if h<=0;% on ajoute au dessus
if size(tab,2)==3;   
e=[[0,tab(1,2),1];[-h,tab(1,2),0];tab];yy=1;
else;
e=[[0,tab(1,2),1,1];[-h,tab(1,2),0,1];tab];yy=1;
end
return;end;


if h>=hh(end);% on ajoute au dessous
if size(tab,2)==3;   
e=[tab;[h-hh(end),tab(end,2),0];[0,tab(end,2),1]];yy=size(e,1);
else;
e=[tab;[h-hh(end),tab(end,2),0,1];[0,tab(end,2),1,1]];yy=size(e,1);
  
end;
return;end;

k=find((hh-h)>0);
k=k(1);% numero de la couche;
hh=[0;hh];% pour k=1
if size(tab,2)==3;
if hh(k+1)>hh(k)*(1+100*eps);n1=ceil(tab(k,3)*(h-hh(k))/(hh(k+1)-hh(k)));n2=ceil(tab(k,3)*(hh(k+1)-h)/(hh(k+1)-hh(k)));else;n1=ceil(tab(k,3)/2);n2=n1;end;
e=[tab(1:k-1,:,:);[h-hh(k),tab(k,2),n1];[0,tab(k,2),1];[hh(k+1)-h,tab(k,2),n2];tab(k+1:end,:,:)];
else;
if hh(k+1)>hh(k)*(1+100*eps);n1=ceil(tab(k,4)*(h-hh(k))/(hh(k+1)-hh(k)));n2=ceil(tab(k,4)*(hh(k+1)-h)/(hh(k+1)-hh(k)));else;n1=ceil(tab(k,4)/2);n2=n1;end;
e=[tab(1:k-1,:,:,:);[h-hh(k),tab(k,2),tab(k,3),n1];[0,tab(k,2),1,1];[hh(k+1)-h,tab(k,2),tab(k,3),n2];tab(k+1:end,:,:,:)];
end;
yy=k+1;
return;end;   % FIN DE ajout d'une couche dans tab ------------------------

if nargin==1;a=init;if length(a)==2;m=a(2);a=a(1);else;m=101;end;init=cell(1,11);init{2}=zeros(1,m);init{end}=struct('dim',1);end;
beta=init{2};

if init{end}.dim==1; %1D
e=zeros(length(beta),1);[prv,e]=retapod(e,a);
if nargout==0;figure;plot(e,'.-');grid;axis tight;title('fonction d''apodisation 1D');xlabel('ordres de fourier');end;
else;  %2D
mx=[1:init{6}];my=[1:init{7}];
[prv,masquex]=retapod(mx.',a(1));
[prv,masquey]=retapod(my.',a(end));
e=masquey.'*masquex;
if nargout==0;figure;retcolor(e);title('fonction d''apodisation 2D');xlabel('ordres de fourier en x');ylabel('ordres de fourier en y');end;
end;
return;
end;  % fin test de la fonction d'apodisation    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    [e,yy,ww,o,s]=retchamp(init,tab,sh,sb,inc)         %[e,yy,ww,o,s]=retchamp(init,a,sh,sb,inc,x,y
if ~isstruct(init{end}); % cas 0 D 'rapide'
switch nargin
case 5;[e,yy,ww,o]=retchamp_0D(init,a,sh,sb,inc,1);
case 6;[e,yy,ww,o]=retchamp_0D(init,a,sh,sb,inc,x);
end
return;end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sh=retio(sh);sb=retio(sb);a=retio(a);

if nargin<=5;  % FORME SIMPLIFIEE pour une seule couche en un seul plan 

if nargin==4; % champ dans un plan    
%  e=retsc(sh,sb,inc,init{1});
e=retsc(a,sh,sb,init{1});return;
end;    

% e=retchamp0(init,a,inc,sens,x);
if nargin<4;inc=[];x=[];end;
e=retchamp0(init,retio(a),sh,sb,inc);return;
end;   % fin forme simplifiee 

% valeurs par defaut des parametres de calcul
und=(init{end}.dim==1); %1D
if nargin<8;h=[];end;if ischar(h);h=[];end;if iscell(h);sssh=h;h=[];else sssh=[];end;

if nargin<9;if und;cale=1:4;else;cale=1:6;end;end;if isempty(cale);cale=0;end;

clonage=any(cale==i);cale=cale(cale~=i);

apod=imag(cale);apod=apod(apod~=0);if isempty(apod);apod=0;end;
cale=real(cale);cale=cale(cale~=0);if isempty(cale);cale=0;end;
calee=all(cale~=0);if calee;ecoe=(cale(1)<0);cale=abs(cale);end;
if nargin<10;calyy=1;end;if isempty(calyy);calyy=0;end;
if nargin<11;calww=1;end;if isempty(calww);calww=0;end;
if nargin<14;met=0;end;if isempty(met);met=0;end;
if nargin<12;calo=i;end;if isempty(calo);calo=0;end;
caloo=all(calo~=0);if caloo;ecoo=(imag(calo)<0);if ecoo;calo=i;end;end;
if nargin<13;cals=[];end;if isempty(cals);cals=1;end;
if nargout<1;calee=0;end;
if nargout<2;calyy=0;end;cel=imag(calyy)~=0;calyy=real(calyy);
if nargout<3;calww=0;end;
if nargout<4;caloo=0;end;
if nargout<5;cals=0;end;
if nargout<6&isempty(sssh);io1=-1;io2=-2;else;io1=2;io2=0;end; % pour effacer ou non les fichiers de sssh
s=[];if cel e={};yy={};ww={};o={};else;if und;e=zeros(0,length(x),length(cale));o=zeros(0,length(x),length(calo));else;e=zeros(0,length(x{1}),length(x{2}),length(cale));o=zeros(0,length(x{1}),length(x{2}),length(calo));end;yy=zeros(1,0);ww=[];end;
if isempty(y);return;end;

% calcul de betaa qui sert a calculer les champs par synthèse de Fourier-Floquet(sauf pour les modes des metaux )  
if ~iscell(sh);betaa=[];else;if calee;betaa=calbetaa(init,x,apod);end;end; 

if ~isempty(h);% une seule couche de hauteur h calculée aux points y
if calee;e=retchamp1(init,retio(a),sh,sb,inc,x,y,h,betaa);e=retclonage(e,clonage,und);
if und;e=e(:,:,cale);else;e=e(:,:,:,cale);end;
if ecoe;e=abs(e).^2;end;
end; 
if calyy;yy=y;end;
if caloo;
o=rettestobjet(init,retio(a),-1,0,x,calo,met);o=reshape(o,[1,size(o)]);o=repmat(o,length(y),1);
if ecoo;o=real(o);end;
end;
if cals;s=retss(sh,retc(a,h),sb);end
if calee;betaa=retio(betaa,-2);end; 
return;end;        % fin une seule couche 
[f,ff]=retfind(isnan(y(:,2)));if isempty(f);z0=0;else;z0=y(f,1);y=y(ff,:);end;
nn=size(y,1);if size(y,2)==3;y=[y,ones(nn,1)];end;

% construction de yreg et wreg et nreg
if y(1,1)<0;y(1,1)=-y(1,1);[yreg,wreg]=retgauss(0,sum(y(:,1)),-abs(y(1,3)));nreg=[];% echantillonnage régulier en y
else yreg=[];wreg=[];nreg=[];nnreg=0;
y0=0;
for ii=nn:-1:1;
if (y(ii,3)~=0)&(imag(y(ii,3))==0);% <----  nb de points non nul et pas de source dans la couche
[yyy,www]=retgauss(0,y(ii,1),y(ii,3),y(ii,4));nreg=[nreg,nnreg+[1:length(yyy)]];nnreg=nnreg+length(yyy);
yreg=[yreg,yyy+y0];wreg=[wreg,www];y(ii,4)=length(yyy);% y(ii,4) contient maintenant le nombre de points 
else;y(ii,4)=0;
end;
y0=y0+y(ii,1);
end;
end;

ii=1;
if init{end}.genre~=2;
while(ii<nn); % mise en forme de y par élimination des textures égales adjacentes(sauf pour MCR)
if isreal(y(ii,3)); % pas de source
f=find(  (y(ii,2)~=y(ii+1:end,2))  |  imag(y(ii+1:end,3))~=0 );
if ~isempty(f);iii=f(1)+ii-1;else iii=nn;end;
if iii>ii;y(ii,1)=sum(y(ii:iii,1));y(ii,4)=sum(y(ii:iii,4));y=y([1:ii,iii+1:nn],:);nn=nn-iii+ii;end; % suppression de lignes
end;
ii=ii+1;
end;
end;
z=z0+[flipud(cumsum(flipud(y(:,1))));0];z=z(2:end); % hauteur depuis le bas

if (calee|cals)&isempty(sssh); % tabulation des matrices S calcul de e
if calee;sssh=cell(1,nn);s_elf=cell(1,nn);sssh{nn}=retio(sh,1);end;
for ii=nn-1:-1:1;
if imag(y(nn-ii,3))~=0; % source dans la couche
aprv=retio(a{y(nn-ii,2)});
ssh=retss(sh,retc(aprv,y(nn-ii,1)/2,z(nn-ii)+y(nn-ii,1)/2),retio(a{imag(y(nn-ii,3))}),retc(aprv,y(nn-ii,1)/2,z(nn-ii)));    
else  %<-- pas de source dans la couche 
aprv=retio(a{y(nn-ii,2)});
switch aprv{end}.type
case 4;[sprv,s_elf{nn-ii}]=retc(init,aprv,a{y(nn-ii-1,2)},a{y(nn-ii+1,2)});ssh=retss(sh,sprv);% elements finis
otherwise;ssh=retss(sh,retc(aprv,y(nn-ii,1),z(nn-ii))); % rcwa
end;
end; %<--  source dans la couche ?
if calee;sssh{ii}=retio(ssh,1);end;
sh=ssh;if cals&(ii==(nn-1));s=ssh;end;
end;
else;if cals;s=sssh{nn-1};end;
end;  % fin tabulation des matrices S calcul de e

y0=0;nby=0;nnreg=1;
for ii=nn:-1:1;%   < ++++++ boucle sur ii
if isempty(nreg);   
f=find((yreg>y0)&(yreg<=(y0+y(ii,1))));if (y0==0)&(yreg(1)==0);f=[1,f];end;if (yreg(end)>y0+y(ii,1))&(yreg(end)<(y0+y(ii,1)+10*eps));f=[f,length(yreg)];end;% TEST pour avoir les points extremes
else;f=find(nreg>=nnreg & nreg<=(nnreg+y(ii,4)-1));nnreg=nnreg+y(ii,4);end;
%f=nby+1:nby+y(ii,4);nby=nby+y(ii,4); % on prend y(ii,4) points
yyy=yreg(f)-y0;www=wreg(f);
if ~isempty(f) & (imag(y(ii,3))==0);% <----  nb de points non nul et pas de source dans la couche
yyyy=y0+yyy;if cel;yyyy={yyyy};www={www};end;if calyy;yy=[yy,yyyy];end;if calww;ww=[ww,www];end;

aprv=retio(a{y(ii,2)});
if (aprv{end}.type~=4) & caloo; % calcul de objet sauf pour les elements finis
if aprv{end}.type==5;  % textures inclinees
if iscell(x);oooo=zeros(length(yyy),length(x{1}),length(x{2}));else;oooo=zeros(length(yyy),length(x));end;
for iii=1:length(yyy);xx=x;if iscell(x);xx{1}=xx{1}-yyy(iii)*aprv{3}(1);xx{2}=xx{2}-yyy(iii)*aprv{3}(2);else;xx=x-yyy(iii)*aprv{3};end;
	oo=rettestobjet(init,retio(a{y(ii,2)}),-1,0,xx,calo,met);if ecoo;oo=real(oo);end;
	oooo(iii,:)=oo(:).';end;	
else;                  % textures droites
oo=rettestobjet(init,retio(a{y(ii,2)}),-1,0,x,calo,met);oo=reshape(oo,[1,size(oo)]);if ecoo;oo=real(oo);end;   
oooo=repmat(oo,size(yyy,2),1);
end;                   % textures inclinees ?
if cel;oooo={retio(oooo,1)};www={www};end;o=[o;oooo];
end;

if calee;    % calcul de e

if aprv{end}.type==4;  % elements finis
[ee,oo]=retelf_champ(init,aprv,retio(sssh{nn+1-ii},io2),sb,inc,x,yyy,s_elf{ii},calo);ee=retclonage(ee,clonage,und);
if caloo;
if ecoo;oo=real(oo);end;
if cel;oo={retio(oo,1)};www={www};end;o=cat(1,o,oo);
end;
else;ee=retchamp1(init,aprv,retio(sssh{nn+1-ii},io2),sb,inc,x,yyy,y(ii,1),betaa,z(ii));ee=retclonage(ee,clonage,und);end;% rcwa
if und;ee=ee(:,:,cale);else;ee=ee(:,:,:,cale);end;
if ecoe;ee=abs(ee).^2;end;
if cel;ee={retio(ee,1)};www={www};end;if ~isempty(ee);e=[e;ee];end;
end;   % fin calcul de e

else;  % <----nb de points nul ou source dans la couche:on efface le fichier
if calee;retio(sssh{nn+1-ii},io1);end;
end;   % <---- 

if ii>1; % < *** calcul de la nouvelle matrice sb pour la couche suivante
aprv=retio(a{y(ii,2)});
if imag(y(ii,3))~=0;% source dans la couche
sb=retss(retc(aprv,y(ii,1)/2,z(ii)+y(ii,1)/2),retio(a{imag(y(ii,3))}),retc(aprv,y(ii,1)/2,z(ii)),sb);    
else;
if aprv{end}.type==4;  % elements finis
sb=retss(retc(init,aprv,a{y(ii-1,2)},a{y(ii+1,2)}),sb);
else;sb=retss(retc(aprv,y(ii,1),z(ii)),sb);end;
end;
y0=y0+y(ii,1);
end; % < ***

end ;   %   < ++++++ boucle sur ii
if calee;betaa=retio(betaa,-2);end; 
if cals;if isempty(s);s=sh;end;s=retss(s,sb);end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=retchamp1(init,a,sh,sb,inc,x,y,h,betaa,z);

if a{end}.type==2;  % < $$$$$$$$$$$$$$$$$ metaux
if init{end}.dim==2; %2D
e=retmchamp(init,a,sh,sb,inc,x{1},x{2},y,h);
else  %1D
e=retmchamp(init,a,sh,sb,inc,x,y,h);
end    
  
else;              % < $$$$$$$$$$$$$$$$$ autres que metaux
	
if init{end}.dim==2;  % < xxxxxxxxxxxxxxxxxxxx  2D 
if a{end}.type==7 & nargin>9;e=retchamp_cylindrique_radial(init,a,sh,sb,inc,x,y,h,betaa,z);% cylindrique radial 	
else;e=ret2champ(init,a,sh,sb,inc,x{1}(:).',x{2}(:).',y,h,betaa);end;
else                  % < xxxxxxxxxxxxxxxxxxxx  1D 
x=x(:).';    
m=init{1};si=init{3};
n=length(init{2});
%p=a{1};pp=a{2};q=a{3};qq=a{4};d=a{5};
if length(a)>6;a3=a{6};a1=a{7};else a3=[];a1=[];end;
e=zeros(size(y,2),size(x,2),4);
if (~isempty(a1))&(a{end}.type~=5)&(a{end}.type~=7);muy=rettestobjet(init,a,-1,0,x,3);end; % pour calcul de hx a partir de bx
vh=[];

for nn=1:size(y,2);% <++++++++++++++ nn
if isempty(sh); % .. forme simplifiee de retchamp
v=y(:,nn);y(nn)=0; % y sert ensuite comme cote,mais on n'ecrase pas les valeurs suivantes de y(:,nn)   
else;           % .. forme generale

if (size(y,2)<4)|(min(abs(a{5}))<eps); %<-- forme non acceleree   
if h==0;ssb=sb;ssh=sh;%epaisseur nulle    
else;
ssb=retss(retc(a,y(nn)),sb);

if a{end}.type==5; % texture inclinee
ssh=retss(sh,rettr(init,retc(a,h-y(nn)),a{3}*y(nn)));
else;
ssh=retss(sh,retc(a,h-y(nn)));
end;
end;
v=retsc(ssh,ssb,inc,m);
else;           %<-- forme acceleree 
if isempty(vh);% initialisation 
sprv=retc(a,h);% sh_prv=retgs(retb(init,a,1.e-3,0),2);
vh=retsc(sh,retss(sprv,sb),inc,m);
vb=retsc(retss(sh,sprv),sb,inc,m);
% passage dans la base modale
if a{end}.type==5; % texture inclinee
vh=(a{2}*(vh.*exp(i*h*a{3}*[init{2},init{2}].')));vb=a{2}*vb;	
else;              % texture droite
vh=.5*[a{4}*vh(1:m);(1./a{5}).*(a{2}*vh(m+1:2*m))];vh=[vh(1:m)-vh(m+1:2*m);vh(1:m)+vh(m+1:2*m)];
vb=.5*[a{4}*vb(1:m);(1./a{5}).*(a{2}*vb(m+1:2*m))];vb=[vb(1:m)-vb(m+1:2*m);vb(1:m)+vb(m+1:2*m)];
end;               % texture inclinee  ?
clear  sh_prv  sprv
end;           % fin initialisation 
[f,ff]=retfind(real(a{5})>0);
if a{end}.type==5; % texture inclinee
vh(ff)=0;vb(f)=0;
vm=vh.*exp(a{5}*(y(nn)-h));vm(ff)=0; % pour eviter les nan
vp=vb.*exp(a{5}*y(nn));vp(f)=0;
else;              % texture droite
vh([f;m+ff])=0;vb([m+f;ff])=0;
vm=vh.*exp([-a{5};a{5}]*(y(nn)-h));vm([f;m+ff])=0;% pour eviter les nan
vp=vb.*exp([-a{5};a{5}]*y(nn));vp([m+f;ff])=0;
end;               % texture inclinee  ?
vm=vm+vp;
f=find(abs(vm)<eps*max(abs(vm)));if length(f)>m;vm(f)=0;vm=sparse(vm);end;
% retour dans la base de Fourier
if a{end}.type==5; % texture inclinee
v=a{1}*vm;v=exp(-i*y(nn)*a{3}*[init{2},init{2}].').*v;
else;              % texture droite
v=[a{3}*(vm(1:m)+vm(m+1:2*m));-a{1}*(a{5}.*(vm(1:m)-vm(m+1:2*m)))];
end;               % texture inclinee  ?

end;             %<--
end;            % .. forme simplifiee ou generale ?

if a{end}.type==5; % texture inclinee
u=exp(i*y(nn)*a{3}*[init{2},init{2}].');uu=exp(-i*y(nn)*a{3}*init{2}.');% translation due à l'inclinaison 
e(nn,:,3)=-i*retff((uu.*(a3*(v.*u))).',betaa);   % Ht
e(nn,:,4)=-i*retff((uu.*(a1*(v.*u))).',betaa);% Bn
mu=rettestobjet(init,a,-1,0,x-y(nn)*a{3},2:4).';
teta=atan(a{3});ct=cos(teta);st=sin(teta);
prv=mu(2,:)*ct^2+mu(1,:)*st^2-(2*st*ct)*mu(3,:);
[e(nn,:,2),e(nn,:,3)]=deal(((st*mu(1,:)-ct*mu(3,:)).*e(nn,:,3)+ct*e(nn,:,4))./prv,...
	((ct*mu(2,:)-st*mu(3,:)).*e(nn,:,3)-st*e(nn,:,4))./prv);%Hx,Hy
e(nn,:,4)=mu(2,:).*e(nn,:,2)+mu(3,:).*e(nn,:,3);% Bx
% e(nn,:,4)=mu(3,:).*e(nn,:,2)+mu(1,:).*e(nn,:,3);% By
e(nn,:,1)=retff(v(1:n).',betaa);%Ez

else;              % texture droite
if ~isempty(a3);e(nn,:,3)=retff((a3*v(1:m)).',betaa);end;       % hy
if ~isempty(a1);e(nn,:,4)=retff((-i*a1*v(m+1:2*m)).',betaa);end;% bx
if ~isempty(si);v=[si{1}*v(1:m);si{1}*v(m+1:2*m)];end;
e(nn,:,1)=retff(v(1:n).',betaa);%ez
e(nn,:,2)=retff((-i*v(n+1:2*n)).',betaa);%hx
if ~isempty(a1);e(nn,:,2)=e(nn,:,4)./(muy.');end;% 'meilleur' calcul de hx
end;                % texture inclinee  ?

end;  % <++++++++++++++ nn
end; % < xxxxxxxxxxxxxxxxxxxx  1D 2D ?
end; %  < $$$$$$$$$$$$$$$$$
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
function e=ret2champ(init,a,sh,sb,inc,x,y,z,h,betaa);
% function e=ret2champ(init,a,sh,sb,inc,x,y,z,h,betaa);
% calcul des champs ex=e(:,:,:,1),ey=e(:,:,2),ez=e(:,:,:,3),hx=e(:,:,:,4),hy=e(:,:,:,5),hz=e(:,:,:,6)
% en x,y,z dans un milieu constant de hauteur h
% inc:vecteur incident du systeme total 
% sh :matrice s du haut de la couche au haut du systeme
% sb :matrice s du bas du systeme au bas de la couche
% le descripteur de texture a  decrivant la couche a ete calculee auparavant avec  retcouche  (parametre c=1)
%  si a a ete calcule sans parametre(ou c=0) le calcul est fait de maniere approchee pour les champs discontinus
%  alors ez et hz ne sont pas calcules(ez=hz=0)
%  betaa sert a calculer les champs par synthese de fourier-floquet 
%

si=init{8};n=size(init{2},2);m=init{1};
%p=a{1};pp=a{2};q=a{3};qq=a{4};d=a{5};
if length(a)>12;ez=retio(a{6});hz=a{7};fex=a{8};fhx=a{9};fey=a{10};fhy=a{11};
xx=a{12}{1};yy=a{12}{2};uu=a{12}{3};pas=a{13};else;[ez,hz,fex,fhx,fey,fhy,xx,yy,uu,pas]=deal([]);end;
io=ischar(ez);

e=zeros(size(z,2),size(x,2),size(y,2),6);

if ~isempty(fey);% calcul de eps et mu (fey marche aussi pour les textures inclinees)
ee=zeros(1,size(x,2),size(y,2),6);
xxx=mod(x/pas(1),1);yyy=mod(y/pas(2),1);
mx=size(xx,2);my=size(yy,2);
for ii=1:mx;if ii==1 x0=0;else x0=xx(ii-1);end;
fx=find((xxx>=x0)&(xxx<=xx(ii)));    
for jj=1:my;if jj==1 y0=0;else y0=yy(jj-1);end;
fy=find((yyy>=y0)&(yyy<=yy(jj)));
for kk=1:6;ee(1,fx,fy,kk)=uu(ii,jj,kk);end;
end;end;% < ii jj

end;% calcul de eps et mu

vh=[];
if a{end}.type==5;decale=1;fey=retio(fey);fhy=retio(fhy);ez=retio(ez);hz=retio(hz);end;

for nn=1:size(z,2);% <*********************************** boucle sur z  (nn)
if isempty(sh); % ....forme simplifiee de retchamp
v=z(:,nn);z(nn)=0; % z sert ensuite comme cote,mais on n'ecrase pas les valeurs suivantes de z(:,nn)   
else;              % .... forme generale
	
if (size(z,2)<4)|(min(abs(a{5}))<eps);  %<-- forme non acceleree  
if h==0;ssb=sb;ssh=sh; % epaisseur nulle    
else;
ssb=retss(retc(a,z(nn)),sb);
if a{end}.type==5; % texture inclinee
ssh=retss(sh,rettr(init,retc(a,h-z(nn)),a{3}*z(nn)));
decale=retdiag(exp(-i*z(nn)*(a{3}*init{2})));
else;
ssh=retss(sh,retc(a,h-z(nn)));
end;
end;
v=retsc(ssh,ssb,inc,m);

if a{end}.type==5; % texture inclinee on ramene à l'origine pour ensuite decaler
v=exp(((i*z(nn)*a{3})*a{9})).'.*v;
end;

else;           %<-- forme acceleree
if isempty(vh);% initialisation 
sprv=retc(a,h);
vh=retsc(sh,retss(sprv,sb),inc,m);
vb=retsc(retss(sh,sprv),sb,inc,m);
% passage dans la base modale
if a{end}.type==5; % texture inclinee
vh=(a{2}*(vh.*exp(i*h*(a{3}*a{9}).')));vb=a{2}*vb;   
else;              % texture droite
vh=.5*[a{4}*vh(1:m);(1./a{5}).*(a{2}*vh(m+1:2*m))];vh=[vh(1:m)-vh(m+1:2*m);vh(1:m)+vh(m+1:2*m)];% passage dans la base de Fourier
vb=.5*[a{4}*vb(1:m);(1./a{5}).*(a{2}*vb(m+1:2*m))];vb=[vb(1:m)-vb(m+1:2*m);vb(1:m)+vb(m+1:2*m)];% passage dans la base de Fourier
end;               % texture inclinee  ?
clear  sh_prv  sprv
end;           % fin initialisation 
% vm=vh.*exp(a{5}.*(z(nn)-h));vp=vb.*exp(-a{5}.*z(nn));
% v=[a{3}*(vm+vp);a{1}*((vm-vp).*a{5})];
[f,ff]=retfind(real(a{5})>0);% cas des vp pathologiques
if a{end}.type==5; % texture inclinee
vh(ff)=0;vb(f)=0;
vm=vh.*exp(a{5}*(z(nn)-h));vm(ff)=0; % pour eviter les nan
vp=vb.*exp(a{5}*z(nn));vp(f)=0;
decale=retdiag(exp(-i*z(nn)*(a{3}*init{2})));
else;              % texture droite
vh([f;m+ff])=0;vb([m+f;ff])=0;
vm=vh.*exp([-a{5};a{5}].*(z(nn)-h));vm([f;m+ff])=0;% pour eviter les nan
vp=vb.*exp([-a{5};a{5}].*z(nn));vp([m+f;ff])=0;
%v=[[a{3},a{3}]*(vm+vp);[a{1},a{1}]*((vm+vp).*[-a{5};a{5}])];
end;               % texture inclinee  ?
vm=vm+vp;
f=find(abs(vm)<eps*max(abs(vm)));if length(f)>m;vm(f)=0;vm=sparse(vm);end;
% retour dans la base de Fourier
if a{end}.type==5; % texture inclinee
v=a{1}*vm;
else;              % texture droite
v=[a{3}*(vm(1:m)+vm(m+1:2*m));-a{1}*(a{5}.*(vm(1:m)-vm(m+1:2*m)))];
end;               % texture inclinee  ?
end;             % <-- forme acceleree ?
end;             % .... forme simplifiee ou generale ?


kx=1:length(x);ky=1:length(y);propre=0;
%champs continus

if  (a{end}.type~=5) & ~isempty(hz);  %calcul propre des champs HZ et EZ 
e(nn,:,:,6)=ret2ff(kx,ky,retio(hz)*v(1:m),betaa);    % HZ
e(nn,:,:,3)=ret2ff(kx,ky,retio(ez)*v(m+1:2*m),betaa);% EZ
end
if ~isempty(si);v=[si{1}*v(1:m);si{3}*v(m+1:2*m)];end;

if ~isempty(fey)|a{end}.type==6;propre=1;  % < ############################# calcul propre des champs discontinus 

switch 	a{end}.type;
	
case 6; % <********************** cylindres Popov
e(nn,:,:,:)=retchamp_popov(init,a,v,x,y,betaa);	
case 5; % <********************** texture inclinee 
% re calcul de ee
offset=z(nn)*a{3};
ep=zeros(1,size(x,2),size(y,2),3,3);mu=zeros(1,size(x,2),size(y,2),3,3);
xxx=mod((x-offset(1))/pas(1),1);yyy=mod((y-offset(2))/pas(2),1);
mx=size(xx,2);my=size(yy,2);
for ii=1:mx;if ii==1 x0=0;else x0=xx(ii-1);end;
fx=find((xxx>=x0)&(xxx<=xx(ii)));    
for jj=1:my;if jj==1 y0=0;else y0=yy(jj-1);end;
fy=find((yyy>=y0)&(yyy<=yy(jj)));
for kk=1:3;ep(1,fx,fy,kk,kk)=uu(ii,jj,kk+3);mu(1,fx,fy,kk,kk)=uu(ii,jj,kk);end;
if size(uu,3)>6; % anisotrope
ep(1,fx,fy,1,2)=uu(ii,jj,10);ep(1,fx,fy,2,1)=uu(ii,jj,10);%      4   10  12
ep(1,fx,fy,2,3)=uu(ii,jj,11);ep(1,fx,fy,3,2)=uu(ii,jj,11);%  ep= 10   5  11
ep(1,fx,fy,1,3)=uu(ii,jj,12);ep(1,fx,fy,3,1)=uu(ii,jj,12);%      12  11   6
mu(1,fx,fy,1,2)=uu(ii,jj,7);mu(1,fx,fy,2,1)=uu(ii,jj,7); %     1 7 9 
mu(1,fx,fy,2,3)=uu(ii,jj,8);mu(1,fx,fy,3,2)=uu(ii,jj,8); % mu= 7 2 8
mu(1,fx,fy,1,3)=uu(ii,jj,9);mu(1,fx,fy,3,1)=uu(ii,jj,9); %     8 9 3 
end;
end;end;% < ii jj
	
mm=length(v)/2;v(mm+1:2*mm)=i*v(mm+1:2*mm);% clonage attention que mm n'est par egal à m si on utilise la symetrie !	
%[Ze,Zh,f_champe,f_champh]=deal(fey,fhy,ez,hz);


XYe={fey(1:n,:)*v;fey(n+1:2*n,:)*v;fey(2*n+1:3*n,:)*v};
XYh={fhy(1:n,:)*v;fhy(n+1:2*n,:)*v;fhy(2*n+1:3*n,:)*v};

if a{end}.sens==1; %  < SENS = 1

for jj=1:my;% < jj    --------
if jj==1 y0=0;else y0=yy(jj-1);end;
f=find((yyy>=y0)&(yyy<=yy(jj)));
if ~isempty(f);
XY={ez{jj}{1,1}*XYe{1}+ez{jj}{1,2}*XYe{2}+ez{jj}{1,3}*XYe{3},...
ez{jj}{2,1}*XYe{1}+ez{jj}{2,2}*XYe{2}+ez{jj}{2,3}*XYe{3},...	
ez{jj}{3,1}*XYe{1}+ez{jj}{3,2}*XYe{2}+ez{jj}{3,3}*XYe{3}};
XY={ret2ff(kx,f,decale*XY{1},betaa);ret2ff(kx,f,decale*XY{2},betaa);ret2ff(kx,f,decale*XY{3},betaa)};
XY=retprod_cell(retinv_cell({reshape(ep(1,:,f,1,1),length(xxx),length(f))*a{8}.ny(1)+reshape(ep(1,:,f,3,1),length(xxx),length(f))*a{8}.ny(3),...
	                         reshape(ep(1,:,f,1,2),length(xxx),length(f))*a{8}.ny(1)+reshape(ep(1,:,f,3,2),length(xxx),length(f))*a{8}.ny(3),...
	                         reshape(ep(1,:,f,1,3),length(xxx),length(f))*a{8}.ny(1)+reshape(ep(1,:,f,3,3),length(xxx),length(f))*a{8}.ny(3);...
	                         0,1,0;a{8}.n(1),a{8}.n(2),a{8}.n(3)}),XY);
e(nn,:,f,1)=reshape(XY{1},[1,size(XY{1})]);
e(nn,:,f,2)=reshape(XY{2},[1,size(XY{2})]);
e(nn,:,f,3)=reshape(XY{3},[1,size(XY{3})]);
XY={hz{jj}{1,1}*XYh{1}+hz{jj}{1,2}*XYh{2}+hz{jj}{1,3}*XYh{3},...
hz{jj}{2,1}*XYh{1}+hz{jj}{2,2}*XYh{2}+hz{jj}{2,3}*XYh{3},...	
hz{jj}{3,1}*XYh{1}+hz{jj}{3,2}*XYh{2}+hz{jj}{3,3}*XYh{3}};
XY={ret2ff(kx,f,decale*XY{1},betaa);ret2ff(kx,f,decale*XY{2},betaa);ret2ff(kx,f,decale*XY{3},betaa)};
XY=retprod_cell(retinv_cell({reshape(mu(1,:,f,1,1),length(xxx),length(f))*a{8}.ny(1)+reshape(mu(1,:,f,3,1),length(xxx),length(f))*a{8}.ny(3),...
	                         reshape(mu(1,:,f,1,2),length(xxx),length(f))*a{8}.ny(1)+reshape(mu(1,:,f,3,2),length(xxx),length(f))*a{8}.ny(3),...
	                         reshape(mu(1,:,f,1,3),length(xxx),length(f))*a{8}.ny(1)+reshape(mu(1,:,f,3,3),length(xxx),length(f))*a{8}.ny(3);...
	                         0,1,0;a{8}.n(1),a{8}.n(2),a{8}.n(3)}),XY);
e(nn,:,f,4)=-i*reshape(XY{1},[1,size(XY{1})]);% declonnage
e(nn,:,f,5)=-i*reshape(XY{2},[1,size(XY{2})]);
e(nn,:,f,6)=-i*reshape(XY{3},[1,size(XY{3})]);
end;
end      % < jj    --------


else;             %  < SENS = -1

for jj=1:mx;% < jj    --------
if jj==1 x0=0;else x0=xx(jj-1);end;
f=find((xxx>=x0)&(xxx<=xx(jj)));
if ~isempty(f);
XY={ez{jj}{1,1}*XYe{1}+ez{jj}{1,2}*XYe{2}+ez{jj}{1,3}*XYe{3},...
ez{jj}{2,1}*XYe{1}+ez{jj}{2,2}*XYe{2}+ez{jj}{2,3}*XYe{3},...	
ez{jj}{3,1}*XYe{1}+ez{jj}{3,2}*XYe{2}+ez{jj}{3,3}*XYe{3}};
XY={ret2ff(f,ky,decale*XY{1},betaa);ret2ff(f,ky,decale*XY{2},betaa);ret2ff(f,ky,decale*XY{3},betaa)};
XY=retprod_cell(retinv_cell({1,0,0;...
	                         reshape(ep(1,f,:,2,1),length(f),length(yyy))*a{8}.nx(2)+reshape(ep(1,f,:,3,1),length(f),length(yyy))*a{8}.nx(3),...
	                         reshape(ep(1,f,:,2,2),length(f),length(yyy))*a{8}.nx(2)+reshape(ep(1,f,:,3,2),length(f),length(yyy))*a{8}.nx(3),...
	                         reshape(ep(1,f,:,2,3),length(f),length(yyy))*a{8}.nx(2)+reshape(ep(1,f,:,3,3),length(f),length(yyy))*a{8}.nx(3);...
	                         a{8}.n(1),a{8}.n(2),a{8}.n(3)}),XY);
e(nn,f,:,1)=reshape(XY{1},[1,size(XY{1})]);
e(nn,f,:,2)=reshape(XY{2},[1,size(XY{2})]);
e(nn,f,:,3)=reshape(XY{3},[1,size(XY{3})]);
XY={hz{jj}{1,1}*XYh{1}+hz{jj}{1,2}*XYh{2}+hz{jj}{1,3}*XYh{3},...
hz{jj}{2,1}*XYh{1}+hz{jj}{2,2}*XYh{2}+hz{jj}{2,3}*XYh{3},...	
hz{jj}{3,1}*XYh{1}+hz{jj}{3,2}*XYh{2}+hz{jj}{3,3}*XYh{3}};
XY={ret2ff(f,ky,decale*XY{1},betaa);ret2ff(f,ky,decale*XY{2},betaa);ret2ff(f,ky,decale*XY{3},betaa)};
XY=retprod_cell(retinv_cell({1,0,0;...
	                         reshape(mu(1,f,:,2,1),length(f),length(yyy))*a{8}.nx(2)+reshape(mu(1,f,:,3,1),length(f),length(yyy))*a{8}.nx(3),...
	                         reshape(mu(1,f,:,2,2),length(f),length(yyy))*a{8}.nx(2)+reshape(mu(1,f,:,3,2),length(f),length(yyy))*a{8}.nx(3),...
	                         reshape(mu(1,f,:,2,3),length(f),length(yyy))*a{8}.nx(2)+reshape(mu(1,f,:,3,3),length(f),length(yyy))*a{8}.nx(3);...
	                         a{8}.n(1),a{8}.n(2),a{8}.n(3)}),XY);
e(nn,f,:,4)=-i*reshape(XY{1},[1,size(XY{1})]);% declonnage
e(nn,f,:,5)=-i*reshape(XY{2},[1,size(XY{2})]);
e(nn,f,:,6)=-i*reshape(XY{3},[1,size(XY{3})]);
end;
end      % < jj    --------
end;              %  < SENS ?


otherwise;              % <********************** texture droite
%champs continus en x
%           EY
ff=fey;propre=(length(ff)>=mx);
if propre;
for ii=1:mx;
if ii==1 x0=0;else x0=xx(ii-1);end;
fx=find((xxx>=x0)&(xxx<=xx(ii)));    
if ~isempty(fx);fff=retio(ff{ii});e(nn,fx,:,2)=ret2ff(fx,ky,fff*v(n+1:2*n),betaa);end
end;
e(nn,:,:,2)=e(nn,:,:,2)./ee(1,:,:,5);
end;
%           HY
if propre;ff=fhy;propre=propre&(length(ff)>=mx);end;
if propre;
for ii=1:mx;
if ii==1 x0=0;else x0=xx(ii-1);end;
fx=find((xxx>=x0)&(xxx<=xx(ii)));    
if ~isempty(fx);fff=retio(ff{ii});e(nn,fx,:,5)=ret2ff(fx,ky,fff*v(3*n+1:4*n),betaa);end
end;
e(nn,:,:,5)=e(nn,:,:,5)./ee(1,:,:,2);
end;
%champs continus en y

%            EX
if propre;ff=fex;propre=propre&(length(ff)>=my);end;
if propre;
for jj=1:my;
if jj==1 y0=0;else y0=yy(jj-1);end;
fy=find((yyy>=y0)&(yyy<=yy(jj)));
if ~isempty(fy);fff=retio(ff{jj});e(nn,:,fy,1)=ret2ff(kx,fy,fff*v(1:n),betaa);end
end
e(nn,:,:,1)=e(nn,:,:,1)./ee(1,:,:,4);
end;
%            HX
if propre;ff=fhx;propre=propre&(length(ff)>=my);end;
if propre;
for jj=1:my;
if jj==1 y0=0;else y0=yy(jj-1);end;
fy=find((yyy>=y0)&(yyy<=yy(jj)));
if ~isempty(fy);fff=retio(ff{jj});e(nn,:,fy,4)=ret2ff(kx,fy,fff*v(2*n+1:3*n),betaa);end
end
e(nn,:,:,4)=e(nn,:,:,4)./ee(1,:,:,1);
end;

clear ff;
end;               % <********************** texture inclinee  ?


end; % < #############################  fin calcul propre

if ~propre; %calcul approché des champs discontinus
e(nn,:,:,2)=ret2ff(kx,ky,v(n+1:2*n),betaa);  %EY
e(nn,:,:,5)=ret2ff(kx,ky,v(3*n+1:4*n),betaa);%HY
e(nn,:,:,1)=ret2ff(kx,ky,v(1:n),betaa);      %EX
e(nn,:,:,4)=ret2ff(kx,ky,v(2*n+1:3*n),betaa);%HX
end    

end;% <*********************************** fin boucle sur z  (nn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function e=retmchamp(init,a,sh,sb,inc,x,y,z,h);
% CALCUL DES CHAMPS dans les metaux infiniment conducteurs dans une couche de hauteur h
%
% cas 1D 
%  function e=retmchamp(init,a,sh,sb,inc,x,y,h);
%   calcul des champs en y,x (vecteurs ligne) 
%  dans le cas de polarisation E// (pol=0)  
%   ez=e(:,:,1),hx=e(:,:,2),hy=e(:,:,3),bx=e(:,:,4)
%  dans le cas de polarisation H// (pol=2)  
%   hz=e(:,:,1),ex=-e(:,:,2),ey=-e(:,:,3),dx=-e(:,:,4) 
%
% cas 2D :
% function e=retmchamp(init,a,sh,sb,inc,x,y,z,h);
% calcul des champs en z,x,y (vecteurs ligne) 
% ex=e(:,:,1),ey=e(:,:,2),ez=e(:,:,3),hx=e(:,:,4),hy=e(:,:,5),hz=e(:,:,6) 
%
% le descripteur de texture a decrivant la couche a ete calculee auparavant avec  retcouche  (parametre c=1)
% sh :matrice s du haut de la couche au haut du systeme
%    si sh est un nombre c'est le numero d'un seul mode a tracer (appel depuis rettmode)
% sb :matrice s du bas du systeme au bas de la couche                                      
% inc:vecteur incident du systeme total 
% y:vecteur des hauteurs dans la couche consideree(y=0:bas de la couche  y=h:haut de la couche)

d=init{end}.d;dim=init{end}.dim;beta0=init{end}.beta0;if dim==1;h=z;z=y;y=0;end;% dim=1;1D  2:2D

dd=a{3};m=a{4};n=a{5};
chx=a{6}{1};
chdx=a{6}{2};
chalp=a{6}{3};
chtp=a{6}{4};
cha=a{6}{5};

mx=length(x);my=length(y);mz=length(z);
if isempty(chx); %metal massif
if dim==1;e=zeros(mz,mx,4);else;e=zeros(mz,mx,my,6);end
return;end;

% calcul des amplitudes des modes montants et descendants
         % importance de la troncature pour la stabilité numerique
if iscell(sh);
sbb=retss(retb(init,a,1.e-6,0),rettronc(sb,0,[],-1));shh=retss(rettronc(sh,0,[],1),retb(init,a,-1.e-6,0));
ddd=exp(-dd*h);
shh{1}(:,1:m)=shh{1}(:,1:m)*diag(ddd);
nb3=sbb{5};sbb{2}(:,nb3+1:nb3+m)=sbb{2}(:,nb3+1:nb3+m)*diag(ddd);
v=retsc(shh,sbb,inc,m);
iim=1:m;else;v=zeros(2*m,1);v(sh)=1;iim=sh;end;%pour accelerer retmode pour les metaux

if dim==1;e=zeros(mz,mx*4);prov0=zeros(1,4);else;e=zeros(mz,mx*my*6);prov0=zeros(4,2);end;
for im=iim;
if dim==2; % 2D
prov=[chx(im,:);chdx(im,:);chalp(im,:);chtp(im,:)];% on ne fait ce calcul qu'une fois
if ~all(prov(:,1)==prov0(:,1));d0=d(1);xy=x;x0=chx(im,1);dx=chdx(im,1);alp=chalp(im,1);tp=chtp(im,1);[sx,cx]=calsc(d0,xy,x0,dx,alp,tp,beta0(1));end;
if ~all(prov(:,2)==prov0(:,2));d0=d(2);xy=y;x0=chx(im,2);dx=chdx(im,2);alp=chalp(im,2);tp=chtp(im,2);[sy,cy]=calsc(d0,xy,x0,dx,alp,tp,beta0(2));end;
cs=reshape(cx.'*sy,1,mx*my);
sc=reshape(sx.'*cy,1,mx*my);
ss=reshape(sx.'*sy,1,mx*my);
cc=reshape(cx.'*cy,1,mx*my);
prov0=prov;
ep=[cs*cha(im,1,1),sc*cha(im,2,1),ss*cha(im,3,1),sc*cha(im,4,1),cs*cha(im,5,1),cc*cha(im,6,1)];
em=[cs*cha(im,1,2),sc*cha(im,2,2),ss*cha(im,3,2),sc*cha(im,4,2),cs*cha(im,5,2),cc*cha(im,6,2)];
else;% 1 D
prov=[chx(im),chdx(im),chalp(im),chtp(im)];if ~all(prov==prov0);% on ne fait ce calcul qu'une fois
d0=d;xy=x;x0=chx(im);dx=chdx(im);alp=chalp(im);tp=chtp(im);[s,c]=calsc(d0,xy,x0,dx,alp,tp,beta0);
prov0=prov;end;
% ep=[c*cha(im,1,1),c*cha(im,2,1),s*cha(im,3,1),c*cha(im,4,1)];% changement de signe -tp a confirmer ...
% em=[c*cha(im,1,2),c*cha(im,2,2),s*cha(im,3,2),c*cha(im,4,2)];
ep=[c*cha(im,1,1),c*cha(im,2,1),-tp*s*cha(im,3,1),c*cha(im,4,1)];
em=[c*cha(im,1,2),c*cha(im,2,2),-tp*s*cha(im,3,2),c*cha(im,4,2)];
end;
e=e+(v(im)*exp(-z.'*dd(im))*ep+v(m+im)*exp((z.'-h)*dd(im))*em); 

end; % boucle sur im
if dim==1;e=reshape(e,mz,mx,4);else;e=reshape(e,mz,mx,my,6);end;


function [s,c]=calsc(d0,xy,x0,dx,alp,tp,beta0);% calcul de fonctions trigonometriques utilisees par retmchamp
fac=exp(-i*beta0*d0*round((x0-xy)/d0));% dephasage de periode en periode ...
if abs(tp)==1;xy=mod(xy+d0/2-x0,d0)-d0/2;f=find(abs(xy)<=dx/2);c=zeros(size(xy));s=c;end;
if tp==1;c(f)=sqrt(2/dx)*cos(alp*(xy(f)+dx/2));s(f)=sqrt(2/dx)*sin(alp*(xy(f)+dx/2));if alp==0;c=c/sqrt(2);end;end;
if tp==2;c=exp(i*alp*xy)/sqrt(d0);s=-i*c;end;
if tp==-1;s(f)=sqrt(2/dx)*cos(alp*(xy(f)+dx/2));c(f)=sqrt(2/dx)*sin(alp*(xy(f)+dx/2));if alp==0;s=s/sqrt(2);end;end;
if tp==-2;s=exp(i*alp*xy)/sqrt(d0);c=-i*s;end;
s=s.*fac;c=c.*fac;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=retchamp0(init,a,inc,sens,x);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% FORME SIMPLIFIEE pour une seule couche en un seul plan 
%
% function e=retchamp(init,a,inc,x,sens);
n=init{1};m=size(init{2},2);

if sens~=0;
switch  a{end}.type   
case 3;inc=a{3}*inc;  % modes de bloch non symetriques
case 5;inc=a{1}*inc;% texture inclinee
otherwise;inc=[a{3}*(inc(1:n,:)+inc(n+1:2*n,:));-a{1}*(diag(a{5})*(inc(1:n,:)-inc(n+1:2*n,:)))];
end;
end;
% inc contient maintenant les composantes du champ dans la base de Fourier symetrique

if isempty(x); % e base de Fourier vraie
if init{end}.dim==1; % 1D
si=init{3};
a3=a{6};a1=a{7};
e=zeros(m,size(inc,2),4);
if ~isempty(a3);e(:,:,3)=a3*inc(1:n,:);end;%hy
if ~isempty(a1);e(:,:,4)=-i*a1*inc(n+1:2*n,:);end;%bx
if ~isempty(si);
e(:,:,1)=si{1}*inc(1:n,:);%ez
e(:,:,2)=-i*si{1}*inc(n+1:2*n,:);%hx
else;
e(:,:,1)=inc(1:n,:);%ez
e(:,:,2)=-i*inc(n+1:2*n,:);%hx
end;

else;  % 2D
si=init{8};
ez=a{6};hz=a{7};
e=zeros(m,size(inc,2),6);
if isempty(si);
e(:,:,1)=inc(1:n/2,:);          %EX
e(:,:,2)=inc(n/2+1:n,:);        %EY
e(:,:,4)=inc(n+1:3*n/2,:);      %HX
e(:,:,5)=inc(3*n/2+1:2*n,:);    %HY
else;
e(:,:,1)=si{1}(1:m,:)*inc(1:n,:);        %EX
e(:,:,2)=si{1}(m+1:2*m,:)*inc(1:n,:);    %EY
e(:,:,4)=si{3}(1:m,:)*inc(n+1:2*n,:);    %HX
e(:,:,5)=si{3}(m+1:2*m,:)*inc(n+1:2*n,:);%HY
end;
if ~isempty(hz);  %calcul propre des champs HZ et EZ 
e(:,:,3)=ez*inc(n+1:2*n,:);%EZ
e(:,:,6)=hz*inc(1:n,:);    %HZ
end;
end;
else; % champs en x   
betaa=calbetaa(init,x,0);e=retchamp1(init,a,[],[],[],x,inc,0,betaa);betaa=retio(betaa,-2); 
end;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ff=retff(f,betaa);  %synthese de fourier-floquet en x f et ff vecteurs ligne (betaa calcule par calbetaa)
ff=f*retio(betaa);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ff=ret2ff(kx,ky,f,betaa);
% synthese de fourier-floquet en x,y (betaa calcule par calbetaa)
% kx et ky sont des tableaux de numeros de valeurs de x
betaa=retio(betaa);
mx=betaa{3};my=betaa{4};
ff=repmat(betaa{1}(kx,:),1,my)*(retdiag(f)*(reshape(repmat(betaa{2}(ky,:),mx,1),length(ky),mx*my)).'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function betaa=calbetaa(init,x,apod);% calcul de betaa qui sert à calculer les champs par synthèse de Fourier-Floquet  
if init{end}.genre==1;% Bessel cylindres Popov
N=init{end}.nhanckel;L=init{2};k=init{3};wk=init{4};Pml=init{10};
K=retdiag(k);W=retdiag(wk);
[X,Y]=ndgrid(x{1},x{2});R=sqrt(X.^2+Y.^2);R=R(:);
[R,prv,numR]=retelimine(R,1.e-13);% pour gain de temps
if ~isempty(Pml);% Pml reelles
fac_r=ones(size(R));f=find(R>0);
fac_r(f)=retinterp_popov(R(f),Pml,2)./R(f);R=R.*fac_r;% R numerique
else;fac_r=1;end;

Teta=atan2(Y,X);Teta=Teta(:);sinTeta=sin(Teta);cosTeta=cos(Teta);
kR=R*k;F=exp(i*L*Teta);
JL=besselj(L,kR)*K*W;JLp1=besselj(L+1,kR)*K*W;JLm1=besselj(L-1,kR)*K*W;

% if apod(1)~=0;apod=retchamp([apod(1),2*N-1]);apod=retdiag(apod(N:2*N-1));JL=JL*apod;JLp1=JLp1*apod;JLm1=JLm1*apod;end;% Modif 2010
if apod(1)~=0;n_apod=200;apod=retdiag(interp1(-n_apod:n_apod,retchamp([apod(1),2*n_apod+1]),k*(n_apod-1)/max(k)));JL=JL*apod;JLp1=JLp1*apod;JLm1=JLm1*apod;end;
betaa={R,numR,sinTeta,cosTeta,F,JL,JLp1,JLm1,fac_r};

else;% Fourier ou cylindrique radial

	
beta=init{2};

if init{end}.dim==1; %1D
betaa=exp(i*beta.'*x);
if apod(1)~=0;betaa=retapod(betaa,apod(1));end;  % apodisation
else;  %2D
mx=init{6};my=init{7};
y=x{2};x=x{1};
betaa=cell(4,1);betaa{3}=mx;betaa{4}=my;
betaa{1}=exp(i*x.'*beta(1,1:mx));
betaa{2}=exp(i*y.'*beta(2,1:mx:mx*my));
if apod(1)~=0;betaa{1}=retapod(betaa{1}.',apod(1)).';end;if apod(end)~=0;betaa{2}=retapod(betaa{2}.',apod(end)).';end;  % apodisation
end;
% betaa=retio(betaa,1);% supression de l'ecriture sur le disque 9_2010 (perte de temps considerable)
end;   % Bessel ou Fourier

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=retclonage(e,clonage,und);% clonage  
if clonage;
if und;e(:,:,2:4)=i*e(:,:,2:4);
else;e(:,:,:,4:6)=i*e(:,:,:,4:6);
end;    
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%                  cas 0 D 'rapide'                      %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [e,y,w,o]=retchamp_0D(init,tab,Th,Tb,inc,npts);if isempty(tab);tab=zeros(0,2);end;
%[rh,th,rb,tb,T,Th,Tb,E,Y,W,O]=retabeles(pol,n,h,beta,k0,teta,parm);
if abs(inc(1))~=0; % incident du bas
if iscell(npts);npts{1}=sum(tab(:,1))-npts{1}(end:-1:1);end;% npts{1}= points en entrée
[e,y,w,o]=retchamp_0D(init,flipud(tab),retrenverse(Tb,3),retrenverse(Th,2),[0,inc(1)],npts);
e0=e(end:-1:1,:,:,:,:);e0(:,:,2,:,:)=-e0(:,:,2,:,:);y=sum(tab(:,1))-y(end:-1:1);w=w(end:-1:1);o=o(end:-1:1);
else;e0=[];
end;
if abs(inc(2))~=0; % incident du haut
[prv1,prv2,prv3,prv4,T,prv5,prv6,E,y,w,o]=retabeles(init{1},tab(:,2),tab(:,1),init{2},init{3},[],0,npts);
% calcul du champ en bas
Th=retss(Th,T);Tb=retsp(Tb,-1);
Th(1,:)=Tb(1,:);
Th=retsp(Th,-1);

for ii=1:4;Th{ii}(isnan(Th{ii}))=0;Tb{ii}(isnan(Tb{ii}))=0;end;


Eb=zeros([size(T{1,1}),2,2]);
% Eb(:,:,1)=Th{1,1}*inc(1)+Th{1,2}*inc(2);% E bas
% Eb(:,:,2)=Th{2,1}*inc(1)+Th{2,2}*inc(2);% H bas
Eb(:,:,1)=Th{1,2}*inc(2);% E bas
Eb(:,:,2)=Th{2,2}*inc(2);% H bas
sz=size(E);
E=reshape(E,sz(1)*sz(2),sz(3),2,2);Eb=reshape(Eb,sz(1)*sz(2),2,2);
e=zeros(sz(1)*sz(2),sz(3),3);
beta=repmat(init{2},length(init{3}),1);
e(:,:,1)=E(:,:,1,1).*repmat(Eb(:,1),1,sz(3))+E(:,:,1,2).*repmat(Eb(:,2),1,sz(3));
e(:,:,2)=-i*(E(:,:,2,1).*repmat(Eb(:,1),1,sz(3))+E(:,:,2,2).*repmat(Eb(:,2),1,sz(3)));
e(:,:,3)=-e(:,:,1).*(beta(:)*(1./((o.').^init{1})));

k0=repmat(sqrt(init{3}),1,length(init{2}));
e(:,:,1)=retdiag(k0)*e(:,:,1);e(:,:,2)=retdiag(1./k0)*e(:,:,2);e(:,:,3)=retdiag(1./k0)*e(:,:,3); % normalisation a flux de poynting=1
e=reshape(e,sz(1),sz(2),sz(3),1,3);
e=permute(e,[3,4,5,1,2]);% plus facile a utiliser comme cela quand on a un seul k0 et un seul beta 

else;% pas incident du haut
e=e0;return;
end; % incident du haut ?


if ~isempty(e0);e=e+e0;end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%                  CYLINDRES POPOV                       %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=retchamp_popov(init,a,p,x,y,betaa);
N=init{end}.nhanckel;
L=init{2};k=init{3};wk=init{4};r=a{7}{1};ep=a{7}{2};
[Ez,epErp,epErm,Hz,muHrp,muHrm,Psim,Psip]=deal(a{6}{:});Ez=Ez*diag(k);Hz=Hz*diag(k);

betaa=retio(betaa);[R,numR,sinTeta,cosTeta,F,JL,JLp1,JLm1,fac_r]=deal(betaa{:});clear betaa; 

r=[inf,r,0];
Ep=zeros(numel(R),1);Mu=zeros(numel(R),1);
for ii=1:length(r)-1;jj=find((R>=r(ii+1)) & (R<r(ii)));Ep(jj)=ep(4,ii);Mu(jj)=ep(1,ii);end;
Ep_0=ep(4,end);Mu_0=ep(1,end);
[e,o]=deal(zeros(numel(R),6));
for ii=1:3;o(:,ii)=Mu;o(:,ii+3)=Ep;end;
ep=p(1:N);
em=p(N+1:2*N);
hp=p(2*N+1:3*N);
hm=p(3*N+1:4*N);
% % 
% figure;
% kkk=init{3}.';kkk=1;
% subplot(2,2,1);prv=hp.*kkk;texte=('hp');plot(init{3},real(prv),'.-k',init{3},imag(prv),'.-r');grid;title(texte);
% subplot(2,2,2);prv=hm.*kkk;texte=('hm');plot(init{3},real(prv),'.-k',init{3},imag(prv),'.-r');grid;title(texte);
% subplot(2,2,3);prv=ep.*kkk;texte=('ep');plot(init{3},real(prv),'.-k',init{3},imag(prv),'.-r');grid;title(texte);
% subplot(2,2,4);prv=em.*kkk;texte=('em');plot(init{3},real(prv),'.-k',init{3},imag(prv),'.-r');grid;title(texte);

% figure;
% kkk=init{3}.';kkk=1;
% subplot(2,2,1);prv=em.*kkk;texte=('em');plot(init{3},real(prv),'.-k',init{3},imag(prv),'.-r');grid;title(texte);
% subplot(2,2,2);prv=ep.*kkk;texte=('ep');plot(init{3},real(prv),'.-k',init{3},imag(prv),'.-r');grid;title(texte);
% subplot(2,2,3);prv=(epErm*(em+Psip*ep)).*kkk;texte=('(em+Psip*ep)');plot(init{3},real(prv),'.-k',init{3},imag(prv),'.-r');grid;title(texte);
% subplot(2,2,4);prv=(epErp*(ep+Psim*em)).*kkk;texte=('ep+Psim*em');plot(init{3},real(prv),'.-k',init{3},imag(prv),'.-r');grid;title(texte);


if L>=0;% E_r
e(:,1)=(-i./Ep).*fac_r.*(JLm1*epErm*(em+Psip*ep));
else;
e(:,1)=(i./Ep).*fac_r.*(JLp1*epErp*(ep+Psim*em));
end;

e(:,2)=fac_r.*(JLp1*ep+JLm1*em);        % E_teta
e(:,3)=JL*Ez*(hp-hm);                   % E_z 

% E_z
if L>=0;% H_r
e(:,4)=(-i./Mu).*fac_r.*(JLm1*muHrm*(hm+Psip*hp));
else;
e(:,4)=(i./Mu).*fac_r.*(JLp1*muHrp*(hp+Psim*hm));
end;


e(:,5)=fac_r.*(JLp1*hp+JLm1*hm);      % H_teta
e(:,6)=JL*Hz*(ep-em);                 % H_z

% 
% vraies valeurs de R et phase en angle
switch init{end}.sym*(L~=0);
case 0;FF=F;
case 1;FF=real(F);F=i*imag(F);		
case -1;FF=i*imag(F);F=real(F);		
end;	

e=e(numR,:);%o=o(numR,:);
e(:,1)=FF.*e(:,1);e(:,2)=F.*e(:,2);e(:,3)=FF.*e(:,3);
e(:,4)=F.*e(:,4);e(:,5)=FF.*e(:,5);e(:,6)=F.*e(:,6);
% retour aux coordonnees cartesiennes
% 
[e(:,1),e(:,2)]=deal(cosTeta.*e(:,1)-sinTeta.*e(:,2),sinTeta.*e(:,1)+cosTeta.*e(:,2));% Ex Ey
[e(:,4),e(:,5)]=deal(cosTeta.*e(:,4)-sinTeta.*e(:,5),sinTeta.*e(:,4)+cosTeta.*e(:,5));% Hx Hy

%o=reshape(o,[1,length(x),length(y),6]);
e=reshape(e,[1,length(x),length(y),6]);
e(:,:,:,4:6)=-i*e(:,:,:,4:6);% declonage



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%                  ELEMENTS FINIS  1 D                   %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [e,o]=retelf_champ(init,a_elf,sh,sb,inc,xx,yy,s_elf,calo);
if init{end}.dim==2;[e,o]=retelf_champ_2D(init,a_elf,sh,sb,inc,xx,yy,s_elf,calo);return;end;

%a_elf={a,aa,mh,mb,m,me,Hx,Hy,Bx,grad_grad,x,maille,indices,ep};
%s_elf={s,Hh,Hb,Eh,Eb};

n=init{1};d=init{end}.d;beta0=init{end}.beta0;

a=a_elf{1};mh=a_elf{3};mb=a_elf{4};m=a_elf{5};me=a_elf{6};
Hx=a_elf{7};Hy=a_elf{8};Bx=a_elf{9};grad_grad=a_elf{10};
x=a_elf{11};maille=a_elf{12};indices=a_elf{13};ep=a_elf{14};
s=s_elf{1};Hh=s_elf{2};Hb=s_elf{3};Eh=s_elf{4};Eb=s_elf{5};

clear a_elf s_elf;

% calcul de EE HHx HHy BBx

uv=retsc(retss(sh,s),sb,inc,n);Hb=Hb*uv(n+1:2*n);
uv=retsc(sh,retss(s,sb),inc,n);Hh=Hh*uv(n+1:2*n);
H=sparse(((m-mb-mh+1):m)',ones(mb+mh,1),[Hh;-Hb],m,1,mb+mh);
[l,v,p,q]=lu(a);E=full(p'*(v\(l\(q'*H))));clear l v p q;
Hx=Hx*E;Hy=Hy*E;Bx=Bx*E;

% interpolation des champs en xx yy
xx=xx(:);yy=yy(:);
mx=length(xx);my=length(yy);

prv=mod(xx,d);fac=exp(i*beta0*(xx-prv));xx=prv;
[xx,yy]=meshgrid(xx,yy);xx=xx(:);yy=yy(:);

e=zeros(mx*my,4);o=ones(mx*my,3);


f=find(x(:,1)==0);e(:,1)=griddata([x(:,1);x(f,1)+d],[x(:,2);x(f,2)],[E(:);E(f)*exp(i*beta0*d)],xx,yy);

xxx=zeros(3*me,2);xxx(:,1)=x(abs(maille),1);xxx(:,2)=x(abs(maille),2);
f=find(maille<0);xxx(f,1)=xxx(f,1)+d;
ccc=(xxx(1:me,:)+xxx(me+1:2*me,:)+xxx(2*me+1:3*me,:))/3;

% recherche du triangle correspondant a chaque point 
f=find(maille<0);
mmaille=maille;
[m1,i1,i2]=retelimine(abs(maille(f)));
mmaille(f)=m+i2;
xxx=x(m1,:);xxx(:,1)=xxx(:,1)+d;
xxx=[x;xxx];
prv=tsearch(xxx(:,1),xxx(:,2),mmaille,xx,yy);
%f=find(~isnan(prv));prv=prv(f);
ff=find(isnan(prv));
if ~isempty(ff);% en cas d'erreur on utilise dsearch
centres=(xxx(mmaille(:,1),:)+xxx(mmaille(:,2),:)+xxx(mmaille(:,3),:))/3;
prv(ff)=dsearchn(centres,[xx(ff),yy(ff)]);
end;

e(:,2)=Hx(prv);e(:,3)=Hy(prv);e(:,4)=Bx(prv);
o(:,1)=ep(indices(prv),1);o(:,2)=ep(indices(prv),2);o(:,3)=ep(indices(prv),3);


fac=repmat(fac.',my,1);
for ii=1:4;e(:,ii)=e(:,ii).*fac(:);end;
e=reshape(e,my,mx,4);

o(:,3)=1./o(:,3);
o=reshape(o,my,mx,3);
if calo==i;o=sqrt(o(:,:,1).*o(:,:,3));;else;o=o(:,:,calo);end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%                  ELEMENTS FINIS  2 D                   %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [e,o]=retelf_champ_2D(init,a_elf,sh,sb,inc,xx,zz,s_elf,calo);
%a_elf={a,aa,maillage,Ah1.',Ah2,Ab1.',Ab2,Bh1.',Bh2,Bb1.',Bb2,Ch1.',Ch2,Cb1.',Cb2,si_elf,ssi_elf,si_bord,ssi_bord,alphax,alphay,EH,ep,struct('dim',1,'type',3,'sog',sog,'pol',parm.pol)};
%       1 2     3      4    5    6     7   8     9   10   11   12   13   14   15   16     17       18      19       20     21   22 23
% s_elf={s,Hh,Hb,Eh,Eb};
yy=xx{2};xx=xx{1};

a=retio(a_elf{1});aa=retio(a_elf{2});maillage=a_elf{3};EH=retio(a_elf{22});ep=a_elf{23};
si_elf=a_elf{16};ssi_elf=a_elf{17};si_bord=a_elf{18};ssi_bord=a_elf{19};pol=a_elf{end}.pol;
n=init{1};d=init{end}.d;beta0=init{end}.beta0;
cao=init{end}.cao;
if isempty(cao);cao=[d/2,0,0];end;
xcao=mod(cao(1),d(1));ycao=mod(cao(2),d(2));lcaox=real(cao(3));lcaoy=real(cao(4));
caoix=imag(cao(3));caoiy=imag(cao(4));
fx=ones(size(xx));fy=ones(size(yy));
prv=mod(abs(xx-xcao),d(1));f=find(prv>d(1)/2);prv(f)=d(1)-(prv(f));f=find(prv<lcaox/2);fx(f)=(1-(cos(pi*prv(f)/lcaox)).^2).*(1-(1-1/(1+i*caoix))*(cos(pi*prv(f)/lcaox)).^2);
prv=mod(abs(yy-ycao),d(2));f=find(prv>d(2)/2);prv(f)=d(2)-(prv(f));f=find(prv<lcaoy/2);fy(f)=(1-(cos(pi*prv(f)/lcaoy)).^2).*(1-(1-1/(1+i*caoiy))*(cos(pi*prv(f)/lcaoy)).^2);
%figure;subplot(2,1,1);plot(xx,real(fx),xx,imag(fx));subplot(2,1,2);plot(yy,real(fy),yy,imag(fy));


me=maillage.centre.me;m=maillage.centre.m;mh=maillage.haut.m;mb=maillage.bas.m;
indices=maillage.centre.indices;

s=s_elf{1};Hh=s_elf{2};Hb=s_elf{3};Eh=s_elf{4};Eb=s_elf{5};
% clear a_elf s_elf;

% calcul de EE HHx HHy BBx
uv=retsc(retss(sh,s),sb,inc,n);if pol==0;Hb=Hb*uv(n+1:2*n);else;Hb=Hb*uv(1:n);end;
uv=retsc(sh,retss(s,sb),inc,n);if pol==0;Hh=Hh*uv(n+1:2*n);else;Hh=Hh*uv(1:n);end;

if isempty(si_elf); % pas de symetrie
H=sparse((m-mb-mh+1:m)',ones(mb+mh,1),[Hh;-Hb],m,1,mb+mh);
[l,v,p,q]=lu(a);E=full(p'*(v\(l\(q'*H))));%clear l v p q;
else;

m_centre=maillage.nb_sym.n_sym;m_bord=maillage.nb_sym.nh_sym+maillage.nb_sym.nb_sym;
H=sparse((m_centre+1:m_centre+m_bord)',ones(m_bord,1),[Hh;-Hb],m_centre+m_bord,1,m_bord);
[l,v,p,q]=lu(blkdiag(ssi_elf,ssi_bord)*a*blkdiag(si_elf,si_bord));E=blkdiag(si_elf,si_bord)*(p'*(v\(l\(q'*H))));
end;

% interpolation des champs en xx yy zz
xx=xx(:);yy=yy(:);zz=zz(:);
mx=length(xx);my=length(yy);mz=length(zz);

[zz,xx,yy]=ndgrid(zz,xx,yy);

xx=xx(:);yy=yy(:);zz=zz(:);
prv=mod(xx,d(1));fac=exp(i*beta0(1)*(xx-prv));xx=prv;
prv=mod(yy,d(2));fac=fac.*exp(i*beta0(2)*(yy-prv));yy=prv;

e=zeros(mx*my*mz,6);o=ones(mx*my*mz,6);
%f=find(x(:,1)==0);e(:,1)=griddata([x(:,1);x(f,1)+d],[x(:,2);x(f,2)],[E(:);E(f)*exp(i*beta0*d)],xx,yy);

% recherche du tetraedre correspondant a chaque point 
prv=rettsearchn(maillage.x,maillage.centre.noeuds,[xx,yy,zz]);
% %f=find(~isnan(prv));prv=prv(f);
% ff=find(isnan(prv));ff,xx(ff),yy(ff),zz(ff)
% if ~isempty(ff);% en cas d'erreur on utilise dsearchn
% centres=(maillage.x(maillage.centre.noeuds(:,1),:)+maillage.x(maillage.centre.noeuds(:,2),:)+maillage.x(maillage.centre.noeuds(:,3),:)+maillage.x(maillage.centre.noeuds(:,4),:))/4;
% prv(ff)=dsearchn(centres,[xx(ff),yy(ff),zz(ff)]);
% end;


ex=EH{1}*E;ey=EH{2}*E;ez=EH{3}*E;hx=EH{4}*E;hy=EH{5}*E;hz=EH{6}*E;
e(:,4)=2*hx(prv);e(:,5)=2*hy(prv);e(:,6)=2*hz(prv);
e(:,1)=ex(prv)-hz(prv).*yy+hy(prv).*zz;
e(:,2)=ey(prv)+hz(prv).*xx-hx(prv).*zz;
e(:,3)=ez(prv)+hx(prv).*yy-hy(prv).*xx;
for ii=1:6;o(:,ii)=ep(indices(prv),ii);end;
for ii=1:3;e(:,3+ii)=e(:,3+ii)./o(:,ii);end;% H=B/mu
for ii=1:6;e(:,ii)=e(:,ii).*fac;end;% semi periodicite
e=reshape(e,mz,mx,my,6);o=reshape(o,mz,mx,my,6);
% changement de coordonnees
if lcaox>0;for ii=1:mx;e(:,ii,:,[1,5,6])=fx(ii)*e(:,ii,:,[1,5,6]);end;end;
if lcaoy>0;for ii=1:my;e(:,:,ii,[2,4,6])=fy(ii)*e(:,:,ii,[2,4,6]);end;end;
if pol==2;e=e(:,:,:,[4,5,6,1,2,3]);o=o(:,:,:,[4,5,6,1,2,3]);end;   % changement de polarisation
e(:,:,:,4:6)=-i*e(:,:,:,4:6);        %  declonage (a faire apres le changement de polarisation)

if calo==i;o=sqrt(prod( sqrt(o(:,:,:,[1,2,4,5])),4));else;o=o(:,:,:,calo);end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%                  cylindrique radial                    %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function e=retchamp_cylindrique_radial(init,a,sh,sb,inc,x,y,h,betaa,r1);
m=init{1};L=init{end}.L;si=init{8};
if ~isempty(si);si=si(5:8);end;
r2=r1+h;
% vh=retsc(sh,retss(sprv,sb),inc,m);
% vb=retsc(retss(sh,sprv),sb,inc,m);
r=y(:).'+r1;
nr=length(r);
[Q_HE,Q_HH,Q_EH,Q_EE,D,P_E,P_H,N_EE,N_EH,N_HE,N_HH,u,O_E,O_H,Iep,Imu,Ieep,Immu,K]=deal(a{1:19});
ne=size(Q_EE,1);nh=size(Q_HE,1);m=ne+nh;

if r1==0; % traitement special du centre
% if init{end}.sog==1;sb{1}(:)=0;
% vh=retsc(retss(sh,retb(init,a,-2,r2)),sb,inc,m);% base y j
% else;
% vh=retsc(retss(sh,retb(init,a,-2,r2)),retss(retb(init,a,2,r2),sb),inc,m);    
%     
% end;
%vh=reteval(retss(sh,retb(init,a,-2,r2,[],0)));if ~isempty(vh);vh=vh*inc(1:size(vh,2));vh=[zeros(m,1);vh(end-m+1:end)];else;vh=zeros(2*m,1);end;

vh=reteval(retss(sh,sb))*inc(:);vh=[zeros(m,1);vh(end-m+1:end)];
%vh=retsc(retss(sh,retb(init,a,-1,r2)),retss(retb(init,a,1,r2),sb),inc,m);

vb=zeros(size(vh));
f=1:length(D);ff=[];
else; % pas au centre;
sprv=retc(a,h,r1);
vh=retsc(retss(sh,retb(init,a,-1,r2)),retss(retb(init,a,1,r2),sprv,sb),inc,m);
vb=retsc(retss(sh,sprv,retb(init,a,-1,r1)),retss(retb(init,a,1,r1),sb),inc,m);
%[f,ff]=retfind(real(D)>0);
f=1:length(D);ff=[];%test
vh([f;m+ff])=0;vb([m+f;ff])=0;
end;

if ne==0;H1_E=zeros(0,nr,3);H2_E=zeros(0,nr,3);else;
H1_E=retbessel('h',[L-1,L,L+1],1,retcolonne(1i*D(1:ne)*r),1);H1_E=reshape(H1_E,ne,nr,3);
H2_E=retbessel('j',[L-1,L,L+1],retcolonne(1i*D(1:ne)*r),1);H2_E=reshape(H2_E,ne,nr,3);
end;
if nh==0;H1_H=zeros(0,nr,3);H2_H=zeros(0,nr,3);else;
H1_H=retbessel('h',[L-1,L,L+1],1,retcolonne(1i*D(ne+1:ne+nh)*r),1);H1_H=reshape(H1_H,nh,nr,3);
H2_H=retbessel('j',[L-1,L,L+1],retcolonne(1i*D(ne+1:ne+nh)*r),1);H2_H=reshape(H2_H,nh,nr,3);
end;

% calcul de epz et muz pour calcul propre de Hz et Ez
z=mod(x{1}/init{end}.d(1),1);[eepz,mmuz]=deal(zeros(size(z)));
for ii=1:length(u{1});if ii==1 z0=0;else z0=u{1}(ii-1);end;
fz=find((z>=z0)&(z<=u{1}(ii)));
eepz(fz)=u{3}(ii,1,4);mmuz(fz)=u{3}(ii,1,1);% attention c'est bien la composante x
end;
eepz=retdiag(1./eepz);mmuz=retdiag(1./mmuz);
betaa=retio(betaa);betaa=betaa{1};
e=zeros(nr,length(x{1}),length(x{2}),6);
for ii=1:nr;% Dz,Eteta,Er,Bz,Hteta,Hr
MN=[[zeros(nh,ne),O_H*retdiag(H1_H(:,ii,2)),zeros(nh,ne),O_H*retdiag(H2_H(:,ii,2))];...
[Q_EE*retdiag(H1_E(:,ii,1)-H1_E(:,ii,3)),Q_EH*retdiag(H1_H(:,ii,1)+H1_H(:,ii,3)),Q_EE*retdiag(H2_E(:,ii,1)-H2_E(:,ii,3)),Q_EH*retdiag(H2_H(:,ii,1)+H2_H(:,ii,3))];...
[N_EE*retdiag(H1_E(:,ii,1)+H1_E(:,ii,3)),N_EH*retdiag(H1_H(:,ii,1)-H1_H(:,ii,3)),N_EE*retdiag(H2_E(:,ii,1)+H2_E(:,ii,3)),N_EH*retdiag(H2_H(:,ii,1)-H2_H(:,ii,3))];...
[O_E*retdiag(H1_E(:,ii,2)),zeros(ne,nh),O_E*retdiag(H2_E(:,ii,2)),zeros(ne,nh)];...
[Q_HE*retdiag(H1_E(:,ii,1)+H1_E(:,ii,3)),Q_HH*retdiag(H1_H(:,ii,1)-H1_H(:,ii,3)),Q_HE*retdiag(H2_E(:,ii,1)+H2_E(:,ii,3)),Q_HH*retdiag(H2_H(:,ii,1)-H2_H(:,ii,3))];...
[N_HE*retdiag(H1_E(:,ii,1)-H1_E(:,ii,3)),N_HH*retdiag(H1_H(:,ii,1)+H1_H(:,ii,3)),N_HE*retdiag(H2_E(:,ii,1)-H2_E(:,ii,3)),N_HH*retdiag(H2_H(:,ii,1)+H2_H(:,ii,3))]];
if r1==0; % traitement special du centre
vm=vh.*exp([-D;abs(real(D))]*(r(ii)-r2));vm([f;m+ff])=0;% pour eviter les nan
vp=zeros(size(vm));
% vp=vh.*exp([-D;abs(real(D))]*(r(ii)-r2));vp([m+f;ff])=0;

else;
% Dprv=[-D;D];Dprv(ff)=-abs(real(Dprv(ff)));Dprv(m+f)=abs(real(Dprv(m+f)));
% vm=vh.*exp(Dprv*(r(ii)-r2));vm([f;m+ff])=0;% pour eviter les nan
% vp=vb.*exp(Dprv*(r(ii)-r1));vp([m+f;ff])=0;
vm=vh.*exp([-D;abs(real(D))]*(r(ii)-r2));vm([f;m+ff])=0;% pour eviter les nan
vp=vb.*exp([-D;abs(real(D))]*(r(ii)-r1));vp([m+f;ff])=0;
end;
prv=MN*(vp+vm);

% if ~isempty(si);v=[si{1}*v(1:m);si{1}*v(m+1:2*m)];end;
% e(nn,:,1)=retff(v(1:n).',betaa);%ez
if isempty(si);	prv=reshape(prv,ne,6);
else;prv=[si{3}*prv(1:nh),si{1}*prv(nh+1:ne+nh),si{1}*prv(ne+nh+1:2*ne+nh),si{1}*prv(2*ne+nh+1:3*ne+nh),si{3}*prv(3*ne+nh+1:3*ne+2*nh),si{3}*prv(3*ne+2*nh+1:3*ne+3*nh)];
end;
for jj=1:6;e(ii,:,1,jj)=prv(:,jj).'*betaa.';end; % TF
e(ii,:,1,1)=e(ii,:,1,1)*eepz;
e(ii,:,1,4)=e(ii,:,1,4)*mmuz;
end

e(:,:,:,4:6)=-i*e(:,:,:,4:6);% declonage

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SOURCES MULTIPLES
% [Tab,Sh,Sb,Lh,Lb]=sources_multiples(tab,zs,sh,sb,ah,ab,source,ns);
% modification de tab sh sb pour utiliser les sources multiples
% ENTREES
% tab ancien tab
% zs coordonnees des sources l'origine etant le bas de tab ( z=0 de retchamp)
% sh sb sans les eventuelles sources exterieures au tab
% ah,ab descripteur de texture des milieux invariants en haut et en bas 
% source matrice S de la source à répéter
% ns numero mis dans tab pour designer la source
% SORTIES
% Tab,Sh,Sb :les nouveaus
% Lh,Lb longueurs supplementaires en haut et en bas ( eventuelles sources exterieures au tab)
% 

function [tab,sh,sb,Lh,Lb]=sources_multiples(tab,xs,sh,sb,ah,ab,source,ns);
htab=sum(tab(:,1));
xs=sort(xs);ds=xs(2)-xs(1);
fs_b=find(xs<=0);
fs_h=find(xs>=htab);
fs_tab=find(xs>0 & xs<htab);
if size(tab,2)==3;tabs=[0,1,ns*i];else;tabs=[0,1,ns*i,1];end;
for ii=1:length(fs_tab);
[tab,k]=retchamp(tab,xs(fs_tab(ii)));tab(k,:)=tabs;
end;
if ~isempty(fs_b);
Lb=-xs(fs_b(1));
prv=retss(retc(ab,ds),source);sb=retss(retc(ab,-xs(fs_b(end))),source,retsp(prv,length(fs_b)-1),sb);
else;Lb=0;end;	
if ~isempty(fs_h);
Lh=xs(fs_h(end))-htab;
prv=retss(source,retc(ah,ds));sh=retss(sh,retsp(prv,length(fs_h)-1),source,retc(ah,xs(fs_h(1))-htab));
else;Lh=0;end;	
% FIN DES SOURCES MULTIPLES
