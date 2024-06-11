function [e,Z,o,w,PP,P,p,X,Y,wxy]=res3(X,Y,aa,profil,einc,parm);

%
%                  %%%%%%%%%%%%%%
%                  %   en 2 D   %       
%                  %%%%%%%%%%%%%%
%
%   function [e,Z,o,w]=res3(X,Y,aa,profil,einc,parm);
%
%  calcul  du champ ee
%   X, Y tableaux des coordonnees en X et Y
%
%  calcul des champs EX=e(:,:,:,1),EY=e(:,:,:,2),EZ=e(:,:,:,3),HX=e(:,:,:,4),HY=e(:,:,:,5),HZ=e(:,:,:,6)   en Z,X,Y
%  et de l'indice complexe de l'objet o(:,:,:) aux memes points
%      Z est calcule par le programme (X Y Z unites metriques)
%
%
% calcul des pertes:
%--------------------
% [e,Z,o,w,PP,P,p,XX,YY,wxy]=res3(X,Y,aa,profil,einc,parm);
% X,Y :bornes (ou vecteurs) En fait on utilise uniquement le min et le max de X et Y
% Le programme génére des points de Gauss entre ces bornes en tenant compte des discontinuités.
% par défaut: Gauss de degres 10 
% peut être precisé avec parm.res3.gauss_x et parm.res3.gauss_y (par defaut parm.res3.gauss_y=parm.res3.gauss_x )
% Ces groupes sont repetes de l'ordre de 6 fois par longueur d'onde.
% Les points effectivement calculés sont retournés dans les tableaux XX et YY
% De même le vecteur Z n'est plus constitué de points equidistants par texture, mais de points de Gauss
% dont le degrés est donné par parm.res3.npts (10 par defaut).
% 
% 
% PP pertes par tranches de profil
% P pertes en Z
% p:pertes par point en Z,XX
% w: poids pour l'intégration en Z ( sum(P.*w)=sun(PP) )
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                  %%%%%%%%%%%%%%
%                  % en conique %       
%                  %%%%%%%%%%%%%%
%
%   function [e,Z,o,w]=res3(X,aa,profil,einc,parm);
%
%  calcul  du champ ee
%   X tableau des coordonnees en X 
%
%  calcul des champs EX=e(:,:,,1),EY=e(:,:,2),EZ=e(:,:,3),HX=e(:,:,4),HY=e(:,:,5),HZ=e(:,:,6)   en Z,X
%  et de l'indice complexe de l'objet o(:,:) aux memes points
%      Z est calcule par le programme (X Z unites metriques)
%
% calcul des pertes:
%--------------------
%
% [e,Z,o,w,PP,P,p,XX,wx]=res3(X,aa,profil,einc,parm);
% X :bornes (ou vecteurs) En fait on utilise uniquement le min et le max de X
% Le programme génére des points de Gauss entre ces bornes en tenant compte des discontinuités.
% par défaut: Gauss de degres 10 
% peut être precisé avec parm.res3.gauss_x
% Ces groupes sont repetes de l'ordre de 6 fois par longueur d'onde.
% Les points effectivement calculés sont retournés dans le tableau XX
% De même le vecteur Z n'est plus constitué de points equidistants par texture, mais de points de Gauss
% dont le degrés est donné par parm.res3.npts (10 par defaut).
%
% PP pertes par tranches de profil
% P pertes en Z
% p:pertes par point en Z,XX
% w: poids pour l'intégration en Z ( sum(P.*w)=sun(PP) )
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                  %%%%%%%%%%%%%%
%                  %  en 1 D    %       
%                  %%%%%%%%%%%%%%
%
%   function [e,Z,o,w]=res3(X,aa,profil,einc,parm);
%
%  calcul  du champ ee
%   X tableau des coordonnees en X 
%
%  calcul des champs:
%               en TE:EY=e(:,:,1),HX=e(:,:,2),HZ=e(:,:,3)                en Z,X
%               en TM:HY=e(:,:,1),EX=e(:,:,2),EZ=e(:,:,3)                en Z,X
%  et de l'indice complexe de l'objet o(:,:) aux memes points
%      Z est calcule par le programme (X  Z unites metriques)
%
%
% calcul des pertes:
%--------------------
%
% [e,Z,o,w,PP,P,p,XXwx]=res3(X,aa,profil,einc,parm);
% X :bornes (ou vecteurs) En fait on utilise uniquement le min et le max de X
% Le programme génére des points de Gauss entre ces bornes en tenant compte des discontinuités.
% par défaut: Gauss de degres 10 
% peut être precisé avec parm.res3.gauss_x
% Ces groupes sont repetes de l'ordre de 6 fois par longueur d'onde.
% Les points effectivement calculés sont retournés dans le tableau XX
% De même le vecteur Z n'est plus constitué de points equidistants par texture, mais de points de Gauss
% dont le degrés est donné par parm.res3.npts (10 par defaut).
% 
% 
% PP pertes par tranches de profil
% P pertes en Z
% p:pertes par point en Z,XX
% w: poids pour l'intégration en Z ( sum(P.*w)=sun(PP) )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%
%  aa structure calculees precedemment par res1
%  einc:composantes du champ e incident dans le repere u v (vecteur ligne de size (1,2)) de l'onde incidente 
%  parm.res3.sens    1:incident du haut   -1  du bas (par defaut 1) 
%
%  parm.res3.npts    nombre de points dans les couches pour le calcul 
%   si une seule valeur ou vecteur de longueur differente au nombre de hauteurs dans profil: la meme dans toutes les couches 
%   sinon c'est le vecteur des valeurs dans chaque couche 
%   par defaut 10
% Il est aussi possible (et recommandé ) de definir des points de Gauss répétés un nombre entier de fois 
%     par exemple:parm.res3.npts=[[0,10,0,12];[1,8,1,5]]; signifie:
%     premiére texture (en haut) 0 point
%     seconde texture gauss de degres 10 repeté 8 fois (80 points)
%     troisiéme texture 0 point 
%     quatriéme texture gauss de degres 12 repeté 5 fois (60 points)
%
%  parm.res3.cale  parametres pour le calcul de e (voir retchamp) par defaut [1:6] si [] pas calcul du champ (o seulement) 
%  parm.res3.calo  parametres pour le calcul de o (voir retchamp) par defaut i   (o=indice) 
%
%  parm.res3.trace=1   trace automatique des champs (par defaut 0)
%  parm.res3.champs   parametres pour ce trace des champs (voir retchamp) par defaut [1:6] 
%
%  parm.res3.cale=[] ;signifie que on ne calcule pas le champ alors e contient TAB  (utile pour verifier le profil)
%  parm.res3.caltab=1;   on ne calcule que tab et meme pas l'objet o
%
    


if ~isnumeric(Y);         % entrees simplifiees dans le cas conique [ee,ZZ,o]=res3(X,aa, profil,  einc, parm); 
conique=1;                % au lieu de [ee,ZZ,o]=res3( X, Y,   aa,   profil,  einc,parm); 
if nargin<5;einc=[];end;  % attention au changement de nom des variables.. 
if nargin<4;profil=[1,1];end;
parm=einc;einc=profil;profil=aa;aa=Y;Y=0;
else;conique=0;
if nargin<6;parm=[];end;
if nargin<5;einc=[1,1];end;
end; 

if isempty(parm);parm=res0;end;
parm.res2.calef=0;parm.res2.cals=0;if isempty(parm.res3.cale);parm.res2.cale=0;end;% si on ne calcule pas e on ne calcule pas sh et sb
[ef,TAB,sh,sb]=res2(aa,profil,parm);% calcul de ef (forme simplifiee), TAB sh sb
if parm.res3.caltab==1;e=TAB;Z=[];o=[];w=[];return;end;  % uniquement calcul de  TAB

aa=retio(aa);
a=aa{1};init=aa{2};n=aa{3};UN=aa{4};D=aa{5};
 
tab=TAB;
if any(size(parm.res3.npts)==1);% un seul vecteur
if length(parm.res3.npts)==size(tab,1);npts=parm.res3.npts;else;npts=parm.res3.npts(1)*ones(size(tab,1),1);end;
npts=npts(:);% reshape(npts,prod(size(npts)),1);  % nombre de points dans les couches pour le calcul
else;npts=parm.res3.npts.';
end;
tab=[tab,npts];

tab(:,1)=tab(:,1)/UN;
num_tranches=size(tab,1);f_tranches=find((tab(:,1)~=0)|(tab(:,3)~=0));
tab=tab(f_tranches,:,:);  % elimination des tranches d'epaisseur nulle ou on ne veut pas de point
if isempty(parm.res3.cale);sh=rettronc(rets1(init),[],[],1);sb=rettronc(rets1(init),[],[],-1);inc=[];% on ne calcule que o
else;  % determination de inc fonction de einc champ incident 

if init{end}.dim==1;einc=einc(1);end; % en 1 D...    
if parm.res3.sens==1;  % incident du haut
if isempty(ef.Einch);inc=[];else;inc=ef.Einch\einc.';end;   
f=find(abs(inc)>eps);inc=inc(f);sh=rettronc(sh,f,[],1);sb=rettronc(sb,[],[],-1);
else;  % incident du bas
if isempty(ef.Eincb);inc=[];else;inc=ef.Eincb\einc.';end; 
f=find(abs(inc)>eps);inc=inc(f);sh=rettronc(sh,[],[],1);sb=rettronc(sb,f,[],-1);
end;
end;
cal_pertes=nargout>4;if cal_pertes;parm.res3.gauss=1;end;  
if parm.res3.gauss~=1;tab(:,3)=-tab(:,3);end;  % points regulierement espaces (par defaut) ou methode de gauss (parm.res3.gauss=1)
%  ou methode de gauss (parm.res3.gauss=1) les poids pour l'integration en Z sont dans w
cale=parm.res3.cale;calo=parm.res3.calo;champs=[parm.res3.champs,i];
if parm.res3.trace&~isempty(cale);cale=1:6;end;

if init{end}.dim==2;xy={X/UN,Y/UN};else;xy=X/UN;cale=cale(real(cale)<4);calo=calo(real(calo)<4);champs=champs(real(champs)<4);end;

if cal_pertes;% on calcule les pertes
d=init{end}.d;
if init{end}.dim==1;  % 1D
cale=1:3;calo=1:3;	
% recherche des points de discontinuité 
x_disc=[];
for ii=1:size(tab,1);aprv=retio(aa{1}{tab(ii,2)});% au cas où aa{tab(ii,2)} est sur le disque  
if aprv{end}.type==1;x_disc=[x_disc,aprv{8}{1}];end;% dielectriques
if aprv{end}.type==2;prv=[aprv{7}{1}+.5*aprv{7}{2};aprv{7}{1}-.5*aprv{7}{2}];x_disc=[x_disc,prv(:,1).'];end;% metaux
clear aprv;
end;
x1=min(X/UN);x2=max(X/UN);[xy,wx]=retgauss(x1,x2,parm.res3.gauss_x,1+floor(x2-x1),x_disc,d);
else;% 2D
if isnan(parm.res3.gauss_y);parm.res3.gauss_y=parm.res3.gauss_x;end;    
cale=1:6;calo=4;
% recherche des points de discontinuité 
x_disc=[];y_disc=[];
for ii=1:size(tab,1);
aprv=retio(aa{1}{tab(ii,2)});% au cas où aa{tab(ii,2)} est sur le disque  
if aprv{end}.type==1;x_disc=[x_disc,d(1)*aprv{12}{1}];y_disc=[y_disc,d(2)*aprv{12}{2}];end;% dielectriques
if aprv{end}.type==2;prv=[aprv{7}{1}+.5*aprv{7}{2};aprv{7}{1}-.5*aprv{7}{2}];x_disc=[x_disc,prv(:,1).'];y_disc=[y_disc,prv(:,2).'];end;% metaux
clear aprv;
end
x_disc=retelimine(x_disc);y_disc=retelimine(y_disc);
x1=min(X/UN);x2=max(X/UN);[x,wx]=retgauss(x1,x2,parm.res3.gauss_x,1+floor(x2-x1),x_disc,d(1));
if conique;y=0;wy=1;else;y1=min(Y/UN);y2=max(Y/UN);[y,wy]=retgauss(y1,y2,parm.res3.gauss_y,1+floor(y2-y1),y_disc,d(2));wy=wy*UN;end;
xy={x,y};wxy=wx.'*wy;
end;
end;% calcul des pertes

cale=cale+parm.res3.apod_champ*i;
%[e,z,w,o]=retchamp(init,a,sh,sb,inc,xy,tab,[],cale,1,1,calo,0,nan);if isempty(e);e=[];end;% pb si size(e)=[0,..])
[e,z,w,o]=retchamp(init,a,sh,sb,inc,xy,tab,[],cale,1,1,calo);if isempty(e);e=[];end;% pb si size(e)=[0,..])
if cal_pertes;% on calcule les pertes
if init{end}.dim==1;  % 1D
p=.5*(abs(e(:,:,1)).^2.*imag(o(:,:,1))+abs(e(:,:,2)).^2.*imag(o(:,:,3))+abs(e(:,:,3)).^2.*imag(o(:,:,2)));p(isnan(p))=0;
P=p*wx(:);
o=sqrt(o(:,:,1).*o(:,:,3));% indice
X=xy*UN;Y=wx*UN;
else;% 2D
p=.5*(abs(e(:,:,:,1)).^2+abs(e(:,:,:,2)).^2+abs(e(:,:,:,3)).^2).*imag(o(:,:,:));p(isnan(p))=0;	
P=reshape(p,length(z),[])*wxy(:);
o=sqrt(o);% indice
X=x*UN;Y=y*UN;
end;
P=P.';
w=w*UN; 
% pertes par domaine
PP=zeros(1,num_tranches);
zlim=[flipud(cumsum(flipud(tab(:,1))));0];
for ii=1:length(f_tranches);f=find(z<zlim(ii) & z>zlim(ii+1));PP(f_tranches(ii))=sum(P(f).*w(f));end;
end; % calcul des pertes

Z=z*UN;


% changement des signes  en 1 D 
%   cas TE:  Ez,Hx,Hy    --> Ey ,-Hx,-Hz pour que le repere reste direct 
%   cas TM:  Hz,-Ex,-Ey  --> Hy ,Ex,Ez donc pas de changement de signe
%  attention:apres on n'a plus droit a retpoynting en 1 D ...
if init{end}.dim==1 & aa{end}==0 & size(e,3)>2;e(:,:,2:3)=-e(:,:,2:3);end;
terf=(length(Z)>1)+(length(X)>1)+(length(Y)>1);
if parm.res3.trace;if init{end}.dim==1;rettchamp(e,o,X,Z,aa{end},champs);else;if terf<3;rettchamp(e,o,X,Y,Z,champs);else;disp('you are attempting to plot a field distribution with x,y,z all vectors of length larger than 1 : it is impossible, see the documentation');end;end;end;
if ~isempty(parm.res3.cale)&conique==1&init{end}.dim==2;sz=size(e);e=reshape(e,sz([1,2,4]));o=reshape(o,sz([1,2]));end;

if cal_pertes
p=p/UN;   
if init{end}.dim==2;
wxy=wxy*UN;if conique;Y=wxy;end;
end;
end;
