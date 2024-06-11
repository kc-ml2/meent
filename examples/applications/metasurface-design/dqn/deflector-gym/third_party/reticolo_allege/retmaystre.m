function [k_sortie,ub_sortie]=retmaystre(cal,fich,w,xs,ys,zs,pols,nh,nb,mm,d,sym,cao,xubk,kbornes,ubornes,parm)

% 1 D function [k,ub]=retmaystre(cal,fich,w,xs,ys,pol,pols,nh,nb,mm,d,sym,cao,xubk,kbornes,ubornes,parm);
% 2 D function [k,ub]=retmaystre(cal,fich,w,xs,ys,zs,pols,nh,nb,mm,d,sym,cao,xubk,kbornes,ubornes,parm);
%
%  calcul du diagramme de dispersion  d'un tronçon de maillage w
%    k=2pi/ld  ub=cos(K*h)          Bloch:  ub(k) k reel  Brillouin  k(ub) ub reel
%  (la distinction entre les 2 cas se fait avec la dimension de ubornes et kbornes)
%
% cal  1 nouveau calcul (on ecrase fich)
% cal  2 suite d'un calcul pour d'autres valeurs de xubk (les valeurs deja traitees ne sont pas refaites)
%         si le fichier n'existe pas il est cree
% cal<=0 tracé du diagramme de dispersion et  possibilitee de choisir sur le diagramme des points pour tracer le champ 
%  (si cal  <0 elimination des accumulations de points (eclaircissemint) avant le tracé)
%  si variables en sortie possibilite de sortir les valeurs de points choisis sur le graphe
%
%  fich:nom du fichier ou on met les resultats
%  w :maillage de tronçon decrivant l'objet(obtenu par retautomatique)
%  si w={wh,wcentre,wb}:cl des modes de bloch en haut et en bas
%  xs,ys,zs: coordonnees du point ou on met la source
%  en 1 D remplacer zs par pol: 0 E//   2 H//
%  pols tableau des polarisations de la source pour recherche du pole
%  nh nb indices extremes pour tracé du cone de lumiere
%  mm,d,sym,cao pour init
%  si d(1)<0 on recherche les modes d'un guide invariant en x alors on ne peut tracer que les figures 1 4 et 5
%
%  si mm est un cell array les premieres valeurs sont utilisees pour 'degrossir' la recherche initiale
%
%  xubk  tableau des valeurs de ub de depart
%  kbornes:  kbornes(1)<real(k)<kbornes(2)   et si Brillouin: kbornes(3)<imag(k)<kbornes(4) 
%  ubornes:  ubornes(1)<real(ub)<ubornes(2) et si Bloch:   ubornes(3)<imag(ub)<ubornes(4) 
%    ces bornes peuvent etre modifiees a chaque calcul ou tracé (non stockés sur le fichier)
%  parm parametres (facultatifs) 
%
% parm=struct('periodique',[],'kbornes',[],'ubornes',[],'delta',1.e-2,'itermax',40+1.e-10,'ermax',1.e-6,'itermax0',20+1.e-8,'ermax0',1.e-6,...
% 'deltabornes',[1.e-6,5.e-2],'nreal',10,'nimag',5,'sens',[-1,1],'tol',1.e-3,'poltrace',[],'ntrace',100,'parmtrace',[],'nrep',[1,0,1],...
% 'figures',[1,1,1,1,1],'coupe',[],'extrapole',[2,2],'nbk',1,'dk',1.e-6,'zeros',0,'imp',1);  % par defaut
%     parm.periodique  1 si on impose des conditions de periodicite entre le haut et le bas de l'objet
%                     -1 si conditions d'antiperiodicite
%                      0 ou [] sinon (valeur par defaut )
%     parm.delta    valeur initiale*du pas pour le suivi 
%     parm.itermax  nb max d'iterations (et tolerance) pour la recherche des poles
%                       (parm.ermax   tolerance sur la fonction)
%     parm.itermax0  nb max d'iterations (et tolerance) pour la recherche des poles dans la recherche initiale
%                       (parm.ermax0   tolerance sur la fonction)
%     parm.deltabornes  critere d'arret sur le pas
%         deltabornes(1):arret  
%         abs(deltabornes(2)):valeur max du pas quand il augmente.
%         Sert aussi a eliminer les points trop pres des courbes deja tracees.Si on ne veut pas utiliser cette option, faire deltabornes(2)<0
%     parm.nreal, parm.nimag  nombre de valeurs reelles et imaginaire pour la recherche au hasard initiale
%     parm.kbornes (ou parm.ubornes) bornes des ces valeurs (par defaut(abs ou [])  parm.kbornes=kbornes   parm.ubornes=ubornes )
%       dans tous les cas la premiere valeur de la partie imaginaire est la demi somme de parm.ubornes ou parm.kbornes (commode pour imposer 0)
%     parm.sens  sens de la recherche pour le suivi (par defaut [ -1 1] )
%         si sens=0,on ne calcule qu'un point et les valeurs de kk et uu sont en sortie
%     parm.tol  dans le cas des modes de bloch (defaut 1.e-3)
%     parm.poltrace  polarisations a tracer des champs  ( cf rettchamp )
%     parm.ntrace  nombre de coupes dans le tracé des champs
%     parm.parmtrace  parametres pour le tracé des champs( cf rettchamp)
%     parm.nrep  repetition pour le tracé(en x y z  ou x y)
%             nrep=[ n1, n2, 0] coupe en z=parm.coupe (hauteur source par defaut)   n1 periodes en x  n2 periodes en y
%             nrep=[ n1, 0, n3] coupe en y=parm.coupe  (0 par defaut)  n1 periodes en x  n3 periodes en z
%             nrep=[ 0, n2, n3] coupe en x=parm.coupe  (0 par defaut)  n2 periodes en y  n3 periodes en z
%     parm.extrapole  parametre de retinterpl pour l'extrapolation (suivi)
%     parm.nbk  parm.dk pour trier les modes de bloch avec plusieurs k
%     parm.zeros=1  alternance de calcul de zeros et de poles dans la recherche initiale
%     parm.figures=[ 1 1 1 1 1 ]  figures à tracer
%            1 : points en real(k) real(cosKh) (celle sur laquelle on pointe les points )
%            2 : points en k cosKh  real et imaginaire
%            3 : courbes interpollees k0*periode/lambda  K*periode/(2*pi)
%            4 : courbes interpollees en real(k) real(cosKh)
%            5 : points 3D avec la partie imaginaire 
%
%  PRINICPE DE LA METHODE
%   Brillouin :ub etant donne, on choisi des valeurs de k au hasard dans le domaine defini par parm.kbornes
%   Bloch :k etant donne, on choisi des valeurs de ub au hasard dans le domaine defini par parm.ubornes
%  on cherche un pole en partant de chacune de ces valeurs
%      (par defaut parm.kbornes=kbornes  parm.ubornes=ubornes sinon on l'introduit dans parm ce qui permet de 'cibler' un point)
%  les poles obtenus sont tries ( limites, elimination des egaux)
%  Dans une seconde etape on part de chacun de ces poles pour faire un suivi  des 2 cotes 
%   si on veut un seul sens:   parm.sens=1   ou    parm.sens=-1
% 
% % EXEMPLE: Brillouin pour un demi probleme reseau de trous carres dans du metal
% kbornes=[7,9,-1,1];ubornes=[-1.5,1.5];
% d=[.8,.8];a=[.4,.4];nh=1;nm=retindice(.75,1);
% h0=.1;fich='prv'
% uh=retu(d,{nh});ub=retu(d,{nm,[0,0,a,nh,1]});
% w={{h0,uh},{h0,ub}};
% xs=0;ys=0;zs=h0;nb=nan;
% mm={[-4,4,-4,4],[-6,6,-6,6]};
% sym=[0,1,0,0];cao=[];
% pols=[1:6];cal=2;
% xxx=retx([],ubornes(1),ubornes(2),20);
% retmaystre(cal,fich,w,xs,ys,zs,pols,nh,nb,mm,d,sym,cao,xxx,kbornes,ubornes,struct('itermax',20+1.e-8,'itermax0',20+1.e-8,'delta',1.e-2,'deltabornes',[1.e-6,5.e-3],'nreal',4,'nimag',1,'sens',[1,-1],'tol',.5,'nbk',2,'poltrace',[pols,i]));
% 
% Remarque:il est possible d'utiliser la partie tracé des diagrammes:
% retmaystre(kbornes,ubornes,periode,ub,k,nh,nb,texte);(texte facultatif)
% 
% See also: RETTCHAMP,RETBRILLOUIN,RETBLOCH,RETCADILHAC,RETZP,RETTBLOCH,RETDECOUPE



if nargin<9;if nargin<8;nh='';end;tracer(cal,fich,1,nh,xs,ys,[],[],[],[],[],[],[],[],[],[],w,[],zs,pols,[],[],[],[],[1,1,1,0,1,1]);return;end;% seulement tracé des diagrammes

bloch=length(kbornes)<3;

if cal==1;kk=[];uub=[];xx0=[];end;                    
if cal==2;if exist([fich,'.mat'],'file')==2;try [uub,kk,lex,lex0,pol,h]=retsave(fich);end;else kk=[];uub=[];xx0=[];end;if bloch;xx0=kk;else;xx0=uub;end;end;
if cal<=0;
if exist([fich,'.mat'],'file')==2;try [uub,kk,lex,lex0,pol,h]=retsave(fich);catch load(fich,'uub','kk','lex','lex0','pol','h');retsave(fich,uub,kk,lex,lex0,pol,h);end;else;return;end;if bloch;xx0=kk;else;xx0=uub;end;
if cal<0;z=real(uub)+i*real(kk);[z,ii]=retelimine(z,-cal);uub=uub(ii);kk=kk(ii);cal=0;end;% si cal<0   eclaircissement des fichiers avant le tracé
end;
if length(d)==2;pol=1;source=[xs,ys,zs];else pol=zs;source=[xs,ys];end;
if ~iscell(mm);mm={mm};end;
if nargin<17;parm=struct([]);end;

pparm=struct('periodique',[],'kbornes',[],'ubornes',[],'delta',1.e-2,'itermax',40+1.e-10,'ermax',1.e-6,'itermax0',20+1.e-8,'ermax0',1.e-6,...
'deltabornes',[1.e-6,5.e-2],'nreal',10,'nimag',5,'sens',[-1,1],'tol',1.e-3,'poltrace',[],'ntrace',100,'parmtrace',[],'nrep',[1,0,1],...
'figures',[1,1,1,1,1],'coupe',[],'extrapole',[2,2],'nbk',1,'dk',1.e-6,'zeros',0);% par defaut
 
periodique=retoptimget(parm,'periodique',pparm);if ~isempty(periodique)&(periodique==0);periodique=[];end;  
kbornes1=retoptimget(parm,'kbornes',pparm);if isempty(kbornes1);kbornes1=kbornes;end;
kbornes1(1:2:end)=max(kbornes(1:2:end),kbornes1(1:2:end));kbornes1(2:2:end)=min(kbornes(2:2:end),kbornes1(2:2:end));
ubornes1=retoptimget(parm,'ubornes',pparm);if isempty(ubornes1);ubornes1=ubornes;end;
ubornes1(1:2:end)=max(ubornes(1:2:end),ubornes1(1:2:end));ubornes1(2:2:end)=min(ubornes(2:2:end),ubornes1(2:2:end));
delta=retoptimget(parm,'delta',pparm);
itermax=retoptimget(parm,'itermax',pparm);
ermax=retoptimget(parm,'ermax',pparm);
itermax0=retoptimget(parm,'itermax0',pparm);
ermax0=retoptimget(parm,'ermax0',pparm);
deltabornes=retoptimget(parm,'deltabornes',pparm);
nreal=retoptimget(parm,'nreal',pparm);
nimag=retoptimget(parm,'nimag',pparm);
sens=retoptimget(parm,'sens',pparm);
tol=retoptimget(parm,'tol',pparm);
poltrace=retoptimget(parm,'poltrace',pparm);if isempty(poltrace);poltrace=[1:3*length(d),i];end;
ntrace=retoptimget(parm,'ntrace',pparm);
parmtrace=retoptimget(parm,'parmtrace',pparm);
nrep=retoptimget(parm,'nrep',pparm);
coupe=retoptimget(parm,'coupe',pparm);if isempty(coupe)&(length(nrep)==3);if nrep(3)==0;coupe=source(3);else;coupe=0;end;end;nrep=[nrep,coupe];
figures=retoptimget(parm,'figures',pparm);if nargout>0;figures(1)=1;end;
extrapole=retoptimget(parm,'extrapole',pparm);
nbk=retoptimget(parm,'nbk',pparm);
dk=retoptimget(parm,'dk',pparm);
zeros=retoptimget(parm,'zeros',pparm);

% definition de l'objet
% recherche de la texture ou on met la source
hh=0;wbb={};whh={};ws={};
if iscell(w{1}{1});wh=w{1};wb=w{3};w=w{2};else wh=w(1);wh{1}{1}=0;wb=w(end);wb{1}{1}=0;end;
if source(end)<0;  % source en dessous
ws=w(1);ws{1}{1}=0;
whh=ws;whh{1}{1}=-source(end);
else      % source dans l'objet ou en dessus

for ii=length(w):-1:1;hh=hh+w{ii}{1};hhh=hh-source(end);
if hhh>0;if isempty(ws);ws=w(ii);ws{1}{1}=0;end;

if isempty(whh);% texture trouvee
wbb=[ws,wbb];wbb{1}{1}=w{ii}{1}-hhh;
whh=ws;whh{1}{1}=hhh;
else %construction de whh
whh=[w(ii),whh];
end;
else 
wbb=[w(ii),wbb];
end;
end;

if hhh<0;% source dans le milieu du haut
ws=w(1);ws{1}{1}=0;
whh=ws;whh{1}{1}=hhh;
wbb=[whh,wbb];
whh=ws;
end;
end;% fin construction whh
xs=source(1:end-1);% coordonnees de la source
h=d(1);d=abs(d); % hauteur totale si <0,modes d'un guide sinon modes de Bloch
if h<0;figures=[1,0,0,1,1]&figures;end;
lex0=[];
if ~isempty(periodique);
[s,a,tab,lex]=retauto([],{ws,whh,wbb},-inf);lex0=periodique; % calcul de lex
else 
[s,a,tab,lex]=retauto([],{ws,wh,wb,whh,wbb},-inf);% calcul de lex
% calcul de lex0 pour tri des modes de bloch avec plusieurs k
if ((length(wh)>1)|(length(wb)>1))&(nbk>1);[s,a,tab,lex0]=retauto([],{wh,wb},-inf);lex0={nbk,dk,lex0};
end;% calcul de lex0
end;
if cal==1;retsave(fich,uub,kk,lex,lex0,pol,h);end;

if cal ~=0;% calcul  < xxxxxxxxxxxxxxxxxxxxxxx


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ùù
figure;
for jj=1:length(xubk);% <++++ xubk


if bloch;   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %  BLOCH   k--->  cos(K*h)    % <---------------------
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if retcal(xubk(jj),xx0);% calcul non deja fait <*****************
k0=xubk(jj);
%recherche des valeurs de depart

x=retx([],ubornes1(1),ubornes1(2),nreal);y=retx([],ubornes1(3),ubornes1(4),nimag);y(1)=(ubornes1(3)+ubornes1(4))/2;
[x,y]=meshgrid(x,y);uu0=x+i*y;

for jjj=1:length(mm);   %  < ** jjj essai successif des mm
u0=[];uuu0=[];u_zero=[];
for ii=1:length(uu0(:));u=uu0(ii);  %  < ++ ii
%zero
if (jjj==1)&(zeros==1);
[u_z,iter_z,er_z,erfonc_z,test_z]=retcadilhac(@maystre_bloch,struct('fonc','z=1/z','niter',floor(itermax0),'tol',itermax0-floor(itermax0),'tolf',ermax0,'bornes',ubornes),u,uuu0,u_zero,h,d,mm{jjj},sym,cao,k0,lex,lex0,xs,pol,pols,tol);
disp(rettexte('zero',k0,u_z,iter_z,er_z,abs(erfonc_z),test_z,mm{jjj}));% impression
if all(test_z(1:3));u_zero=[u_zero,u_z];end;
end;   
    
% pole    
[u,iter,er,erfonc,test]=retcadilhac(@maystre_bloch,struct('niter',floor(itermax0),'tol',itermax0-floor(itermax0),'tolf',ermax0,'bornes',ubornes),u,uuu0,u_zero,h,d,mm{jjj},sym,cao,k0,lex,lex0,xs,pol,pols,tol);
if jjj>0;disp(rettexte('pole',k0,u,iter,er,abs(erfonc),test,mm{jjj}));end;% impression
if all(test);uuu0=[uuu0,u];% <--
if ~isempty(uub);distmin=min(sqrt(abs(uub(:)-u).^2+abs(kk(:)-k0).^2));else;distmin=inf;end; 
if distmin>deltabornes(2);%< ... si on n'a pas deja le point ...  
u0=[u0,u];
u0=retelimine(u0( (real(u0)>ubornes(1))  &(real(u0)<ubornes(2))  &(imag(u0)>ubornes(3))&(imag(u0)<ubornes(4))   ),itermax0-floor(itermax0));
end;
if jjj==length(mm);uub=[uub,u];kk=[kk,k0];retsave(fich,uub,kk,lex,lex0,pol,h);
end;%  <-- dans tous les cas on garde la valeur si all(test) et jjj==length(mm)
end;  %< ... si on n'a pas deja le point
end;  %  < ++ ii
uu0=u0;
end;  %  < ** jjj

if all(sens==0);k_sortie=kk;ub_sortie=uub;return;end;% calcul d'un seul point
disp(rettexte(u0));
for ii=1:length(u0);
for ssens=sens;  % on part d'un sens ou de l'autre en k (suivant sens)
[kk,uub]=calk(u0(ii),lex,lex0,d,mm{end},sym,cao,xs,pol,pols,tol,k0,h,ssens,delta,itermax,ermax,extrapole,ubornes,kbornes,deltabornes,nh,nb,uub,kk,fich);
if ~isempty(kk);tracer(kbornes,ubornes,-1,fich,uub,kk,lex,lex0,d,mm,sym,cao,xs,pol,pols,tol,h,deltabornes,nh,nb,poltrace,ntrace,parmtrace,nrep,figures);drawnow;end;
end;end;
xx0=[xx0,xubk(jj)];
end; % calcul deja fait  ? <*****************


else        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %  BRILLOUIN  cos(K*h) -->k   % <---------------------
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if retcal(xubk(jj),xx0);% calcul non deja fait <*****************
u=xubk(jj);
%recherche des valeurs de depart
if length(d)==1;beta0=ACOS(xubk(jj),h);else beta0=[ACOS(xubk(jj),h),0];end;
init=cell(size(mm));
for jjj=1:length(mm);init{jjj}=retinit(d,mm{jjj},beta0,sym,cao);end;   %  < ** jjj essai successif des mm

x=retx([],kbornes1(1),kbornes1(2),nreal);y=retx([],kbornes1(3),kbornes1(4),nimag);y(1)=(kbornes1(3)+kbornes1(4))/2;
[x,y]=meshgrid(x,y);kk0=x+i*y;
for jjj=1:length(mm);% < ** jjj essai sucessif des mm
k0=[];kkk0=[];k_zero=[];
for ii=1:length(kk0(:));k=kk0(ii);  %  < ++ ii
% zero
if (jjj==1)&(zeros==1);
[k_z,iter_z,er_z,erfonc_z,test_z]=retcadilhac(@maystre_brillouin,struct('fonc','z=1/z','niter',floor(itermax0),'tol',itermax0-floor(itermax0),'bornes',kbornes),k,kkk0,k_zero,init{jjj},lex,lex0,xs,pol,pols,tol);
disp(rettexte('zero',u,k_z,iter_z,er_z,abs(erfonc_z),test_z,mm{jjj}));% impression
if all(test_z(1:3));k_zero=[k_zero,k_z];end;
end;


% pole
[k,iter,er,erfonc,test]=retcadilhac(@maystre_brillouin,struct('niter',floor(itermax0),'tol',itermax0-floor(itermax0),'bornes',kbornes),k,kkk0,k_zero,init{jjj},lex,lex0,xs,pol,pols,tol);
if jjj>0; disp(rettexte(k,iter,er,abs(erfonc),test,mm{jjj}));end;% impression
if all(test);kkk0=[kkk0,k]; % <--
if ~isempty(uub);distmin=min(sqrt(abs(uub(:)-u).^2+abs(kk(:)-k).^2));else distmin=inf;end;    
if distmin>deltabornes(2);%< ... si on n'a pas deja le point ...  
k0=[k0,k];
k0=retelimine(k0( (real(k0)>kbornes(1))  &(real(k0)<kbornes(2))  &(imag(k0)>kbornes(3))&(imag(k0)<kbornes(4))   ),itermax0-floor(itermax0));
end;
if jjj==length(mm);uub=[uub,u];kk=[kk,k];retsave(fich,uub,kk,lex,lex0,pol,h);
end; % <-- dans tous les cas on garde la valeur si all(test)
end;%< ... si on n'a pas deja le point ...  
end;  %  < ++ ii
kk0=k0;
end;   %  < ** jjj

if all(sens==0);k_sortie=kk;ub_sortie=uub;return;end;% calcul d'un sel point
disp(rettexte(k0));
for ii=1:length(k0);
for ssens=sens;  % on part d'un sens ou de l'autre en u (suivant sens)
[kk,uub]=calk(xubk(jj),lex,lex0,d,mm{end},sym,cao,xs,pol,pols,tol,k0(ii),h,ssens,delta,itermax,ermax,extrapole,ubornes,kbornes,deltabornes,nh,nb,uub,kk,fich);
if ~isempty(kk);tracer(kbornes,ubornes,-1,fich,uub,kk,lex,lex0,d,mm,sym,cao,xs,pol,pols,tol,h,deltabornes,nh,nb,poltrace,ntrace,parmtrace,nrep,figures);drawnow;end;
end;end;

xx0=[xx0,xubk(jj)];
end;% calcul deja fait  ? <*****************


end;% bloch ou brillouin <---------------------
end;  % <++++ xubk
else  % cal==0 :tracé < xxxxxxxxxxxxxxxxxxxxx

num=0;
[num,fig,uub,kk]=tracer(kbornes,ubornes,num,fich,uub,kk,lex,lex0,d,mm{end},sym,cao,xs,pol,pols,tol,h,deltabornes,nh,nb,poltrace,ntrace,parmtrace,nrep,figures);
drawnow;
end;  % cal==0 ?       < xxxxxxxxxxxxxxxxxxxxx

if nargout>0;  % on cherche des points
try;
k_sortie=[];ub_sortie=[];  
disp(' ');
disp('ENTREE DE MODES');
suite=1;
while 1
figure(fig);drawnow; 
disp('ON PEUT ZOUMER ou DEZOUMER avant d''entrer la donnée (''input'' 3 lignes au dessous) ');
disp('  quand on entre des modes,pour arreter ou refaire un zoom, cliquer à droite');

suite=input('  pour arreter definitivement entrer 0, pour continuer d''entrer les modes entrer 1    ?     ');
if suite==0;break;end
while 1;
ax=axis;
[ub_cherche,k_cherche,button]=ginput(1);if button>1;break;end;
[prv,jj]=min(((real(uub(:))-ub_cherche)/(ax(2)-ax(1))).^2+((real(kk(:))-k_cherche)/(ax(4)-ax(3))).^2);
k_sortie=[k_sortie,kk(jj)];
ub_sortie=[ub_sortie,uub(jj)];
plot(real(uub(jj)),real(kk(jj)),'og');
end;% boucle sans fin
end;% suite=1
catch set(gcf,'Pointer','arrow');end;
end;   % on cherche des points   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function q=maystre_bloch(u,u0,u_zero,h,d,mm,sym,cao,k0,varargin);% pour bloch
if length(d)==1;beta0=ACOS(u,h);else beta0=[ACOS(u,h),0];end;
init=retinit(d,mm,beta0,sym,cao);
q=maystre(k0,init,varargin{:})*(polyval(poly(u_zero),u)/polyval(poly(u0),u));% on elimine les autres zeros
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function q=maystre_brillouin(k,k0,k_zero,varargin);% pour brillouin
q=maystre(k,varargin{:})*(polyval(poly(k_zero),k)/polyval(poly(k0),k));% on elimine les autres zeros
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function q=maystre(k0,init,lex,lex0,xs,pol,pols,tol,num,poltrace,ntrace,parmtrace,nrep,vg,varargin);
%if isempty(varargin);c=0;else;c=1+i;preserves=retio;end;% tracé du champ
if isempty(varargin);c=i;else;c=1+i;end;preserves=retio;% tracé du champ
ssh={};ssb={};
if iscell(lex0); % bloch avec plusieurs k
nbk=lex0{1};dk=lex0{2};lex0=lex0{3};
ssh=cell(1,nbk-1);ssb=cell(1,nbk-1);
%[s,a,tab,lex0]=retauto(init,{wh,wb},-inf)
for ii=1:nbk-1;sprv=retauto(init,[],i,lex0,k0+i*(dk*4^(ii-1)));ssh{ii}=sprv{1};ssb{ii}=sprv{2};end;
end;
%[s,a,tab,lex]=retauto(init,{ws,wh,wb,whh,wbb},-inf)
[s,a,tab]=retauto(init,[],c,lex,k0);

as=a{tab{1}(1,2)};source=rets(init,xs,as,pols);

if length(s)==5; 
if  size(tab{2},1)>1;ah=retbloch(init,[{s{2}},ssh],sum(tab{2}(:,1)),tol);else;ah=a{tab{2}(1,2)};end;sh=retb(init,ah,1.e-3,0,[],[]);
if abs(retcompare(tab{2},tab{3}))<1.e-13;sb=retrenverse(sh,2);else;% cas des milieux identiques en haut et en bas
if  size(tab{3},1)>1;ab=retbloch(init,[{s{3}},ssb],sum(tab{3}(:,1)),tol);else;ab=a{tab{3}(1,2)};end;sb=retb(init,ab,-1.e-3,0,[],[]);
end;
q=reteval(retss(sh,s{4},source,s{5},sb));q=1/trace(q);
else %periodique
[qq,ss]=retperiodique(retss(s{2},source,s{3}),lex0);q=1/trace(qq);
end;
 
if isempty(varargin);retio(preserves,-4);return;end;
%  tracé du champ   varargin:  valeurs de x
K=init{end}.beta0(1);d=init{end}.d;cosKh=cos(K*d(1));
a=[a,{source}];
if length(s)==5;  
if ~isreal(nrep);tabb=retchamp([tab{2};tab{4};[0,tab{1}(1,2),i*length(a)];tab{5};tab{3}],real(nrep));% coupe dans le plan d-nrep
else tabb=[repmat(tab{2},nrep,1);tab{4};[0,tab{1}(1,2),i*length(a)];tab{5};repmat(tab{3},nrep,1)];
end;inc=ones(length(pols),1);
else % periodique
if ~isreal(nrep);tabb=retchamp([tab{2};[0,tab{1}(1,2),i*length(a)];tab{3}],real(nrep));
else tabb=repmat([tab{2};[0,tab{1}(1,2),i*length(a)];tab{3}],nrep,1);end;
inc=ss(:,1:length(pols))*ones(length(pols),1);inc(length(inc)/2:end)=inc(length(inc)/2:end)*(lex0^(nrep-1));
inc=[inc;zeros(nrep*length(pols),1)];
sh=rettronc(rets1(init),0,[],1);sb=rettronc(rets1(init),0,[],-1);
end;
if isreal(nrep);hh=sum(tabb(:,1));for ii=1:size(tabb,1);if tabb(ii,3)==0;tabb(ii,3)=ceil(ntrace*tabb(ii,1)/hh);end;end;end;

for ii=1:length(varargin);
switch(length(poltrace));
case 1;texte=rettexte(num,k0,K,cosKh,vg);
case 2;texte={rettexte(num,k0),rettexte(K,cosKh,vg)};
otherwise;texte={rettexte(num,k0),rettexte(K,cosKh),rettexte(vg)};
end;

[e,z,w,o]=retchamp(init,a,sh,sb,inc,varargin{ii},tabb);
if length(d)==2;x=varargin{ii}{1};y=varargin{ii}{2};rettchamp(e,o,x,y,z,poltrace,parmtrace,[],texte); %  2 D
else
x=varargin{ii};rettchamp(e,o,x,z,pol,poltrace,parmtrace,[],texte);end;           %  1 D
end;
retio(preserves,-4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [kk,uub]=calk(ub,lex,lex0,d,mm,sym,cao,xs,pol,pols,tol,k0,h,sens,delta,itermax,ermax,extrapole,ubornes,kbornes,deltabornes,nh,nb,uub,kk,fich);
uuub=ub;kkk=k0;% pour suivi local
bloch=length(kbornes)<3;
if bloch;ubornes1=ubornes;% elargissement des bornes de 1.1
ubornes1(1)=ubornes1(1)-.1*(ubornes1(2)-ubornes1(1));
ubornes1(2)=ubornes1(2)+.1*(ubornes1(2)-ubornes1(1));
ubornes1(3)=ubornes1(3)-.1*(ubornes1(4)-ubornes1(3));
ubornes1(4)=ubornes1(4)+.1*(ubornes1(4)-ubornes1(3));
else kbornes1=kbornes;
kbornes1(1)=kbornes1(1)-.1*(kbornes1(2)-kbornes1(1));
kbornes1(2)=kbornes1(2)+.1*(kbornes1(2)-kbornes1(1));
kbornes1(3)=kbornes1(3)-.1*(kbornes1(4)-kbornes1(3));
kbornes1(4)=kbornes1(4)+.1*(kbornes1(4)-kbornes1(3));
end;

pas=0;iiter=6*ones(1,5);

while 1  %  boucle sans fin
delt=delta;
test=[0,0,0,0];ppas=0;
while ~all(test(1:3));ppas=ppas+1;%  <...............
if bloch;% < bloch
[xubk,k,ub1,k1]=suivi(uuub,kkk,0,delt*sens,extrapole);
[ub,iter,er,erfonc,test]=retcadilhac(@maystre_bloch,struct('niter',floor(itermax),'tol',itermax-floor(itermax),'tolf',ermax,'bornes',ubornes1),xubk,[],[],h,d,mm,sym,cao,k,lex,lex0,xs,pol,pols,tol);
disp(rettexte(k,iter,er,abs(erfonc),test,ub,ppas,'bloch'));   % impression
else     % < brillouin
[ub,k0,ub1,k1]=suivi(uuub,kkk,delt*sens,0,extrapole);
if length(d)==1;beta0=ACOS(ub,h);else beta0=[ACOS(ub,h),0];end;
init=retinit(d,mm,beta0,sym,cao);
[k,iter,er,erfonc,test]=retcadilhac(@maystre_brillouin,struct('niter',floor(itermax),'tol',itermax-floor(itermax),'tolf',ermax,'bornes',kbornes1),k0,[],[],init,lex,lex0,xs,pol,pols,tol);
disp(rettexte(k,iter,er,abs(erfonc),test,ub,ppas,'brillouin'));   % impression
end; % < bloch ou brillouin
if ppas>5;    return;end; % si en 5 fois on ne trouve pas le pole on arrete
delt=delt/2; % essai avec pas plus petit

end;%  <...............

uub=[uub,ub];kk=[kk,k];retsave(fich,uub,kk,lex,lex0,pol,h);

if bloch;  % bloch
if (real(ub)<ubornes(1))|(real(ub)>ubornes(2))|(imag(ub)<ubornes(3))|(imag(ub)>ubornes(4))|(k<kbornes(1))|(k>kbornes(2))return;end; % arret
else      % brillouin
if (ub<ubornes(1))|(ub>ubornes(2))|(real(k)<kbornes(1))|(real(k)>kbornes(2))|(imag(k)<kbornes(3))|(imag(k)>kbornes(4));return;end; % arret
end;

if pas>4;%  <********************
v1=[real(uuub(end-1)),imag(uuub(end-1)),real(kkk(end-1)),imag(kkk(end-1))];
v2=[real(uuub(end)),imag(uuub(end)),real(kkk(end)),imag(kkk(end))];
v3=[real(ub),imag(ub),real(k),imag(k)];
accepte=(sum((v3-v2).*(v2-v1)))>0.95*norm(v3-v2)*norm(v2-v1);% critere sur l''angle'

else accepte=1;end;

if accepte;%  < accepte
delt=delta;pas=pas+1;
uuub=[uuub,ub];kkk=[kkk,k];
iiter=[iiter(2:end),iter]
if sum(iiter)>8*length(iiter);iiter(:)=6;delta=delta/1.5,end;
if sum(iiter)<4*length(iiter);iiter(:)=6;delta=delta*1.5,end;

if ppas>=2;delta=delt/(2^(ppas-1)),end;
delta=min(delta,abs(deltabornes(2)));
if delta<deltabornes(1);disp(rettexte(delta,deltabornes(1)));return;end;
% adaptation du pas,  impression
% tracé de controle
if 0
if pas>5;aaa=sum((v3-v2).*(v2-v1))/(norm(v3-v2)*norm(v2-v1))-1;else aaa=0;end;
if bloch;uprevu=xubk;kprevu=k;else uprevu=ub;kprevu=k0;end;
hold off;subplot(2,1,1);plot(real(uuub),real(kkk),'.k',real(uuub(end)),real(kkk(end)),'ok');title(rettexte(delta,aaa));
%subplot(2,1,2);plot(real(uuub(max(1:end-3):end)),real(kkk(max(1:end-3):end)),'.',real(uprevu),real(kprevu),'o');drawnow;
subplot(2,2,3);plot(real(uuub(max(1:end-3):end)),real(kkk(max(1:end-3):end)),'.k',real(uprevu),real(kprevu),'ok',real(ub1),real(k1),'*k');
subplot(2,2,4);plot(real(uuub(end)),real(kkk(end)),'.k',real(uprevu),real(kprevu),'ok',real(ub1),real(k1),'*k');drawnow;
end

else delta=delta/2;%  < pas garde 
if delta<deltabornes(1);disp(rettexte(delta,deltabornes(1)));return;end;
end;%  < accepte ?

end;% boucle sans fin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ub,k,ub1,k1]=suivi(uub,kk,deltau,deltak,extrapole);delta=deltau+deltak;
switch(length(uub));
case 1;ub=uub+deltau/5;k=kk+deltak/5;ub1=ub;k1=k;
case 2;ub=2*uub(2)-uub(1);k=2*kk(2)-kk(1);ub1=ub;k1=k;
otherwise;
nn=max(1,length(uub)-max(2*sum(extrapole),3));
uub=uub(nn:end);kk=kk(nn:end);
l=sqrt(abs(uub(2:end)-uub(1:end-1)).^2+abs(kk(2:end)-kk(1:end-1)).^2);l=[0,l];l=cumsum(l);ll=l(end)+abs(delta);
ub=retinterp(l,uub,ll,extrapole);k=retinterp(l,kk,ll,extrapole);% extrapolation de degre extrapole fonction de l'abscisse curviligne
ub1=ub;k1=k;

if isreal(uub);k=retcadilhac(kk(end-2:end).',uub(end-2:end).'-ub);
else;ub=retcadilhac(uub(end-2:end).',kk(end-2:end).'-k);end;
[ub,k,ub1,k1]=retpermute(ub1,k1,ub,k);
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [num,fig,uub,kk]=tracer(kbornes,ubornes,num,fich,uub,kk,lex,lex0,d,mm,sym,cao,xs,pol,pols,tol,h,deltabornes,nh,nb,poltrace,ntrace,parmtrace,nrep,figures);
bloch=length(kbornes)<3;
fig=0;

if h>0;texte={'bloch k0 --> cos(K*periode)   ','real(cos(K*periode))','imag(cos(K*periode))','brillouin  cos(K*periode)-->k0  ','cos(K*periode)'};
else texte={'bloch k0 --> K','real(K)','imag(K)','brillouin  K-->k0  ','K'};end;


if bloch;
f=find((kk<kbornes(2))&(kk>kbornes(1))&(real(uub)<ubornes(2))&(real(uub)>ubornes(1))&(imag(uub)>ubornes(3))&(imag(uub)<ubornes(4)) );
else 
f=find((real(kk)<kbornes(2))&(real(kk)>kbornes(1))&(imag(kk)>kbornes(3))&(imag(kk)<kbornes(4))&(uub<ubornes(2))&(uub>ubornes(1)));
end;
kk=kk(f);uub=uub(f);
kkk=linspace(kbornes(1),kbornes(2),1000);uclh=COS(kkk*nh,h);uclb=COS(kkk*nb,h);


if figures(1);% <-------------figure 1
if num>=0;fig=figure;else fig=0;end;
hold on;
hand=plot(real(uub),real(kk),'.r',uclh,kkk,'--k',uclb,kkk,'--k',[0,0],kbornes(1:2),'--k',[-1,-1],kbornes(1:2),'--k',[1,1],kbornes(1:2),'--k');
if isfinite(nh)&isfinite(nb);
t=linspace(0,pi/2,100);
for p=0:kbornes(2)*h*max(nh,nb)/pi;
plot((-1)^p*cosh(p*pi*cos(t)),(p*pi/(nh*h))*sin(t),'-.k','LineWidth',2)
plot((-1)^p*cosh(p*pi*cos(t)),(p*pi/(nb*h))*sin(t),'-.k','LineWidth',2)
end;
end;
if bloch;ylabel('k  ');xlabel(rettexte(texte{2},fich),'interpreter','none');else ylabel('real(k0)  ');xlabel(rettexte(texte{5},fich),'interpreter','none');end;
grid on;
set(hand(end-3),'linewidth',2);set(hand(end-1),'linewidth',.5);set(hand(end),'linewidth',.5);axis([ubornes(1:2),kbornes(1:2)]);
if h>0; 
ax1=gca;ax2=axes('XAxisLocation','top','Xlim',get(ax1,'Xlim'),'XTick',cos((.5:-.05:0)*2*pi),'XTickLabel',{'.5';'.45';'.4';'.35';'.3';'.25';'.2';'.15';'.1';'.05';'.0'},'Ylim',get(ax1,'Ylim')*h/(2*pi),'YAxisLocation','right','color','none','TickDir','out','Xcolor','r','Ycolor','r','fontsize',8);
prv=get(ax2,'Yticklabel');Yticklabel=cell(1,size(prv,1));for ii=1:size(prv,1);Yticklabel{ii}=prv(ii,:);end;
set(ax2,'Ytick',((2*pi)/h)*str2num(prv),'Yticklabel',Yticklabel);
ylabel('periode/lambda  ','fontsize',8);
if bloch;title('real(K)*periode/(2*pi)   ','fontsize',8,'color','r');else title('K*periode/(2*pi)','fontsize',8,'color','r');end;
linkaxes([ax1,ax2]);axes(ax1);
end;

end;  % <-------------figure 1
if num<0;return;end;

if figures(2);figure;% <-------------figure 2
subplot(1,2,1);hold on;
hand=plot(real(uub),real(kk),'.k',uclh,kkk,'--k',uclb,kkk,'--k',[0,0],kbornes(1:2),'-.k',[-1,-1],kbornes(1:2),'-.k',[1,1],kbornes(1:2),'-.k');ylabel('real(k0)');xlabel('cos(K*h)');
title(rettexte(fich),'interpreter','none','fontsize',7);grid on;
set(hand(end-4),'linewidth',2);set(hand(end-3),'linewidth',2);set(hand(end-2),'linewidth',1);set(hand(end-1),'linewidth',1);set(hand(end),'linewidth',1);axis([ubornes(1:2),kbornes(1:2)]);

if isfinite(nh)&isfinite(nb);
t=linspace(0,pi/2,100);
for p=0:kbornes(2)*h*max(nh,nb)/pi;
plot((-1)^p*cosh(p*pi*cos(t)),(p*pi/(nh*h))*sin(t),'--k','LineWidth',2);
if nh~=nb;plot((-1)^p*cosh(p*pi*cos(t)),(p*pi/(nb*h))*sin(t),'-.k','LineWidth',2);end;
end;
end;

if bloch; %  bloch
subplot(1,2,2);plot(imag(uub),kk,'.k');
ylabel('k0');xlabel('imag(cos(K*h))');axis([-inf,inf,kbornes(1:2)]);grid on;
title(rettexte(pol,sym,fich),'interpreter','none','fontsize',7);
else      %  brillouin
subplot(1,2,2);plot(imag(kk),real(kk),'.k');
ylabel('real(k0)');xlabel('imag(k0)');axis([-inf,inf,kbornes(1:2)]);grid on;
title(rettexte(pol,sym,fich),'interpreter','none','fontsize',7);
end;      %  bloch ou brillouin
end;  % <-------------figure 2


%  prolongement
if ~isempty(lex);
if bloch;x=[real(uub(:)),real(kk(:)),imag(uub(:))];else x=[real(uub(:)),real(kk(:)),imag(kk(:))];end;
[xx,xx_seul]=lignes(x);
uub1=xx(:,1);kk1=xx(:,2);uub1_seul=xx_seul(:,1);kk1_seul=xx_seul(:,2);
else;uub1=uub;kk1=kk;uub1_seul=[];kk1_seul=[];xx_seul=zeros(0,2);end;

% pour les Physiciens ...
if figures(3);figure;hold on;% <-------------figure 3
kclh=kkk*nh;kclb=kkk*nb;
plot([.5,.5],kbornes(1:2)*h/(2*pi),'-.k',[0,0],kbornes(1:2)*h/(2*pi),'-.k','linewidth',2);
K=real(ACOS(uub1,h));K(abs(uub1)>1+1.e-9)=nan;
f=find(abs(uub1_seul)<=1);K_seul=real(ACOS(xx_seul(f,1),h));
if ~isempty(lex);plot([K*h/(2*pi);K*h/(2*pi)+1;-K*h/(2*pi);-K*h/(2*pi)+1],repmat(real(kk1),4,1)*h/(2*pi),'r','linewidth',2);
else;plot([K*h/(2*pi),K*h/(2*pi)+1,-K*h/(2*pi),-K*h/(2*pi)+1].',repmat(real(kk1.'),4,1)*h/(2*pi),'.r');
end;
plot([K_seul*h/(2*pi);K_seul*h/(2*pi)+1;-K_seul*h/(2*pi);-K_seul*h/(2*pi)+1],repmat(real(kk1_seul(f)),4,1)*h/(2*pi),'.r');
if length(d)==2;if isempty(cao);cao=[0,0,0,0];end;if real(cao(4))==0;dd=d(1)/d(2);Lmax=2;else;dd=0;Lmax=0;end;else;dd=0;Lmax=0;end;
for L=-1:1;for LL=0:Lmax;prv=linspace(-.5,1,100);
%plot(kclh*h/(2*pi)+L,kkk*h/(2*pi),'--k',-kclh*h/(2*pi)+L,kkk*h/(2*pi),'--k','linewidth',2);
%if nb~=nh;plot(kclb*h/(2*pi)+L,kkk*h/(2*pi),'-.k',-kclb*h/(2*pi)+L,kkk*h/(2*pi),'-.k','linewidth',2);end;
plot(prv,sqrt((prv+L).^2+(LL*dd)^2)/real(nh),'--k','linewidth',2);
if nb~=nh;plot(prv,sqrt((prv+L).^2+(LL*dd)^2)/real(nb),'--k','linewidth',2);end;
end;end;
ylabel('courbes interpolees   periode/lambda');title(['Courbes classiques  ',fich],'fontsize',10,'interpreter','none');
ax=axis;nlignes=100;xx=repmat([ax(1);ax(2);nan],1,nlignes);xxx=repmat([ax(2);ax(1);nan],1,nlignes);vg=.1;
y0=linspace(ax(3)-vg*(ax(2)-ax(1)),ax(4)+vg*(ax(2)-ax(1)),nlignes);yy=[y0;y0+(ax(2)-ax(1))*vg;y0+nan];yyy=[y0+.5*(ax(2)-ax(1))*vg;y0+.5*(ax(2)-ax(1))*vg;y0+nan];
plot([xx(:);nan;xx(:);nan;xxx(:)],[yyy(:);nan;yy(:);nan;yy(:)],':g');
xlabel(rettexte('K*periode/(2*pi)  pointilles verts:',vg));
axis([0,.5,kbornes(1:2)*h/(2*pi)]);
end;  %                        <-------------figure 3

% courbes en cosKh apres prolongement
if figures(4);figure;hold on;% <-------------figure 4
kkk=linspace(kbornes(1),kbornes(2),1000);uclh=COS(kkk*nh,h);uclb=COS(kkk*nb,h);
plot(real(uub1),real(kk1),'-r',real(uub1_seul),real(kk1_seul),'.r',uclh,kkk,'--k',uclb,kkk,'--k','LineWidth',2);

if isfinite(nh)&isfinite(nb);
t=linspace(0,pi/2,100);
for p=0:kbornes(2)*h*max(nh,nb)/pi;
plot((-1)^p*cosh(p*pi*cos(t)),(p*pi/(nh*h))*sin(t),'--k','LineWidth',2)
if nb~=nh;plot((-1)^p*cosh(p*pi*cos(t)),(p*pi/(nb*h))*sin(t),'-.k','LineWidth',2);end;
end;
end;
ylabel('k0  ');xlabel(['courbes interpolees  ',texte{5}]);axis([ubornes(1:2),kbornes(1:2)]);
if h>0;% modes de Block
plot([0,0],kbornes(1:2),'--k',[-1,-1],kbornes(1:2),'--k',[1,1],kbornes(1:2),'--k','LineWidth',2);
ax1=gca;ax2=axes('XAxisLocation','top','Xlim',get(ax1,'Xlim'),'XTick',cos((.5:-.05:0)*2*pi),'XTickLabel',{'.5';'.45';'.4';'.35';'.3';'.25';'.2';'.15';'.1';'.05';'.0'},'Ylim',get(ax1,'Ylim')*h/(2*pi),'YAxisLocation','right','color','none','TickDir','out','Xcolor','r','Ycolor','r','fontsize',8);
prv=get(ax2,'Yticklabel');Yticklabel=cell(1,size(prv,1));for ii=1:size(prv,1);Yticklabel{ii}=prv(ii,:);end;
set(ax2,'Ytick',((2*pi)/h)*str2num(prv),'Yticklabel',Yticklabel);
ylabel('periode/lambda  ','fontsize',8);
if bloch;title('real(K)*periode/(2*pi)  ','fontsize',8,'color','r');else;title('K*periode/(2*pi)','fontsize',8,'color','r');end;
linkaxes([ax1,ax2]);axes(ax1);
end;
grid on;
end;  % <-------------figure 4

if figures(1);figure(fig);drawnow; % <-------------figure 1 (deja ouverte)
if ~isempty(lex);suite=input('tracé de n modes ? entrer n puis choisir sur le graphe (si clic a droite pas de tracé),0 pour sortir,[] pour terminer sans imprimer les autres figures)  ');else;suite=0;end;
if isempty(suite);return;end;
while suite~=0;
figure(fig);drawnow;  
[xxubk,kk0,button]=ginput(suite);
for nmode=1:suite;xubk=xxubk(nmode);k0=kk0(nmode);num=num+1;
figure(fig);drawnow;
ax=axis;
[prv,iii]=min((real(uub-xubk)/(ax(2)-ax(1))).^2+(real(kk-k0)/(ax(4)-ax(3))).^2);
plot(real(uub(iii)),real(kk(iii)),'ok');text(uub(iii),real(kk(iii)),[' \leftarrow ',int2str(num)],'FontSize',18);

% calcul de vg=(dk/dK) et GVD=-(dvg/dK )/vg^3 =-(d2k/dK2) / vg^3
if length(uub)>2;
iiii=find(uub~=uub(iii));[prv,iii0]=min(abs(uub(iiii)-uub(iii)).^2+abs(kk(iiii)-kk(iii)).^2);iii0=iiii(iii0);
iiii=find((uub~=uub(iii))&(uub~=uub(iii0)));[prv,iii1]=min(abs(uub(iiii)-uub(iii)).^2+abs(kk(iiii)-kk(iii)).^2);iii1=iiii(iii1);
uuub=uub([iii,iii0,iii1]);kkkk=kk([iii,iii0,iii1]);[prv,iiii]=sort(real(uuub));uuub=uuub(iiii);kkkk=kkkk(iiii);
fit=polyfit(uuub,kkkk,2);
fit=polyder(fit);der1=polyval(fit,uub(iii));fit=polyder(fit);der2=polyval(fit,uub(iii));% derivees
if h>0;vg=-der1*h*sqrt(1-uub(iii)^2);vg3_GVD=h^2*(uub(iii)*der1-der2*(1-uub(iii)^2));GVD=vg3_GVD/vg^3;
else vg=der1;vg3_GVD=der2;GVD=-der2/(der1^3);end;

if real(vg<0);vg=-vg;GVD=-GVD;end;
else vg=nan;GVD=nan;vg3_GVD=nan;end;
K=ACOS(uub(iii),h);K_periode_sur_2pi=K*h/(2*pi);periode_sur_ld=kk(iii)*h/(2*pi);cosKh=uub(iii);k0=kk(iii);ld=2*pi/k0;
if h>0;rettexte(K_periode_sur_2pi,periode_sur_ld,vg,GVD,vg3_GVD,cosKh,k0,ld)
else rettexte(vg,vg3_GVD,GVD,k0,ld),end;
% maystre    tracé du champ
if button(nmode)==1;
if length(d)==1;beta0=ACOS(uub(iii),h);xy={linspace(-nrep(1)*d/2,nrep(1)*d/2,101)};nnrep=nrep(2);
else beta0=[ACOS(uub(iii),h),0];
if nrep(2)==0;xy={{linspace(-nrep(1)*d(1)/2,nrep(1)*d(1)/2,101),nrep(end)}};nnrep=nrep(3);
else if nrep(3)==0;xy={{linspace(-nrep(1)*d(1)/2,nrep(1)*d(1)/2,101),linspace(-nrep(2)*d(2)/2,nrep(2)*d(2)/2,101)}};nnrep=i+nrep(4);
else xy={{nrep(end),linspace(-nrep(2)*d(2)/2,nrep(2)*d(2)/2,101)}};nnrep=nrep(3);end;

end;
end;
init=retinit(d,mm,beta0,sym,cao);
q=maystre(kk(iii)*(1+1.e-7),init,lex,lex0,xs,pol,pols,tol,num,poltrace,ntrace,parmtrace,nnrep,vg,xy{:});
end;
end; % nmode
figure(fig);drawnow;  
suite=input('tracé de n modes ? entrer n puis choisir sur le graphe (0 pour sortir ,[] pour terminer sans imprimer les autres figures)  ');
end;
end;    % <-------------figure 1



if figures(5);figure;hold on;  % <-------------figure 5
plot3(real(uclh),kkk,imag(uclh),'-.k','LineWidth',2);plot3(real(uclb),kkk,imag(uclb),'-.k','LineWidth',2);
if isfinite(nh)&isfinite(nb);
t=linspace(0,pi/2,100);
for p=0:kbornes(2)*h*max(nh,nb)/pi;
ucl=(-1)^p*cosh(p*pi*cos(t));kcl=(p*pi/(nh*h))*sin(t);
f=find((ucl>ubornes(1))&(ucl<ubornes(2))&(kcl>kbornes(1))&(kcl<kbornes(2)));plot(ucl(f),kcl(f),'-.k','LineWidth',2);
ucl=(-1)^p*cosh(p*pi*cos(t));kcl=(p*pi/(nb*h))*sin(t);
f=find((ucl>ubornes(1))&(ucl<ubornes(2))&(kcl>kbornes(1))&(kcl<kbornes(2)));plot(ucl(f),kcl(f),'-.k','LineWidth',2);
end;
end;

[uub_c,kk_c]=eclairci(uub,kk,ubornes,kbornes);

if bloch;% tracé 3 D bloch
plot3(real(uub),kk,imag(uub),'.k','MarkerSize',12);
plot3(real(uub),kk,0*imag(uub),'.k','MarkerSize',3);
for jj=1:length(uub_c);plot3([real(uub_c(jj)),real(uub_c(jj))],[kk_c(jj),kk_c(jj)],[0,imag(uub_c(jj))],'-k','LineWidth',.5);end
title(rettexte(texte{1},fich),'fontsize',8,'interpreter','none');

xlabel(texte{2});ylabel('k0');zlabel(texte{3});
else   % tracé 3 D brillouin
plot3(uub,real(kk),imag(kk),'.k','MarkerSize',12);
plot3(uub,real(kk),0*imag(kk),'.k','MarkerSize',3);
for jj=1:length(uub_c);plot3([uub_c(jj),uub_c(jj)],[real(kk_c(jj)),real(kk_c(jj))],[0,imag(kk_c(jj))],'-k','LineWidth',.5);end;
title(rettexte(texte{4},fich),'fontsize',8,'interpreter','none');
xlabel(texte{5});ylabel('real(k0)');zlabel('imag(k0)');
end;   % bloch ou brillouin

[xx,yy]=meshgrid(linspace(ubornes(1),ubornes(2),10),linspace(kbornes(1),kbornes(2),10));zz=zeros(size(xx));mesh(xx,yy,zz,'EdgeColor','k','Facecolor','none');
if h>0;
plot3([-1,-1],kbornes(1:2),[0,0],'--k','LineWidth',2);
plot3([1,1],kbornes(1:2),[0,0],'--k','LineWidth',2);
end;
plot3([0,0],kbornes(1:2),[0,0],'--k','LineWidth',2);

axis([ubornes(1:2),kbornes(1:2)]);
alpha(.7);
end;  % <-------------figure 5

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % maintenant dans reticolo (avec une variante)
% function [s,ss]=retperiodique(s,a) 
% retio(s);
% n=s{2};s=s{1};nsi=size(s,2)-2*n;nsd=size(s,1)-2*n;
% ss=[[a*eye(n)-s(1:n,1:n),-s(1:n,n+1:2*n)];[-s(n+nsd+1:2*n+nsd,1:n),(1/a)*eye(n)-s(n+nsd+1:2*n+nsd,n+1:2*n)]]\[s(1:n,2*n+1:2*n+nsi);s(n+nsd+1:2*n+nsd,2*n+1:2*n+nsi)];
% s=s(n+1:n+nsd,1:2*n)*ss+s(n+1:n+nsd,2*n+1:2*n+nsi);
%  
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [uub_c,kk_c]=eclairci(uub,kk,ubornes,kbornes) %uub_c=uub(1:20:end);kk_c=kk(1:20:end);return;
tol=1.e-2;
n=length(uub);
n0=0;uub_c=zeros(1,n);kk_c=zeros(1,n); 
ur=abs(ubornes(2)-ubornes(1));
kr=abs(kbornes(2)-kbornes(1));
if length(kbornes)>2;
ki=abs(kbornes(4)-kbornes(3));
while ~isempty(uub);n0=n0+1;uub_c(n0)=uub(1);kk_c(n0)=kk(1);
f=find((abs(uub(1)-uub)/ur+abs(real(kk(1))-real(kk))/kr+abs(imag(kk(1))-imag(kk))/ki)>tol);
uub=uub(f);kk=kk(f);
end;
else 
ui=abs(ubornes(4)-ubornes(3));
while ~isempty(uub);n0=n0+1;uub_c(n0)=uub(1);kk_c(n0)=kk(1);
f=find((abs(uub(1)-uub)/ur+abs(real(kk(1))-real(kk))/kr+abs(imag(uub(1))-imag(uub))/ui)>tol);
uub=uub(f);kk=kk(f);
end;
end;
uub_c=uub_c(1:n0);kk_c=kk_c(1:n0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,x_seul]=lignes(x)
dim=size(x,2);x_seul=zeros(0,dim);if size(x,1)<3;return;end;
nann=repmat(nan,1,dim);

% tri
n=size(x,1);k=1;l=n;kk=0;
alpha=0.2;alpha1=alpha;
while k<n;
xx=x(k+1:n,:)-repmat(x(k,:),l-1,1);
dist=sqrt(sum(xx.^2,2));
[prv,ii]=min(dist);
if (k>2)&(prv>1.e-6)
t=xx./repmat(dist,1,dim);
t0=(x(k,:)-x(k-1,:))./(sqrt(sum((x(k,:)-x(k-1,:)).^2,2))+eps);
c=t-repmat(t0,l-1,1);c=sqrt(sum(c.^2,2));
[prv,ii]=min((1-alpha1)*dist+alpha1*c);
%si angle trop grand, on en tient pas compte de l'angle au pas suivant
if c(ii)<.3;alpha1=alpha;else alpha1=0;end;
end;
ii=ii+k;
[x(k+1,:),x(ii,:)]=retpermute(x(ii,:),x(k+1,:));
k=k+1;l=l-1;
end;

d=sqrt(sum((x(2:end,:)-x(1:end-1,:)).^2,2));
f=find(d>eps);x=x([1;f+1],:);d=d(f);% elimination des egaux
t=(x(2:end,:)-x(1:end-1,:))./repmat(d,1,dim);% tg
c=sqrt(sum((t(2:end,:)-t(1:end-1,:)).^2,2));
c=[c(1);c];% variation de la tangente

% etude statistique locale de d et c
nm=min(10,length(d)-1);

mmm=length(d);
for kkk=1:2;if kkk==1;prv=d.';else prv=c.';end;


pprv=zeros(2*nm,mmm);
f=1;for iii=1:20;if isempty(f);break;end;% 'lissage'
for ii=1:nm;
pprv(-ii+nm+1,:)=prv(1+mod([0:mmm-1]-ii,mmm));
pprv(ii+nm,:)=prv(1+mod([0:mmm-1]+ii,mmm));
end;
f=find(prv>3*(mean(pprv)+std(pprv)));
prv(f)=min(pprv(nm:nm+1,f));
end;
if kkk==1;fd=find((d.'>3*(mean(pprv)+std(pprv))|(d.'>norm((max(x)-min(x)))/min(50,mmm/20))));
else fc=find((c.'>5*(mean(pprv)+std(pprv))|(c.'>.3)));end;
end;% kkk

f_test=[];

f=retelimine(union(f_test,union(fc,fd)));
if isempty(f);x=[nann;x];return;
else 
xx={x(1:f(1),:)};dd={[0;d(1:f(1)-1)]};
for ii=1:length(f)-1;
if f(ii+1)>f(ii)+1;
xx=[xx,{x(f(ii)+1:f(ii+1),:)}];
dd=[dd,{[0;d(f(ii)+1:f(ii+1)-1)]}];
else;
xx=[xx,{x(f(ii)+1,:)}];
dd=[dd,{[]}];
end;
end;
if f(end)<(n-1);xx=[xx,{x(f(end)+1:end,:)}];dd=[dd,{[0;d(f(end)+1:end)]}];end;
end;
[f,ff]=retfind(cellfun('size',xx,1)<2);% on ne garde que les branches avec plus de 2 points
for ii=1:length(f);x_seul=[x_seul;xx{f(ii)}];end;
xx=xx(ff);dd=dd(ff);

% interpolation
L=eps;for ii=1:length(xx);L=L+sum(dd{ii});end;
for ii=1:length(xx);
di=cumsum(dd{ii});
t=sort([linspace(di(1),di(end),floor(100000*di(end)/L))';di]);
xxx=zeros(length(t),dim);
for jj=1:dim;xxx(:,jj)=retinterp(di,xx{ii}(:,jj),t,'cubic');end;
for xlim=[-1,1];% on ajoute xx=-1 et xx=1
if (min(xxx(:,1))<xlim)&(max(xxx(:,1))>xlim);
for iii=2:size(xxx,1);if (min(xxx((iii-1):iii,1))<xlim)&(max(xxx((iii-1):iii,1))>xlim);
xxxx=[xlim,retinterp(xxx((iii-1):iii,1),xxx((iii-1):iii,2),xlim),retinterp(xxx((iii-1):iii,1),xxx((iii-1):iii,3),xlim)];
xxx=[xxx(1:iii-1,:);xxxx;xxx(iii:end,:)];
end;end;
end;
end;
xx{ii}=xxx;
end;

x=nann;for jj=1:length(xx);x=[x;nann;xx{jj}];end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u=ACOS(u,h);% determination avec des coupures adaptees
if h>0;
fm=find((imag(u)<-eps)&(real(u)<-1));
fp=find((imag(u)<eps)&(real(u)>1));
u=acos(u);
u(fm)=2*pi-u(fm);
u(fp)=-u(fp);
u=u/h;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u=COS(u,h);if h>0;u=cos(u*h);end;
