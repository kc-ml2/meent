function [a,d,w,maillage]=retcouche(init,w,c,li,maillage);
% function [a,d,w]=retcouche(init,w,c,li);
%
%  calcul du 'descripteur de texture' a, utilisee par retb, retc, retchamp pour calculer les champs
%  si on n'utilise que retb il suffit de prendre c=-1 (ce qui diminue la memoire et le calcul) 
%  si on veut tracer les champs c=1 mais plus long et gourmand en memoire
%  si on ne veut pas tracer les champs c=0 (option par defaut) 
%  init calcule par retinit
%  w est le maillage de texture decrivant l'objet calculee par retu
%  c,li parametres
%
% si c=-inf matrices vides pour tests
%
% POUR LES DIELECTRIQUES
%........................
%
%  pour calculer 'proprement' les champs prendre c=1,mais ceci alourdit les calculs 
%     ( pour les sources  en 2 D si on veut calculer l'energie diffractee par la source il faut prendre c=1
%         ou bien c=[1,x,y] ou x y sont les coordonnees de la source      )
%
%  d:valeurs propres
%  la  matrice m est sous la forme  m=[0,a1;a2,0]=.5*[q,q;p*d,-p*d]*[d,0;0,-d]*[qq,(1/d)*pp;qq,-(1/d)*pp]  
%  en 1D a={    p,pp,q,qq,              d,                      a3,a1,         w                 parm};
%           base vecteurs propres  valeurs propres     matrices servant au calcul des champs   
%                                                        (vides si c=-1ou 0)
%
%  en 2D a={    p,pp,q,qq,               d,              ez,hz,fex,fhx,fey,fhy,  w            pas,       parm};
%           base vecteurs propres  valeurs propres     matrices servant au calcul des champs   pas      
%                                                        (vides si c=-1ou 0)
%   parm=struct('dim',dim,'genre',2,'type',1,'sog',sog);
%   type:  1 dielectriques  ,2 metaux infiniment conducteurs ,3 modes de bloch non symetriques,4 elements finis,5 textures inclinees
%   genre 2  descripteur de texture
%
%  ce programme necessite en general la DIAGONALISATION d'une matrice n*n 
%  (sauf pour les metaux et les milieux homogenes isotropes)
% 
%  en 2D 
%
%  si on prend imag(c)~=0 ,les matrices sont stockés sur disque(ne pas oublier de faire retio a la fin pour liberer les fichiers)
%            si de plus imag(c)=-2  en 2 D  on utilise une bibliotheque
%            pour effacer la bibliotheque :retcouche;  
%            pour recuperer cette bibliotheque :bibli=retcouche;  
%            pour regenerer cette bibliotheque :retcouche(bibli);
%                      attention si bibli contient des fichiers a ce qu'ils ne soient pas effaces  
%
% si w est mis en sortie il devient: w={w{1},w{2},w{3},eepz,mmuz,muy,mux,epy,epx,hx,ez,hz,fex,fhx,fey,fhy}
% ce qui permet de stocker les divers tableaux eepz,...
% dans un  nouvel appel ils ne sont plus recalcules
%  si de plus c=-2 on ne fait pas la diagonalisation ( utile a retbrillouin qui n'utilise que eepz,mmuz,muy,mux,epy,epx)  
% ceci permet d'eviter des calculs inutiles pour des objets compliques quand lambda varie 
%  ATTENTION les eps ne doivent pas avoir varié entre temps...
% si de plus imac(c)~=0  eepz,...etc sont stockés sur disque  (ne pas oublier de faire retio a la fin )%
%
%   li(12):parametres(facultatif) pour la methode de li (1 -1 ou 0) par defaut li=zeros(1,12) ou 12 fois li(1) si 1 seul terme 
%   si 0 le choix du sens depend de la continuitee des fonctions
%    li         | 1   |  2  | 3 | 4 | 5 | 6 |            li        |  7 | 8 | 9 | 10| 11| 12|
%               -----------------------------                      --------------------------
%    matrices   |eepz | mmuz|muy|mux|epy|epx|   calcul des champs: |eepz|muz|fex|fhx|fey|fhy|
% 
% POUR LES TEXTURES INCLINEES
% le maillage de texture w décrivant l'objet, calculée par retu doit être remplacé par:
% en 1D   {w,struct('teta',teta,'fac',fac,'Maystre',Maystre)}
% en 2D   {w,struct('teta',teta,'phi',phi,'sens',sens,'fac',fac,'Maystre',Maystre)}
%   teta, phi,angles en radians,
%   Maystre angle de la coupure en radians (facultatif sinon coupure fonction de teta)
%   sens = 1 ou -1 suivant que l'on travaille en x ou en y
%   fac=[1+.01i ,1+.1i] pour tri des modes par exemple on diagonalise pour k0 k0*(1+.01i) et on extrapole à k0*(1+.1i)
%   sens et fac sont facultatifs (par défaut sens = 1 fac=[])
%
% RECAPITULATIF pour le parametre c
% |   c   |retchamp|retc|retb|rets|retbrillouin|retneff|cpu                             |imag(c)|
% ----------------------------------------------------------                            ---------
% |   1   | oui    |oui |oui |oui |oui         |oui    |1      calcul complet           | ~=0   | stockage sur disque
% |[1,x,y]|approché|oui |oui |oui |oui         |oui    |1      champ sur la source      |    -2 | bibliotheque(obsolete)  
% |   0   |approché|oui |oui |non |oui         |oui    |.99    le champ est calculé approximativement
% |  -1   |non     |non |oui |non |oui         |oui    |.9     seulement conditions aux limites(retb) 
% |  -2   |non     |non |non |non |oui         |non    |.01    pas diagonalisation 
% |  -3   |non     |non |non |non |non         |oui    |.5     seulement les valeurs propres  
% | -inf  |non     |non |non |non |non         |non    |0      matrices vides pour tests  
%
%..............................................
% POUR LES METAUX INFINIMENT CONDUCTEURS
%..............................................
%  function [a,d]=retcouche(init,w,c);
%  calcul du descripteur de texture a utilisee par  retc pour le metal infiniment conducteur
%  a={sparse(g1),sparse(g2),dd,m,n,champ,www};
%  dd:valeurs propres
%  c:0 ou 1 si on veut calculer les champs
%
%
%  See also:RETB,RETC,RETBLOCH,RETCHAMP,RETDEPLACEMENT

persistent bibli;  % utilisation d'une bibliotheque
if nargin==0;if nargout==0;bibli=retio(bibli,-1);else a=bibli;end;return;end;
if nargin==1;bibli=init;end; % on regenere la bibliotheque

if nargin<3;c=0;end;if length(c)>1;xsource=c(2:end);c=c(1);else xsource=[];end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<4;li=0;end;if isempty(li);li=0;end;if length(li)<12;li=li(1)*ones(1,12);end; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if c==-inf; % matrices vides pour tests
if size(w,2)==5 ;% metaux
a=cell(1,7);a{7}=w;
else;            % dielectriques
if init{end}.dim==2; %2D
a=cell(1,14);a{12}=w;
else;
a=cell(1,9);a{8}=w;
end;
end;
d=[];
return;end;  % fin matrices vides    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
if size(w,2)==5 ;% < +++++++ metaux
[a,d]=retmcouche(init,w,real(c));
else;            % < +++++++  dielectriques
if init{end}.dim==2; % < ******************* 2D 
if imag(c)==-2; % utilisation d'une bibliotheque
if isempty(init{10});iinit=[init{2}(:);init{end}.sym(:)];else;iinit=[init{2}(:);init{end}.sym(:);init{10}{1}(:);init{10}{2}(:)];end;
if length(w)==5;ww=[w{1}(:);w{2}(:);w{3}(:);w{4}(:);w{5}(:)];else;ww=[w{1}(:);w{2}(:);w{3}(:)];end; 
if ~isempty(bibli);
if abs(retcompare(iinit,bibli{1},2))<10*eps;    
for ii=2:length(bibli);
if (abs(retcompare(ww,bibli{ii}{1},2))<10*eps)&(real(c)<=bibli{ii}{2});% valeur trouvee
a=bibli{ii}{3};w=bibli{ii}{4};return;end;
end;
else;bibli=retio(bibli,-1);end;
end;
end; 
if ~isempty(init{end}.L);if nargin>3;c=c+i;end;[a,d]=retcouche_cylindrique_radial(init,w,c);return;end;% cylindriquee_radial

if nargout==3;[a,d,w]=ret2couche(init,w,c,li,xsource);else [a,d]=ret2couche(init,w,c,li,xsource);end;
if imag(c)==-2; % stoquage dans la bibliotheque
a=retio(a,1);w=retio(w,1);
if isempty(bibli);bibli={iinit,{ww,real(c),a,w}};else bibli={bibli{:},{ww,real(c),a,w}};end;
end;
    

else                       %< ******************* 1D 
beta=init{2};si=init{3};cao=init{5};sog=init{end}.sog;n=size(beta,2);x=w{1}/init{4};ep=w{2};
c=real(c);

pp=[];qq=[];q=[];
if length(x)<=1 & isempty(si) & isempty(cao); % <------- milieu homogene et pas de symetries et pas cao: diagonalisation analytique
d=retsqrt(-ep(1)/ep(3)+(beta.'.^2)./(ep(2)*ep(3)),1);
%d=retbidouille(d);% <<<<<<<<<<<<<<<<<< MODIF

p=eye(n);q=p/ep(3);pp=eye(n);qq=pp*ep(3);   
if c==1;
[fmu,fmmu,fep]=retc1(x,ep,beta);if size(w,2)==2;[a1,a2,k,aa2]=retc2(fmu,fmmu,fep,beta,cao);else;[a1,a2,k]=retc2(fmu,fmmu,fep,beta,cao,w{3},w{4});end;
a3=-inv(rettoeplitz(fmu))*diag(beta); % matrice a3 permettant le calcul de hy=(1/mu)*(du/dx)
if ~isempty(si);a3=a3*si{1};a1=a1*si{1};end
else
a3=[];a1=[];
end 

else                                     % <------- milieu non homogene ou symetries ou cao
[fmu,fmmu,fep]=retc1(x,ep,beta);if size(w,2)==2;[a1,a2,k,aa2]=retc2(fmu,fmmu,fep,beta,cao);else;[a1,a2,k]=retc2(fmu,fmmu,fep,beta,cao,w{3},w{4});end;
if ~isempty(si);          % < ............ utilisation des proprietes de symetrie
aa1=a1*si{1};a1=si{2}*aa1;a2=si{2}*a2*si{1};
end    
if length(x)<=1 & isempty(cao)  %<.....  milieu homogene dans le cas d'utilisation de symetries: diagonalisation analytique
d=retsqrt(diag(a2*a1),1);
%d=retbidouille(d);% <<<<<<<<<<<<<<<<<< MODIF

p=eye(init{1});q=a1;    
else;                           %<..... diagonalisation numerique    
[p,d]=retc3(a1,a2,c);
end;                            %<..... diagonalisation numerique ?
if c>=0;q=a1*p;pp=retpinv(p);qq=retpinv(q);end;

if c==1;
%a3=-inv(rettoeplitz(fmu))*diag(beta);% matrice a3 permettant le calcul de hy=i/mu*du/dx
a3=-inv(rettoeplitz(fmu))*k;% matrice a3 permettant le calcul de hy=i/mu*du/dx avec changement de coordonnees
if ~isempty(si);a3=a3*si{1};a1=aa1;end
else
a3=[];if ~isempty(si);a1=aa1;end
end;
end;                                    % <------- milieu homogene et pas de symetries   ? ?

a={p,pp,q,qq,d,a3,a1,w,init{end}.d,init{2},struct('dim',1,'genre',2,'type',1,'sog',sog)};
if nargout==3;w={w{1},w{2},a1,aa2};end;
end;                       %< ******************* 2D 1D 

end;           % < +++++++  metaux ou dielectriques

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a,d,w]=ret2couche(init,w,c,li,xsource) 
io=imag(c)~=0;c=real(c);
beta=init{2};ordre=init{3};nx=init{6};ny=init{7};n1=nx*ny;si=init{8};pas=init{end}.d;n=init{1};cao=init{10};delt=init{end}.delt;sog=init{end}.sog;
x=w{1};y=w{2};u=w{3};
nw=(size(w,2)==3);  % on recalcule eepz... 


kbidouillemax=16;
kbidouille=1;
while (kbidouille>0)&(kbidouille<kbidouillemax);%  <-----------BOUCLE SUR KBIDOUILLE

if ~nw;             % on reutilise des valeurs deja calculees
eepz=w{4};mmuz=w{5};muy=w{6};mux=w{7};epy=w{8};epx=w{9};hx=w{10};ez=w{11};hz=w{12};fex=w{13};fhx=w{14};fey=w{15};fhy=w{16};
else;
eepz=[];mmuz=[];muy=[];mux=[];epy=[];epx=[];hx=[];ez=[];hz=[];fex=[];fhx=[];fey=[];fhy=[];
end;    

%parm=max([1,sqrt(max(sum(abs(beta.^2),1)))*.2]); % pour retbidouille
parm=kbidouille*max([1,sqrt(max(sum(abs(beta.^2),1)))*.2]); % pour retbidouille

%  < _________couche non homogene ou changement de coordonnees
if ~(all(all(all(repmat(u(1,1,:),size(u,1),size(u,2))==u)))&(u(1)==u(2))&(u(1)==u(3))&(u(4)==u(5))&(u(4)==u(6))&isempty(cao)); %guide

%calcul de la matrice m=[0,a1;a2,0] associee au passage dans un milieu periodique 2D invariant en z
if ~isempty(cao); % changement de coordonnees
kx=retmat(apodisecao(cao{1})*retdiag(beta(1,1:nx)),ny);
ky=retmat(apodisecao(cao{2})*retdiag(beta(2,1:nx:n1)),-nx);
else
kx=retdiag(beta(1,:));ky=retdiag(beta(2,:));
end;
[prv,version]=retversion;if version>7.3;nxnymax=inf;else;nxnymax=4000;end;   % limite du calcul complexe dans l'inversion de ret2li. Peut creer des instabilités
if ~isempty(si);  % < XXXX  utilisation des proprietes de symetrie 

if isempty(eepz);eepz=ret2li(x,y,1./u(:,:,6),nx,ny,0,0,li(1),si,nxnymax,3);else;eepz=retio(eepz);end;  %calcul de eepz
a1=i*(retprod(si{2}(:,1:n1)*kx,eepz,ky*si{3}(1:n1,:))-retprod(si{2}(:,1:n1)*kx,eepz,kx*si{3}(n1+1:2*n1,:))+retprod(si{2}(:,n1+1:2*n1)*ky,eepz,ky*si{3}(1:n1,:))-retprod(si{2}(:,n1+1:2*n1)*ky,eepz,kx*si{3}(n1+1:2*n1,:)));
if c==1;ez=retprod(eepz,ky*si{3}(1:nx*ny,:))-retprod(eepz,kx*si{3}(nx*ny+1:end,:));ez=retio(ez,io);end; %tableaux utilises pour calculer les champs
if (nargout==3)&nw;eepz=retio(eepz,io);else;retio(eepz,-1);clear eepz;end;

if isempty(muy);muy=ret2li(x,y,u(:,:,2),nx,ny,1,0,li(3),si,nxnymax,5);else;muy=retio(muy);end; %muy
a1=retfullsparse(a1);a1=a1+i*retprod(si{2}(:,1:n1),muy,si{3}(n1+1:2*n1,:));
if (nargout==3)&nw;muy=retio(muy,io);else;retio(muy,-1);clear muy;end;

if isempty(mux);mux=ret2li(x,y,u(:,:,1),nx,ny,0,1,li(4),si,nxnymax,4);else;mux=retio(mux);end;  % mux
a1=retfullsparse(a1);a1=a1-i*retprod(si{2}(:,n1+1:2*n1),mux,si{3}(1:n1,:));
a1=full(a1);
if (nargout==3)&nw;mux=retio(mux,io);else;retio(mux,-1);clear mux;end;

if isempty(mmuz);mmuz=ret2li(x,y,1./u(:,:,3),nx,ny,0,0,li(2),si,nxnymax,6);else;mmuz=retio(mmuz);end;  %calcul de mmuz
a2=i*(retprod(-si{4}(:,1:n1)*kx,mmuz,ky*si{1}(1:n1,:))+retprod(si{4}(:,1:n1)*kx,mmuz,kx*si{1}(n1+1:2*n1,:))-retprod(si{4}(:,n1+1:2*n1)*ky,mmuz,ky*si{1}(1:n1,:))+retprod(si{4}(:,n1+1:2*n1)*ky,mmuz,kx*si{1}(n1+1:2*n1,:)));

if c==1;hz=-retprod(mmuz,ky*si{1}(1:n1,:))+retprod(mmuz,kx*si{1}(n1+1:2*n1,:));hz=retio(hz,io);end; %tableaux utilises pour calculer les champs
if (nargout==3)&nw;mmuz=retio(mmuz,io);else;retio(mmuz,-1);clear mmuz;end;

if isempty(epy);epy=ret2li(x,y,u(:,:,5),nx,ny,1,0,li(5),si,nxnymax,2);else;epy=retio(epy);end; %calcul de epy
a2=retfullsparse(a2);a2=a2-i*retprod(si{4}(:,1:n1),epy,si{1}(n1+1:2*n1,:));
if (nargout==3)&nw;epy=retio(epy,io);else;retio(epy,-1);clear epy;end;
 
if isempty(epx);epx=ret2li(x,y,u(:,:,4),nx,ny,0,1,li(6),si,nxnymax,1);else;epx=retio(epx);end;  %calcul de epx
a2=retfullsparse(a2);a2=a2+i*retprod(si{4}(:,n1+1:2*n1),epx,si{1}(1:n1,:));
if (nargout==3)&nw;epx=retio(epx,io);else;retio(epx,-1);clear epx;end;clear kx ky;
a2=full(a2);


else;  % < XXXX  non utilisation des proprietes de symetrie 

if isempty(eepz);  %calcul de eepz
eepz=ret2li(x,y,1./u(:,:,6),nx,ny,0,0,li(1));else;eepz=retio(eepz);
end;
a1=i*[[kx*eepz*ky,-kx*eepz*kx];[ky*eepz*ky,-ky*eepz*kx]];
if c==1;ez=[eepz*ky,-eepz*kx];ez=retio(ez,io);end; %tableaux utilises pour calculer les champs

if (nargout==3)&nw;eepz=retio(eepz,io);else;clear eepz;end;

if isempty(muy);muy=ret2li(x,y,u(:,:,2),nx,ny,1,0,li(3));else;muy=retio(muy);end; %muy
a1=retfullsparse(a1);a1(1:n1,n1+1:2*n1)=a1(1:n1,n1+1:2*n1)+i*muy;
if (nargout==3)&nw;muy=retio(muy,io);else;clear muy;end;

if isempty(mux);mux=ret2li(x,y,u(:,:,1),nx,ny,0,1,li(4));else;mux=retio(mux);end;  % mux
a1=retfullsparse(a1);a1(n1+1:2*n1,1:n1)=a1(n1+1:2*n1,1:n1)-i*mux;
if (nargout==3)&nw;mux=retio(mux,io);else;clear mux;end;

if isempty(mmuz);mmuz=ret2li(x,y,1./u(:,:,3),nx,ny,0,0,li(2));else;mmuz=retio(mmuz);end;  %calcul de mmuz
a2=i*[[-kx*mmuz*ky,kx*mmuz*kx];[-ky*mmuz*ky,ky*mmuz*kx]];
if c==1;hz=[-mmuz*ky,mmuz*kx];hz=retio(hz,io);end; %tableaux utilises pour calculer les champs
if (nargout==3)&nw;mmuz=retio(mmuz,io);else;clear mmuz;end;

if isempty(epy);epy=ret2li(x,y,u(:,:,5),nx,ny,1,0,li(5));else;epy=retio(epy);end; %calcul de epy
a2=retfullsparse(a2);a2(1:n1,n1+1:2*n1)=a2(1:n1,n1+1:2*n1)-i*epy;
if (nargout==3)&nw;epy=retio(epy,io);else;clear epy;end;
 
if isempty(epx);epx=ret2li(x,y,u(:,:,4),nx,ny,0,1,li(6));else;epx=retio(epx);end;  %calcul de epx
a2=retfullsparse(a2);a2(n1+1:2*n1,1:n1)=a2(n1+1:2*n1,1:n1)+i*epx;
if (nargout==3)&nw;epx=retio(epx,io);else;clear epx,end;clear kx ky;
    
    
end;   % < XXXX   utilisation des proprietes de symetrie  ?

moharam=0;sens=0;
if ~isempty(si);ne=size(si{6},1);nh=size(si{8},1);else;ne=init{1}/2;nh=ne;end
if size(u,1)==1 & nx==1 & ne>0;moharam=nh;sens=-1;end;
if size(u,2)==1 & ny==1 & nh>0;moharam=ne;sens=1;end;

if c~=-2;[p,d]=retc3(a1,a2,c,moharam,sens);if c~=-3;q=a1*p;else;q=[];end;else p=[];q=[];d=[];end;  % diagonalisation


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   utile pour les milieux homogenes non isotropes ...
%        % orthogonalisation au sens du flux de Poynting des modes de vp egales
%        bb=find(abs(real(d))<1.e-6&abs(imag(d))>eps);
%        if ~isempty(bb);
%        q=a1*p;
%        if isempty(si);p1=p(:,bb);q1=q(:,bb);else;p1=si{3}*p(:,bb);q1=si{1}*q(:,bb);end;
%        n2=length(bb);masque=zeros(1,n2);
%        for ii=1:n2;
%        if masque(ii)==0;
%        jj=find(d(bb)==d(bb(ii)));masque(jj)=1;    
%        if length(jj)>1;
%        b=bb(jj);    
%        [cho,test]=chol(p1(:,jj)'*[[zeros(n1/2),-eye(n1/2)];[eye(n1/2),zeros(n1/2)]]*q1(:,jj)*diag(d(b)));
%        if test==0;p(:,b)=p(:,b)*inv(cho);end;
%        end;end;end;
%        end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


else  % < _________  espace homogene et isotrope


if c==1; %< ....tableaux utilises pour calculer les champs
kx=retdiag(beta(1,:));ky=retdiag(beta(2,:));
if isempty(eepz);eepz=ret2li(x,y,1./u(:,:,6),nx,ny,0,0,li(7));else;eepz=retio(eepz);end;
if isempty(mmuz);mmuz=ret2li(x,y,1./u(:,:,3),nx,ny,0,0,li(8));else;mmuz=retio(mmuz);end;
ez=[eepz*ky,-eepz*kx];if ~isempty(si);ez=ez*si{3};end;ez=retio(ez,io);
hz=[-mmuz*ky,mmuz*kx];if ~isempty(si);hz=hz*si{1};end;hz=retio(hz,io);
end;     %< ....
if (nargout==3)&nw;% <---
muy=ret2li(x,y,u(:,:,2),nx,ny,1,0,li(3));muy=retio(muy,io);  % muy
mux=ret2li(x,y,u(:,:,1),nx,ny,0,1,li(4));mux=retio(mux,io);  % mux
epy=ret2li(x,y,u(:,:,5),nx,ny,1,0,li(5));epy=retio(epy,io);  % epy
epx=ret2li(x,y,u(:,:,4),nx,ny,0,1,li(6));epx=retio(epx,io);  % epx
eepz=ret2li(x,y,1./u(:,:,6),nx,ny,0,0,li(7));eepz=retio(eepz,io);
mmuz=ret2li(x,y,1./u(:,:,3),nx,ny,0,0,li(8));mmuz=retio(mmuz,io);
else;              % <---
retio(eepz,-1);retio(mux,-1);retio(muy,-1);retio(mmuz,-1);retio(epx,-1);retio(epy,-1);    
clear eepz mux muy mmuz epx epy
end;               % <---
clear kx ky;



if isempty(si); % < XXXX  pas de symetrie   decomposition E et H pas de symetrie
ux=beta(1,:);uy=beta(2,:);
d=repmat(retsqrt(ux.^2+uy.^2-u(1,1,1)*u(1,1,4),1).',2,1);
%d=retbidouille(d);% <<<<<<<<<<<<<<<<<< MODIF

uu=sqrt(ux.^2+uy.^2);
f=find(abs(uu)<100*eps);ux(f)=cos(delt);uy(f)=sin(delt);uu(f)=1;% onde normale  cas degenere angle delt par defaut   
ux=ux./uu;uy=uy./uu;
p=[[diag(ux);diag(uy)],[diag(uy);diag(-ux)]];
d2=-retbidouille(d([1:n/2]).',1,parm).^2;

p=[[diag(ux);diag(uy)],[diag(uy);diag(-ux)]];
q=i*[[diag(uy);diag(-ux)]*u(1,1,1),[diag(-ux);diag(-uy)]*diag(d2/u(1,1,4))];

else;  % < XXXX   cas des symetries
p=eye(n);q=zeros(n);p1=si{3};d=zeros(n,1);
masque=zeros(1,n);
for ii=1:n;  % <*** boucle ii
if masque(ii)==0;  % <+++    masque=0
masque(ii)=1;bm=find(masque==0);
ff=find(p1(:,ii)~=0);
f=ff(1);if f>n1;f=f-n1;end;
k=find((p1(f,bm)~=0)|(p1(f+n1,bm)~=0));
masque(bm(k))=1;
ux=beta(1,f);uy=beta(2,f); 
d2=retsqrt(ux^2+uy^2-u(1,1,1)*u(1,1,4),1);
d([ii,bm(k)])=d2;
%d([ii,bm(k)])=retbidouille(d2);% <<<<<<<<<<<<<<<<<< MODIF
d2=-retbidouille(d2,1,parm).^2;

uu=sqrt(ux^2+uy^2);
if abs(uu)<100*eps; ux=cos(delt);uy=sin(delt);else ux=ux/uu;uy=uy/uu;end; % onde normale  cas degenere angle delt par defaut   
 
if ~isempty(k);      % <---- k=[]
kk=bm(k(1));    
prv=(p1([f,f+n1],[ii,kk])\([[ux;uy],[uy;-ux]]));  %passage E//  H//
p(:,[ii,kk])=p(:,[ii,kk])*prv;
q(:,[ii,kk])=si{2}*[p1(1+n1:2*n1,[ii,kk]);-p1(1:n1,[ii,kk])]*prv*[[i*u(1,1,1);0],[0;i*d2/u(1,1,4)]];
else          % <---une seule op dans le mode
if abs(ux*p1(f,ii)+uy*p1(f+n1,ii))<abs(uy*p1(f,ii)-ux*p1(f+n1,ii));%cas H//
prv=i*d2/u(1,1,4);
else;%cas E//
prv=i*u(1,1,1);
end;
q(:,ii)=prv*si{2}*([p1(1+n1:2*n1,ii);-p1(1:n1,ii)]);
end;                  % <---- k=[]
end;  % <+++    masque=0

end;    % <*** boucle ii
p=sparse(p);q=sparse(q);
end;% < XXXX  symetries ?
end; % < _________

rc=min(rcond(full(p)),rcond(full(q)));if rc<=5*eps;kbidouille=2*kbidouille;else kbidouille=0;end;

end;  %  <-----------BOUCLE SUR KBIDOUILLE (on recommence si p ou q mal conditionnees)



if c<0;pp=[];qq=[];else pp=retpinv(p);qq=retpinv(q);end;

if c==1; %tableaux utilises pour calculer les champs non encore calcules
mx=size(x,2);my=size(y,2);mxx=1;myy=1;
if ~isempty(xsource);  % tableaux reduits pour les sources 
xxx=mod(xsource(1)/pas(1),1);yyy=mod(xsource(2)/pas(2),1);

for iii=1:mx;if iii==1 x0=0;else x0=x(iii-1);end;
fx=find((xxx>=x0)&(xxx<=x(iii)));    
for jjj=1:my;if jjj==1 y0=0;else y0=y(jjj-1);end;
fy=find((yyy>=y0)&(yyy<=y(jjj)));
if ~isempty(fx)&~isempty(fy) iiii=iii;jjjj=jjj;end;
end;end;
else;mxx=1:mx;myy=1:my;iiii=mxx;jjjj=myy;% tableaux complets
end;  %tableaux utilises pour calculer les champs non encore calcules

if isempty(fex);
l_mxx=length(mxx);l_myy=length(myy);
fex=cell(1,l_myy);
for ii=1:l_myy; % fonctions discontinues en x devenant continues
trouve=1;for jj=1:ii-1;if all(u(:,jjjj(myy(ii)),4)==u(:,jjjj(myy(jj)),4));fex{myy(ii)}=fex{myy(jj)};trouve=0;break;end;end;	
if trouve;fex{myy(ii)}=retio(ret2li(x,1,u(:,jjjj(myy(ii)),4),nx,ny,0,1,li(9)),io);end;
end;
fey=cell(1,l_mxx);
for ii=1:l_mxx; % fonctions discontinues en y devenant continues 
trouve=1;for jj=1:ii-1;if all(u(iiii(mxx(ii)),:,5)==u(iiii(mxx(jj)),:,5));fey{mxx(ii)}=fey{mxx(jj)};trouve=0;break;end;end;	
if trouve;fey{mxx(ii)}=retio(ret2li(1,y,u(iiii(mxx(ii)),:,5),nx,ny,1,0,li(11)),io);end;
end;
fhx=cell(1,l_myy);
for ii=1:l_myy; % fonctions discontinues en x devenant continues
trouve=1;for jj=1:ii-1;if all(u(:,jjjj(myy(ii)),1)==u(:,jjjj(myy(jj)),1));fhx{myy(ii)}=fhx{myy(jj)};trouve=0;break;end;end;	
if trouve;fhx{myy(ii)}=retio(ret2li(x,1,u(:,jjjj(myy(ii)),1),nx,ny,0,1,li(10)),io);end;
end;
fhy=cell(1,l_mxx);
for ii=1:l_mxx; % fonctions discontinues en y devenant continues
trouve=1;for jj=1:ii-1;if all(u(iiii(mxx(ii)),:,2)==u(iiii(mxx(jj)),:,2));fhy{mxx(ii)}=fhy{mxx(jj)};trouve=0;break;end;end;	
if trouve;fhy{mxx(ii)}=retio(ret2li(1,y,u(iiii(mxx(ii)),:,2),nx,ny,1,0,li(12)),io);end;
end;
end;
end;

a={p,pp,q,qq,d,ez,hz,fex,fhx,fey,fhy,w,pas,init{2},init{11},struct('dim',2,'genre',2,'type',1,'sog',sog)};


%end;  %  <-----------BOUCLE SUR KBIDOUILLE (on recommence si p ou q mal conditionnees)% (pourquoi j'avais mis cette ligne la ???????????????)

if (nargout==3)&nw;
w={w{1},w{2},w{3},eepz,mmuz,muy,mux,epy,epx,hx,ez,hz,fex,fhx,fey,fhy};
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [aa,dd]=retmcouche(init,www,cc);
%  function [aa,dd]=retmcouche(init,www,cc);
%  calcul du descripteur de texture aa utilisee par  retc pour le metal infiniment conducteur

n=init{1};beta=init{2};x=www{1};dx=www{2};ep=www{3};pol=www{4};mm=www{5};

if isempty(x);dd=[];% metal non troue
if pol==0; % metal electrique
g1=[eye(n),zeros(n)];g2=zeros(n,0);
%g1=[eye(n),1.e-5*eps*eye(n)];g2=zeros(n,0);% modif 10 2010 pour la stabilite
else; % metal magnetique
g1=[zeros(n),eye(n)];g2=zeros(n,0);
%g1=[1.e-5*eps*eye(n),eye(n)];g2=zeros(n,0);% modif 10 2010 pour la stabilite
end;    
aa={g1,g2,[],0,n,cell(1,5),www,struct('dim',init{end}.dim,'genre',2,'type',2,'sog',0)};
return;
end 


nt=size(x,1);% nombre de trous
champ=cell(1,5);
if init{end}.dim==1; %1D
               %%%%%%%%%%%%%%%%%%%%
               %      1  D        % 
               %%%%%%%%%%%%%%%%%%%%

si=init{3};d=init{end}.d;
n1=size(beta,2);if isempty(mm);mm=repmat(n1,nt,1);end;
mmax=sum((mm+pol/2));
sc=zeros(mmax,n1);scm=zeros(mmax,n1);
dd=zeros(mmax,1);ww=zeros(mmax,1);
if cc==1;cha=zeros(mmax,4,2);chx=zeros(mmax,1);chdx=zeros(mmax,1);chalp=zeros(mmax,1);chtp=zeros(mmax,1);end;

% calcul de sc et sm

parm=pol;tp=1;
m=0;w=[];  %  nombre de modes
for it=1:nt;
for mx=1-pol/2:mm(it);
m=m+1;
alp=mx*pi/dx(it);bet=beta;x0=x(it);d0=dx(it);
[s,c,sm,cm]=calsc(d0,bet,x0,dx,alp,tp,parm);
sc(m,:)=s;scm(m,:)=sm;
dd(m)=retbidouille(retsqrt(alp^2-ep(it,1)/ep(it,3),1),1);
w(m)=dd(m)*ep(it,3);

if cc==1;
% amplitudes des champs: 1 montants: 2 descendants
cha(m,:,1)=[1,i*w(m),i*alp*ep(it,3),i*dd(m)];
cha(m,:,2)=[1,-i*w(m),i*alp*ep(it,3),-i*dd(m)];
chx(m)=x(it);     % centre
chdx(m)=dx(it);    % largeur
chalp(m)=alp;     % alpha
chtp(m)=pol-1;% tp: 1 cos  -1 sin
end;

end;
end;
sc=sc(1:m,:);scm=scm(1:m,:);dd=dd(1:m);w=w(1:m);
if cc==1;cha=cha(1:m,:,:);chx=chx(1:m);chdx=chdx(1:m);chalp=chalp(1:m);chtp=chtp(1:m);end;
% construction de la matrice G

if pol==0;
g1=diag(1./w)*sc;g2=scm.'/d;
if ~isempty(si);g1=g1*si{1};g2=si{2}*g2;end; %  utilisation des symetries
f=find(max(abs(g2).^2,[],1)>eps);m=size(f,2);% elimination des modes nulls (dans le cas des symetries)
g1=g1(f,:);g2=g2(:,f);dd=dd(f);
g1=[[zeros(m,n),g1];[eye(n),zeros(n)]];g2=[[-eye(m),eye(m)];[g2,g2]];
else; 
g1=sc;g2=scm.'*diag(w)/d;
if ~isempty(si);g1=g1*si{1};g2=si{2}*g2;end; %  utilisation des symetries
f=find(max(abs(g1).^2,[],2)>eps);m=size(f,1);% elimination des modes nulls (dans le cas des symetries)
g1=g1(f,:);g2=g2(:,f);dd=dd(f);
g1=[[g1,zeros(m,n)];[zeros(n),eye(n)]];g2=[[eye(m),eye(m)];[-g2,g2]];
end;
if cc==1;cha=cha(f,:,:);chx=chx(f);chdx=chdx(f);chalp=chalp(f);chtp=chtp(f);end;
else;
               %%%%%%%%%%%%%%%%%%%%
               %      2  D        % 
               %%%%%%%%%%%%%%%%%%%%
nx=init{6};ny=init{7};si=init{8};d=init{end}.d;
n1=nx*ny;if isempty(mm);mm=repmat([nx,ny],nt,1);end;
sc=zeros(0,n1);cs=zeros(0,n1);scm=zeros(0,n1);csm=zeros(0,n1);
alpha=zeros(0,2);
u=zeros(0,1);v=zeros(0,1);w=zeros(0,1);dd=zeros(0,1);
if cc==1;chax1=zeros(0,6);chay1=zeros(0,6);chax2=zeros(0,6);chay2=zeros(0,6);chx=zeros(0,2);chdx=zeros(0,2);chalp=zeros(0,2);chtp=zeros(0,2);end;

% calcul de sc scm cs et csm (beta peut etre complexe donc scm n'est pas toujours conj(sc) ...)
parm=1;
m=0;  %  nombre de modes
for it=1:nt;
if dx(it,1)>d(1);mmx=nx-1;parmx=0;else;mmx=mm(it,1);parmx=1;end; 
if dx(it,2)>d(2);mmy=ny-1;parmy=0;else;mmy=mm(it,2);parmy=1;end; 
for mx=0:mmx;

if parmx==1;
alp=mx*pi/dx(it,1);tp=1;d0=dx(it,1);
else;
alp=beta(1,mx+1);tp=2;d0=d(1);    
end;
bet=beta(1,1:nx);x0=x(it,1);[s,c,sm,cm]=calsc(d0,bet,x0,dx,alp,tp,parm);

sx=s;cx=c;sxm=sm;cxm=cm;alpx=alp;

for my=0:mmy;
m=m+1; % creation d'un mode
if parmy==1;
alp= my*pi/dx(it,2);tp=1;d0=dx(it,2);
else;
alp=beta(2,1+nx*my);tp=2;d0=d(2);
end;
bet=beta(2,1:nx:n1);x0=x(it,2);[s,c,sm,cm]=calsc(d0,bet,x0,dx,alp,tp,parm);
alpha=[alpha;[alpx,alp]];

sc=[sc;reshape(sx.'*c,1,n1)];cs=[cs;reshape(cx.'*s,1,n1)];
scm=[scm;reshape(sxm.'*cm,1,n1)];csm=[csm;reshape(cxm.'*sm,1,n1)];

n2=ep(it,3)*ep(it,6);ww=i/ep(it,6-3*pol/2);
dd=[dd;retbidouille(retsqrt(alpha(m,1)^2+alpha(m,2)^2-n2,1),1)];
u=[u;ww*alpha(m,1)*alpha(m,2)/dd(m)];
v=[v;ww*(alpha(m,2)^2-n2)/dd(m)];
w=[w;ww*(alpha(m,1)^2-n2)/dd(m)];

if cc==1;
% amplitudes des champs
if pol==0; % metal electrique
chax1=[chax1;[u(m),v(m),-i/ep(it,6)*alpha(m,2),-1,0,-alpha(m,1)/dd(m)]];
chay1=[chay1;[-w(m),-u(m),i/ep(it,6)*alpha(m,1),0,-1,-alpha(m,2)/dd(m)]];
chax2=[chax2;[u(m),v(m),i/ep(it,6)*alpha(m,2),1,0,-alpha(m,1)/dd(m)]];
chay2=[chay2;[-w(m),-u(m),-i/ep(it,6)*alpha(m,1),0,1,-alpha(m,2)/dd(m)]];
else; % metal magnetique
chax1=[chax1;[1,0,alpha(m,1)/dd(m),u(m),v(m),-i/ep(it,3)*alpha(m,2)]];
chay1=[chay1;[0,1,alpha(m,2)/dd(m),-w(m),-u(m),i/ep(it,3)*alpha(m,1)]];
chax2=[chax2;[1,0,-alpha(m,1)/dd(m),-u(m),-v(m),-i/ep(it,3)*alpha(m,2)]];
chay2=[chay2;[0,1,-alpha(m,2)/dd(m),w(m),u(m),i/ep(it,3)*alpha(m,1)]];
end;
chx=[chx;x(it,:)];          % centre
chdx=[chdx;min(dx(it,:),d)];           % largeur
chalp=[chalp;alpha(m,:)];     % alpha
chtp=[chtp;[(2-parmx)*(1-pol),(2-parmy)*(1-pol)]];%  si pol=0 (electrique): tp= 2 (exp)  tp=  1(cos) 
                                                  %  si pol=2 (magnetique): tp=-2 (exp)  tp= -1(cos)
end;

end;end;end;
if cc==1;cha=cat(3,[chax1;chay1],[chax2;chay2]);chx=[chx;chx];chdx=[chdx;chdx];chalp=[chalp;chalp];chtp=[chtp;chtp];end

% construction de la matrice G
g1=[[sc;zeros(m,n1)],[zeros(m,n1);cs]];
g2=[[csm.'*diag(u);scm.'*diag(v)],[-csm.'*diag(w);-scm.'*diag(u)]];
if pol==0; % metal electrique
if ~isempty(si);g1=g1*si{3};g2=si{2}*g2;end; %  utilisation des symetries
f=find(max(abs(g1).^2,[],2)>eps);m=size(f,1);% elimination des modes nulls 
dd=[dd;dd];dd=dd(f);
g1=g1(f,:);g2=g2(:,f)/(d(1)*d(2));
g1=[[zeros(m,n),g1];[eye(n),zeros(n)]];g2=[[-eye(m),eye(m)];[g2,g2]];
else; % metal magnetique
if ~isempty(si);g1=g1*si{1};g2=si{4}*g2;end; %  utilisation des symetries
f=find(max(abs(g1).^2,[],2)>eps);m=size(f,1);% elimination des modes nulls 
dd=[dd;dd];dd=dd(f);
g1=g1(f,:);g2=g2(:,f)/(d(1)*d(2));
g1=[[g1,zeros(m,n)];[zeros(n),eye(n)]];g2=[[eye(m),eye(m)];[g2,-g2]];
end;
if cc==1;cha=cha(f,:,:);chx=chx(f,:);chdx=chdx(f,:);chalp=chalp(f,:);chtp=chtp(f,:);end;
end;
if cc==1;champ={chx,chdx,chalp,chtp,cha};end;
aa={sparse(g1),sparse(g2),dd,m,n,champ,www,struct('dim',init{end}.dim,'genre',2,'type',2,'sog',0)};



function [s,c,sm,cm]=calsc(d0,bet,x0,dx,alp,tp,parm);% calcul de 'l'integrale sin*epp..' si parm=0 s sm  (E//)  si parm=2 c cm (H//)  s1 parm=1  les 2 (2D)
c=[];cm=[];
if tp==2;c=(bet==alp)*sqrt(d0);s=i*c;cm=c;sm=-s;else;
prv1=retsinc((bet+alp)*d0/(2*pi));prv2=retsinc((bet-alp)*d0/(2*pi));
if parm==1;
c=sqrt(d0/2)*exp(i*bet*x0).*(exp(i*alp*d0/2)*prv1+exp(-i*alp*d0/2)*prv2);
cm=sqrt(d0/2)*exp(-i*bet*x0).*(exp(i*alp*d0/2)*prv2+exp(-i*alp*d0/2)*prv1);
s=-i*sqrt(d0/2)*exp(i*bet*x0).*(exp(i*alp*d0/2)*prv1-exp(-i*alp*d0/2)*prv2);
sm=-i*sqrt(d0/2)*exp(-i*bet*x0).*(exp(i*alp*d0/2)*prv2-exp(-i*alp*d0/2)*prv1);
if alp==0;c=c/sqrt(2);cm=cm/sqrt(2);end;
end;
if parm==2;
s=sqrt(d0/2)*exp(i*bet*x0).*(exp(i*alp*d0/2)*prv1+exp(-i*alp*d0/2)*prv2);
sm=sqrt(d0/2)*exp(-i*bet*x0).*(exp(i*alp*d0/2)*prv2+exp(-i*alp*d0/2)*prv1);
if alp==0;s=s/sqrt(2);sm=sm/sqrt(2);end;
end;
if parm==0;
s=-i*sqrt(d0/2)*exp(i*bet*x0).*(exp(i*alp*d0/2)*prv1-exp(-i*alp*d0/2)*prv2);
sm=-i*sqrt(d0/2)*exp(-i*bet*x0).*(exp(i*alp*d0/2)*prv2-exp(-i*alp*d0/2)*prv1);
end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ffff=ret2li(xx,yy,u,nx,ny,cx,cy,sens,si,nxnymax,pol);
% calcul de la matrice ffff
%  tf(g)=ffff*tf(f)
%     g=u*f
% cx:f continue en x  cy:f continue en y
% les continuitees de g sont inversees par multiplication par u
% sens:parametre indiquant le sens du calcul(facultatif)
%

if nargin<8 sens=1;end;
if nargin<9 si=[];end;
if nargin<10 nxnymax=inf;end;

% essai de simplification de xx yy u
mx=size(xx,2);my=size(yy,2);
x=xx(mx);uu=u(mx,:);
for ii=mx-1:-1:1;if ~all(u(ii,:)==u(ii+1,:));uu=[u(ii,:);uu];x=[xx(ii),x];end;end;
u=uu(:,my);y=yy(my);
for jj=my-1:-1:1;if ~all(uu(:,jj)==uu(:,jj+1));u=[uu(:,jj),u];y=[yy(jj),y];end;end;

n=nx*ny;
if length(retelimine(u(:)))==1;ffff=u(1)*speye(n);return;end;% cas constant (mu sans pml)
mx=size(x,2);my=size(y,2);	
alx=(-nx+1:nx-1)*(2*pi);aly=(-ny+1:ny-1)*(2*pi);
if nargin<8;sens=0;end;
if sens==0;sens=cx|(~cy);end;% permet d'eviter l' inversion finale (sauf si cx=cy=0)

topx=rettoeplitz(1:2*nx-1);topy=rettoeplitz(1:2*ny-1);% pour gain de temps

if sens==1 % de X en Y
if cy;f=retf(y,u,aly);else f=retf(y,1./u,aly);end;    
ff=zeros(ny,ny,mx);
if (cx&cy)|((~cx)&(~cy)) 
for ii=1:mx;ff(:,:,ii)=rettoeplitz(f(ii,:),topy);end
else
for ii=1:mx;ff(:,:,ii)=inv(rettoeplitz(f(ii,:),topy));end
end;
fff=reshape(retf(x,reshape(ff,ny^2,mx),alx),ny,ny,2*nx-1);

if n<nxnymax;  % <-------version 'normale' 
ffff=zeros(nx,nx,ny,ny);
for ii=1:ny;for jj=1:ny; 
ffff(:,:,ii,jj)=rettoeplitz(fff(ii,jj,:),topx);
end;end;
ffff=permute(ffff,[1,3,2,4]);
ffff=reshape(ffff,n,n);
if ~cx;if isempty(si);ffff=inv(ffff);else;ffff=retblkinv(ffff,si{11},si{9},si{10},si{12}(pol));end;end;
% if ~cx; ffff=inv(ffff);end;

ffffi=[];

else;         % <----version 'economique' en memoire
ffff=zeros(nx,nx,ny,ny);
for ii=1:ny;for jj=1:ny;
ffff(:,:,ii,jj)=rettoeplitz(real(fff(ii,jj,:)),topx);
end;end
ffff=permute(ffff,[1,3,2,4]);ffff=reshape(ffff,n,n);ffff=retio(ffff,1);
ffffi=zeros(nx,nx,ny,ny);
for ii=1:ny;for jj=1:ny;
ffffi(:,:,ii,jj)=rettoeplitz(imag(fff(ii,jj,:)),topx);
end;end
ffffi=permute(ffffi,[1,3,2,4]);ffffi=reshape(ffffi,n,n);ffffi=retio(ffffi,1);
if ~cx;if isempty(si);[ffff,ffffi]=retinv(ffff,ffffi);else;[ffff,ffffi]=retblkinv(ffff,si{11},si{9},si{10},si{12}(pol),ffffi);end;end;
end;              % <-----version 'economique'  ?   
       
    
else  % de Y en X
ff=zeros(nx,nx,my);
if cx;f=retf(x,u.',alx);else;f=retf(x,1./u.',alx);end;
if (cx&cy)|((~cx)&(~cy));
for ii=1:my;ff(:,:,ii)=rettoeplitz(f(ii,:),topx);end;
else
for ii=1:my;ff(:,:,ii)=inv(rettoeplitz(f(ii,:),topx));end;
end;
fff=reshape(retf(y,reshape(ff,nx^2,my),aly),nx,nx,2*ny-1);

if n<nxnymax; % <----- version 'normale'
ffff=zeros(ny,ny,nx,nx);
for ii=1:nx;for jj=1:nx;
ffff(:,:,ii,jj)=rettoeplitz(fff(ii,jj,:),topy);
%ffff(ii,:,jj,:)=rettoeplitz(fff(ii,jj,:),topy);
end;end;
ffff=permute(ffff,[3,1,4,2]);ffff=reshape(ffff,n,n);
if ~cy;if isempty(si);ffff=inv(ffff);else;ffff=retblkinv(ffff,si{11},si{9},si{10},si{12}(pol));end;end; 
ffffi=[];

else;          % <----- version 'economique' en memoire
ffff=zeros(ny,ny,nx,nx);
for ii=1:nx;for jj=1:nx;
ffff(:,:,ii,jj)=rettoeplitz(real(fff(ii,jj,:)),topy);
%ffff(ii,:,jj,:)=rettoeplitz(real(fff(ii,jj,:)),topy);
end;end;
ffff=permute(ffff,[3,1,4,2]);ffff=reshape(ffff,n,n);ffff=retio(ffff,1);
ffffi=zeros(ny,ny,nx,nx);
for ii=1:nx;for jj=1:nx;
ffffi(:,:,ii,jj)=rettoeplitz(imag(fff(ii,jj,:)),topy);
%ffffi(ii,:,jj,:)=rettoeplitz(imag(fff(ii,jj,:)),topy);
end;end;
ffffi=permute(ffffi,[3,1,4,2]);ffffi=reshape(ffffi,n,n);ffffi=retio(ffffi,1);
%if ~cy;[ffff,ffffi]=retinv(ffff,ffffi);end;
if ~cy;if isempty(si);[ffff,ffffi]=retinv(ffff,ffffi);else;[ffff,ffffi]=retblkinv(ffff,si{11},si{9},si{10},si{12}(pol),ffffi);end;end;
end;
end;          % <-----version 'economique'  ?  


%if n>=nxnymax;ffff={ffff,ffffi};end;% ???????????
if n>=nxnymax;ffff=retio(ffff)+i*retio(ffffi);end;
ffff=retsparse(ffff);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ffff=ret2li_matrice(xx,yy,u,nx,ny,mx,my,sens);
% u=ret2li_matrice(x,y,U,nx,ny,mx,my,sens)
	% calcul de la matrice ffff
	%  ffff*{tf(g)}={tf(f)}
	% si sens=1: {f} tableau de p fonctions continues en x qui multiplies par une matrice carree u devient
	% un tableau de fonctions continues en y
	% si sens=- 1: {f} tableau de p fonctions continues en y qui multiplies par une matrice carree u devient
	% un tableau de fonctions continues en x
% La matrice est decrite par x y {U} points de discontinuite en x et y et valeurs à droite
% U est un cell array: size(U)=[p,p]  (ici p=3 )
% on obtient la matrice INVERSE
%  G  = U * F  espace reel
%  u *  g  =  f  espace de fourier (matrices)
% sens = -1  F continu en x  G continue en y
% sens = 1 F continu en y  G continue en x

if nargin<8;sens=-1;end;
if iscell(xx);x=xx{1};else;x=xx;end;if iscell(yy);y=yy{1};else;y=yy;end;% pour intégration de gauss
alx=(-nx+1:nx-1)*(2*pi);aly=(-ny+1:ny-1)*(2*pi);
p=size(u,2);
u_elimine=cell(size(u));n_elimine=zeros(size(u));for iii=1:length(u_elimine(:));u_elimine{iii}=retelimine(u{iii});n_elimine(iii)=length(u_elimine{iii})==1;end;	

if all(n_elimine(:));% cte pour gagner du temps;
for iii=1:numel(n_elimine);n_elimine(iii)=u_elimine{iii}(1);end; 
if sens~=0;n_elimine=inv(n_elimine);end;	
ffff=cell(size(u));[ffff{:}]=deal(sparse(nx*ny,nx*ny));for iii=1:numel(n_elimine);ffff{iii}=n_elimine(iii)*speye(nx*ny);end;	
return;end;

topx=rettoeplitz(1:2*nx-1);topy=rettoeplitz(1:2*ny-1);% pour gain de temps
switch sens;
case 1;% <*******   ffff * Tf(fonction continue en y)= Tf(fonction continue en x)
ff=zeros(p*nx,p*nx,my);
for iii=1:p;for jjj=1:p;     % tf en x fonction de y
if 	n_elimine(iii,jjj);for iy=1:my;ff([1:nx]+(iii-1)*nx,[1:nx]+(jjj-1)*nx,iy)=u_elimine{iii,jjj}(1)*speye(nx);end;else;
f=retf(xx,u{iii,jjj}.',alx);
for iy=1:my;ff([1:nx]+(iii-1)*nx,[1:nx]+(jjj-1)*nx,iy)=rettoeplitz(f(iy,:),topx);end;
end;end;end;                      % iii jjj
for iy=1:my;ff(:,:,iy)=inv(ff(:,:,iy));end;
fff=cell(p);[fff{:}]=deal(zeros(nx,nx,ny));
for iii=1:p;for jjj=1:p;fff{iii,jjj}=ff([1:nx]+(iii-1)*nx,[1:nx]+(jjj-1)*nx,:);end;end;
for iii=1:p;for jjj=1:p;fff{iii,jjj}=reshape(retf(yy,reshape(fff{iii,jjj},nx^2,my),aly),nx,nx,2*ny-1);end;end;
ffff=cell(p);[ffff{:}]=deal(zeros(ny,ny,nx,nx));
for iii=1:p;for jjj=1:p;
for ii=1:nx;for jj=1:nx;ffff{iii,jjj}(:,:,ii,jj)=rettoeplitz(fff{iii,jjj}(ii,jj,:),topy);end;end;
ffff{iii,jjj}=permute(ffff{iii,jjj},[3,1,4,2]);		
%for ii=1:nx;for jj=1:nx;ffff{iii,jjj}(ii,:,jj,:)=rettoeplitz(fff{iii,jjj}(ii,jj,:),topy);end;end;

ffff{iii,jjj}=reshape(ffff{iii,jjj},nx*ny,nx*ny);
end;end;

case -1     % <*******   ffff * Tf(fonction continue en x)= Tf(fonction continue en y)
ff=zeros(p*ny,p*ny,mx);
for iii=1:p;for jjj=1:p; % tf en y fonction de x
if 	n_elimine(iii,jjj);for ix=1:mx;ff([1:ny]+(iii-1)*ny,[1:ny]+(jjj-1)*ny,ix)=u_elimine{iii,jjj}(1)*speye(ny);end;else;
f=retf(yy,u{iii,jjj},aly);
for ix=1:mx;ff([1:ny]+(iii-1)*ny,[1:ny]+(jjj-1)*ny,ix)=rettoeplitz(f(ix,:),topy);end;
end;end;end;                 % iii jjj
for ix=1:mx;ff(:,:,ix)=inv(ff(:,:,ix));end;
fff=cell(p);[fff{:}]=deal(zeros(nx,nx,ny));
for iii=1:p;for jjj=1:p;fff{iii,jjj}=ff([1:ny]+(iii-1)*ny,[1:ny]+(jjj-1)*ny,:);end;end;
for iii=1:p;for jjj=1:p;fff{iii,jjj}=reshape(retf(xx,reshape(fff{iii,jjj},ny^2,mx),alx),ny,ny,2*nx-1);end;end;
ffff=cell(p);[ffff{:}]=deal(zeros(nx,nx,ny,ny));
%ffff=cell(p);[ffff{:}]=deal(zeros(nx,ny,nx,ny));
for iii=1:p;for jjj=1:p;
for ii=1:ny;for jj=1:ny;ffff{iii,jjj}(:,:,ii,jj)=rettoeplitz(fff{iii,jjj}(ii,jj,:),topx);end;end;
%for ii=1:ny;for jj=1:ny;ffff{iii,jjj}(:,ii,:,jj)=rettoeplitz(fff{iii,jjj}(ii,jj,:),topx);end;end;
ffff{iii,jjj}=permute(ffff{iii,jjj},[1,3,2,4]);		
ffff{iii,jjj}=reshape(ffff{iii,jjj},nx*ny,nx*ny);
end;end;

case 0;   % <*******  Tf(fonction )= ffff *Tf(fonction continue en x et en y)
ffff=cell(1,p);[ffff{:}]=deal(zeros(nx,ny,nx,ny));% u est un vecteur ligne
ff=zeros(ny,ny,mx);fff=zeros(nx,nx,ny);
for iii=1:p;
f=retf(yy,u{iii},aly); % tf en y fonction de x
for ix=1:mx;ff(:,:,ix)=rettoeplitz(f(ix,:),topy);end;
fff=reshape(retf(xx,reshape(ff,ny^2,mx),alx),ny,ny,2*nx-1);
for ii=1:ny;for jj=1:ny;ffff{iii}(:,ii,:,jj)=rettoeplitz(fff(ii,jj,:),topx);end;end;
ffff{iii}=reshape(ffff{iii},nx*ny,nx*ny);
end;
end;     % <*******
ffff=retsparse(ffff);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [a,b]=retinv(a,b);
%inversion a+ib
a=retio(a,-2);b=retio(b,-2);
am=max(a(:))-min(a(:));bm=max(b(:))-min(b(:));
if am>bm;
if bm<10*eps*am;b=spalloc(size(b,1),size(b,2),0);a=retio(inv(a),1);
else
aa=b*inv(a);
a=inv(a+aa*b);b=-a*aa;
a=retio(a,1);b=retio(b,1);
end;

else
if am<10*eps*bm;a=spalloc(size(a,1),size(a,2),0);b=-inv(retio(b));
else 
bb=-a*inv(b);
b=inv(-b+bb*a);a=b*bb;
a=retio(a,1);b=retio(b,1);
end;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [a,ai]=retblkinv(a,decoupe,N,M,n_bloc,ai);
% b=retblkinv(a,decoupe,N,M,n_bloc);
% M*a*N  blocs de longueur decoupe
% M=inv(N);
% b=inv(a);
% si n_bloc est precisé on ne calcule que la contribution du bloc numero n_bloc 
% a=inv(a);
% si ai est precisé inversion economique en mémoire

if nargin<2;a=inv(a);return;end;
if nargin<5;n_bloc=0;end;

n=length(decoupe);
n2=cumsum(decoupe);n1=1+[0,n2(1:n-1)];
if nargin<6;% version normale                  ****************
	if n_bloc==0;
	b=M*a*N;
	a=0;
	for n_bloc=1:n;if n2(n_bloc)>=n1(n_bloc);a=a+N(:,n1(n_bloc):n2(n_bloc))*inv(b(n1(n_bloc):n2(n_bloc),n(n_bloc):n2(n_bloc)))*M(n1(n_bloc):n2(n_bloc),:);end;end;
	else;
	if n2(n_bloc)>=n1(n_bloc);a=N(:,n1(n_bloc):n2(n_bloc))*inv(M(n1(n_bloc):n2(n_bloc),:)*a*N(:,n1(n_bloc):n2(n_bloc)))*M(n1(n_bloc):n2(n_bloc),:);end
	end;	
	
else;       % version economique en memoire    ****************
	if n_bloc==0;
	b=M*retio(a)*N;bi=M*retio(ai)*N;
	a=0;ai=0;
	for n_bloc=1:n;% <<<<<<<<<<<<
	[prv,prvi]=retinv(b(n1(n_bloc):n2(n_bloc),n(n_bloc):n2(n_bloc)),bi(n1(n_bloc):n2(n_bloc),n(n_bloc):n2(n_bloc)));
	a=a+N(:,n1(n_bloc):n2(n_bloc))*retio(prv)*M(n1(n_bloc):n2(n_bloc),:);
	ai=ai+N(:,n1(n_bloc):n2(n_bloc))*retio(prvi)*M(n1(n_bloc):n2(n_bloc),:);
	end;          % <<<<<<<<<<<<
	else;
	[prv,prvi]=retinv(M(n1(n_bloc):n2(n_bloc),:)*retio(a)*N(:,n1(n_bloc):n2(n_bloc)),M(n1(n_bloc):n2(n_bloc),:)*retio(ai)*N(:,n1(n_bloc):n2(n_bloc)));
	a=N(:,n1(n_bloc):n2(n_bloc))*retio(prv)*M(n1(n_bloc):n2(n_bloc),:);
	ai=N(:,n1(n_bloc):n2(n_bloc))*retio(prvi)*M(n1(n_bloc):n2(n_bloc),:);
	end;
	a=retio(a,1);ai=retio(ai,1);
end;        % version economique en memoire ?  ****************

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%diagonalisation
function [p,d]=retc3(a1,a2,c,moharam,sens);
if nargin<4;moharam=0;end;
if nargin<3;c=1;end;
a=full(a2*a1);clear a1 a2	
%moharam=0;
if moharam>0;% diagonalisation moharam en 2 blocs
n=size(a,1);
if c==-3;p=[];d=[reteig(a(1:moharam,1:moharam));reteig(a(moharam+1:end,moharam+1:end))];
else;
[p1,d1]=reteig(a(1:moharam,1:moharam));[p2,d2]=reteig(a(moharam+1:end,moharam+1:end));d=diag([diag(d1);diag(d2)]);
if sens>0;% a12=0
pp=((p2\a(moharam+1:end,1:moharam))*p1)./(repmat(diag(d1).',n-moharam,1)-repmat(diag(d2),1,moharam));p=[p1,zeros(moharam,n-moharam);p2*pp,p2];
else;    % a21=0
pp=((p1\a(1:moharam,moharam+1:end))*p2)./(repmat(diag(d2).',moharam,1)-repmat(diag(d1),1,n-moharam));p=[p1,p1*pp;zeros(n-moharam,moharam),p2];
end;
% retcompare(a*p,p*d)
d=diag(d);
end;

else; % diagonalisation complete
if c==-3;d=reteig(a);p=[];else;[p,d]=reteig(a);d=diag(d);end;
end;
d=retsqrt(d,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a1,a2,k,aa2]=retc2(mu,mmu,ep,beta,cao,a1,aa2);%n=length(beta);
%calcul de la matrice m=[0,a1;a2,0] associee au passage dans un milieu periodique
%invariant en y
%mu :coefficients de fourier de mu (2*n-1 valeurs le coefficient d'ordre 0 etant range en mu(n))
%mmu:idem pour 1/mu   ep:idem pour epsilon
%beta:n constantes de propagation en x du developpement de rayleigh

if length(cao)>0;
%k=rettoeplitz(cao)*diag(beta); %  sans apodisation
% apod=7+n/1500;if apod>7.5;apod=9;end;apod=9;
%cao(n)=cao(n)-1;k=(diag(retchamp([7+min(.5,n/1500),n]))*rettoeplitz(cao)+eye(n))*diag(beta);% avec apodisation 
%cao(n)=cao(n)-1;k=(diag(retchamp([apod,n]))*rettoeplitz(cao)+eye(n))*diag(beta);% avec apodisation 
k=apodisecao(cao)*diag(beta);% avec apodisation 
else k=diag(beta);end;
if nargin<7;aa2=[];end;if isempty(aa2);
a1=inv(rettoeplitz(mmu));aa2=inv(rettoeplitz(mu));
end;
a2=k*aa2*k-rettoeplitz(ep);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function cao=apodisecao(cao)
% n=ceil(length(cao)/2);
% apod=7+n/1500;if apod>7.5;apod=9;end;
% cao(n)=cao(n)-1;cao=diag(retchamp([apod,n]))*rettoeplitz(cao)+eye(n);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cao=apodisecao(cao)
n=ceil(length(cao)/2);
apod=7+n/1500;if apod>7.5;apod=1;end;
cao(n)=cao(n)-1;
pprv=sum(abs(rettoeplitz(cao)).^2,2)-sum(abs(cao).^2);
for ii=1:5;pprv=(pprv([1,1:end-1])+pprv([2:end,end]))/2;end; % lissage
if n<100;cao=diag(retchamp([apod,n]))*rettoeplitz(cao)+eye(n);
else;cao=diag(10.^(-20*sqrt(abs(pprv))))*rettoeplitz(cao)+eye(n);end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fmu,fmmu,fep]=retc1(x,ep,beta);n=length(beta);alpha=(-n+1:n-1)*(2*pi);
fmu=retf(x,ep(2,:),alpha);fmmu=retf(x,ep(3,:),alpha);fep=retf(x,ep(1,:),alpha);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

