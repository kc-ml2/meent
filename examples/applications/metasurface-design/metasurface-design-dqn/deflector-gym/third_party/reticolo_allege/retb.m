function [s,b,bet,c,angles,ray]=retb(init,a,sens,norm,fi,fd);
% function [s,b,bet,c,angles,ray]=retb(init,a,sens,norm,fi,fd);
%
% calcul de la matrice S (ou G) aux bornes en haut ou en bas (conditions aux limites) 
%   real(sens)<0 en bas  (modes->champ) 
%   real(sens)>0 en haut   (champ->modes)
%   a le descripteur de la couche peut avoir ete stocké sur fichier par retio(a,1)
%   norm =1 ou absent: normalisation des modes propagatifs
%   pour le choix du repere u v dans le cas degenere de l'onde plane normale (par defaut 0)
%     (uniquement cas 2D si angles est indique )
%   
%   fi,fd permettent de faire suivre betb d'une troncature(rettronc(sb,fi,fd,sign(sens))
%   (important pour les problemes de memoire car la natrice totale n'est pas cree en S) 
%*******************************************************************************
% le cell array  ray permet d'ecrire le developpement de RAYLEIGH des modes (diriges vers le bas)
%
%   en 1D  ray={beta,EZ,i*HX}            cell array  de matrice colonnes
%   en 2D  ray={beta(:,2),EX,EY,HX,HY}   cell array  de matrice colonnes(sauf beta)
%
%   
%   beta;composantes du developpement de pseudo fourier
%   les champs correspondent au mode se dirigeant vers le BAS 
%   pour le mode se dirigeant vers le haut;E inchange H devient -H
%      
%    les colonnes correspondent aux modes      
%    les lignes a la base de fourier  
%   dans le cas de symetries il y a plus de lignes que de colonnes
%   ATTENTION; ray demande beaucoup de memoire ...
%*******************************************************************************
%   certains modes consideres comme propagatifs peuvent etre selectionnes 
%   si norm est absent ou =1,ces modes selectionnes sont normalises (flux poynting)
%      La matrice S donne alors directement les efficacitee: abs(S).^2
%   S'il n'y a pas de pertes et que l'on a conservé tous les ordres propagatifs,
%   on doit avoir sum(abs(S).^2)=[1,1,  1]
%   Dans le cas d'un milieu homogène, leur forme est 'en général simple': 
% 	Par exemple, pour une onde normale au plan du reseau,une composante de E ou H vaut 1.
% 	Il est néanmoins tres conseillé d' utiliser la sortie c qui donne les composantes de ces champs.
% 	
%   abs(real(sens)):tolérance sur la partie reelle des modes selectionnes
%*******************************************************************************
%    ON PEUT DE PLUS OBTENIR DES INFORMATIONS SUR LES MODES SELECTIONNES
%*******************************************************************************
%   b(:,1):leurs numeros    b(:,2):leurs constantes de propagation (en z)
%
%   pour les ondes planes:
%   bet: projection horizontale du vecteur d'onde
%   c:champ a l'origine en 2D:(EX,EY,HX,HY)   en 1D:(EZ,i*HX) 
%   dans le cas d'utilisation de symetries bet et c se referent a UNE des ondes planes constituant le mode
%   (si le mode n'est pas une onde plane on prend la composante de fourier dominante)
%   si imag(sens)~=0 trace de ces modes
%
%*******************************************************************************
%*******************************************************************************
%  mise en forme des ondes planes incidentes et diffractees pour un reseau en 2D
%
%   angles={teta,delta,psip,kp,up,vp,eep,hhp,eeep,hhhp,psim,km,um,vm,eem,hhm,eeem,hhhm};
%             1    2     3   4  5  6  7   8    9   10   11  12 13 14  15  16  17   18
%   dans le cas d'utilisation de symetries ces donnees
%   se referent a UNE des ondes planes constituant le mode
%   la polarisation est alors telle que h soit porte soit par ox soit par oy
%
%*******************************************************************************
% kp vecteur d'onde UNITAIRE dirige vers le haut
% vp unitaire perpendiculaire au plan d'incidence  (le triedre 0z, kp ,vp etant direct)  
% up=vp vect. kp (le triedre up vp kp est direct) 
% pour l'incidence normale   up=[cos(delt),sin(delt)]   vp=[-sin(delt),cos(delt)]   kp=oz
%   
% eep:composantes de E sur up et vp
% hhp:composantes de H sur up et vp
% psip:angle(oriente par kp) entre up et la direction principale de polarisation de E
%  (en general 0 ou pi/2 exactement)
%
% eeep  hhhp  amplitudes de E et H dans le repere deduit de up vp par rotation d'angle psip
%
%*************************************************************
% km vecteur d'onde UNITAIRE dirige vers le bas
% vm=vp unitaire perpendiculaire au plan d'incidence  (le triedre 0z, km ,vm est direct)  
% um=vm vect. km (le triedre um vm km est direct) 
% pour l'incidence normale  up=[-cos(delt),-sin(delt)]   vp=[-sin(delt),cos(delt)]   kp=-oz
%   
% eem:composantes de E sur um et vm
% hhm:composantes de H sur um et vm
% psim:angle(oriente par km) entre um et la direction principale de polarisation de E
%  (en general 0 ou pi/2 exactement)
%
% eeem  hhhm  amplitudes de E et H dans le repere deduit de um vm par rotation d'angle psim
%
%*************************************************************
% teta=angle de oz avec kp (0 a pi/2)
% delta:angle de oy avec vp oriente par oz  (-pi a pi
% psi:angle de up avec abs(eep) (0 a pi/2)
%*******************************************************************************
%*******************************************************************************
%  mise en forme des ondes planes incidentes et diffractees pour un reseau en 1D
%
%   angles={teta,kp,up,eep,hhp,km,um,eem,hhm};
%
%   dans le cas d'utilisation de symetries ces donnees
%   se referent a UNE des ondes planes constituant le mode
%
%*******************************************************************************
% kp vecteur d'onde UNITAIRE dirige vers le haut
% up= perpendiculaire a kp  (oz up kp direct) 
% eep:composante de E sur oz 
% hhp:composante de H sur up
%
% km vecteur d'onde UNITAIRE dirige vers le bas
% um= perpendiculaire a km  (oz um km direct) 
%
% teta=angle de oy avec kp (-pi/2 a pi/2)
%
%*******************************************************************************
% See also: RETRESEAU,RETHELP_POPOV


if ~isstruct(init{end}); % cas 0 D 'rapide'
if isfinite(a);
if sens>0;[prv1,prv2,prv3,prv4,prv5,s]=retabeles(init{1},[a,a],0,init{2},init{3},[],0);      % en haut
else;[prv1,prv2,prv3,prv4,prv5,prv6,s]=retabeles(init{1},[a,a],0,init{2},init{3},[],0);end;  % en bas
else;
s={zeros(length(init{3}),length(init{2})),zeros(length(init{3}),length(init{2}));zeros(length(init{3}),length(init{2})),zeros(length(init{3}),length(init{2}))};
if sens>0;
if (a>0 & init{1}==0) | (a<0 & init{1}==2);s{2,1}(:)=1.e100; % metal electrique
else;s{2,2}(:)=1.e100;end;                                   % metal magnetique  
else;
	
if (a>0 & init{1}==0) | (a<0 & init{1}==2);s{2,1}(:)=1;s{2,2}(:)=1;% metal electrique
else;s{1,1}(:)=1;s{1,2}(:)=1;end;                                  % metal magnetique
end;
end;
return;end;       % fin cas 0 D 'rapide'

if isfield(init{end},'L')&~isempty(init{end}.L);s=retb_cylindrique_radial(init,a,sens,norm);if nargin>4;s=rettronc(s,fi,fd,sign(real(sens)));end;return;end;% cylindrique_radial

a=retio(a);

if a{end}.type==3;% modes de bloch non symetriques ou milieux homogenes non isotropes 
if real(sens)>0;s=a{2};else;s=a{1};end;b=[];bet=[];c=[];angles=[];ray=[];
if nargin>4;s=rettronc(s,fi,fd,sign(real(sens)));end;
return;
end; 

if a{end}.type==5;% textures inclinees
n=init{1};sog=init{end}.sog;
if real(sens)>0;s=retgs(a{2},1-sog);else;s=retgs(a{1},1-sog);end;
if nargin>4;s=rettronc(s,fi,fd,sign(real(sens)));end;
return;
end;

if nargin<4;norm=1;end;
if a{end}.type==2;  % <---metaux-------------------
g1=a{1};g2=a{2};d=a{3};m=a{4};n=a{5};
b=find(abs(real(d))<abs(real(sens)));
if isempty(b);b=zeros(0,2);else;b=[b,d(b)];end;
if init{end}.dim==1;% 1 D
angles=cell(1,9);ray=cell(1,3);bet=zeros(0,1);c=zeros(0,2);
else;                % 2 D
angles=cell(1,19);ray=cell(1,5);bet=zeros(0,2);c=zeros(0,4);
end;
if real(sens)>0;  % dessus  champ->modes
s={g1;g2;n;n;m;m};
else;             % dessous  modes->champ
s={g2;g1;m;m;n;n};
end
%troncature  
if nargin>4;s=rettronc(s,fi,fd,sign(real(sens)));end; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

else;            %<----dielectriques -------------------   
n=init{1};beta=init{2};
parm=nargout;
               %%%%%%%%%%%%%%%%%%%%
               %      1  D        % 
               %%%%%%%%%%%%%%%%%%%%

if init{end}.dim==1; %1D

si=init{3};sog=init{end}.sog;p=a{1};q=a{3};d=a{5};%dd=init{end}.d;
angles=cell(1,9);ray=cell(1,3);b=zeros(0,2);bet=zeros(0,1);c=zeros(0,2);

b=find(abs(real(d))<abs(real(sens))&abs(d)>eps);%&abs(imag(d))>eps);

if ~isempty(b);  % traitement des modes propagatifs

if ~isempty(si) ;% passage dans la base de fourier pour calculer la norme
p1=si{1}*p(:,b);q1=si{1}*q(:,b);
if parm>2;
[prov,bet]=sort(-abs(p1),1);bet=bet(1,:).'; %numero de la composante de fourier dominante
c=zeros(length(b),2);
for ii=1:length(b);c(ii,:)=[q1(bet(ii),ii),d(b(ii))*p1(bet(ii),ii)];end;% champ EZ  i*HX  en 0
end;
if norm==1;bb=sqrt(abs(imag(sum(q1.*conj(p1*diag(d(b))),1)))).*exp(.5i*angle(sum(q1(:,:).*(p1(end:-1:1,:)*diag(d(b))),1)))*sqrt(i);;end;% norme 
else;  % pas de symetries   
if parm>2;
[prov,bet]=sort(-abs(p(:,b)),1);bet=bet(1,:).';  % numero de la composante de fourier dominante
c=zeros(length(b),2);
for ii=1:length(b);c(ii,:)=[q(bet(ii),b(ii)),d(b(ii))*p(bet(ii),b(ii))];end;% champ EZ  i*HX  en 0
end;
if norm==1;bb=sqrt(abs(imag(sum(q(:,b).*conj(p(:,b)*diag(d(b))),1)))).*exp(.5i*angle(sum(q(:,b).*(p(end:-1:1,b)*diag(d(b))),1)))*sqrt(i);end;% norme 
%if norm==1;bb=sqrt(abs(imag(sum(q(:,b).*conj(p(:,b)*diag(d(b))),1))));end;% norme 
end;

if norm==1; % normalisation
if ~isempty(bb);bb=repmat(bb,n,1);
f=find(abs(bb(1,:))>eps&abs(imag(d(b).'))>eps);
p(:,b(f))=p(:,b(f))./bb(:,f);q(:,b(f))=q(:,b(f))./bb(:,f);
if parm>2;c(f,:)=c(f,:)./(bb(1,f).'*[1,1]);end;
end;
end;
if parm>2;bet=beta(bet).';end;

b=[b,d(b)];

else;
b=zeros(0,2);bet=zeros(0,1);c=zeros(0,2);
end;  % fin de traitement des modes propagatifs    

if parm>=6;% matrice du developpement de Rayleigh
if ~isempty(si);ray={beta.',sparse(si{1}*q),sparse(si{1}*p*diag(d))};else;ray={beta.',sparse(q),sparse(p*diag(d))};end;
end;    


%  mise en forme des ondes planes incidentes et diffractees pour un reseau en 1D
if parm>4&~isempty(b);
km=[bet,b(:,2)/i];km=km./repmat(sqrt(sum((km.^2),2)),1,size(km,2));% vecteurs d'onde normalises
um=[km(:,2),-km(:,1)];% um= -ez vect.  km
hhm=zeros(size(bet,1),1);f=find(abs(um(:,1))>eps);
hhm(f)=-i*c(f,2)./um(f,1);eem=c(:,1); % on multiplie par -i car c(:,2)est i HX
kp=km;kp(:,2)=-kp(:,2);
up=um;up(:,1)=-up(:,1);
eep=eem;hhp=hhm;
teta=atan2(real(kp(:,1)),real(kp(:,2)));
angles={teta,kp,up,eep,hhp,km,um,eem,hhm};
end

%trace des modes si imag(sens)=~=0
if imag(sens)~=0;rettmode(init,a,b(:,1));drawnow;end;

               %%%%%%%%%%%%%%%%%%%%
               %      2D        % 
               %%%%%%%%%%%%%%%%%%%%
else; %2 D


angles=cell(1,19);ray=cell(1,5);b=zeros(0,2);bet=zeros(0,2);c=zeros(0,4);
si=init{8};p=a{1};q=a{3};d=a{5};delt=init{end}.delt;sog=init{end}.sog;
b=find(abs(real(d))<abs(real(sens))&abs(d)>eps);%&abs(imag(d))>eps);

if ~isempty(b);  % traitement des modes propagatifs

if ~isempty(si);%passage dans la base de fourier pour calculer la norme (cas d' utilisation des symetries)
p1=si{3}*p(:,b);q1=si{1}*q(:,b);n1=size(si{1},1);
if parm>2;  % calcul de bet et c dans le cas d'utilisation de symetries
[prov,bet]=sort(-abs(p1(1:n1/2,:))-abs(p1(n1/2+1:n1,:)),1);bet=bet(1,:).'; %numero de la composante de fourier dominante
c=zeros(length(b),4);
for ii=1:length(b);c(ii,:)=[q1(bet(ii),ii),q1(bet(ii)+n1/2,ii),d(b(ii))*p1(bet(ii),ii),d(b(ii))*p1(bet(ii)+n1/2,ii)];end;%champ EX EY HX HY  en 0,0
end
if norm==1;bb=sqrt(abs(real(sum(q1(1:n1/2,:).*conj(p1(n1/2+1:n1,:)*diag(d(b)))-q1(n1/2+1:n1,:).*conj(p1(1:n1/2,:)*diag(d(b))),1)))).*exp(.5i*angle(sum(q1(1:n1/2,:).*(p1(n1:-1:n1/2+1,:)*diag(d(b)))-q1(n1/2+1:n1,:).*(p1(n1/2:-1:1,:)*diag(d(b))),1)))*i;end;
% 'terme de phase' pour normalisation du produit scalaire si termes de fourier symetriques(de -n a n)
else;  % pas de symetries    

if parm>2; % calcul de bet et c 
[prov,bet]=sort(-abs(p(1:n/2,b))-abs(p(n/2+1:n,b)),1);bet=bet(1,:).'; %numeros de la composante de fourier dominante
c=zeros(length(b),4);
for ii=1:length(b);c(ii,:)=[q(bet(ii),b(ii)),q(bet(ii)+n/2,b(ii)),d(b(ii))*p(bet(ii),b(ii)),d(b(ii))*p(bet(ii)+n/2,b(ii))];end;%champ EX EY HX HY  en 0,0
end
if norm==1;bb=sqrt(abs(real(sum(q(1:n/2,b).*conj(p(n/2+1:n,b)*diag(d(b)))-q(n/2+1:n,b).*conj(p(1:n/2,b)*diag(d(b))),1)))).*exp(.5i*angle(sum(q(1:n/2,b).*(p(n:-1:n/2+1,b)*diag(d(b)))-q(n/2+1:n,b).*(p(n/2:-1:1,b)*diag(d(b))),1)))*i;end;
% 'terme de phase' pour normalisation du produit scalaire si termes de fourier symetriques
end


if norm==1; % normalisation
if ~isempty(bb);bb=repmat(bb,n,1);
f=find(abs(bb(1,:))>eps&abs(imag(d(b).'))>eps);
p(:,b(f))=p(:,b(f))./bb(:,f);q(:,b(f))=q(:,b(f))./bb(:,f);
if parm>2;c(f,:)=c(f,:)./(bb(1,f).'*[1,1,1,1]);end
end;
end;
if parm>2;bet=beta([1,2],bet).';end;

b=[b,d(b)];

else;
b=zeros(0,2);bet=zeros(0,2);c=zeros(0,4);
end;  % fin de traitement des modes propagatifs    

if parm>=6;% matrice du developpement de Rayleigh
if ~isempty(si);n1=size(si{1},1);
ray={beta.',sparse(si{1}(1:n1/2,:))*q,sparse(si{1}(1+n1/2:n1,:))*q,sparse(si{3}(1:n1/2,:)*p*diag(d)),sparse(si{3}(1+n1/2:n1,:)*p*diag(d))};
else;
ray={beta.',sparse(q(1:n/2,:)),sparse(q(1+n/2:n,:)),sparse(p(1:n/2,:)*diag(d)),sparse(p(1+n/2:n,:)*diag(d))};
end;

end;    

%  mise en forme des ondes planes incidentes et diffractees pour un reseau en 2D
if parm>4&~isempty(b);
km=[bet,b(:,2)/i];km=km./repmat(sqrt(sum((km.^2),2)),1,size(km,2));%vecteurs d'onde normalises
vm=[-km(:,2),km(:,1),zeros(size(km,1),1)];
ii=find(abs(km(:,1).^2+km(:,2).^2)<100*eps);vm(ii,1)=-sin(delt);vm(ii,2)=cos(delt);%incidence normale: delt valeur par defaut de delta
vm=vm./repmat(sqrt(sum((vm.^2),2)),1,size(vm,2));
um=[km(:,3).*vm(:,2),-km(:,3).*vm(:,1),km(:,2).*vm(:,1)-km(:,1).*vm(:,2)];%um= vm vect.  km

em=zeros(size(bet,1),3);hm=em;ep=em;hp=em;f=find(abs(km(:,3))>eps);
em(f,:)=[c(f,1),c(f,2),-(c(f,1).*km(f,1)+c(f,2).*km(f,2))./km(f,3)];
hm(f,:)=[c(f,3),c(f,4),-(c(f,3).*km(f,1)+c(f,4).*km(f,2))./km(f,3)];
eem=[sum(em.*um,2),sum(em.*vm,2)];
hhm=[sum(hm.*um,2),sum(hm.*vm,2)];
kp=km;kp(:,3)=-kp(:,3);
vp=vm;up=um;up(:,1:2)=-up(:,1:2);

ep(f,:)=[c(f,1),c(f,2),-(c(f,1).*kp(f,1)+c(f,2).*kp(f,2))./kp(f,3)];
hp(f,:)=-[c(f,3),c(f,4),-(c(f,3).*kp(f,1)+c(f,4).*kp(f,2))./kp(f,3)];% - car onde dans l'autre sens(par convention E est inchange et H oppose)
eep=[sum(ep.*up,2),sum(ep.*vp,2)];
hhp=[sum(hp.*up,2),sum(hp.*vp,2)];

  % analyse des polarisations
eeep=eep;eeem=eem;hhhp=hhp;hhhm=hhm;
psip=.5*atan2(real(eep(:,1)).*real(eep(:,2))+imag(eep(:,1)).*imag(eep(:,2)),.5*(abs(eep(:,1)).^2-abs(eep(:,2)).^2));
eeep(:,1)=eep(:,1).*cos(psip)+eep(:,2).*sin(psip);eeep(:,2)=-eep(:,1).*sin(psip)+eep(:,2).*cos(psip);
ii=find(abs(eeep(:,2))>abs(eeep(:,1)));psip(ii)=psip(ii)+pi/2;[eeep(ii,1),eeep(ii,2)]=deal(eeep(ii,2),-eeep(ii,1));
ii=find(psip<=-pi/2+100*eps);psip(ii)=psip(ii)+pi;eeep(ii,:)=-eeep(ii,:);
hhhp(:,1)=hhp(:,1).*cos(psip)+hhp(:,2).*sin(psip);hhhp(:,2)=-hhp(:,1).*sin(psip)+hhp(:,2).*cos(psip);
ii=find(abs(psip-pi/2)<100*eps);psip(ii)=pi/2;ii=find(abs(psip)<100*eps);psip(ii)=0;

psim=.5*atan2(real(eem(:,1)).*real(eem(:,2))+imag(eem(:,1)).*imag(eem(:,2)),.5*(abs(eem(:,1)).^2-abs(eem(:,2)).^2));
eeem(:,1)=eem(:,1).*cos(psim)+eem(:,2).*sin(psim);eeem(:,2)=-eem(:,1).*sin(psim)+eem(:,2).*cos(psim);
ii=find(abs(eeem(:,2))>abs(eeem(:,1)));psim(ii)=psim(ii)+pi/2;[eeem(ii,1),eeem(ii,2)]=deal(eeem(ii,2),-eeem(ii,1));
ii=find(psim<=-pi/2+100*eps);psim(ii)=psim(ii)+pi;eeem(ii,:)=-eeem(ii,:);
hhhm(:,1)=hhm(:,1).*cos(psim)+hhm(:,2).*sin(psim);hhhm(:,2)=-hhm(:,1).*sin(psim)+hhm(:,2).*cos(psim);
ii=find(abs(psim-pi/2)<100*eps);psim(ii)=pi/2;ii=find(abs(psim)<100*eps);psim(ii)=0;

teta=acos(kp(:,3));
delta=atan2(-real(vm(:,1)),real(vm(:,2)));
angles={teta,delta,psip,kp,up,vp,eep,hhp,eeep,hhhp,psim,km,um,vm,eem,hhm,eeem,hhhm};
else;angles=cell(1,18);end;

%trace des modes si imag(sens)=~=0
if imag(sens)~=0&~isempty(b);rettmode(init,a,b(:,1));end


end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   remplissage de la matrice S ou G    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear a init si;
d=retbidouille(d,2);
if sog==1 % matrices s
if nargin<5;fi=1:n;fd=1:n;end;if length(fi)==1&fi==0;fi=1:n;end;% pour troncatures
if (length(fd)==1)&(fd==0);fd=1:n;end;ni=length(fi);nd=length(fd);

%calcul des matrices s
sp=(issparse(p)&issparse(q));
s1=speye(n);s1=s1(fd,fi);
if (real(sens)>0);%dessus
q=retpinv(q);p=p*retdiag(d);s=p*q;p=p(:,fi);
s=[-s,2*p];clear p d;q=q(fd,:);
s={[[q,-s1];s];n;nd;1};
if ~issparse(q);s{1}=full(s{1});end;
else %dessous
p=retdiag(1./d)*retpinv(p);s=q*p;q=q(:,fi);
s=[2*q,s];clear q d;p=p(fd,:);
s={[s;[s1,p]];ni;n;1};
end
if ~sp;s{1}=full(s{1});end; % important pour le temps de calcul de retss !!

else; %matrices g
% calcul des matrices g
p=p*diag(d);
if (real(sens)>0);%dessus
s={eye(2*n);[[q;-p],[q;p]];n;n;n;n};
else %dessous
s={[[q;-p],[q;p]];eye(2*n);n;n;n;n};
end
%troncature  
if nargin>4;s=rettronc(s,fi,fd,sign(real(sens)));end; 
    
end;    

end; % <-------dielectriques--------------------


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%                  cylindrique radial                    %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function s=retb_cylindrique_radial(init,a,sens,r);
L=init{end}.L;

a=retio(a);
[Q_HE,Q_HH,Q_EH,Q_EE,D,P_E,P_H]=deal(a{1:7});clear a;
ne=size(Q_EE,1);nh=size(Q_HE,1);

H1_E=retbessel('h',[L-1,L,L+1],1,retcolonne(i*D(1:ne)*r),1);
H1_H=retbessel('h',[L-1,L,L+1],1,retcolonne(i*D(ne+1:ne+nh)*r),1);

H2_H=retbessel('j',[L-1,L,L+1],retcolonne(i*D(ne+1:ne+nh)*r),1);
H2_E=retbessel('j',[L-1,L,L+1],retcolonne(i*D(1:ne)*r),1);

% premier indice vp
% second indice [L-1,L,L+1]

ne=size(Q_EE,1);nh=size(Q_HE,1);
M11=[[zeros(nh,ne),P_H*retdiag(H1_H(:,2))];[Q_EE*retdiag(H1_E(:,1)-H1_E(:,3)),Q_EH*retdiag(H1_H(:,1)+H1_H(:,3))]];
M12=[[zeros(nh,ne),P_H*retdiag(H2_H(:,2))];[Q_EE*retdiag(H2_E(:,1)-H2_E(:,3)),Q_EH*retdiag(H2_H(:,1)+H2_H(:,3))]];
M21=[[P_E*retdiag(H1_E(:,2)),zeros(ne,nh)];[Q_HE*retdiag(H1_E(:,1)+H1_E(:,3)),Q_HH*retdiag(H1_H(:,1)-H1_H(:,3))]];
M22=[[P_E*retdiag(H2_E(:,2)),zeros(ne,nh)];[Q_HE*retdiag(H2_E(:,1)+H2_E(:,3)),Q_HH*retdiag(H2_H(:,1)-H2_H(:,3))]];
n=ne+nh;
    
if init{end}.sog==1;% matrices S
if sens>0;% en haut (exterieur)
%s={[[-M11,zeros(n)];[-M21,eye(n)]]\[[-eye(n),M12];[zeros(n),M22]];n;n;1};	
s={retantislash([[-M11,zeros(n)];[-M21,eye(n)]],[[-eye(n),M12];[zeros(n),M22]]);n;n;1};	
else;      % en bas (interieur)
%s={[[eye(n),-M12];[zeros(n),-M22]]\[[M11,zeros(n)];[M21,-eye(n)]];n;n;1};
s={retantislash([[eye(n),-M12];[zeros(n),-M22]],[[M11,zeros(n)];[M21,-eye(n)]]);n;n;1};
end;

else;% matrices G
if (real(sens)>0);%dessus
s={eye(2*n);[M11,M12;M21,M22];n;n;n;n};
else %dessous
s={[M11,M12;M21,M22];eye(2*n);n;n;n;n};
end;

end;



