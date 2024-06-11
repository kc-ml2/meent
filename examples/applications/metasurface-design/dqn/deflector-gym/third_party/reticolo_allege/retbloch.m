function [a,d,v,test]=retbloch(init,s,h,tol);
%  function [a,d,v,test]=retbloch(init,s,h,tol);
%
%  calcul du descripteur de texture a equivalent à un tronçon de longueur h
%   test permet de verifier si s=retc(a,h) (doit etre petit)
%
%  si real(tol)=inf:seulement valeurs propres des modes du milieu equivalent (a 2*pi*i/h pres..)
%  d=retbloch(init,s,h,tol);avec tol=inf ou tol=i+inf
%
%    si imag(tol)~0   maille non symetrique 
%         si imag(tol)>0  beta = 0  on utilise le mode reciproque
%         si imag(tol)<0  beta ~= 0   
%    si imag(tol)==0  maille symetrique 
%
%  s: matrice S (ou G) de la maille
%  d: valeurs propres des modes du milieu equivalent (à 2*pi*i/h pres..)
%  v:vecteurs propres associees
%  si t=retgs(s,2) est la matrice t associee à  s ,      t*v = v*diag(exp(d*h))
%                        w=v;w(n+1:2*n,:)=-w(n+1:2*n,:); t*w = w*diag(exp(-d*h))
%
%  tri en 'entrants' et 'sortants' des vecteurs propres (pouvant poser probléme) dont la partie reelle de la valeur propre est <real(tol) 
%  ATTENTION:il s'agit de real(log(valeur_propre_matrice_T)) et non des d ( d=log(valeur_propre_matrice_T)/h    tol_sur_d=tol*h   )
%     si s est une seule matrice S:suivant le signe du flux du vecteur de poynting 
%     si s ={ s0,   s1,  s2...  }  est forme de plusieurs matrices S calculees pour k=k0+kk0*i k=k0+kk1*i  k0+kk2*i  k0+kk3*i ...
%           { kk0, kk1,  kk2...}}  (si les kk1 ne sont pas indiques :0,1,4,4^2..
%  dans tous les cas les modes de bloch sont ceux de la PREMIERE matrice S les autres ne servant que pour le tri 
%  par defaut tol=0 
%
% exemple: nbk=3;kz=[0,2.^[0:nbk-1]];for ii=nbk:-1:1;k=k0+kz(ii)*1.e-6i;calcul pour k ...  s{ii}= ...;end
%   ceci permet de terminer par le calcul a k=k0 reel et d'utiliser les valeurs dans la suite des calculs
%  remarque: s  peut avoir ete stocké sur fichier par retio(s,1)
%
%  NORMALISATION
%  si les ordres de Fourier sont symetriques (-n:n ) les modes entrantes et sortants ont un produit scalaire imaginaire pur (-4i flux de poynting) 
%
% USAGE PARTICULIER: construire les vecteurs propres d'un maillage de texture a
%  [a,d,v]=retbloch(a); d valeurs propres  v vecteurs propres vers le bas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            en 0D
%    [sh_bloch,sb_bloch,d,ASB]=retbloch(init,s,h,n_bloch); 
%   s matrice de passage du tronçon de hauteur h ,n_bloch: indice du milieu ou on veut calculer ASB (falcultatif)
%    sh_bloch :passage champs  --> modes de bloch  
%    sb_bloch :passage  modes de bloch --> champs 
%     ( Les modes de bloch propagatifs ont un flux du vecteur de poynting egal à .5)
%    d(2,nk,nb)   valeurs propres des modes du milieu equivalent (a 2*pi*i/h pres..)
%      (  d(1,:,:) : vers le haut,d(2,:,:) : vers le bas )
%    ASB(2,2,nk,nb),composantes sur les ondes planes (vers le haut ,vers le bas )du milieu d'indice n_bloch 
%           des 2 modes de Bloch ( vers le haut ,vers le bas ) 
%           n_bloch peut etre remplace par un sh
% See also:RETTBLOCH,RETMAYSTRE,RETABELES

% valeurs propres et vecteurs propres d'un descripteur de texture
if nargin<2;
a=init;d=a{5};v=[a{3};a{1}*retdiag(d)];	
return;end;



% 0D vectorialise [sh,sb,D,ASB]=retblochD(init,s,h,n_bloch);
if size(init,2)<4;if nargin<4;tol=[];end;
[a,d,v,test]=retbloch0D(init,s,h,tol,nargout);
return;end;

s=retio(s);if length(s)==1;s=retio(s{1});end;
n=init{1};
if nargin<4;tol=0;end;test=0;

if imag(tol)==0;   %  MAILLE SYMETRIQUE

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %     MAILLE  SYMETRIQUE            %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(s,2)==1; % 1 seul k0
s=retgs(s);
if ~isfinite(real(tol));a=reteig(s{1}(1:n,1:n));a=acosh(1./a)/h;return;end;% seulement les modes
[q,d]=reteig(s{1}(1:n,1:n));
d=acosh(1./diag(d));
p=-s{1}(n+1:2*n,1:n)*q*diag(1./tanh(d));
% p est calcule ici car on en a besoin pour poynting si d change de signe on multiplie la colonne de p par -1
if  nargin<3;a=q;v=p;return;end;% seulement diagonalisation
%  tri  des modes entrants et sortants
else; % s={s(k0), s(k0+i*epsilon) } tri fonction de k0+i*epsilon
[q,d,p]=retbloch(init,s{1,1});
ff=find(real(d)<tol); % valeurs de d a trier

if ~isempty(ff);% on traite k0+i*epsilon pour faire le tri
dtri=d(ff);if size(s,1)==1;t=0;else;t=s{2,1};end;
for ii=2:size(s,2);[qi,di]=retbloch(init,s{1,ii}); 
[prv,ffi]=max(abs(qi\q(:,ff)));% numeros correspondants aux ff dans di
dtri=[dtri,di(ffi)];
if size(s,1)==1;t=[t,4^(ii-2)];else;t=[t,s{2,ii}];end;
end;
f=ff(tri(dtri,t)); %tri

d(f)=-d(f);p(:,f)=-p(:,f);
end;
end; % fin du tri avec k0 et k0+i*epsilon

% calcul du flux du vecteur de poynting 
if (nargin>=4)&(size(s,2)==1);f=find(abs(real(d))<tol);if ~isempty(f);
p1=p(:,f);q1=q(:,f);
if init{end}.dim==1; %1D
si=init{3};
if ~isempty(si);p1=si{1}*p1;q1=si{1}*q1;end    
f=f(find(imag(sum(q1.*conj(p1),1))<0));
else;               %2D
si=init{8};
if ~isempty(si);p1=si{3}*p1;q1=si{1}*q1;end;
n1=size(q1,1);
f=f(find(real(sum(q1(1:n1/2,:).*conj(p1(n1/2+1:n1,:))-q1(n1/2+1:n1,:).*conj(p1(1:n1/2,:)),1))>0));
end;
d(f)=-d(f);p(:,f)=-p(:,f);
end;end;% fin du tri en fonction du flux du vecteur de poynting 

d=d/h;
if nargout>2;v=[q;p];end;
p=p*retdiag(1./d);
qq=inv(q);pp=inv(p);
if init{end}.dim==1; %1D
a={p,pp,q,qq,d,[],[],[],init{end}.d,init{2},struct('dim',1,'genre',2,'type',1,'sog',init{end}.sog)};
else;               %2D
a={p,pp,q,qq,d,[],[],[],[],[],[],{[],[],[]},init{end}.d,init{2},init{11},struct('dim',2,'genre',2,'type',1,'sog',init{end}.sog)};
end; 

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %  MAILLE NON SYMETRIQUE    BETA=0        %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else;if imag(tol)>0;
tol=real(tol);  

if size(s,2)==1; % 1 seul k0
s=retgs(s);
if ~isfinite(real(tol));a=calvd(s)/h;return;end;% seulement les modes

[v,d]=calvd(s,n);
if nargin<3;return;end;% seulement diagonalisation

%  tri  des modes entrants et sortants
% ATTENTION ce tri suppose que pour les modes 'litigieux'  on puisse etablir 
% une correspondance entrants <--> sortants  ce qui n'est pas toujours possible si beta~=0

else; % s={s(k0), s(k0+i*epsilon  } tri fonction de k0+i*epsilon

[v,d]=calvd(s{1},n);
ff=find(abs(real(d(1:n)))<tol); % valeurs de d a trier

if ~isempty(ff);% on traite k0+i*epsilon pour faire le tri
dtri=d(ff);if size(s,1)==1;t=0;else;t=s{2,1};end;
for ii=2:size(s,2);[vi,di]=calvd(s{1,ii},n);
[prv,ffi]=max(abs(vi\v(:,ff)));% numeros correspondants aux ff dans di
dtri=[dtri,di(ffi)];
if size(s,1)==1;t=[t,4^(ii-2)];else;t=[t,s{2,ii}];end;
end;

f=ff(tri(dtri,t)); %tri

d([f;f+n])=-d([f+n;f]);v(:,[f;f+n])=v(:,[f+n;f]);% intervertion entrants <---> sortants
end;end; % fin du tri avec k0 et k0+i*epsilon

% calcul du flux du vecteur de poynting 
if (nargin>=4)&(size(s,2)==1);f=find(abs(real(d(1:n)))<tol);if ~isempty(f);
q1=v(1:n,f);p1=v(n+1:2*n,f);
if init{end}.dim==1; %1D
si=init{3};
if ~isempty(si);p1=si{1}*p1;q1=si{1}*q1;end    
f=f(find(imag(sum(q1.*conj(p1),1))>0));
else;               %2D
si=init{8};
if ~isempty(si);p1=si{3}*p1;q1=si{1}*q1;end;
n1=size(q1,1);
f=f(find(real(sum(q1(1:n1/2,:).*conj(p1(n1/2+1:n1,:))-q1(n1/2+1:n1,:).*conj(p1(1:n1/2,:)),1))<0));
end;                   % 1D 2 D
d([f;f+n])=-d([f+n;f]);v(:,[f;f+n])=v(:,[f+n;f]);% intervertion entrants <---> sortants
end;end;% fin du tri en fonction du flux du vecteur de poynting 
d=d/h;
v=normalise(v,n,init,d);% normalisation
if init{end}.sog==1;sb=retgs(v);sh=retgs(inv(v));else;sb=retgs(v,1);sh=retgs(inv(v),1);end;
a={sb,sh,v,[],d,struct('dim',init{end}.dim,'genre',2,'type',3,'sog',init{end}.sog)};

else;
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %  MAILLE NON SYMETRIQUE    BETA~=0       %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tol=real(tol);  

if size(s,2)==1; % 1 seul k0
if init{end}.sog==0;s_g=s;s=retgs(s);;end;
if ~isfinite(real(tol));a=calvd(s)/h;return;end;% seulement les modes

[v,d]=calvd(s,-n);masque=2*(real(d)>0)-1;

if nargin<3;return;end;% seulement diagonalisation

%  tri  des modes entrants et sortants

else; % s={s(k0), s(k0+i*epsilon  } tri fonction de k0+i*epsilon

[v,d]=calvd(s{1},-n);masque=2*(real(d)>0)-1;
ff=find(abs(real(d))<tol); % valeurs de d a trier

if ~isempty(ff);% on traite k0+i*epsilon pour faire le tri
dtri=d(ff).*masque(ff);if size(s,1)==1;t=0;else;t=s{2,1};end;
for ii=2:size(s,2);[vi,di]=calvd(s{1,ii},-n);mmasque=2*(real(di)>0)-1;
[prv,ffi]=max(abs(vi\v(:,ff)));% numeros correspondants aux ff dans di
dtri=[dtri,di(ffi).*mmasque(ffi)];
if size(s,1)==1;t=[t,4^(ii-2)];else;t=[t,s{2,ii}];end;
end;
f=ff(tri(dtri,t)); %tri

masque(f)=-masque(f);
end; %  ff non vide
end; % fin du tri avec k0 et k0+i*epsilon

% calcul du flux du vecteur de poynting 
if (nargin>=4)&(size(s,2)==1);f=find(abs(real(d))<tol);
if ~isempty(f);
q1=v(1:n,f);p1=v(n+1:2*n,f);
if init{end}.dim==1; %1D
si=init{3};
if ~isempty(si);p1=si{1}*p1;q1=si{1}*q1;end    
f=f(find((imag(sum(q1.*conj(p1),1))).*(masque(f)).'>0));
else;               %2D
si=init{8};
if ~isempty(si);p1=si{3}*p1;q1=si{1}*q1;end;
n1=size(q1,1);

f=f(find((real(sum(q1(1:n1/2,:).*conj(p1(n1/2+1:n1,:))-q1(n1/2+1:n1,:).*conj(p1(1:n1/2,:)),1))).*(masque(f)).'<0));
end;
masque(f)=-masque(f);
end;%  f non vide
end;% fin du tri en fonction du flux du vecteur de poynting 


[fp,fm]=retfind(masque>0);mp=length(fp);mm=length(fm);
% tentative de mise en ordre
if mp<=mm;
[prv,ii]=sort(abs(real(d(fp))));fp=fp(ii); % classement par attenuation croissante
i2=couples(d(fp),-d(fm));fm=fm(i2);
else;
[prv,ii]=sort(abs(real(d(fm))));fm=fm(ii);% classement par attenuation croissante
i2=couples(d(fm),-d(fp));fp=fp(i2);
end;

v=v(:,[fp;fm]);

%d=[d(fp);-d(fm)];
%mp=length(fp);mm=length(fm);
%if mp>mm;d(n+1:mp)=abs(real(d(n+1:mp)))+i*imag(d(n+1:mp));end;% cas pathologique
%if mp<mm;d(mp:n)=abs(real(d(mp:n)))+i*imag(d(mp:n));end;% cas pathologique

v=normalise(v,n,init,d);% normalisation

if init{end}.sog==1;sb=retgs(v);sh=retgs(inv(v));else;sb=retgs(v,1);sh=retgs(inv(v),1);s=s_g;end;
% recalcul des valeurs propres 
if size(s,2)==1;ss=reteval(retss(sh,s,sb));else;ss=reteval(retss(sh,s{1},sb));end;d=-log(diag(ss))/h;

a={sb,sh,v,[],d,struct('dim',init{end}.dim,'genre',2,'type',3,'sog',init{end}.sog)};

end;end; % FIN SYMETRIE DE LA MAILLE  

if nargout>=4;if size(s,2)==1;test=retcompare(retgs(s),retgs(retc(a,h)));else;test=retcompare(retgs(s{1}),retgs(retc(a,h)));end;end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v,d]=calvd(s,n);s=retgs(s);
if nargout<2;v=log(retds(s));return;end;% seulement valeurs propres
[v,d]=retds(s);dd=log(diag(d));% diagonalisation
if n<0;d=-dd;return;end;
% tri des valeurs propres
[f1,ff0]=retfind(real(dd)>=.1);[prv,iii]=sort(real(dd(f1)));f1=f1(iii);% f1:modes croissants du moins au plus attenue   
[f2,f0]=retfind(real(dd(ff0))<=-.1);f2=ff0(f2);[prv,iii]=sort(real(dd(f2)));f2=f2(iii);% f2:modes decroissants du plus au moins attenue   
f0=ff0(f0);[prv,iii]=sort(-imag(dd(f0)));f0=f0(iii);% f0 modes propagatifs (tries en 2 groupes)

nn=length(f0);if mod(nn,2)==0;nn=nn/2;% tri'plus soigne' des propagatifs
i2=couples(dd(f0),-dd(f0));
f=find(i2==1:2*nn|i2(i2)~=1:2*nn);nnn=floor(length(f)/2);i2(f(1:2:2*nnn))=f(2:2:2*nnn);i2(f(2:2:2*nnn))=f(1:2:2*nnn); % traitement des pas normauxo
k=[];kk=[];
for ii=1:2*nn; 
if retcal(ii,[kk;k])&retcal(i2(ii),[kk;k]);k=[k;ii];kk=[kk;i2(ii)];end;
end;
if length(k)==nn&length(kk)==nn;f0=f0([k;flipud(kk)]);end;
end; % fin du tri soigne

f=[f2;f0;f1];%   montants    propagatifs  descendants
v=v(:,f); % mise en ordre des vecteurs propres  

% s dans la base modale
sb=retgs(v);sh=retgs(inv(v));ss=reteval(retss(sh,s,sb));d=-log(diag(ss));
d1=d(1:n);d2=d(n+1:2*n);
[prv,i1]=sort(real(d1.'));

%recherche des valeurs propres 'homologues'
i2=couples(d1(i1),d2);
d=[d1(i1);d2(i2)];v=v(:,[i1,i2+n]);

%correction et intervertion des d a partie reelle<0
f=find(real(d(1:n))<0);
d([f;f+n])=-d([f+n;f]);v(:,[f;f+n])=v(:,[f+n;f]);
%f=find(real(d(1:n))<0);d(f)=d(f+n);% cas pathologique ...
f=find(real(d(1:n))<0);d(f)=-conj(d(f));% cas pathologique ...


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function i2=couples(d1,d2); % tri de d1 et d2 par couples :d1,d2(i2) (length(d1)<=length(d2))
n1=length(d1);n2=length(d2);dd2=d2;i2=zeros(1,n1);err=zeros(1,n1);
for ii=1:n1;err(ii)=min(abs2ipi(d1(ii)-d2));end;[errr,iiii]=sort(err);% classement des erreurs
for ii=iiii;[prv,i2(ii)]=min(abs2ipi(d1(ii)-dd2));dd2(i2(ii))=nan;end;% le tri est fait dans l'ordre des couples les plus proches
if n2>n1;i2=[i2,setdiff(1:n2,i2)];end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=abs2ipi(z); % distance modulo 2*i pi
a=sqrt(real(z).^2+(mod(imag(z)+pi,2*pi)-pi).^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f=tri(dtri,t);
[g,gg]=retfind(real(dtri(:,1))~=0);dd=exp(dtri(gg,:));
ff=find(abs(dd(:,1)-1./dd(:,2))<abs(dd(:,1)-dd(:,2)));
d=cosh(dtri(g,:));
fff=zeros(size(d,1),1);for ii=1:length(fff);fff(ii)=choix(t.',d(ii,:).');end;f=find(fff);
f=sort([g(f);gg(ff)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v=normalise(v,n,init,d);
v=v*retdiag(1./sqrt(sum(abs(v).^2,1)));% normalisation de v
% phase de v telle que le 'produit scalaire' des modes soit imaginaire pur
q1=v(1:n,:);p1=v(n+1:2*n,:);
if init{end}.dim==1; %1D
si=init{3};
if ~isempty(si);p1=si{1}*p1;q1=si{1}*q1;end;
bb=exp(-i*(-pi/2+angle(sum(q1(:,n+1:2*n).*p1(end:-1:1,1:n),1)-sum(q1(:,1:n).*p1(end:-1:1,n+1:2*n),1))));

else;               %2D
si=init{8};
if ~isempty(si);p1=si{3}*p1;q1=si{1}*q1;end;
n1=size(q1,1);
bb=exp(-i*angle(sum(q1(1:n1/2,n+1:2*n).*p1(n1:-1:n1/2+1,1:n)-q1(n1/2+1:n1,n+1:2*n).*p1(n1/2:-1:1,1:n),1)-sum(q1(1:n1/2,1:n).*p1(n1:-1:n1/2+1,n+1:2*n)-q1(n1/2+1:n1,1:n).*p1(n1/2:-1:1,n+1:2*n),1)));
end;                   % 1D 2 D
v(:,1:n)=v(:,1:n)*retdiag(bb);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f=choix(t,z);% prolongement parametrique 
t=t-t(1);
n=length(t);aa=repmat(t(2:n),1,n-1).^repmat([n-1:-1:1],n-1,1);aa=aa\(z(2:end)-z(1));aa=[aa;z(1)]; % approximation polynomiale(plus precise) attention si z(1)=0
%n=length(t);aa=repmat(t,1,n).^repmat([n-1:-1:0],n,1);aa=aa\z; % approximation polynomiale
%aa=polyfit(t,z,length(t)-1); % approximation polynomiale
%aa=(repmat(t,1,length(t)).^repmat(length(t)-1:-1:0,length(t),1))\z; % approximation polynomiale (plus precise ? )
zz=roots(imag(aa));
zz=zz(imag(zz)==0);%  zeros reels
zz=zz(zz>=0);%  zeros reels>0 


if ~isempty(zz);
[prv,ii]=min(abs(zz));
zz=zz(ii);
if length(t)>2;
if abs(zz)>3*t(end);f=choix(t([1;end]),z([1;end]));return;end;% si zz est trop grand on se contente de 2 termes
if abs(zz)<100*eps*t(end);f=choix(t([1:2]),z([1:2]));return;end;  % si zz est nul on se contente de 2 termes
end;
f=abs(real(polyval(aa,zz)))<1; % real<1  ??
else f=0;end;
% pour tests
% tt=linspace(t(1),t(end)+10*(t(end)-t(1)),100);zzz=polyval(aa,tt);z0=polyval(aa,t);
% figure;plot(real(zzz),imag(zzz),'.',real(z0(1)),imag(z0(1)),'or',real(z0(2)),imag(z0(2)),'og',real(z0(end)),imag(z0(end)),'ob');grid;title(rettexte(f,z(1),zz,z0,t),'fontsize',7);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





function [Th,Tb,D,ASB]=retbloch0D(init,T,h,n_bloch,ngout); % cas 0D
tol=1.e-10;% pour tri des modes propagatifs
sz=size(T{1,1});nk=sz(1);nb=sz(2);ne=nk*nb;
D=zeros(ne,2);Tb=zeros(ne,2,2);Th=zeros(ne,2,2);
b=(T{1,1}+T{2,2})/2;c=T{1,1}.*T{2,2}-T{1,2}.*T{2,1};
b=b(:);c=c(:);
d=sqrt(b.^2-c);
[f,ff]=retfind(abs(b+d)>abs(b-d));% meilleure precision du calcul
D(f,1)=b(f)+d(f);D(f,2)=c(f,1)./D(f,1);
D(ff,1)=b(ff)-d(ff); D(ff,2)=c(ff,1)./D(ff,1);
f=find(abs(D(:,2))<abs(D(:,1)));D(f,:)=D(f,[2,1]);% tri du sens des atténuées (abs(D(:,1))<=abs(D(:,2))  le premier mode va vers le haut



Tb(:,1,1)=T{1,2}(:);Tb(:,2,1)=D(:,1)-T{1,1}(:);
Tb(:,1,2)=T{1,2}(:);Tb(:,2,2)=D(:,2)-T{1,1}(:);
Th(:,1,1)=D(:,1)-T{2,2}(:);Th(:,2,1)=T{2,1}(:);
Th(:,1,2)=D(:,2)-T{2,2}(:);Th(:,2,2)=T{2,1}(:);
f=find((abs(Th(:,1,1)).^2+abs(Th(:,2,1)).^2)>(abs(Tb(:,1,1)).^2+abs(Tb(:,2,1)).^2));Tb(f,:,1)=Th(f,:,1);
f=find((abs(Th(:,1,2)).^2+abs(Th(:,2,2)).^2)>(abs(Tb(:,1,2)).^2+abs(Tb(:,2,2)).^2));Tb(f,:,2)=Th(f,:,2);
% 
f=find((abs(Tb(:,1,1)).^2+abs(Tb(:,2,1)).^2+abs(Tb(:,1,2)).^2+abs(Tb(:,2,2)).^2)<100*eps^2);
Tb(f,1,1)=1;Tb(f,2,1)=i;Tb(f,1,2)=1;Tb(f,2,2)=-i;% cas dégénéré

[ff,fff]=retfind(D==0);D(ff)=-1.e100;D(fff)=log(D(fff));
[ff,fff]=retfind(abs(real(D(:,1)))<tol);% test de poynting ff:propagatifs , fff: non propagatifs
poynting=-imag(Tb(ff,1,1).*conj(Tb(ff,2,1)));
f=find(poynting<0);f=ff(f);  % numeros des modes 'à renverser'

poynting=1./sqrt(abs(poynting));% normalisation des propagatifs 
for jj=1:2;prv=exp(-i*angle(Tb(ff,1,jj))).*poynting;for ii=1:2;Tb(ff,ii,jj)=Tb(ff,ii,jj).*prv;end;end;
poynting=1./sqrt(abs(Tb(fff,1,1)).^2+abs(Tb(fff,1,1)).^2);% normalisation des non propagatifs
for jj=1:2;prv=exp(-i*angle(Tb(fff,1,jj))).*poynting;for ii=1:2;Tb(fff,ii,jj)=Tb(fff,ii,jj).*prv;end;end;

[D(f,2),D(f,1)]=deal(D(f,1),D(f,2));
[Tb(f,:,2),Tb(f,:,1)]=deal(Tb(f,:,1),Tb(f,:,2));


b=1./(Tb(:,1,1).*Tb(:,2,2)-Tb(:,1,2).*Tb(:,2,1));% Th=inv(Tb) 
Th(:,1,1)=Tb(:,2,2).*b;Th(:,2,2)=Tb(:,1,1).*b;
Th(:,1,2)=-Tb(:,1,2).*b;Th(:,2,1)=-Tb(:,2,1).*b;


Tb={reshape(Tb(:,1,1),nk,nb),reshape(Tb(:,1,2),nk,nb);reshape(Tb(:,2,1),nk,nb),reshape(Tb(:,2,2),nk,nb)};
Th={reshape(Th(:,1,1),nk,nb),reshape(Th(:,1,2),nk,nb);reshape(Th(:,2,1),nk,nb),reshape(Th(:,2,2),nk,nb)};

% DD=exp(D*h);DD=cat(3,cat(4,DD(:,:,1),zeros(size(DD(:,:,1)))),cat(4,zeros(size(DD(:,:,2))),DD(:,:,2)));
% retcompare(T,retss(Tb,DD,Th))

if ngout>2;D=D/h;D=reshape(D,nk,nb,2);D=permute(D,[3,1,2]);end;
ASB=[];
if (ngout>3)&(~isempty(n_bloch));% calcul de ASB
	
if isnumeric(n_bloch);ASB=retgs(retss(retb(init,n_bloch,1),Tb),2);else;ASB=retgs(retss(n_bloch,Tb),2);end;
end;




