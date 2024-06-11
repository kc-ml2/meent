function [q,jj]=retfpm(hh,d,sh,sb,n0,jmax,tol,np,nl);
% function  [q,jj]=retfpm(hh,d,sh,sb,n0,jmax,tol);          recherche des resonances  (premiere forme)
% function  [h,y]=retfpm(p,d,sh,sb,n0,jmax,tol,np,nl);  recherche des p premieres resonances (seconde forme)
%  recherche des resonances du F P generalise troncature automatique (tol)
%
%  le champ 'incident' est le 'mode interne' de numero abs(n0)
%  d: valeurs propres du milieu intermediaire (obtenues par [aa,d]=retcouche(...)
%  sh: matrice s du demi probleme haut (le milieu intermediaire est le bas  le haut est le haut)
%  sb: matrice s du demi probleme bas  (le bas est le bas   le milieu intermediaire est le haut)
%  sb et sh sont de type mode->mode de preference tronquees a l'exterieur([],[])  MAIS PAS A L'INTERIEUR 
%        si jmax est absent ou [] jmax=0
%        si tol est absent ou [] tol=1.e-6
%        si sb est absent ou [] sb=retrenverse(sh)
%
% remarque: sh et sb et a peuvent avoir ete stockés sur fichier par retio(s,1)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  PREMIERE FORME imag(n0)=0 function  [q,jj]=retfpm(hh,d,sh,sb,n0,jmax,tol);
%   les resonances correspondent aux zeros de q
%   si jmax=0    q est la 'reponse du mode interne'   pour les hauteurs hh
%
%   si jmax~=0   hh+q sont les hauteurs complexes voisine de hh qui annulent la  'reponse du mode interne n0' 
%      obtenue en jj<jmax iterations (hh q jj vecteurs de meme dimension)
%
%        si jmax est absent ou [] jmax=0
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SECONDE FORME  imag(n0)~=0 function  [h,y]=retfpm(p,d,sh,sb,n0,jmax,tol,np,nl);
%  recherche des premieres resonances en h du F P obtenues en jmax iterations au plus 
%  si y existe il retourne une repartition de points 'bien repartis' pour le trace de le transmission du F P
%   (np points sur nl fois la largeur) par defaut np=31,nl=30
%        si jmax est absent ou [] jmax=30
% ...................................................................................................
%  si imag(n0)>0:
%  si p est un scalaire  on recherche les p premieres resonances
%  si p est un vecteur de dimension 2 on cherche les resonances dont la partie reelle est comprise entre p(1) et p(2)
%  (quand les resonances ne sont pas obtenues en jmax iterations les valeurs de h sont estimees en fonction de d(n0))
% ...................................................................................................
%  si imag(n0)<0:
%  p:vecteur de complexes,les valeurs de h retournees sont les resonances les plus proches de p
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% See also RETFP,RETFPS


sh=retio(sh);if isempty(sb);sb=retrenverse(sh);end;sb=retio(sb);
if ~isempty(hh);p=length(hh);end;
if nargin<7;tol=[];end;if isempty(tol);tol=1.e-6;end;
if nargin<6;jmax=[];end;

if imag(n0)~=0;  % p premieres resonances (seconde forme)
if isempty(jmax);jmax=30;end;
if nargin<8;np=31;end;
if nargin<9;nl=30;end;
if imag(n0)>0;[q,jj]=retfph(hh,d,sh,sb,real(n0),jmax,tol,[],np,nl);else;[q,jj]=retfph(length(hh),d,sh,sb,real(n0),jmax,tol,hh,np,nl);end
return;
end;



if isempty(jmax);jmax=0;end;


d=retbidouille(d,5);
jj=0;
sh=retgs(sh);sb=retgs(sb);
sh=rettronc(sh,[],[],1);sb=rettronc(sb,[],[],-1);

q=zeros(size(hh));jj=zeros(size(hh));
if jmax==0;% calcul pour h0
for ii=1:length(hh);q(ii)=fonc(hh(ii),d,sh,sb,n0,tol,0);end;
else; % recherche de h complexe tel que fonc(h+q...)=0
for ii=1:length(hh);
er=inf;z=[];zz=[];h=hh(ii);jj(ii)=0;
while((er>1.e-10)&(jj(ii)<jmax));jj(ii)=jj(ii)+1;z=[z;h];zz=[zz;fonc(h,d,sh,sb,n0,tol,1)];[h,z,zz,er]=retcadilhac(z,zz);end;
q(ii)=h-hh(ii);
end;
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function q=fonc(h,d,sh,sb,n0,tol,parm);tol=-log(tol);
dd=-d*h;ddd=dd;ddd(n0)=0;f=find(abs(real(ddd))<tol);nn0=find(f==n0);sh=rettronc(sh,f,f,-1);sb=rettronc(sb,f,f,1);
if parm==0;
dd=retdiag(exp(dd(f)));
s=(eye(length(f))-sh{1}*dd*sb{1}*dd)\sh{1}(:,nn0);q=1/s(nn0);
else;
dm=retdiag(exp(dd(f).*(real(dd(f))<=0)));dp=retdiag(exp(-dd(f).*(real(dd(f))>0)));
s=zeros(2*length(f),1);s(nn0)=dp(nn0,nn0);s=[[dp,-sb{1}*dm];[sh{1}*dm,-dp]]\s;q=1/s(nn0);
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [h,y]=retfph(p,d,sh,sb,n0,jmax,tol,hh,np,nl);
%function  [h,y]=retfph(p,d,sh,sb,n0,jmax,tol,hh,np,nl);
%  recherche des premieres resonances en h du F P obtenues en jmax iterations au plus 
%   si p est un scalaire  on recherche les p premieres resonances
%   si p est un vecteur de dimension 2 on cherche les resonances dont la partie reelle est comprise entre p(1) et p(2)
%  d: valeurs propres du milieu intermediaire (obtenues par [aa,d]=retcouche(...)
%  sh: matrice s du demi probleme haut (le milieu intermediaire est le bas  le haut est le haut)
%  sb: matrice s du demi probleme bas  (le bas est le bas   le milieu intermediaire est le haut)
%  sb et sh sont de type mode->mode de preference tronquees a l'exterieur([],[])  MAIS PAS A L'INTERIEUR 
%  si les resonances ne sont pas obtenues en jmax iterations les valeurs de h sont estimees en fonction de d(n0)
%  si hh est precise ou non vide,les valeurs de h retournees sont les resonances les plus proches
%   (alors p ne sert plus  p==length(hh))
%
%        si jmax est absent jmax=30
%        si tol est absent tol=1.e-6
%        si sb est absent sb=retrenverse(sh)
%  si y existe il retourne une repartition de points 'bien repartis' pour le trace de le transmission du F P
%   (np points sur nl fois la largeur) par defaut np=31,nl=30

dh=pi/abs(imag(d(n0)));
if length(p)==1;hmin=0;p=round(p);hmax=(p+1)*dh;else hmin=min(p);hmax=max(p);p=inf;end;

kmax=10;%nombre max d'essais 
if isempty(hh);  % pas de valeurs de depart
h=[];kk=0;
h0=max(hmin-dh/2,3*dh);h0=min(h0,hmax+dh/2);a=1;% en decroissant
while (real(h0)>hmin-dh/2)&(kk<kmax);
hh0=h0;hz=dh;ii=jmax;k=1;while (ii>=jmax)&(abs(hz)>dh/2)&(k<kmax);[hz,ii]=retfpm(hh0,d,sh,sb,n0,jmax,tol);hz=hz+hh0;hh0=h0+a*dh*(rand-.5);k=k+1;end;

if k>=kmax;  %echec
if isempty(h);h=hmax;end;
while (real(h(1))>hmin)&(length(h)<p);h=[h(1)-dh,h];end;h0=hmin-dh; 
else; % valeur acceptee
h0=hz;
if (real(h0)<=hmax);h=[h0,h];a=.1;kk=0;else;kk=kk+1;end;% stockage
if (length(h)>=2)&(abs(h(2)-h(1))>.5*dh)&(abs(h(2)-h(1))<1.5*dh);h0=2*h(1)-h(2);else;h0=h0-dh;end;%estimation 
end;    
end;
h=h(find((real(h)>=hmin)&(real(h)<=hmax)));

if isempty(h);h0=hmin-dh;a=1;else;h0=h(end)+dh;a=.1;end;% en croissant
while (length(h)<p)&(real(h0)<hmax+dh/2);
hh0=h0;hz=dh;ii=jmax;k=1;while (ii>=jmax)&(abs(hz)>dh/2)&(k<kmax);[hz,ii]=retfpm(hh0,d,sh,sb,n0,jmax,tol);hz=hz+hh0;hh0=h0+a*dh*(rand-.5);k=k+1;end;

if k>=kmax;  %echec
if isempty(h);h=hmin;end;
while (real(h(end))<hmax)&(length(h)<p);h=[h,h(end)+dh];end;h0=hmax+dh; 
else; % valeur acceptee
h0=hz;
h=[h,h0];a=.1;% stockage
if (length(h)>=2)&(abs(h(end)-h(end-1))>.5*dh)&(abs(h(end)-h(end-1))<1.5*dh);h0=2*h(end)-h(end-1);else;h0=h0+dh;end;%estimation 
end;    
end;

h=h(find((real(h)>=hmin)&(real(h)<=hmax)));
if ~isempty(h);h=h(1:min(p,length(h)));else;h=dh*[0:p-1];end;
if isfinite(p)&(length(h)<p);while (length(h)<p);h=[h,h(end)+dh];end;end; 


else;  % valeurs de depart donnees
h=hh;  
for m=1:p;h0=hh(m);hh0=h0;
ii=jmax;k=1;while (ii>=jmax)&(k<kmax);[hz,ii]=retfpm(hh0,d,sh,sb,n0,jmax,tol);h(m)=hh0+hz;hh0=h0+.7*dh*(rand-.5);k=k+1;end;if k>=kmax;h(m)=hh(m);end;    
end;    
end;    
    

y=0;
if nargout==2;
for ii=1:length(h);
if imag(h(ii))~=0;
y=[y,retlorentz(h(ii),np,nl/2)];    
else;
y=[y,real(h(ii))+linspace(dh/np-dh/2,dh/2,np)];    
end;
end ;   
y=y(find((y>=hmin)&(y<=hmax)));y=sort(y);y=[y(find(y(1:end-1)~=y(2:end))),y(end)];
end ;   

