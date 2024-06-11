function [q,jj]=retfps(init,h0,a,x,sh,sb,ii,jmax);
%function [q,jj]=retfps(init,h0,a,x,sh,sb,ii,jmax);
% recherche des resonances dans une couche invariante de descripteur a
%   les resonances correspondent aux zeros de q
%   le champ incident est une source ponctuelle placee dans la couche dont on annule la 'reponse'
%
%   x: position de la source (dans le plan transversal) ATTENTION a respecter les symetries...
%   en hauteur la source est au milieu (h0/2)   
%   
% sh:matrice S du haut de la couche au haut du systeme  (champ->modes)  
% sb:matrice S du bas du systeme au bas de la couche (modes->champs)  si absent on renverse sh   
%  (sh et sb et a peuvent avoir ete stockés sur fichier par retio(s,1))
%
% h0:hauteur de la couche
%
%   si jmax=0 
%      q est la 'reponse de la source' placee au milieu de la couche      pour les hauteurs h0(vecteur)
%
%   si jmax~=0   h0+q sont les hauteurs complexes voisine de h0 qui annulent la  'reponse de la source' 
%      obtenue en jj<jmax iterations h0 q jj vecteurs de meme dimension
%        si jmax est absent jmax=0
%
%   a: descripteur de la couche
%
%   ii: tableau d' indices correspondant a la polarisation de la source 
%      (on travaille sur 1/(somme des inverses des reponses))
%       si ii est absent ou nul ou prend toutes les valeurs
%       si ii(1) a une partie imaginaire:c'est l'apodisation de la source
%       (les polarisations sont alors real(ii))
%   en 1D :EZ,HX,HY  ii prend les valeurs 1,2 ou 3
%   en 2D :EX,EY,EZ,HX,HY,HZ  ii prend les valeurs 1,2,3,4,5 ou 6
%
% See also RETFP,RETFPM

sh=retio(sh);if nargin<6;sb=[];end;if isempty(sb);sb=retrenverse(sh,2);end;sb=retio(sb);
if nargin<7;ii=[];end;if isempty(ii);if init{end}.dim==1;ii=[1:6];else;ii=[1:3];end;end;
if nargin<8;jmax=0;end;
a=retio(a);

n=init{1};
source=rets(init,x,a,real(ii),real(ii),imag(ii(1)));
sh=rettronc(sh,[],[],1);sb=rettronc(sb,[],[],-1);

q=zeros(size(h0));jj=zeros(size(h0));

if jmax==0;% calcul pour h0
q=zeros(size(h0));
for ii=1:length(h0);q(ii)=fonc(h0(ii),sh,sb,a,source);end;
else; % recherche des h tel que  fonc(h+q...)=0
for ii=1:length(h0);
er=inf;z=[];zz=[];h=h0(ii);jj(ii)=0;
while((er>1.e-10)&(jj(ii)<jmax));jj(ii)=jj(ii)+1;z=[z;h];zz=[zz;fonc(h,sh,sb,a,source)];[h,z,zz,er]=retcadilhac(z,zz);end;
q(ii)=h-h0(ii);
end;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function q=fonc(h,sh,sb,a,source);
ss=retc(a,h/2);s=retss(sh,ss,source,ss,sb);s=retgs(s);q=1./sum(diag(s{1}));


