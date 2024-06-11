
function [n,beta,d]=retneff(d,bornes,attenuation,estimee);
% function [n,beta,d]=retneff(d,bornes,attenuation);
%  recherche des valeurs propres d'une texture
%  dont la partie reelle (absorption) est inferieure en valeur absolue a: attenuation (facultatif)
%  et dont la partie imaginaire est :
% --->    si bornes est un scalaire: le plus voisine(en valeur absolue) de bornes
% --->    si bornes est de dimension 2 comprise(en valeur absolue) strictement entre bornes(1) et bornes(2);
%             (et classes par abs(imag) decroissantes)
%
%  en entree :d est soit le tableau des valeurs propres
%  soit le descripteur de la texture calcule par retcouche
% ( eventuellement stocké sur disque par retio)
%  en sortie :d est le tableau des valeurs propres (non triees)
%  n et beta :numeros et valeurs des valeurs propres selectionnees et triees
%
% si une seule entree :  d=retneff(a)
%  ou [n,beta,d]=retneff(d) les beta=d(n) etant classes par ordre d'ATTENUATION DECROISSANTE 
%  ( et pour ceux d'attenuation semblable(à 1.e-8) par ordre de PROPAGATION CROISSANTE )    
%
% (ATTENTION dans bornes et attenuation à multiplier les indices effectifs 'physiques ' par k=l2*pi/ld si ld~= 2*pi)
%   d et beta sont les VALEURS PROPRES et non les indices effectifs
%   en pratique: neff= i*d / (2*pi/ld)
%
%*********************************************************************************************************
% autre utilisation: chercher l'indice effectif neff_vrai,d'un mode de Bloch,
% à partir de la valeur neff definie à 2*pi/(k0*h) pres, et compte tenu d'une valeur estimee 
% neff_vrai=retneff(neff,k0,h,estimee);
%
%
%   exemples
%%%%%%%%%%%%%%
% d=retneff(a)         d: valeurs propres de la texture de descripteur a
% [n,beta,d]=retneff(a) ou retneff(d)    d: valeurs propres de la texture de descripteur a (non triees)
%                    beta=d(n) :ces vp classees par ATTENUATIUON decroissante
% [n,beta,d]=retneff(a,b1)  beta=:le plus proche de b1 ( n son numero) ,d les valeurs propres (non triees)
% [n,beta,d]=retneff(d,[b1,b2]);  beta=:les beta compris entre b1 et b2  classes par PROPAGATION croissante (et  n leurs numeros)
%
% [n,beta,d]=retneff(a,b1,attenuation)   [n,beta,d]=retneff(d,[b1,b2],attenuation);
% idem mais en en gardant que les beta dont la partie reelle (absorption) est inferieure en valeur absolue a attenuation 
% 
% See also: RETTNEFF

% neff=retneff(neff,k0,h,estimee);neff=d;k0=bornes;h=attenuation
if nargin==4;n=d+round((estimee-d).*bornes.*attenuation/(2*pi))*2*pi./(bornes.*attenuation);return;end;

d=retio(d);if iscell(d);if d{end}.type==2;d=d{3};else;d=d{5};end;end;  % metaux

if nargin==1 & nargout<=1;n=d;return;end;
if nargin==1 & nargout>1;
beta=1+abs(real(d))+i*abs(imag(d));[b,k,kk]=retelimine(real(beta));beta=b(kk)+i*imag(beta);
%beta=1+abs(real(d))-i*imag(d);[b,k,kk]=retelimine(real(beta));beta=b(kk)+i*imag(beta);
[prv,nn]=sort(-imag(beta));[prv,n]=sort(real(beta(nn)));n=nn(n);beta=d(n);   
return;end;

if nargin<3;attenuation=inf;end;
nn=find(abs(real(d))<=attenuation);
if length(bornes)==1; % on cherche le plus voisin de bornes
%[prv,n]=min(abs(abs(imag(d(nn)))-bornes));n=nn(n);beta=d(n);

[prv,n]=min(abs(d(nn)+i*bornes));n=nn(n);beta=d(n);
%[prv,n]=min(abs(-imag(d(nn))-bornes));n=nn(n);beta=d(n);
else;                 % on cherche ceux compris entre bornes(1) et bornes(2);
% n=nn(find((abs(imag(d(nn)))>bornes(1))&(abs(imag(d(nn)))<bornes(2))));
% [prv,nn]=sort(-abs(imag(d(n))));n=n(nn);beta=d(n);
n=nn(find((-imag(d(nn))>bornes(1))&(-imag(d(nn))<bornes(2))));   
[prv,nn]=sort(imag(d(n)));n=n(nn);beta=d(n);
end;
