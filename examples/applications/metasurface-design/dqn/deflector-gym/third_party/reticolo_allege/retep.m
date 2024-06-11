function ep=retep(indice,pol,k);
% function ep=retep(indice,pol,k);
% calcul de ep=[epsilon,mux,1/muy] a partir de l'indice(eventuellement vecteur ligne)
%   attention mux est en fait muy et muy mux des notations 2D
% pol:polarisation:0  E//  2 H// par defaut E//
% la pml s'obtient en multipliant ep par une constante complexe
%
%  k=2pi/lambda   changement d'echelle facultatif(pol doit etre indique si on utilise k)
%
% pour les milieux anisotropes
%---------------------------------
%  faire  retep(struct('k3',4)) les ep auront alors 4 colonnes au lieu de 3  
%  cette valeur correspond au terme non diagonal de la matrice ( mu_xy ) et vaut initialement 0
%  une fois modifiee on peut construire u_anisotrope={u_anisotrope,struct('teta',0)}
% avant: a_anisotrope=retcouche(init,u_anisotrope,1);
%
% See also:RET2EP,RETAUTOMATIQUE,RETU

persistent k3;if isempty(k3);k3=3;end;if isstruct(indice);if isfield(indice,'k3');k3=indice.k3;end;ep=k3;return;end;

% indice=indice(:).';
% if nargin<2;pol=0;end;
% if pol==0;ep=[indice.^2;ones(size(indice));ones(size(indice))];
% else;ep=[ones(size(indice));indice.^2;indice.^-2];end;
% if nargin>=3;ep(1:2,:)=ep(1:2,:)*k;ep(3,:)=ep(3,:)/k;end;

indice=indice(:).';
if nargin<2;pol=0;end;
ep=zeros(k3,length(indice));
if pol==0;ep(1:3,:)=[indice.^2;ones(size(indice));ones(size(indice))];
else;ep(1:3,:)=[ones(size(indice));indice.^2;indice.^-2];end;
if nargin>=3;ep(1:2,:)=ep(1:2,:)*k;ep(3,:)=ep(3,:)/k;end;
