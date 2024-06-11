function ep=ret2ep(indice,ax,ay,k);
% function ep=ret2ep(indice,ax,ay,k);
% calcul de ep=[mux;muy;muz;epx;epy;epz] a partir de l'indice (eventuellement vecteur)
% si indice=0 et que l'on a qu'un seul argument, ep=[0;0;0;0;0;0]
%
%  si ax et ay existent calcul des ep associes aux PML
%  ret2ep(indice,ax,1) --> PML en x associee a indice  ax :pente du changement de variable  exemple: ax=2+.5i  ax=1:pas de PML
%  ret2ep(indice,1,ay) --> PML en y associee a indice  ay :pente du changement de variable  
%  si ep=ret2ep(indice)   les ep de la PML associee sont: ep.*ret2ep(1,ax,ay) 
%
%  k=2pi/lambda   changement d'echelle facultatif(peut etre precise en l'absence de ax et ay: ret2ep(indice,k)...)
%
% pour les milieux anisotropes
%---------------------------------
%  faire  k6=ret2ep(struct('k6',12)) les ep auront alors 12 colonnes au lieu de 6
% les 6 colonnes supplementaires correspondent à  mu_xy, mu_yz, mu_xz, ep_xy, ep_yz, ep_xz,et vallent initialement 0
%  cette valeur correspond au terme non diagonal de la matrice ( mu_xy )
%  une fois modifiee on peut construire u_anisotrope={u_anisotrope,struct('teta',0)}
%  avant: a_anisotrope=retcouche(init,u_anisotrope,1);
%
%
% See also:RETEP,RETAUTOMATIQUE,RETU

persistent k6;if isempty(k6);k6=6;end;if isstruct(indice);if isfield(indice,'k6');k6=indice.k6;end;ep=k6;return;end;

l=nargin;indice=indice(:).';
ep=zeros(k6,length(indice));
for ii=1:length(indice);
if (l==1)&(indice(ii)==0);ep(:,ii)=zeros(k6,1);else;ep(1:6,ii)=[1;1;1;indice(ii)^2;indice(ii)^2;indice(ii)^2];end;
if l>=3;
ep(1:6,ii)=ep(1:6,ii).*repmat([ay/ax;ax/ay;ax*ay],2,1);
end 
end
if l==2;ep=ep*ax;end;if l==4;ep=ep*k;end;% changement d'echelle