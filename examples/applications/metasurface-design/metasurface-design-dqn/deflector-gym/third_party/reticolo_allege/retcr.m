function [ee,o]=retcr(e,o,z,r,x,y,L,sym);
% 
% %  METHODE CYLINDRIQUE RADIALE
% on définit init en 2D conique, incidence normale de la manière suivante:
% init=retinit(d,[-mm,mm,0,0],[0,0,0,L],sym,cao);Le parametre L intervient ici
%  retu, retcouche comme RCWA 
% l'intégration se fait sur r du centre vers l'extérieur
% les variables X Y Z de RCWA sont z theta et r
% ona donc des champs: e(r,z,theta,1:6)
% % Nouveau:
% dans retc il faut donner la valeur de depart du rayon: retc(a,delta_r,r_de_depart)
% dans retb il faut donner le rayon: retb(init,a,sens,rayon,Inc,Dif);
% source=rets(init,[hs,rs],a,parametres habituels...); hs rs: hauteur de la source et rayon
% % Source à l'origine
% La limite d'une source circulaire quand rs->0 est numériquement instable.
% Il est preferable d'utiliser retcr_centre:
% [S2,Inc]=retcr_centre(init,zs,a,sh,R,polinc,poldif,apod);
% qui donne le champ diffracté genere par la source, à la distance R dans la base modale
%     (utile pour tracer le champ sb ne doit pas alors être tronqué )
% ainsi qu'une matrice S2 qui permet de calculer l'énergie P2 emise par une source S 
% en plus de ce qu'elle emmettrait dans le milieu stratifié infini en x y (retbahar):  P2=.5*imag(S'*S2*S)
% ( voir le help)
% % calcul et tracé du champ
% tab= 'comme d'habitude ...en 2D conique '
% sh=retb(init,a_ext,1,sum(tab(:,1)),[],[]);
% sb=retb(init,a_int,-1,tab(end,1),[],0); % très important de ne pas tronquer en bas pour calculer le champ dans le disque central 
% 
% % transformations en coordonnées xyz (comme 'Popov')
% [ee,oo]=retcr(e,o,z,r,x,y,L,sym);%
% e,o,z,r, calculés par retchamp en avec theta=0
% deviennent ee oo, en x, y ,z avec l'option de symétrie sym=-1,0,ou 1
% Attention, cette manipulation utilise une interpolation des champs sur la variable r 
% suseptible de produire des artefacts (sauf si on reste dans le plan x=0). Il est preferable que o ait toutes ses composantes (on interpole les fonctions
% continues)
%
%  passage Popov -->cr;[e_cr,oo]=retcr(e_Popov,1);
%  passage cr -->Popov;[e_Popov,oo]=retcr(e_cr,-1);
%
% 
% % Source ponctuelle decentrée obtenue en sommant plusieurs L de -Lmax à Lmax
% Il suffit de faire le calcul pour L>=0
% Purcell, le flux diffracté se somment (grace à Parceval)
% Si L~=0 il faut multiplier par 2 (L et -L)
% Pour calculer le developpement en ondes planes,il faut pour chaque L calculer:
% u=k0*n_air*sin(teta);v=0; angles{L}=retop(n,z,u,v,  e_r,r_r,z_r,wr_r,    e_z,r_z,z_z,wz_z, k0,sens,L,0);
% et utiliser retcr_op qui combine les diagrammes compte tenu d'une symetrie et peut eventuellement faire le tracé ( voir le help)
% 
% See also: RETCR_CENTRE RETCR_OP


if nargin<3;% passage popov -->cr
if nargin<2;o=1;end;    
if o==1;% passage popov -->cr
ee=permute(e(:,:,:,[3,2,1,6,5,4]),[2,1,3,4]);
%ee(:,:,:,[2,5])=-ee(:,:,:,[2,5]);
else;% passage cr --> popov    
% retour aux coordonnees cartesiennes
ee=permute(e(:,:,:,[3,2,1,6,5,4]),[2,3,1,4]);
%ee(:,:,:,[2,5])=-ee(:,:,:,[2,5]);
end
return;
end;
	
if nargin<8;sym=0;end;if L==0;sym=0;end;
[X,Y]=ndgrid(x,y);R=sqrt(X.^2+Y.^2);R=R(:);
[R,prv,numR]=retelimine(R,1.e-13);% pour gain de temps

Teta=atan2(Y,X);Teta=Teta(:);sinTeta=sin(Teta);cosTeta=cos(Teta);
nz=length(z);

%oo=zeros(length(z),length(x),length(y),size(o,4));
ee=nan(length(R),size(e,2),size(e,3),size(e,4));
[r,ii]=retelimine(r);e=e(ii,:,:,:);o=o(ii,:,:,:);% pout interp1
f=find(R>=min(r)&R<=max(r));
if length(r)>1 & abs(retcompare(r,R(f)))>1.e-10;
    if size(o,4)==6;
    e(:,:,:,3)=e(:,:,:,3).*o(:,:,:,6);    
    e(:,:,:,6)=e(:,:,:,6).*o(:,:,:,3);    
    ee(f,:,:,:)=interp1(r,e,R(f));o=interp1(r,o,R,'nearest');% interpolation fonctions continues
    ee(:,:,:,3)=ee(:,:,:,3)./o(:,:,:,6);    
    ee(:,:,:,6)=ee(:,:,:,6)./o(:,:,:,3);    
    else;    
    ee(f,:,:,:)=interp1(r,e,R(f));if ~isempty(o);o=interp1(r,o,R,'nearest');end;% interpolation
    end;    
else;ee=e;
end;
e=ee(numR,:,:,:);


if ~isempty(o);o=o(numR,:,:,:);end;

ee=zeros(numel(X),nz,6);
% retour aux coordonnees cartesiennes
e=e(:,:,:,[3,2,1,6,5,4]);% e était dans l'ordre e_z, e_teta, e_r. Il est maintenant dans l'ordre e_x, e_y, e_z
                         % e(r, teta, z, 1:6) format Popov
% vraies valeurs de R et phase en angle
F=exp(1i*L*Teta);
switch sym;
case 0;FF=F;
case 1 ;FF=real(F);F=i*imag(F);		
case -1;FF=i*imag(F);F=real(F);		
end;	
e(:,:,:,1)=retdiag(FF)*e(:,:,:,1);e(:,:,:,2)=retdiag(F)*e(:,:,:,2);e(:,:,:,3)=retdiag(FF)*e(:,:,:,3);
e(:,:,:,4)=retdiag(F)*e(:,:,:,4);e(:,:,:,5)=retdiag(FF)*e(:,:,:,5);e(:,:,:,6)=retdiag(F)*e(:,:,:,6);
% passage en cartesiennes
ee(:,:,1)=retdiag(cosTeta)*e(:,:,:,1)-retdiag(sinTeta)*e(:,:,:,2);% Ex
ee(:,:,2)=retdiag(sinTeta)*e(:,:,:,1)+retdiag(cosTeta)*e(:,:,:,2);% Ey
ee(:,:,3)=e(:,:,:,3);% Ez
ee(:,:,4)=retdiag(cosTeta)*e(:,:,:,4)-retdiag(sinTeta)*e(:,:,:,5);% Hx
ee(:,:,5)=retdiag(sinTeta)*e(:,:,:,4)+retdiag(cosTeta)*e(:,:,:,5);% Hy
ee(:,:,6)=e(:,:,:,6);% Hz
ee=permute(ee,[2,1,3]);% xy z  -> z xy
if ~isempty(o);o=permute(o,[2,1,3,4]);o=reshape(o,nz,length(x),length(y),[]);end;
ee=reshape(ee,[nz,length(x),length(y),6]);
