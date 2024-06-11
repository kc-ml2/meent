function [s,jj]=retfp(h,d,sh,sb,tol);
%      function [s,jj]=retfp(h,d,sh,sb,tol);
%  F P generalise troncature automatique (tol)
%
%  h:vecteur des valeurs de h a traiter
%
%..........................................................................................
% si imag(tol)==0
%..........................................................................................
%      function s=retfp(h,d,sh,sb,tol);
%  s tableau des matrices s de dimension:(:,:,length(h))
%  (attention il s'agit bien de matrices et non pas de cell array ...)
%
%..........................................................................................
% si imag(tol)=jmax >0
%..........................................................................................
%      function [hz,jj]=retfp(h,d,sh,sb,tol);
%  hz: poles voisins des valeurs h donnees en entree obtenus en moins de jmax iterations
%    (hz{:}(ii)=kz(ii)*h{:}(ii) ou kz est un complexe voisin de 1 identique pour tous les bras)
%    hz:tableau de 'nombre de bras'  lignes
%    jj: nombre d'iterations effectivement faites 
%
%  d: valeurs propres du milieu intermediaire (obtenues par [aa,d]=retcouche(...)
%  sh: matrice s du demi probleme haut (le milieu intermediaire est le bas  le haut est le haut)
%  sb: matrice s du demi probleme bas  (le bas est le bas   le milieu intermediaire est le haut)
%  sb et sh sont des matrices S de type mode <-> mode de preference tronquees 
%
%  parametres facultatifs:
%  si sb n'est pas specifiee (ou sb=[]) on considere le probleme comme symetrique (sb=retrenverse(sh))
%  tol:tolerance sur la troncature des valeurs de d suivant h (exp(d*h)<tol) par defaut tol=1.e-6 
%
%  pour le FP a plusieurs bras:sh est en fait un cell array  {sh ,s1,s2  ..}
%  ou les s1, s2... sont les matrices S intermediaires de type MODE MODE
%  les valeurs propres d{1)  d{2} ..d{mf}  sont mises en  cell array  (sinon on reprend les memes ..) 
%  la matrice S est calculee pour plusieurs valeurs de h{1} h{2}..  h{mf} (de meme dimension) mises en  cell array 
%       (si un seul vecteur h il est repete par defaut)
%
%  sh{1} <----h{1}----> sh{2} <----h{2}----> ....    sh{mf} <----h{mf}----> sb
%             d{1}                 d{2}                          d[mf}
%
%  remarque: les matrices s constituant sh et sb peuvent etre des noms de fichiers crées par retio(sh,1) 
% See also RETFPM,RETFPS
s=[];jj=[];if isempty(h);return;end;
if nargin<5;tol=1.e-6;end;
if real(tol)<=0;tol=1.e-6+i*imag(tol);end;
jmax=imag(tol);cad=(jmax>0);  % recherche des poles
tol=-log(real(tol));
if nargin<4;sb=[];end;
if ~iscell(sh);sh=retio(sh);else;sh{1}=retio(sh{1});end;
if~iscell(sh{1});if isempty(sb);sb=retrenverse(sh);else;sb=retio(sb);end;%if cad;sh=rettronc(sh,[],[],1);sb=rettronc(sb,[],[],-1);end;
	
	sp={sh,sb};
else;if isempty(sb);sb=retrenverse(sh{1});else;sb=retio(sb);end;%if cad;sh{1}=rettronc(sh{1},[],[],1);sb=rettronc(sb,[],[],-1);end;
	sp={sh{:},sb};
end;
%clear sh sb;

mf=size(sp,2)-1; % nombre de bras du FP    
for ii=2:mf;sp{ii}=retio(sp{ii});end;
if ~iscell(h);hh=h;h=cell(1,mf);for ii=1:mf;h{ii}=hh;end;end;
if ~iscell(d);dd=d;d=cell(1,mf);for ii=1:mf;d{ii}=dd;end;end;
mh=length(h{1});
for ii=1:mf;d{ii}=reshape(retbidouille(d{ii},4),1,length(d{ii}));end;
sog=size(sp{1},1)<6;  % S ou G    
jj=zeros(1,mh);

for ii=1:mh; % boucle sur h

er=inf;z=[];zz=[];jj(ii)=0;kz=1;
while((er>1.e-10)&(jj(ii)<=jmax));jj(ii)=jj(ii)+1;z=[z;kz]; % boucle pour le calcul des poles
spp=sp;   
dd1=-d{1}*kz*h{1}(ii);[f1,ff1]=retfind(abs(real(dd1))<tol);nf1=length(f1);dd1=exp(dd1(f1));ddd1=retdiag(dd1);if isempty(ff1);ff1=0;end;
spp{1}=rettronc(spp{1},-ff1,-ff1,-1);
spp{1}{1}(:,1:nf1)=spp{1}{1}(:,1:nf1)*ddd1; % meme calcul en S ou G
for iii=2:mf; % bras    
dd2=-d{iii}*kz*h{iii}(ii);[f2,ff2]=retfind(abs(real(dd2))<tol);nf2=length(f2);dd2=exp(dd2(f2));ddd2=retdiag(dd2);if isempty(ff2);ff2=0;end;
spp{iii}=rettronc(spp{iii},-ff2,-ff1,-ff1,-ff2);
if sog;  %  S
spp{iii}{1}(:,1:nf2+nf1)=spp{iii}{1}(:,1:nf2+nf1)*retdiag([dd2,dd1]);
else;    %  G
mm=spp{iii}{5};
spp{iii}{1}(:,1:nf2)=spp{iii}{1}(:,1:nf2)*ddd2;
spp{iii}{2}(:,mm+1:mm+nf1)=spp{iii}{2}(:,mm+1:mm+nf1)*ddd1;
end;
ff1=ff2;nf1=nf2;dd1=dd2;ddd1=ddd2;
end; % bras

% dernier
spp{mf+1}=rettronc(spp{mf+1},-ff1,-ff1,1);
if sog;  %  S
mm=spp{mf+1}{2};
spp{mf+1}{1}(:,mm+1:mm+nf1)=spp{mf+1}{1}(:,mm+1:mm+nf1)*ddd1;
else;    % G
mm=spp{mf+1}{5};
spp{mf+1}{2}(:,mm+1:mm+nf1)=spp{mf+1}{2}(:,mm+1:mm+nf1)*ddd1;
end;

ss=retss(spp);if ~sog;ss=retgs(ss);end;

if cad;zz=[zz;1./sum(diag(ss{1}))];[kz,z,zz,er]=retcadilhac(z,zz);end;
end; % boucle pour le calcul des poles

if cad;
if ii==1;s=zeros(mf,mh);end;
for iii=1:mf;s(iii,ii)=kz*h{iii}(ii);end;
else;
if ii==1;s=zeros([size(ss{1}),mh]);end;
s(:,:,ii)=ss{1};
end;
end;  % boucle sur h
