function [x_ctobj,y_ctobj,dim]=rettchamp(e,o,x,y,z,k,parm,fig,Text);x_ctobj=[];y_ctobj=[];
%  Tracé des champs et de l'objet
%
%  en 1D  [x_ctobj,y_ctobj]=rettchamp(e,o,x,y,pol,k,parm,fig,Text); 
%  en 2D  [x_ctobj,y_ctobj]=rettchamp(e,o,x,y,z,k,parm,fig,Text); 
% e o x y z calcules par retchamp ( e o eventuellement vides )
% en 2D:  on trace une coupe: l'un des vecteurs x,y,z doit obligatoirement être réduit à un seul élément.
%
% k :numéros des composantes du champ que l'on veut visualiser (par défaut toutes)
%     en 1D E//:  1 E 2 Hx 3 Hy      en 1D H//:  1 H 2 -Ex 3 -Ey       en 2D:  1 Ex 2 Ey 3 Ez 4 Hx 5 Hy 6 Hz  
% pour o on ne visualise que real(indice)  (si o a plusieurs composantes eps mu,  l'indice est calcule)
% si une valeur de k est >10:  si 11 trace de: abs^2 des champs (par défaut)
%                              si 12 trace de: abs des champs
%                              si 13 trace de: real des champs
%                              si 14 trace de: imag des champs
%                              si 15 animation fonction du temps (parm est alors une structure à champ 
%                                            par defaut parm=struct('fac',1,'temps',linspace(0,4*pi,20),'cmap','jet');
% si une valeur de k est >100:  le premier chiffre donne le colormap (1 échelle de gris conforme en noir)
% on prend ensuite les 2 autres chiffres pour k (Ex 113 partie relle avec échelle de gris conforme) 
%   de plus:
%  en 1D si une valeur de k est <0:vecteur de poynting  tracé avec des flèches (quiver)
%  en 2 D:
%      si une valeur de k vaut -1 flux du vecteur de poynting a travers le plan
%      si une valeur de k vaut -2 projection dans le plan du vecteur de poynting  tracé avec des flèches (quiver)
%      si une valeur de k vaut -3 projection dans le plan du vecteur de poynting NORMALISE trace avec des flèches avec
%                    en surcharge le norme du vecteur de poynting
%      si une valeur de k vaut  0  contours de l'objet en surcharge et trace de l'objet
%      si une valeur de k vaut  i  contours de l'objet en surcharge et pas tracé de l'objet
%
%  parm:pour retcolor(par défaut [])
%  fig: numéro de figure (si abs ou [] nouvelle figure)
%  sur les figures le nom des axes est le nom des variables x y z à l'appel
% Text: titre  si Text est un cell array, ces éléments sont repartis sur les divers subplots
%
% x_ctobj,y_ctobj,si une valeur de k vaut i, contiennent les coordonnées du contour de l'objet 
% les arcs étant sépares par nan  (faire plot(x_ctobj,y_ctobj) pour avoir le trace du contour de l'objet)
% 
% EXEMPLES
%
% rettchamp(e,o,x,y,pol,[1:3,i]);% 1D: abs^2 de toutes les composantes et tracé de l'objet en pointillés
% rettchamp(e,o,x,y,z,[1:6,i,12],[],[],'commentaire');% 2D:  abs de toutes les composantes, tracé de l'objet en pointillés,commentaire 
% rettchamp(e,o,x,y,pol,[1,15,i]); % animation fonction du temps 
% rettchamp(e,o,x,y,pol,[2,15,i],struct('fac',.2,'temps',linspace(0,8*pi,50),'cmap','hot')); % animation fonction du temps avec paramétres
%
% See also:RETCHAMP,RETPOYNTING,RETVM,RETPOINT,RETCOLOR,RETFIG


if nargin<9;Text=' ';end;
if nargin<8;fig=[];end;if isempty(fig);fig=0;end;if isreal(fig)&length(fig)==1;if fig<1;figure('color','w');else;retfig(fig);end;end;
if nargin<7;parm=2;cb=1;end;if ~isstruct(parm);cb=real(parm)>=0;end;% colorbar
if nargin<6;k=1:6;end;


Tex={' '};fontsz=8;
if ~isempty(Text);
if iscell(Text);fontsz=0;for ii=1:length(Text);fontsz=fontsz+length(Text{ii});end;
Tex=Text(2:end);Text=Text{1};
else Tex={' '};fontsz=(3+length(Text));
end;
fontsz=max(7,10-floor(fontsz/20));Text=[Text,'  '];
end;
fontsz=max(7,ceil(fontsz/sqrt(length(find(abs(k)<=6)))));
font={'fontsize',fontsz,'interpreter','none'};
font_label={'interpreter','none'};
ko=~isempty(o);
couleur=0;
[f,ff]=retfind(abs(k)<10);if isempty(ff);type=1;else;type=mod(k(ff(1)),10);couleur=floor(k(ff(1))/100);k=k(f);end;
if type==5;cine_champ(e,o,x,y,z,k,parm,Text);return;end;
[f,ff]=retfind(real(k)==0);ct=~isempty(f)&~isempty(o);
if ct;ko=ko&imag(k(f(1)))==0;end;

k=k(ff);
[f,ff]=retfind(k<0);
p=[];
if ~isempty(f); % vecteur de poynting
p=retpoynting(e);
sz=size(p);pn=reshape(p,prod(sz(1:end-1)),sz(end));ppn=sqrt(sum((pn).^2,2));pn=pn./repmat(ppn,1,sz(end));
pn=reshape(pn,sz);sz(end)=1;ppn=reshape(ppn,sz);
else;pn=[];ppn=[];p=[];
end;

% recherche de la dimension
nx=length(x);dx=max(x)-min(x);
ny=length(y);dy=max(y)-min(y);
nz=length(z);dz=max(z)-min(z);
se=size(e);nb_champs=se(end);if prod(se)>nx*ny*nz;se(end)=1;end;sse=ones(1,4);sse(1:length(se))=se;sse(4)=1;
so=size(o);if prod(so)>nx*ny*nz;so(end)=1;end;sso=ones(1,4);sso(1:length(so))=so;sso(4)=1;
ss=max([sse;sso]);
dim=-1;
% if all(ss==[nz,nx,ny,1]);dim=2;end;
% if all(ss==[ny,nx,1,1]);dim=1;end;

if all(ss==[ny,nx,1,1]);dim=1;end; %% modif 1 2011
if all(ss==[nz,nx,ny,1]);dim=2;end;
if nb_champs>4;dim=2;end;%% modif 1 2011

if length(find([nx,ny,nz]>1))==1;ddim=dim;dim=0;end;% courbe
if dim==-1;return;end;
%if (length(size(e))==4)|(length(size(o))==3);dim=2;else dim=1;end;


if dim==2;Texte=Text;   % 2 D
xyz={inputname(3),inputname(4),inputname(5)};
%ch={['E',xyz{1}],['E',xyz{2}],['E',xyz{3}],['H',xyz{1}],['H',xyz{2}],['H',xyz{3}]};
switch(type);
case 1;pparm=1;ch={['abs(E',xyz{1},')^2'],['abs(E',xyz{2},')^2'],['abs(E',xyz{3},')^2'],['abs(H',xyz{1},')^2'],['abs(H',xyz{2},')^2'],['abs(H',xyz{3},')^2']};
case 2;pparm=1;ch={['abs(E',xyz{1},')'],['abs(E',xyz{2},')'],['abs(E',xyz{3},')'],['abs(H',xyz{1},')'],['abs(H',xyz{2},')'],['abs(H',xyz{3},')']};
case 3;pparm=0;ch={['real(E',xyz{1},')'],['real(E',xyz{2},')'],['real(E',xyz{3},')'],['real(H',xyz{1},')'],['real(H',xyz{2},')'],['real(H',xyz{3},')']};
case 4;pparm=0;ch={['imag(E',xyz{1},')'],['imag(E',xyz{2},')'],['imag(E',xyz{3},')'],['imag(H',xyz{1},')'],['imag(H',xyz{2},')'],['imag(H',xyz{3},')']};
end;

ke=~isempty(e)*length(k);kk=ke+ko;
if  ~isempty(o);so=size(o);if length(so)==4&so(end)>1;o=sqrt(prod( sqrt(o(:,:,:,min([1,2,4,5],[],end))),4));end;o=real(o);end;% mise en forme de o

if nx==1; Text1=['  coupe ',xyz{1},'=',num2str(x),'  '];  
d1=dy;d2=dz;[k1,k2,scb]=calk(d1,d2,kk);
if ko;retsubplot(k1,k2,1);hold on;retcolor(y,z,real(o(:,1,:)),parm);
title({Texte,['objet  ',Text1]},font{:});axis equal;axis tight;xlabel(xyz{2},font_label{:});ylabel(xyz{3},font_label{:});set(gca,'fontsize',fontsz);
Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);Text1=' ';
if ct;ctobjet=retcontour(y,z,o(:,1,:),-1);end;
end;
kkkk=ko;for kkk=1:ke;kkkk=kkkk+1;
retsubplot(k1,k2,kkkk);hold on;    
switch k(kkk);    
case -1;retcolor(y,z,p(:,:,1),parm);title({Texte,['flux poynting',Text1]},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);   
case -2;quiver(y,z,p(:,:,2),p(:,:,3));title({Texte,['poynting',Text1]},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);   
%case -3;caxis([-max(ppn(:)),2*max(p(:))]);retcolor(y,z,ppn,parm);quiver(y,z,pn(:,:,2),pn(:,:,3));title({Texte,['poynting normalise',Text1]},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);   
case -3;retcolor(y,z,ppn,parm);quiver(y,z,pn(:,:,2),pn(:,:,3));title({Texte,['poynting normalise',Text1]},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);   
otherwise;retcolor(y,z,fonc(e(:,1,:,k(kkk)),type),parm,pparm);title({Texte,[ch{k(kkk)},Text1]},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);
end;    
if all(parm~=2i);axis equal;axis tight;end;xlabel(xyz{2},font_label{:});ylabel(xyz{3},font_label{:});Text1=' ';set(gca,'fontsize',fontsz);
if ct;ctobjet=retcontour(y,z,o(:,1,:),k(kkk));end;
end;
end;

if ny==1; Text1=['  coupe ',xyz{2},'=',num2str(y),'  '];
d1=dx;d2=dz;[k1,k2,scb]=calk(d1,d2,kk);
if ko;retsubplot(k1,k2,1);hold on;retcolor(x,z,real(o(:,:,1)),parm);
title({Texte,['objet  ',Text1]},font{:});axis equal;axis tight;xlabel(xyz{1},font_label{:});ylabel(xyz{3},font_label{:});set(gca,'fontsize',fontsz);
Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);Text1=' '; 
if ct;ctobjet=retcontour(x,z,o(:,:,1),-1);end;
end;
kkkk=ko;for kkk=1:ke;kkkk=kkkk+1;
retsubplot(k1,k2,kkkk);hold on;
switch k(kkk);    
case -1;retcolor(x,z,p(:,:,2),parm);title({Texte,['flux poynting',Text1]},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);    
case -2;quiver(x,z,p(:,:,1),p(:,:,3));axis equal;axis tight;title({Texte,['poynting',Text1]},font{:});set(gca,'fontsize',fontsz);Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);  
%case -3;caxis([-max(ppn(:)),2*max(p(:))]);retcolor(x,z,ppn,parm);quiver(x,z,pn(:,:,1),pn(:,:,3));axis equal;axis tight;title({Texte,['poynting normalise',Text1]},font{:});set(gca,'fontsize',fontsz);Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);  
case -3;retcolor(x,z,ppn,parm);quiver(x,z,pn(:,:,1),pn(:,:,3));axis equal;axis tight;title({Texte,['poynting normalise',Text1]},font{:});set(gca,'fontsize',fontsz);Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);  
otherwise;retcolor(x,z,fonc(e(:,:,1,k(kkk)),type),parm,pparm);title({Texte,[ch{k(kkk)},Text1]},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);    
end;
if all(parm~=2i);axis equal;axis tight;end;xlabel(xyz{1},font_label{:});ylabel(xyz{3},font_label{:});Text1=' ';set(gca,'fontsize',fontsz); 
if ct;ctobjet=retcontour(x,z,o(:,:,1),k(kkk));end;
end;
end;

if nz==1; Text1=['  coupe ',xyz{3},'=',num2str(z),'  '];
d1=dx;d2=dy;[k1,k2,scb]=calk(d1,d2,kk);
p=permute(p,[2,1,3]);pn=permute(pn,[2,1,3]);ppn=permute(ppn,[2,1,3]);o=permute(o,[1,3,2]);e=permute(e,[1,3,2,4]);% on permute les axes x et y pour une meilleure presentation

if ko;retsubplot(k1,k2,1);hold on; 
retcolor(x,y,real(o(1,:,:)),parm);title({Texte,['objet',Text1]},font{:});axis equal;axis tight;xlabel(xyz{1},font_label{:});ylabel(xyz{2},font_label{:});set(gca,'fontsize',fontsz);
Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);Text1=' ';
if ct;ctobjet=retcontour(x,y,o(1,:,:),-1);end;
end;
kkkk=ko;for kkk=1:ke;kkkk=kkkk+1;
retsubplot(k1,k2,kkkk);hold on;
switch k(kkk);
case -1;retcolor(x,y,p(:,:,3),parm);title({Texte,['flux poynting',Text1]},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);   
case -2;quiver(x,y,p(:,:,1),p(:,:,2));title({Texte,['poynting',Text1]},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);  
%case -3;caxis([-max(ppn(:)),2*max(p(:))]);retcolor(x,y,ppn,parm);quiver(x,y,pn(:,:,1),pn(:,:,2));title({Texte,['poynting',Text1]},font{:});set(gca,'fontsize',fontsz);Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);  
case -3;retcolor(x,y,ppn,parm);quiver(x,y,pn(:,:,1),pn(:,:,2));title({Texte,['poynting',Text1]},font{:});set(gca,'fontsize',fontsz);Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);  
otherwise;retcolor(x,y,fonc(e(1,:,:,k(kkk)),type),parm,pparm);title({Texte,[ch{k(kkk)},Text1]},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);
end;
if all(parm~=2i);axis equal;axis tight;end;xlabel(xyz{1},font_label{:});ylabel(xyz{2},font_label{:});Text1=' ';set(gca,'fontsize',fontsz);
if ct;ctobjet=retcontour(x,y,o(1,:,:),k(kkk));end;
end;
end;
end;%  2 D

if dim==1;Texte=Text; %1 D
xy={inputname(3),inputname(4)};
if nargin<6;k=1:3;end;
pol=z;
if ~isempty(o);so=size(o);if so(end)>1;o=sqrt(o(:,:,1).*o(:,:,end));end;o=real(o);end;% mise en forme de o

switch(type);
case 1;pparm=1;if pol==0;ch={'abs(E)^2',['abs(H',xy{1},')^2'],['abs(H',xy{2},')^2'],['abs(B',xy{1},')^2']};else;ch={'abs(H)^2',['abs(E',xy{1},')^2'],['abs(E',xy{2},')^2'],['abs(D',xy{1},')^2']};end;    
case 2;pparm=1;if pol==0;ch={'abs(E)',['abs(H',xy{1},')'],['abs(H',xy{2},')'],['abs(B',xy{1},')']};else;ch={'abs(H)',['abs(E',xy{1},')'],['abs(E',xy{2},')'],['abs(D',xy{1},')']};end;    
case 3;pparm=0;if pol==0;ch={'real(E)',['real(H',xy{1},')'],['real(H',xy{2},')'],['real(B',xy{1},')']};else;ch={'real(H)',['real(E',xy{1},')'],['real(E',xy{2},')'],['real(D',xy{1},')']};end;    
case 4;pparm=0;if pol==0;ch={'imag(E)',['imag(H',xy{1},')'],['imag(H',xy{2},')'],['imag(B',xy{1},')']};else;ch={'imag(H)',['imag(E',xy{1},')'],['imag(E',xy{2},')'],['imag(D',xy{1},')']};end;    
end;
ke=~isempty(e)*length(k);kk=ko+ke;
d1=dx;d2=dy;[k1,k2,scb]=calk(d1,d2,kk);
if ko;retsubplot(k1,k2,1);retcolor(x,y,real(o),parm);axis equal;axis tight;xlabel(xy{1},font_label{:});ylabel(xy{2},font_label{:});title({Texte,'objet'},font{:});set(gca,'fontsize',fontsz);
Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);
if ct;ctobjet=retcontour(x,y,o,-1);end;
end;
kkkk=ko;for kkk=1:ke;kkkk=kkkk+1;
if k(kkk)<0;retsubplot(k1,k2,kkkk);
retcolor(x,y,ppn,parm);ax=caxis;caxis([-ax(2),ax(2)*2]);hold on;quiver(x,y,pn(:,:,1),pn(:,:,2)); axis equal;axis tight;xlabel(xy{1},font_label{:});ylabel(xy{2},font_label{:});title({Texte,'poynting'});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);set(gca,'fontsize',fontsz); 
else;retsubplot(k1,k2,kkkk);retcolor(x,y,fonc(e(:,:,k(kkk)),type),parm,pparm);
axis equal;axis tight;xlabel(xy{1},font_label{:});ylabel(xy{2},font_label{:});title({Texte,ch{k(kkk)}},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);set(gca,'fontsize',fontsz);  
end;
if ct;ctobjet=retcontour(x,y,o,k(kkk));end;
end;
end; % 1 D

if dim==0;Texte=Text; % 0 D
switch ddim;
case -1;return;
case 1;  % 0 D du 1 D  
xy={inputname(3),inputname(4)};
	if nx>1;num_var=1;var=x;end;% modif 2010
	if ny>1;num_var=2;var=y;end;
if nargin<6;k=1:2;end;
pol=z;
if ~isempty(o);so=size(o);if length(so)==3&so(end)>1;o=sqrt(o(:,:,1).*o(:,:,end));end;o=abs(real(o));end;% mise en forme de o
%if pol==0;ch={'E',['H',xy{1}],['H',xy{2}],['B',xy{1}]};else;ch={'H',['E',xy{1}],['E',xy{2}],['D',xy{1}]};end;    

switch(type);
case 1;if pol==0;ch={'abs(E)^2',['abs(H',xy{1},')^2'],['abs(H',xy{2},')^2'],['abs(B',xy{1},')^2']};else;ch={'abs(H)^2',['abs(E',xy{1},')^2'],['abs(E',xy{2},')^2'],['abs(D',xy{1},')^2']};ylim([0,inf]);end;    
case 2;if pol==0;ch={'abs(E)',['abs(H',xy{1},')'],['abs(H',xy{2},')'],['abs(B',xy{1},')']};else;ch={'abs(H)',['abs(E',xy{1},')'],['abs(E',xy{2},')'],['abs(D',xy{1},')']};ylim([0,inf]);end;    
case 3;if pol==0;ch={'real(E)',['real(H',xy{1},')'],['real(H',xy{2},')'],['real(B',xy{1},')']};else;ch={'real(H)',['real(E',xy{1},')'],['real(E',xy{2},')'],['real(D',xy{1},')']};end;    
case 4;if pol==0;ch={'imag(E)',['imag(H',xy{1},')'],['imag(H',xy{2},')'],['imag(B',xy{1},')']};else;ch={'imag(H)',['imag(E',xy{1},')'],['imag(E',xy{2},')'],['imag(D',xy{1},')']};end;    
end;


ke=~isempty(e)*length(k);kk=ko+ke;
if ko;retsubplot(kk,1,1);plot(var,real(o),'-k','linewidth',2);axis tight;ylim([min(0,1.05*min(real(o))),max(0,1.05*max(real(o)))]);xlabel(xy{num_var},font_label{:});title({Texte,'objet  real(indice)'},font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);end;
kkkk=ko;for kkk=1:ke;kkkk=kkkk+1;
if k(kkk)<0;retsubplot(kk,1,kkkk);plot(var,p(:,1),'-k',var,p(:,2),'--k','linewidth',2);axis tight;xlabel(xy{num_var},font_label{:});ylabel('poynting',font_label{:});title(Texte,font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end); 
else;retsubplot(kk,1,kkkk);plot(var,fonc(e(:,:,k(kkk)),type),'-k','linewidth',2);axis tight;xlabel(xy{num_var},font_label{:});ylabel(ch{k(kkk)},font_label{:});title(Texte,font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end); 
end;
%if ct;ax=axis;o=o-min(o);if max(o)~=0;o=o/max(o);end;o=o-.5;hold on;plot(var,(ax(3)+ax(4))/2+.9*(ax(4)-ax(3))*o,'r--');end;
if ct;ax=axis;if max(o)~=0;o=o/max(o);end;hold on;plot(var,ax(3)+.9*(ax(4)-ax(3))*o,'r--');end;
end;
case 2;  % 0 D du 2 D 
xyz={inputname(3),inputname(4),inputname(5)};
	if nx>1;num_var=1;var=x;end;% modif 2010
	if ny>1;num_var=2;var=y;end;
	if nz>1;num_var=3;var=z;end;
%ch={['E',xyz{1}],['E',xyz{2}],['E',xyz{3}],['H',xyz{1}],['H',xyz{2}],['H',xyz{3}]};

switch(type);
case 1;ch={['abs(E',xyz{1},')^2'],['abs(E',xyz{2},')^2'],['abs(E',xyz{3},')^2'],['abs(H',xyz{1},')^2'],['abs(H',xyz{2},')^2'],['abs(H',xyz{3},')^2']};    
case 2;ch={['abs(E',xyz{1},')'],['abs(E',xyz{2},')'],['abs(E',xyz{3},')'],['abs(H',xyz{1},')'],['abs(H',xyz{2},')'],['abs(H',xyz{3},')']};    
case 3;ch={['real(E',xyz{1},')'],['real(E',xyz{2},')'],['real(E',xyz{3},')'],['real(H',xyz{1},')'],['real(H',xyz{2},')'],['real(H',xyz{3},')']};   
case 4;ch={['imag(E',xyz{1},')'],['imag(E',xyz{2},')'],['imag(E',xyz{3},')'],['imag(H',xyz{1},')'],['imag(H',xyz{2},')'],['imag(H',xyz{3},')']};   
end;


ke=~isempty(e)*length(k);kk=ke+ko;
if  ~isempty(o);so=size(o);if length(so)==4&so(end)>1;o=sqrt(o(:,:,:,1).*o(:,:,:,end));end;o=real(o);end;% mise en forme de o
if ko;retsubplot(kk,1,1);plot(var,real(o),'-k','linewidth',2);xlabel(xyz{num_var},font_label{:});title({Texte,'objet  real(indice)'},font{:});ylim([min(0,1.05*min(real(o))),max(0,1.05*max(real(o)))]);Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end);end;
kkkk=ko;for kkk=1:ke;kkkk=kkkk+1;
if k(kkk)<0;retsubplot(kk,1,kkkk);plot(var,p(:,2),'-k','linewidth',2);xlabel(xyz{num_var},font_label{:});ylabel('poynting',font_label{:});title(Texte,font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end); 
else;retsubplot(kk,1,kkkk);plot(var,fonc(e(:,:,:,k(kkk)),type),'-k','linewidth',2);xlabel(xyz{num_var},font_label{:});ylabel(ch{k(kkk)},font_label{:});title(Texte,font{:});Texte=Tex{1};Tex=[Tex,{' '}];Tex=Tex(2:end); 
end;
if ct;ax=axis;o=o-min(o);if max(o)~=0;o=o/max(o);end;o=o-.5;hold on;plot(y,(ax(3)+ax(4))/2+.9*(ax(4)-ax(3))*o,'r--');end;
end;   
end; % switch ddim;
end; % 0 D

colormap(retcolor(couleur));
if nargout>0;try;x_ctobj=ctobjet(1,:);y_ctobj=ctobjet(2,:);catch;x_ctobj=[];y_ctobj=[];end;end;
%if nargout>0;jj=1;while(jj<=size(ctobjet,2));ii=ctobjet(2,jj);ctobjet(:,jj)=nan;jj=jj+ii+1;end;x_ctobj=ctobjet(1,2:end);y_ctobj=ctobjet(2,2:end);end; % mise en forme de ctobjet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=fonc(e,type);
switch type;
case 1;e=abs(e).^2;    
case 2;e=abs(e);    
case 3;e=real(e);    
case 4;e=imag(e);
end;    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function c=retcontour(x,y,o,k);
[x,ix]=retelimine(x);[y,iy]=retelimine(y);o=o(iy,ix);
st_warning=warning;warning off;
x_ctobj=[];y_ctobj=[];
ax=caxis(gca);
hold on
o=squeeze(abs(o));
co=retelimine(o);nco=length(co);
for ii=1:min(nco,20);
oo=abs(o-co(ii))<1.e-8;
oo=ax(1)+oo*(ax(2)-ax(1));

[prv,vv]=retversion;
if (prv>6)&(vv<7.4);
[c,h]=contour('v6',x,y,oo,1);set(h,'linestyle','-','edgecolor','k','linewidth',1);
if k>0;[c,h]=contour('v6',x,y,oo,1);set(h,'linestyle','--','edgecolor','w','linewidth',1);end;
else;
[c,h]=contour(x,y,oo,1);set(h,'linestyle','-','edgecolor','k','linewidth',1);
if k>0;[c,h]=contour(x,y,oo,1);set(h,'linestyle','--','edgecolor','w','linewidth',1);end;
end;
if isempty(c);c=zeros(2,0);end;% pour versions>v6
jj=1;while(jj<=size(c,2));ii=c(2,jj);c(:,jj)=nan;jj=jj+ii+1;end;x_ctobj=[x_ctobj,c(1,2:end),nan];y_ctobj=[y_ctobj,c(2,2:end),nan]; % mise en forme de ctobjet
end;


c=[x_ctobj;y_ctobj];
hold off;warning(st_warning);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [k1,k2,scb]=calk(d1,d2,kk);
%k1=1;k2=1;calk='if d2>d1;k1=max(1,round(sqrt((d1*kk/d2))));k2=ceil(kk/k1);else;k2=max(1,round(sqrt(d2*kk/d1)));k1=ceil(kk/k2);end;';
%k1=1;k2=1;calk='d1,d2,if d2>d1;k1=max(1,round(sqrt((d1*kk/d2))));k2=ceil(kk/k1);else;k2=max(1,round(sqrt(d2*kk/d1)));k1=ceil(kk/k2);end;';
k1=1:kk;k2=ceil(kk./k1);
[prv,ii]=max(min([1./(k1*d2);1.1./(k2*d1)]));
k1=k1(ii);k2=k2(ii);
if k1*d2>k2*d1;scb='vert';else scb='vert';end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cine_champ(e,o,x,y,z,k,parm,Text);

defaultopt=struct('fac',1,'temps',linspace(0,4*pi,20),'cmap','jet');
fac=retoptimget(parm,'fac',defaultopt,'fast');
temps=retoptimget(parm,'temps',defaultopt,'fast');
cmap=retoptimget(parm,'cmap',defaultopt,'fast');

[xc,yc,dim]=rettchamp(e,o,x,y,z,[0,i],-2,i);hold off;close(gcf);

if isempty(fac);fac=1;end;
if dim==1;A=fac*max(abs(retcolonne(e(:,:,k(1)))));else;A=fac*max(abs(retcolonne(e(:,:,:,k(1)))));end;% une seule composante
fig=figure;set(gcf,'position',get(0,'ScreenSize')+[0,28,0,-95]);
for t=temps;clf(gcf);
rettchamp(e*exp(-i*t),[],x,y,z,[k(1),13],-2,i,Text);colormap(cmap);shading interp;
hold on;plot(xc,yc,'-k','linewidth',1);
caxis([-A,A]);hold off;
drawnow;
end
