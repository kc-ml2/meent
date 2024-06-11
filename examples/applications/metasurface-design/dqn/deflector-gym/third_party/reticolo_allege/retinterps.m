function varargout=retinterps(varargin);
% interpolation des matrices S 2x2
%% PREMIERE FORME
%
%  SS=retinterps(x,S,xx,dim,methode);
% S cell_array de dimension quelconque 
% x variable correspondant à la dimension dim de S (on doit avoir:length(x)=size(S,dim))
% xx nouvelle variable à interpoler
% SS cell_array de meme dimension que S excepté dim qui a la longueur de xx
% 
% si dim<0 les matrices sont symetriques
% si imag(dim)~=0 tentative de suivi
% 
%% SECONDE FORME
%
% [SS,xx]=retinterps(x,S,xx,methode);
% S{ x:variable a interpoler, parametre} --> SS{ xx:variable interpolee, parametre}
% x et xx peuvent etre de taille[:,length(parametre)], mais si ce sont des vecteurs,ils sont répétés
% x doit être en nombre impair et la valeur centrale correspondre à une déformation nulle
%
%% TROISIEME FORME
%
% S=retinterps(var,S,vvar,ordre_unwrap,var_griddata,methode_interp1,methode_griddata,sym);
%
% interpolation multi dimensionnelle de matrices 2 2  SYMETRIQUES (rh=rb)
% avec eventuellement un griddata sur 2 variables
% S cell-array de matrices 2 2
% 
% var cell-array des anciennes variables ( si un element est vide on n'interpole pas sur la dimension correspondante)
% vvar cell-array des nouvelles variables
% ordre_unwrap vecteur de l'ordre des 'unwrap' 
% 	pour les valeurs >0 unwrap dans le sens direct 
% 	pour les valeurs <0 unwrap en sens inverse ( important :il vaut mieux commencer par des configuration regulieres) 
% var_griddata les numeros des 2 variables pour le griddata (peut etre vide) 
% 
% var,vvar,ordre_unwrap ont pour longueur le nombre de dimensions de S
% methode_interp1: pour interp1 ('cubic' par defaut)
% methode_griddata: pour griddata ('cubic' par defaut)
%
% exemple:S=retinterps({var1,var2,var3},S,{vvar1,vvar2,vvar3},[1,2,-3],[1,2]);




if iscell(varargin{1});[varargout{1:nargout}]=Interp(varargin{:});return;end;% interpolation directe matrices multi dimensionnelles
if ~ischar(varargin{4});[varargout{1:nargout}]=interps_rands(varargin{:});return;end;% interpolation directe
[varargout{1:nargout}]=interp_rst(varargin{:});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [SS,xx,rho_L_s_Vg2_s_ld3,gamma_L_s_Vg_s_ld3]=interp_rst(x,S,xx,methode,varargin);
if nargin<4;methode='cubic';end;

nb_r=(size(S,1)-1)/2;nb_k=size(S,2);
if size(xx,1)==1;xx=xx(:);end;
if size(xx,2)==1;xx=repmat(xx,1,nb_k);end;

if size(x,1)==1;x=x(:);end; % modif
if size(x,2)==1;x=repmat(x,1,nb_k);end;

SS=cell(size(xx,1),nb_k);
eval=iscell(S{1,1}); % modif

%for k=1:nb_k;
for k=nb_k:-1:1;
[t,rh,rb]=deal(zeros(1,2*nb_r+1));for ii=1:2*nb_r+1;if eval;prv=reteval(S{ii,k});else;prv=S{ii,k};end;t(ii)=prv(1,1);rb(ii)=prv(2,1);rh(ii)=prv(1,2);end

tt=retinterp(x(:,k),t(nb_r+1)./t,xx(:,k),methode);rrh=retinterp(x(:,k),t(nb_r+1)*rh./t,xx(:,k),methode);rrb=retinterp(x(:,k),t(nb_r+1)*rb./t,xx(:,k),methode);
rrh=rrh./tt;rrb=rrb./tt;tt=t(nb_r+1)./tt;
if eval;for ii=1:size(xx,1);SS{ii,k}={[tt(ii),rrh(ii);rrb(ii),tt(ii)];2;2;1};end;
else;for ii=1:size(xx,1);SS{ii,k}=[tt(ii),rrh(ii);rrb(ii),tt(ii)];end;end;
% figure;
% subplot(2,2,1);plot(x(:,k),real(t(nb_r+1)./t-1),'.k',xx(:,k),real(t(nb_r+1)./tt-1),'-k',x(:,k),imag(t(nb_r+1)./t-1),'.r',xx(:,k),imag(t(nb_r+1)./tt-1),'-r');
% subplot(2,2,2);plot(x(:,k),real(t(nb_r+1)*rh./t),'.k',xx(:,k),real(t(nb_r+1)*rrh./tt),'-k',x(:,k),imag(t(nb_r+1)*rh./t),'.r',xx(:,k),imag(t(nb_r+1)*rrh./tt),'-r');
% subplot(2,2,3);plot(x(:,k),real(t(nb_r+1)*rb./t),'.k',xx(:,k),real(t(nb_r+1)*rrb./tt),'-k',x(:,k),imag(t(nb_r+1)*rb./t),'.r',xx(:,k),imag(t(nb_r+1)*rrb./tt),'-r');

% 	figure;
% 	subplot(2,2,1);plot(x(:,k),real(t./t(nb_r+1)-1),'.k',xx(:,k),real(tt./t(nb_r+1)-1),'-k',x(:,k),imag(t./t(nb_r+1)-1),'.r',xx(:,k),imag(tt./t(nb_r+1)-1),'-r');
% 	subplot(2,2,2);plot(x(:,k),real(rh),'.k',xx(:,k),real(rrh),'-k',x(:,k),imag(rh),'.r',xx(:,k),imag(rrh),'-r');
% 	subplot(2,2,3);plot(x(:,k),real(rb),'.k',xx(:,k),real(rrb),'-k',x(:,k),imag(rb),'.r',xx(:,k),imag(rrb),'-r');
% 

end;


if nargout<3;return;end;
nb_r=(size(SS,1)-1)/2;
[ddrh,ddrb,ddt,dd2t,ddT,dd2T]=deal(zeros(1,nb_k));

for ik=1:nb_k;
[t,rh,rb]=deal(zeros(2*nb_r+1,1));
for ii=1:2*nb_r+1;prv=SS{ii,ik}{1};
t(ii)=prv(2,2);rb(ii)=prv(2,1);rh(ii)=prv(1,2);
end;
t0=t(nb_r+1);
[p,prv,mu]=polyfit(xx(:,ik),rh*t0./t,4);
pp=polyder(p);
ddrh(ik)=polyval(pp,-mu(1)/mu(2))/mu(2);

[p,prv,mu]=polyfit(xx(:,ik),rb*t0./t,4);
pp=polyder(p);
ddrb(ik)=polyval(pp,-mu(1)/mu(2))/mu(2);

[p,prv,mu]=polyfit(xx(:,ik),t0./t,4);
pp=polyder(p);
ppp=polyder(pp);
ddt(ik)=polyval(pp,-mu(1)/mu(2))/mu(2);
dd2t(ik)=polyval(ppp,-mu(1)/mu(2))/mu(2)^2;

[p,prv,mu]=polyfit(xx(:,ik),t./t0,4);
pp=polyder(p);
ppp=polyder(pp);

ddT(ik)=polyval(pp,-mu(1)/mu(2))/mu(2);
dd2T(ik)=polyval(ppp,-mu(1)/mu(2))/mu(2)^2;
end;
rho_L_s_Vg2_s_ld3=(abs(ddrb).^2+abs(ddrh).^2)/2;
gamma_L_s_Vg_s_ld3=max(0,(-abs(ddrb).^2-abs(ddrh).^2+2*real(dd2t)+2*abs(ddt).^2))/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout=convert(varargin);
% [t,rh,rb]=convert(gama1,gama2,gama3,u,tho1,tho2);
%[gama1,gama2,gama3,u,tho1,tho2]=convert(t,rh,rb);

if nargin>4;[varargout{1:nargout}]=gama1gama2gama3rho_2_trhrb(varargin{:});
else;[varargout{1:nargout}]=trhrb_2_gama1gama2gama3rho(varargin{:});end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [t,rh,rb]=gama1gama2gama3rho_2_trhrb(gama1,gama2,gama3,u,tho1,tho2);
t=.5*exp(i*gama3).*sqrt(1-u.^2).*(tho1+tho2.*exp(i*(gama1+gama2-2*gama3+pi)));
rb=.5*exp(i*gama2).*(tho1.*(1-u)-tho2.*(1+u).*exp(i*(gama1+gama2-2*gama3+pi)));
rh=.5*exp(i*gama1).*(tho2.*(1-u)-tho1.*(1+u).*exp(-i*(gama1+gama2-2*gama3+pi)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gama1,gama2,gama3,u,tho1,tho2]=trhrb_2_gama1gama2gama3rho(t,rh,rb,sens);
tho1tho2=abs(t.^2-rh.*rb);tho12ptho22=2*abs(t).^2+abs(rh).^2+abs(rb).^2;
delta=sens*sqrt((abs(rh).^2-abs(rb).^2).^2+4*abs(t.*conj(rh)+conj(t).*rb).^2);
tho1=sqrt((tho12ptho22+delta)/2);tho2=tho1tho2./tho1;%sqrt(2*tho1tho2.^2./(tho12ptho22+delta));
u=zeros(size(t));f=find(abs((abs(rh).^2-abs(rb).^2))>100*eps); 
u(f)=(abs(rh(f)).^2-abs(rb(f)).^2)./delta(f);

g1_P_g2=angle(rh.*rb-t.^2);

psi=real(acos((2*abs(t).^2./(1-u.^2)-.5*tho12ptho22)./tho1tho2));
%ff=find(imag(-exp(-i*g1_P_g2).*t.^2)./delta>0);
ff=find(imag(conj(rh.*rb).*t.^2)*sens<0);
psi(ff)=-psi(ff);
gama3=(g1_P_g2-psi+pi)/2;
gama1=angle(rh./(tho2.*(1-u)-tho1.*(1+u).*exp(-i*psi)));
gama2=angle(rb./(tho1.*(1-u)-tho2.*(1+u).*exp(i*psi)));

[tt,rrh,rrb]=gama1gama2gama3rho_2_trhrb(gama1,gama2,gama3,u,tho1,tho2);retcompare(tt.^2,t.^2),retcompare(rrh,rh),retcompare(rrb,rb),

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function SS=interps_rands(x,S,xx,dim,methode);
% function [SS,xx]=retinterps(x,S,xx,methode);
% interpolation des matrices S 2x2
% S( x:variable à interpoler, parametre) --> SS( xx:variable interpolee, parametre)

% mise en ordre de S pour l'interpolation
suivi=imag(dim)~=0;dim=real(dim);
sym=dim<0;dim=abs(dim);

sz=size(S);sz=[sz,ones(1,max(0,dim-length(sz)))];% cas ou 1 seul element...
nouvel_ordre=[dim,1:dim-1,dim+1:length(sz)];
ancien_ordre=[2:dim,1,dim+1:length(sz)];
ssz=sz(nouvel_ordre);S=reshape(permute(S,nouvel_ordre),ssz(1),[]);
SSz=ssz;SSz(1)=length(xx);

x=x(:);xx=xx(:);
if nargin<5;methode='cubic';end;
% convertion et suivi
S=retconvS(S);t=S{1,1};rh=S{1,2};rb=S{2,1};
if sym;%  <<<<<<<<<<<<<<<<<<<<<  symetrique
gama1=angle(rh-t);gama2=angle(rh+t);tho1=abs(rh+t);tho2=abs(rh-t);
if suivi;
[gama1_p,gama2_p,gama1_m,gama2_m]=deal(gama1,gama2,gama2,gama1);
[tho1_p,tho2_p,tho1_m,tho2_m]=deal(tho1,tho2,tho2,tho1);
test=zeros(1,4);

for jj=1:size(t,2);
for kk=2:size(t,1);
if kk==2;gama1_th=gama1(kk-1,jj);gama2_th=gama2(kk-1,jj);tho1_th=tho1(kk-1,jj);tho2_th=tho2(kk-1,jj);
else;
a=(x(kk-1)-x(kk))/(x(kk-1)-x(kk-2));b=(x(kk)-x(kk-2))/(x(kk-1)-x(kk-2));
gama1_th=a*gama1(kk-2,jj)+b*gama1(kk-1,jj);
gama2_th=a*gama2(kk-2,jj)+b*gama2(kk-1,jj);
tho1_th=a*tho1(kk-2,jj)+b*tho1(kk-1,jj);
tho2_th=a*tho2(kk-2,jj)+b*tho2(kk-1,jj);
end;	
test(1)=abs(exp(i*gama1_th)-exp(i*gama1_p(kk,jj)))<abs(exp(i*gama1_th)-exp(i*gama1_m(kk,jj)));
test(2)=abs(exp(i*gama2_th)-exp(i*gama2_p(kk,jj)))<abs(exp(i*gama2_th)-exp(i*gama2_m(kk,jj)));
test(3)=abs(tho1_th-tho1_p(kk,jj))<abs(tho1_th-tho1_m(kk,jj));
test(4)=abs(tho2_th-tho2_p(kk,jj))<abs(tho2_th-tho2_m(kk,jj));
if sum(test)<3;gama1(kk,jj)=gama1_m(kk,jj);gama2(kk,jj)=gama2_m(kk,jj);tho1(kk,jj)=tho1_m(kk,jj);tho2(kk,jj)=tho2_m(kk,jj);end;


gama1(kk,jj)=gama1(kk,jj)+2*pi*round((gama1(kk-1,jj)-gama1(kk,jj))/(2*pi));
gama2(kk,jj)=gama2(kk,jj)+2*pi*round((gama2(kk-1,jj)-gama2(kk,jj))/(2*pi));
end;% kk
end;% jj
else;
gama1=unwrap(gama1,[],1);gama2=unwrap(gama2,[],1);
end;% suivi
% interpolation
ggama1=interp1(x,gama1,xx,methode);
ggama2=interp1(x,gama2,xx,methode);
ttho1=interp1(x,tho1,xx,'linear');
ttho2=interp1(x,tho2,xx,'linear');

% calcul de tt,rrh,rrb 
rmt=ttho2.*exp(i*ggama1);
rpt=ttho1.*exp(i*ggama2);
[tt,rrh]=deal((rpt-rmt)/2,(rpt+rmt)/2);
rrb=rrh;

else;%  <<<<<<<<<<<<<<<<<<<<<  non symetrique
	
% à interpoler:
% angt2=unwrap(angle(t.^2),[],1);
% angrh=unwrap(angle(rh),[],1);
% angrb=unwrap(angle(rb),[],1);
% g1_P_g2=unwrap(angle(rh.*rb-t.^2));

a2=abs(t.*conj(rh)+conj(t).*rb).^2;
a1=imag(conj(rh.*rb).*t.^2);
Rh=abs(rh).^2;
Rb=abs(rb).^2;
T=abs(t).^2;
tho1tho2=abs(rh.*rb-t.^2);
tho1=sqrt((2*T+Rh+Rb+sqrt((Rh-Rb).^2+4*a2))/2);
tho2=tho1tho2./tho1;
% interpolation

% angt2=interp1(x,angt2,xx,methode);
% angrh=interp1(x,angrh,xx,methode);
% angrb=interp1(x,angrb,xx,methode);
% g1_P_g2=interp1(x,g1_P_g2,xx,methode);

angt2=unwrap(angle(interp1(x,t.^2,xx,methode)));
angrh=unwrap(angle(interp1(x,rh,xx,methode)));
angrb=unwrap(angle(interp1(x,rb,xx,methode)));
g1_P_g2=unwrap(angle(interp1(x,rh.*rb-t.^2,xx,methode)));

a1=interp1(x,a1,xx,methode);
a2=interp1(x,a2,xx,'linear');
Rh=interp1(x,Rh,xx,methode);
Rb=interp1(x,Rb,xx,methode);
T=interp1(x,T,xx,methode);
tho1tho2=interp1(x,tho1tho2,xx,methode);
tho1=interp1(x,tho1,xx,'linear');
tho2=interp1(x,tho2,xx,'linear');
	
% calcul
u=(Rh-Rb)./sqrt((Rh-Rb).^2+4*a2);
psi=real(acos((2*T./(1-u.^2)-.5*(2*T+Rh+Rb))./tho1tho2));
ff=find(a1<0);
psi(ff)=-psi(ff);
gama3=(g1_P_g2-psi+pi)/2;
gama2=angrb-angle(tho1.*(1-u)-tho2.*(1+u).*exp(i*psi));
gama1=angrh-angle(tho2.*(1-u)-tho1.*(1+u).*exp(-i*psi));
	
[tt,rrh,rrb]=convert(gama1,gama2,gama3,u,tho1,tho2);
% kk=1;
% figure; plot(xx,angt2(:,kk),xx,angrh(:,kk),xx,angrb(:,kk),xx,g1_P_g2(:,kk));legend('angt2','angrh','angrb','g1_P_g2');
% figure; plot(xx,a1(:,kk),xx,T(:,kk),xx,Rh(:,kk),xx,Rb(:,kk));legend('a1','T','Rh','Rb');
% figure; plot(xx,a2(:,kk),xx,tho1tho2(:,kk),xx,tho1(:,kk),xx,tho2(:,kk));legend('a2','tho1tho2','tho1','tho2');
% figure; plot(xx,gama1(:,kk),xx,gama2(:,kk),xx,gama3(:,kk),xx,u(:,kk),xx,tho1(:,kk),xx,tho2(:,kk));legend('gama1','gama2','gama3','u','tho1','tho2');
% stop

end;% <<<<<<<<<<<<<<<<<<<<<  symetrique ?

SS=retconvS({tt,rrh;rrb,tt});

% retour au format initial
SS=permute(reshape(SS,SSz),ancien_ordre);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function S=Interp(var,S,vvar,ordre_unwrap,var_griddata,methode_interp1,methode_griddata);
% 
% interpolation multi dimensionnelle de matrices 2 2 SYMETRIQUES (rh=rb thb=tbh) 
if nargin<7;methode_griddata='cubic';end;
if nargin<6;methode_interp1='cubic';end;
if nargin<5;var_griddata=[];end;
ordre_unwrap=retcolonne(ordre_unwrap,1);% vecteur ligne
sz=size(S);nvar=length(sz);
S=reshape(S,sz(1),[]);
S=retconvS(S);t=reshape(S{1,1},sz);rh=reshape(S{1,2},sz);
tho2=abs(rh-t);gama2=angle(rh+t);tho1=abs(rh+t);gama1=angle(rh-t);


 %[Gama1,Gama2,Gama3]=deal(gama1,gama2,gama3);

f_flip=-ordre_unwrap(ordre_unwrap<0);ordre_unwrap=abs(ordre_unwrap);
for ii=f_flip;gama1=flipdim(gama1,ii);gama2=flipdim(gama2,ii);end;
for dim=ordre_unwrap;gama1=unwrap(gama1,[],dim);gama2=unwrap(gama2,[],dim);end;
for ii=fliplr(f_flip);gama1=flipdim(gama1,ii);gama2=flipdim(gama2,ii);end;
%[Gama1,Gama2,Gama3]=deal(gama1,gama2,gama3);

if ~isempty(var_griddata);% <<<<<<<<<<<<<<<<
dim=1:nvar;
autres_variables=setdiff(dim,var_griddata);
dim_nv=[var_griddata,autres_variables];	
[prv,dim_anc]=sort(dim_nv);
gama1=permute(gama1,dim_nv);	
gama2=permute(gama2,dim_nv);	
tho1=permute(tho1,dim_nv);	
tho2=permute(tho2,dim_nv);
[Vvar1,Vvar2]=ndgrid(vvar{var_griddata(1)},vvar{var_griddata(2)});
[Var1,Var2]=ndgrid(var{var_griddata(1)},var{var_griddata(2)});
[ttho1,ttho2,ggama1,ggama2]=deal(zeros(length(vvar{var_griddata(1)}),length(vvar{var_griddata(2)}),prod(sz(autres_variables))));


for kk=1:prod(sz(autres_variables));
ttho1(:,:,kk)=griddata(Var1,Var2,tho1(:,:,kk),Vvar1,Vvar2,'linear');
ttho2(:,:,kk)=griddata(Var1,Var2,tho2(:,:,kk),Vvar1,Vvar2,'linear');
ggama1(:,:,kk)=griddata(Var1,Var2,gama1(:,:,kk),Vvar1,Vvar2,methode_griddata);
ggama2(:,:,kk)=griddata(Var1,Var2,gama2(:,:,kk),Vvar1,Vvar2,methode_griddata);
end;

gama1=permute(ggama1,dim_anc);	
gama2=permute(ggama2,dim_anc);	
tho1=permute(ttho1,dim_anc);	
tho2=permute(ttho2,dim_anc);	

else;                    % <<<<<<<<<<<<<<<<
autres_variables=1:nvar;
end;                     % <<<<<<<<<<<<<<<<
for kk=autres_variables; % interpolation sur les autres variables
tho1=interp_n(var{kk},tho1,vvar{kk},kk,nvar,'linear');
tho2=interp_n(var{kk},tho2,vvar{kk},kk,nvar,'linear');
gama1=interp_n(var{kk},gama1,vvar{kk},kk,nvar,methode_interp1);
gama2=interp_n(var{kk},gama2,vvar{kk},kk,nvar,methode_interp1);

end;
rmt=tho2.*exp(i*gama1);
rpt=tho1.*exp(i*gama2);
tt=(rpt-rmt)/2;
rh=(rpt+rmt)/2;
S=retconvS({tt,rh;rh,tt});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yy=interp_n(var,y,vvar,k,n,methode);
if k>1;dim_nv=[k,2:k-1,1,k+1:n];y=permute(y,dim_nv);end;
sz=size(y);yy=reshape(interp1(var,y(:,:),vvar,methode),[length(vvar),sz(2:end)]);% bugg ?
%yy=interp1(var,y,vvar,methode);
if k>1;yy=permute(yy,dim_nv);end;
















