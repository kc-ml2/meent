function varargout=retgauss(varargin);%varargout=cell(1,nargout);
% function [xx,wx,x_disc,xint,P,C]=retgauss(a,b,n,m,xd,d);
%
% integration methode de gauss degre n de a jusque à    b (éventuellement separée en m morceaux)
%  xx est le tableau des points ou calculer la fonction a integrer (vecteur ligne)
%  wx est le tableau des poids associes (vecteur ligne)
% si n>468 methode de gauss approchee (1.e-5) a n*abs(m) points (deconseillee)
% si n<0 methode des trapezes a abs(n) points (alors m ne sert plus)
% si m<0 on rajoute les bornes a et b avec des poids nuls
%
%  si xd est precise: xd points de discontinuite,(eventuellement definis modulo une periode d )
%   on repartit au moins m*n points entre les discontinuites
%  x_disc contient les points de separation des zones (x_disc est ordonne)
%
% Calcul des integrales indefinies
%**********************************
% [xx,wx,x_disc,xint,C]=retgauss(a,b,n,m,xd,d); % version non economique en memoire
% si z=f(x), l'integrale definie de a à xint de f est C*z(:); vecteur colonne  de meme longueur que xint, C:matrice pleine
% 
% [xx,wx,x_disc,xint,P,C]=retgauss(a,b,n,m,xd,d); % version economique en memoire
%  P et C :sont des matrices creuses permettant de calculer des integrales definies
%  si z=f(x), l'integrale definie de a à xint de f est P*cumsum(wx(:).*z(:))+C*z(:); vecteur colonne
%
%  (xint est en general plus grand que x car il contient tous les points limites où l'integrale est obtenue avec la precision maximale)
%**************************
% si b=inf+bb*i    Gauss Laguerre integration de a à inf de fonctions decroissant comme exp(-abs(x/bb))
% si b=-inf+bb*i   Gauss Laguerre integration de a à -inf de fonctions decroissant comme exp(-abs(x/bb))
% Dans ce cas l'integrale ne peut pas etre separee en morceaux. Si n est trop grand (>nmax=)on integre de a à a+bb par 
% Gauss Legendre et de bb à inf par Laguerre
% par defaut bb=1
%
%%%  Exemples
%%%%%%%%%%%%%%%%
%  [xx,wx]=retgauss(0,1,3)    %  xx=[0.1127,0.5, 0.8873]    wx=[0.27778,0.44444,0.27778]
%  [xx,wx]=retgauss(0,1,3,-1) %  xx=[0 , 0.1127, 0.5, 0.8873,1]    wx=[0, 0.27778 ,0.44444, 0.27778,0]  
%  [xx,wx]=retgauss(0,1,-3)   %  xx=[0 , 0.5 , 1]    wx=[0.25, 0.5 , 0.25]  
%  [xx,wx]=retgauss(0,1,-3,-2)%  xx=[0 , 0.5 , 1]    wx=[0.25, 0.5 , 0.25]  
%
%%% Exemple de calcul d'integrale indefinie:
% a=20*pi*randn;b=20*pi*randn;[x,wx,x_disc,xint,C]=retgauss(a,b,10,20);z=sin(x);y=(C*z(:)).';yy=cumsum(wx.*z);
%% [x,wx,x_disc,xint,P,C]=retgauss(a,b,10,20);z=sin(x);y=(P*cumsum(wx(:).*z(:))+C*z(:)).';yy=cumsum(wx.*z);% forme economique
%  figure;semilogy(xint,abs(y-cos(a)+cos(xint)),'-',x,abs(yy-cos(a)+cos(x)),'--r');legend('avec correction','sans correction',3);title(rettexte(a,b));
%%
%
% See also:QUAD,QUADL,QUADGK,RETFRESNEL,RETSINC,RETINTEGRE


if nargout<4;,[varargout{1:nargout}]=retgauss1(varargin{:});
else;[varargout{1:3}]=retgauss1(varargin{:});
[varargout{4:nargout}]=retintdefinie(varargout{1:3});	
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xx,ww,x_disc,C]=retgauss1(a,b,n,m,xd,d);
persistent x_gauss nmax w_gauss;%pour gain de temps
if nargin<6;d=1;end; 
if nargin<5;xd=zeros(1,0);end; 
if nargin<4;m=1;end; 
if m<0;bornes=1;else;bornes=-1;end;m=abs(m);% pour rajouter les bornes

                       %%%%%%%%%%%%%%%%
if isfinite(real(b));  %   LEGENDRE   %
                       %%%%%%%%%%%%%%%%
	

if ~isreal(a);cal_gauss_legendre;return;end;
x_disc=zeros(1,0);
if nargin<5; % <<<<<<<<  pas de discontinuitees

if n<0;  % methode des trapezes d'ordre abs(n);
n=abs(n);
if n==1;xx=(a+b)/2;ww=(b-a);
else;xx=linspace(a,b,n);ww=((b-a)/(n-1))*ones(1,n);ww([1,n])=.5*ww([1,n]);end;
else;    % methode de gauss ou des rectangles
    
    
if n*m==0;[xx,ww,x_disc,C]=deal(zeros(1,0));return;end;

%load retgauss nmax;
if isempty(x_gauss);load retgauss;x_gauss=x;w_gauss=w;end;%pour gain de temps
%if (n<nmax)&(n>1)% methode de gauss
if (n>1);% methode de gauss
if (n<nmax);%<<.... forme exacte
%load retgauss x w;
x=x_gauss{n+1};w=w_gauss{n+1};
if mod(n,2)==0 x=[-x(n/2:-1:1),x];w=[w(n/2:-1:1),w];else x=[-x((n-1)/2:-1:1),0,x];w=[w((n+1)/2:-1:2),w];end;
else;      %<<.... forme approchee
x=(n/(n+1.5+1.6/n))*linspace(-1,1,n);
w=cos(.5*pi*x);x=sin(.5*pi*x);
w=(2/sum(w))*w;
end;      %<<.... forme exacte?

xx=zeros(1,0);ww=zeros(1,0);
for ii=1:m;% on separe en m morceaux l'intervalle [a  b]
aa=a+(ii-1)*(b-a)/m;bb=a+ii*(b-a)/m;x_disc=[x_disc,aa,bb];    
xx=[xx,(bb+aa)/2+(bb-aa)/2.*x];ww=[ww,w.*((bb-aa)/2)];
end;

else; % methode des rectangles d'ordre n*m
n=n*m;xx=a+((b-a)/n)*([1:n]-.5);ww=(b-a)/n.*ones(1,n);x_disc=a+((b-a)/n)*[0:n];   
end;
end;  % trapezes ou (gauss ou rectangles)

else; %<<<<<<<<<<<<<< discontinuitees en xd
if b<a;
if nargin>5;[xx,ww,x_disc]=retgauss(b,a,n,-m*bornes,xd,d);else;[xx,ww,x_disc]=retgauss(b,a,n,-m*bornes,xd);end;
xx=fliplr(xx);ww=-fliplr(ww);return;end;% donc maintenant b>=a ...    
% 'mise en forme' de xd
xd=retelimine(sort(xd));
if nargin>5;%....discontinuitees definies modulo d
xd=mod(xd,d);
n1=ceil((a-max(xd))/d);
n2=floor((b-min(xd))/d);
xdd=zeros(1,0);
for ii=n1:n2;xdd=[xdd,xd+ii*d];end;
xd=xdd;    
end;      % ....discontinuitees definies modulo d    
xd=xd((xd>a)&(xd<b));  
xd=retelimine([a,sort(xd(:).'),b]);
% calcul des xx et ww    
xdd=xd(2:end)-xd(1:end-1);mm=ceil(m*xdd/sum(xdd)); % repartition proportionnelle aux longueurs   
xx=zeros(1,0);ww=zeros(1,0);;
for ii=1:length(xdd);
[xg,wg,xg_disc]=retgauss(xd(ii),xd(ii+1),n,mm(ii));
xx=[xx,xg];ww=[ww,wg];x_disc=[x_disc,xg_disc];
end;
end; %<<<<<<<<<<<<<< discontinuitees en xd ?
x_disc=retelimine(x_disc);

if bornes>0;
if a~=xx(1);xx=[a,xx];ww=[0,ww];end;
if b~=xx(end);xx=[xx,b];ww=[ww,0];end;
end;
                       %%%%%%%%%%%%%%%%
else;                  %   LAGUERRE   %
                       %%%%%%%%%%%%%%%%
if ~isreal(a);cal_gauss_laguerre;return;end;
% function [xx,ww]=retgauss(a,b,n,m,xd,d);
sens=sign(real(b));b=abs(imag(b));if b==0;b=1;end;
load retgauss_laguerre;
if n>nmax;
[xxx,www]=retgauss(a,a+b*sens,n-nmax+1);[xx,ww]=retgauss(a+b*sens,inf*sens+i*b,nmax-1);xx=[xxx,xx];ww=[www,ww];
else;
x=x{n+1};w=w{n+1};
xx=sens*x*b+a;ww=sens*w*b;
end;
if bornes>0;xx=[a,xx];ww=[0,ww];end; % on ajoute les bornes
x_disc=a;
                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%
end;                   %  LEGENDRE OU LAGUERRE ?  %
                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%

					   
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		%     CALCUL DES INTEGRALES DEFINIES            %
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
function [xint,P,C]=retintdefinie(x,wx,x_disc);
% matrice creuse de correction C 
% si z=f(x), l'integrale definie de a à xint de f est:
% P*cumsum(wx(:).*z(:))+C*z(:);
% 
% si C non precisee: P est pleine, l'integrale est P*z(:)

inversion=x(end)<x(1);if inversion;x=-x;wx=-wx;x_disc=-x_disc;end;x_disc=sort(x_disc);
wx_store=wx;[xint,k]=retelimine([x,x_disc],100*eps);wx=[wx,zeros(size(x_disc))];wx=wx(k);% on ajoute les points de discontinuite
nx=length(x);nxint=length(xint);
P=ones(1,nxint);for ii=2:nxint;f=find(x<=xint(ii));P(ii)=f(end);end;
P=sparse([1:nxint],P,ones(1,nxint),nxint,nx,nxint);% passage de x à xint


x=xint;
C=sparse(length(x),length(x));
for jj=1:length(x_disc)-1; % jj
x1=x_disc(jj);x2=x_disc(jj+1);f=find((x>x1)&(x<x2)&(wx~=0));
if jj==1;
n0=length(f);
T=cos(acos((2*x(f)-x1-x2)/(x2-x1)).'*[0:n0-1]);
[xx,wxx]=retgauss(x1,x2,10,1,x(f));
TT=cos(acos((2*xx-x1-x2)/(x2-x1)).'*[0:n0-1]);
A=repmat(wxx,n0,1);
for ii=1:n0;A(ii,xx>x(f(ii)))=0;end;
C0=(A*TT)/T;fac=x(f(end))-x(f(1));
end;
C(f,f)=C0*(x(f(end))-x(f(1)))/fac-tril(repmat(wx(f),length(f),1));
end;                % jj
C=C*P;C(1,1)=-wx_store(1);% traitement special du premier point
if nargout<3;P=full(C+P*tril(repmat(wx_store,nx,1)));end;

if inversion;xint=-xint;if nargout<3;P=-P;else;C=-C;end;end;
					   
					   
					   
					   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     CALCUL DES ZEROS ET TABULATION DES POIDS             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cal_gauss_legendre;
error('Attention le programme essaye de recalculer les points de Gauss. Risque d''effacer le fichier.Supprimer cette instruction si vous voulez vraiement continuer');
% calcul des coefficients et ecriture sur fichier
nmax=201;
x=cell(1,nmax);w=cell(1,nmax);
x{1}=[];x{2}=0;
for nn=3:nmax;mm=length(x{nn-1});
t=([1:mm-1]/mm);t=.5*(1-.97*t);
x{nn}=x{nn-1};x{nn}(1:mm-1)=x{nn}(1:mm-1).*(1-t)+x{nn}(2:mm).*t;% approximation de depart
for ii=1:length(x{nn});
[z0,itermax,erz,erfonc,test]=retcadilhac(@legendre,struct('niter',30,'nitermin',5,'tol',10*eps,'tolfun',10*eps),x{nn}(ii),nn);
x{nn}(ii)=abs(real(z0));
end;
	
if mod(nn,2)==0;x{nn}=[0,x{nn}];end;
% calcul des poids
    % ancienne version (avec polynomes de tchebitchev)
	% mm=length(x{nn});
	% aaa=zeros(0,mm);bbb=[];
	% for ii=1:mm;aaa=[aaa;cos(acos(x{nn})*2*(ii-1))];bbb=[bbb;1/(1-(2*(ii-1))^2)];end;
	% %for ii=1:mm-1;aaa=[aaa;legendre(x{nn},2*ii+1)];bbb=[bbb;0];end
	% if mod(nn,2)==0;aaa(:,1)=aaa(:,1)/2;end;
	% w{nn}=(aaa\bbb)';
w{nn}=2./((1-x{nn}.^2).*legendre(x{nn},nn,1).^2);
subplot(2,1,1); plot(x{nn},'.');title(rettexte(nn));subplot(2,1,2); plot(w{nn},'.');drawnow;
end;    
for nn=2:2:nmax;x{nn}=x{nn}(2:end);w{nn}=2./((1-x{nn}.^2).*legendre(x{nn},n+1,1).^2);end;
save retgauss nmax x w -v6 ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cal_gauss_laguerre
error('Attention le programme essaye de recalculer les points de Gauss. Risque d''effacer le fichier.Supprimer cette instruction si vous voulez vraiement continuer');
% calcul des coefficients et ecriture sur fichier
nmax=200;
x=cell(1,nmax);w=cell(1,nmax);
x{1}=[];x{2}=[1];x{3}=[2-sqrt(2),2+sqrt(2)];x{4}=[0.41577455678348,2.29428036027904,6.28994508293748];
for nn=3:nmax;mm=length(x{nn-1});
if nn>3;x{nn}=zeros(1,mm+1);x{nn}(1:2)=x{nn-1}(1:2);for ii=3:mm+1;x{nn}(ii)=2*x{nn-1}(ii-1)-x{nn-2}(ii-2);end;end;% approximation de depart

for ii=1:length(x{nn});
[z0,itermax,erz,erfonc,test]=retcadilhac(@laguerre,struct('niter',30,'nitermin',5,'tol',10*eps,'tolfun',10*eps),x{nn}(ii),nn);
x{nn}(ii)=abs(real(z0));
end;
% calcul des poids
w{nn}=exp(x{nn}).*x{nn}./(laguerre(x{nn},nn+1)*nn).^2;
subplot(2,1,1); plot(x{nn},'.');title(rettexte(nn));subplot(2,1,2); plot(w{nn},'.');drawnow;
end;    
save retgauss_laguerre nmax x w  -v6   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
function zz=legendre(z,n,p);% polynomes de legendre d'ordre n-1 et derivee: ordre de la derivee 
if nargin<3;p=0;end;
switch p
case 0
	if n==1;zz=ones(size(z));return;end;
	if n==2;zz=z;return;end;
	zz1=ones(size(z));zz2=z;
	for ii=1:n-2;zz=((2*ii+1)*z.*zz2-ii*zz1)/(ii+1);zz1=zz2;zz2=zz;end;
	return	
case 1
	if n==1;zz=zeros(size(z));return;end;
    if n==2;zz=ones(size(z));return;end;
	zz1=zeros(size(z));zz2=ones(size(z));
	for ii=1:n-2;zz=((2*ii+1)*z.*zz2-ii*zz1+(2*ii+1)*legendre(z,ii+1,p-1))/(ii+1);zz1=zz2;zz2=zz;end;
	return	
otherwise	
	if n==1;zz=zeros(size(z));return;end;
    if n==2;zz=zeros(size(z));return;end;
	zz1=zeros(size(z));zz2=zeros(size(z));
	for ii=1:n-2;zz=((2*ii+1)*z.*zz2-ii*zz1+p*(2*ii+1)*legendre(z,ii+1,p-1))/(ii+1);zz1=zz2;zz2=zz;end;
	return	
end;	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
function zz=laguerre(z,n); % polynome de  Laguerre d' ordre n-1
if n==1;zz=ones(size(z));return;end;
if n==2;zz=1-z;return;end;
zz1=ones(size(z));zz2=1-z;
for ii=1:n-2;zz=((2*ii+1-z).*zz2-ii*zz1)/(ii+1);zz1=zz2;zz2=zz;end;
