
function varargout=retfaisceau(varargin);

% Generation d'une onde plane ou d'un faisceau gaussien au format retchamp 
% 
% ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤  
% ¤  En 3D (reseau 2D)  ¤
% ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤  
% e=retfaisceau(ep,mu,teta,delta,psi,x,y,z,  clone,  x0,y0,z0,w);
% k=[sin(teta)*cos(delta),sin(teta)*sin(delta),cos(teta)] vecteur unitaire dirigeant le faisceau
% teta angle de oz avec la direction du faisceau (0<=teta<=pi  )
% delta rotation autour de oz  (0<=delta<=2*pi )
% psi rotation de E autour de k
%
% Le champ electrique vaut 1 à l'origine x0 y0 z0
%
%   clone: 1 clonage 0 pas clonage(0 par defaut)
%   si clone a une partie imaginaire x y z ont même dimension et ne seront pas 'meshés'
%            (alors [Z,X,Y]=ndgrid(z,x,y) en 2D [Y,X]=ndgrid(y,x) en 1D )
%
%                                                               | teta   | delta | psi |
% onde plane venant du bas d'incidence teta, E porté par oX     | teta   |  pi/2 |  0  |  
% onde plane venant du bas d'incidence teta, H porté par oX     | teta   |   0   |  0  |  
% onde plane venant du haut d'incidence teta, E porté par oX    |pi-teta |  pi/2 |  0  |  
% onde plane venant du haut d'incidence teta, H porté par oX    |pi-teta |   0   |  0  |  
% 
% On peut donner une amplitude E, tableau de 2 elements:
% e=retfaisceau(ep,mu,E,teta,delta,psi,x,y,z, clone, x0,y0,z0,w);
% 
% Faisceaux gaussiens speciaux combinaisons linéaires d'ondes planes
%--------------------------------------------------------------------
%     w=[waist,m,sym,parm,sin_tetamax]
%  waist:     m: exp(i*m*phi), sym: 0 1 ou -1,sin_tetamax=NA/n
%   parm: permet de definir des faisceaux de type particulier
% ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤  
% ¤  En 2D (reseau 1D)  ¤
% ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤  
%  e=retfaisceau(ep,mu,teta,pol,x,y,  clone,  x0,y0,w);
% teta angle de oy avec la direction du faisceau (-pi<=teta<=pi  )
% Le champ electrique vaut 1 à l'origine x0 y0
%  On ne peut pas donner une amplitude   ( faire e=E*retfaisceau(ep,mu,teta,pol,x,y,...)
%
%
%% EXEMPLES
% 
% pol=2;ld=.8;  k0=2*pi/ld; n=1.5;teta=pi/10; 
% x=linspace(-2*ld,2*ld,101);y=linspace(-2*ld,2*ld,101) ;
% e=retfaisceau(k0*n^2,k0,teta,x,y,pol,0,0,0,ld/4);rettchamp(e,[],x,y,pol,[1,15])
%
% 
% ld=.8;  k0=2*pi/ld; n=1.5;teta=pi/8;delta=0;psi=0; 
% x=linspace(-5*ld,5*ld,103);y=0;z=linspace(-10*ld,10*ld,101) ;
% e=retfaisceau(k0*n^2,k0,teta,delta,psi,x,y,z,0,0,0,0,.5*ld);rettchamp(e,[],x,y,z,[1:6,12])
%
% 
% ld=.76;  k0=2*pi/ld; n=1.5;teta=0;delta=0;psi=0; 
% x=linspace(-3*ld,3*ld,103);y=0;z=linspace(-3*ld,3*ld,101) ;
% e=retfaisceau(k0*n^2,k0,teta,delta,psi,x,y,z,0,0,0,0,[.4*ld,2]);rettchamp(e,[],x,y,z,[1:6,12])
% ee=e;ee(:,:,:,1)=sum(abs(e(:,:,:,1:3)).^2,4);rettchamp(ee,[],x,y,z,[1,12])
% See also RETPOINT RETRESEAU RETOP RETB



%if ismember(nargin,[8,9,13])|numel(varargin{3})>1;% 3D
if ismember(nargin,[8,9,13])|numel(varargin{3})>1;% 3D
	if numel(varargin{3})>1;
	[varargout{1:nargout}]=retfaisceau_3D_general(varargin{:});
	else;[varargout{1:nargout}]=retfaisceau_3D(varargin{:});
	end;
else;
	[varargout{1:nargout}]=retfaisceau_2D(varargin{:});
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=retfaisceau_3D_general(ep,mu,E,teta,delta,psi,varargin);
if abs(E(1))>0;e=E(1)*retfaisceau_3D(ep,mu,teta,delta,psi,varargin{:});else e=0;end;
e=e+E(2)*retfaisceau_3D(ep,mu,teta,delta,psi+pi/2,varargin{:});
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=retfaisceau_3D(ep,mu,teta,delta,psi,x,y,z,clone,x0,y0,z0,w);
%                                                               | teta   | delta | psi |
% onde plane venant du bas d'incidence teta, E porté par oX     | teta   |  pi/2 |  0  |  
% onde plane venant du bas d'incidence teta, H porté par oX     | teta   |   0   |  0  |  
% onde plane venant du haut d'incidence teta, E porté par oX    |pi-teta |  pi/2 |  0  |  
% onde plane venant du haut d'incidence teta, H porté par oX    |pi-teta |   0   |  0  |  
% 

if nargin<13;w=inf;end;
if nargin<10;x0=0;y0=0;z0=0;end;
if nargin<9;clone=0;end;xmesh=imag(clone)~=0;clone=real(clone);
if xmesh;xx=x;yy=y;zz=z;else;[zz,xx,yy]=ndgrid(z,x,y);xx=xx(:);yy=yy(:);zz=zz(:);end;
k=sqrt(ep*mu);

e=zeros(length(xx),6);if isempty(xx);return;end;
c_psi=cos(psi);s_psi=sin(psi);c_teta=cos(teta);s_teta=sin(teta);c_delta=cos(delta);s_delta=sin(delta);

[X0,Y0,Z0]=calXYZ(x0,y0,z0,c_psi,s_psi,c_teta,s_teta,c_delta,s_delta);
[X,Y,Z]=calXYZ(xx,yy,zz,c_psi,s_psi,c_teta,s_teta,c_delta,s_delta);X=X(:)-X0;Y=Y(:)-Y0;Z=Z(:)-Z0;


if isfinite(real(w(1)));% faisceau gaussien
if length(w)==1;  % ************* Fresnel
prv=1./(1+(2i/(k*w^2))*Z); e(:,1)=(prv.*exp(i*k*Z-(X.^2+Y.^2).*prv/w^2)).';
e(:,5)=1i/mu*(k-2./(k*w^2)+2/(k*w^4).*(X.^2+Y.^2).*prv.^2).*e(:,1);
e(:,6)=(2/(mu*w^2))*(Y.*prv.*e(:,1));
% e(:,5)=i*(k/mu)*e(:,1);
else;          % ************** somme d'ondes planes JJ Greffet Integration en phi analytique

m=w(2);
if length(w)>2;sym=w(3);else;sym=0;end;
if length(w)>3;parm=w(4);else;parm=0;end;
if length(w)>4;sin_tetamax=w(5);else;sin_tetamax=1;end;
w=w(1);k=real(k);


U=sin_tetamax*min(10/w,k);

switch real(parm)
case 1;U0=imag(parm)+2*sqrt(-log((1+exp(-(U*w/2)^2))/2))/w;[u,wu]=retgauss(0,U,10,2,[2/w,U0]);
otherwise;[u,wu]=retgauss(0,U,10,20,2/w);
end;
parm=real(parm);
wu=u.*wu.*exp(-(u*w/2).^2);

%figure;plot(u,exp(-(u*w/2).^2),'linewidth',2);xlim([0,k]);ylim([0,1]); stop
%wu=wu.*exp(-(u*w/2).^2);
s_tet=u(:).'/k;c_tet=sqrt(1-s_tet.^2);

rho=sqrt(X.^2+Y.^2);phi0=atan2(Y,X);
wu=wu/(2*pi*sum(wu));
if parm==1;wu(u<U0)=-wu(u<U0);end; % parm=1 m=1 => P3D
prv=exp((1i*k)*Z*c_tet)*retdiag(2*pi*wu);% integration et dephasage en z

	if sym==0; % exp( im phi)
	J=reshape(retbessel('j',[m-1,m,m+1],retcolonne(k*rho(:)*s_tet)),length(rho),length(u),3);
	J(:,:,1)=retdiag(exp(1i*(m-1)*(phi0+pi/2)))*J(:,:,1).*prv;
	J(:,:,2)=retdiag(exp(1i*m*(phi0+pi/2)))*J(:,:,2).*prv;
	J(:,:,3)=retdiag(exp(1i*(m+1)*(phi0+pi/2)))*J(:,:,3).*prv;
	[J(:,:,1),J(:,:,3)]=deal(.5*(J(:,:,1)+J(:,:,3)),.5i*(J(:,:,1)-J(:,:,3)));% 1: Ic     2: I      3: Is
	else; % cos( m phi)
	J=reshape(retbessel('j',[m-1,m,m+1,-m-1,-m,-m+1],retcolonne(k*rho(:)*s_tet)),length(rho),length(u),6);
% 	if parm==2;wu(u<2*sqrt(log(2))/w)=-wu(u<2*sqrt(log(2))/w);end;
	J(:,:,1)=retdiag(exp(1i*(m-1)*(phi0+pi/2)))*J(:,:,1).*prv;
	J(:,:,2)=retdiag(exp(1i*m*(phi0+pi/2)))*J(:,:,2).*prv;
	J(:,:,3)=retdiag(exp(1i*(m+1)*(phi0+pi/2)))*J(:,:,3).*prv;
	J(:,:,4)=retdiag(exp(1i*(-m-1)*(phi0+pi/2)))*J(:,:,4).*prv;
	J(:,:,5)=retdiag(exp(-1i*m*(phi0+pi/2)))*J(:,:,5).*prv;
	J(:,:,6)=retdiag(exp(1i*(-m+1)*(phi0+pi/2)))*J(:,:,6).*prv;
	[J(:,:,1),J(:,:,3)]=deal(.5*(J(:,:,1)+J(:,:,3)),.5i*(J(:,:,1)-J(:,:,3)));% 1: Ic     2: I      3: Is
	[J(:,:,4),J(:,:,6)]=deal(.5*(J(:,:,4)+J(:,:,6)),.5i*(J(:,:,4)-J(:,:,6)));% 1: Ic     2: I      3: Is
    end;% sym
	
    %if abs(m)==1;a=1;b=exp(i*m*pi/2);else;a=0;b=exp(i*m*pi/2);;end;
    %a=1;b=1i*sign(m);
    a=1;b=1i*sign(m);%a=0;1;b=1;0*exp(i*m*pi/2);;
  %a=1;b=0;exp(i*m*pi/2);%a=0;1;b=1;0*exp(i*m*pi/2);;% TEST
	e(:,1)=a*J(:,:,1)*c_tet.'-b*sum(J(:,:,3),2);
	e(:,2)=a*J(:,:,3)*c_tet.'+b*sum(J(:,:,1),2);
	e(:,3)=-a*J(:,:,2)*s_tet.';
	e(:,4)=(-1i*k/mu)*(b*J(:,:,1)*c_tet.'+a*sum(J(:,:,3),2));
	e(:,5)=(-1i*k/mu)*(b*J(:,:,3)*c_tet.'-a*sum(J(:,:,1),2));
	e(:,6)=b*(1i*k/mu)*J(:,:,2)*s_tet.';
	if sym~=0;
	a=sym;b=-1i*sym*sign(m);%a=sym;b=-0*sym;
	%a=sym;b=-sym;
	%if abs(m)==1;a=sym;b=sym*exp(-i*m*pi/2);else;a=0;b=sym*exp(i*(m-2)*pi/2);end;% pourquoi exp(i*(m-2)*pi/2) et pas exp(-i*m*pi/2) ???????????
	e(:,1)=e(:,1)+a*J(:,:,4)*c_tet.'-b*sum(J(:,:,6),2);
	e(:,2)=e(:,2)+a*J(:,:,6)*c_tet.'+b*sum(J(:,:,4),2);
	e(:,3)=e(:,3)-a*J(:,:,5)*s_tet.';
	e(:,4)=e(:,4)+(-1i*k/mu)*(b*J(:,:,4)*c_tet.'+a*sum(J(:,:,6),2));
	e(:,5)=e(:,5)+(-1i*k/mu)*(b*J(:,:,6)*c_tet.'-a*sum(J(:,:,4),2));
	e(:,6)=e(:,6)+b*(1i*k/mu)*J(:,:,5)*s_tet.';
	e=e/2;
	end;
end;          % ************** Fresnel ou somme d'ondes planes JJ Greffet Integration en phi analytique
else;   % onde plane
e(:,1)=exp(i*k*Z).';
e(:,5)=i*(k/mu)*e(:,1);
end;
[e(:,1),e(:,2),e(:,3)]=calxyz(e(:,1),e(:,2),e(:,3),c_psi,s_psi,c_teta,s_teta,c_delta,s_delta);
[e(:,4),e(:,5),e(:,6)]=calxyz(e(:,4),e(:,5),e(:,6),c_psi,s_psi,c_teta,s_teta,c_delta,s_delta);

if clone==0;e(:,4:6)=-i*e(:,4:6);end;% declonage
if ~xmesh e=reshape(e,length(z),length(x),length(y),6);end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=retfaisceau_2D(ep0,mu0,teta,x,y,pol,clone,x0,y0,w);
if nargin<10;w=inf;end;
if nargin<8;x0=0;y0=0;end;
if nargin<7;clone=0;end;xmesh=imag(clone)~=0;clone=real(clone);
if xmesh;xx=x;yy=y;else;[yy,xx]=ndgrid(y,x);xx=xx(:);yy=yy(:);end;
k=sqrt(ep0*mu0);if pol==2;[ep,mu]=deal(mu0,ep0);else;[ep,mu]=deal(ep0,mu0);end;
c_teta=cos(teta);s_teta=sin(teta);
[X0,Y0]=calXY(x0,y0,c_teta,s_teta);
[X,Y]=calXY(xx,yy,c_teta,s_teta);X=X(:)-X0;Y=Y(:)-Y0;
e=zeros(length(X),3);	
if isfinite(real(w));% faisceau gaussien
if imag(w)==0;% fresnel
prv=1./(1+(2i/(k*w^2))*Y);
e(:,1)=(sqrt(prv).*exp(i*k*Y-(X.^2).*prv/w^2)).';
e(:,2)=1i/mu*(k-1./(k*w^2)+2/(k*w^4).*X.^2.*prv.^2).*e(:,1);
e(:,3)=(2/(mu*w^2))*(X.*prv.*e(:,1));
%e(:,2)=1i*k/mu*e(:,1);
else;% somme d'ondes planes JJ Greffet
parm=imag(w);w=real(w);k=real(k);	
U=min(10/w,k);
[u,wu]=retgauss(-U,U,10,10);wu=wu.*exp(-(u*w/2).^2);wu=wu/sum(wu);
if parm==-1;wu=(wu.*u/U);end;
%if parm==-1;wu=wu.*sign(u);end;
e=zeros(length(xx),3);
for ii=1:length(u);
e=e+wu(ii)*retfaisceau(ep0,mu0,asin(u(ii)/k),X,Y,pol,1+i);
end;
end;

else;% onde plane
e(:,1)=exp(i*k*Y).';e(:,2)=(1i*k/mu)*e(:,1);
end;
[e(:,2),e(:,3)]=calxy(e(:,2),e(:,3),c_teta,s_teta);

if clone==0;e(:,2:3)=-i*e(:,2:3);end;% declonage
if ~xmesh e=reshape(e,length(y),length(x),3);end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X,Y,Z]=calXYZ(x,y,z,c_psi,s_psi,c_teta,s_teta,c_delta,s_delta);
% changement de repere x y z ->X Y Z
x1=x*c_delta+y*s_delta;
y1=-x*s_delta+y*c_delta;
x2=x1*c_teta-z*s_teta;
Z=x1*s_teta+z*c_teta;
X=x2*c_psi+y1*s_psi;
Y=-x2*s_psi+y1*c_psi;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,y,z]=calxyz(X,Y,Z,c_psi,s_psi,c_teta,s_teta,c_delta,s_delta);
% changement de repere X Y Z -> x y z 
x2=X*c_psi-Y*s_psi;
y2=X*s_psi+Y*c_psi;
x1=x2*c_teta+Z*s_teta;
z=-x2*s_teta+Z*c_teta;
x=x1*c_delta-y2*s_delta;
y=x1*s_delta+y2*c_delta;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X,Y]=calXY(x,y,c_teta,s_teta);
% changement de repere x y ->X Y 
X=x*c_teta+y*s_teta;
Y=-x*s_teta+y*c_teta;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,y]=calxy(X,Y,c_teta,s_teta);
% changement de repere X Y  -> x y  
x=X*c_teta-Y*s_teta;
y=X*s_teta+Y*c_teta;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
