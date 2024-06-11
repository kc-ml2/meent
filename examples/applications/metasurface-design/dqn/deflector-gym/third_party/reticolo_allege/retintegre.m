function a=retintegre(x,y,g,parm,varargin);
% PREMIERE VERSION
% function a=retintegre(x,y,g,parm);
% calcul de l'integrale a de la fonction y(x)*exp(g*x) de x(1) a x(end) 
% pour une fonction y(x)ayant des poles voisins de l'axe reel
% x et y vecteurs de meme dimension x reel, y complexe, g:vecteur
% (les points x ne sont pas necessairement equidistants au moins 2 ,et de preference en nombre impair)
% a est un vecteur de meme dimension que g
%
% la fonction y est approchee sur x(ii),x(ii+1),x(ii+2) par (aa(1)*x+aa(2))/(aa(3)*x+aa(4))
% si un pole est distant de moins que(x(ii+2)-x(ii))*parm:le produit de cette fonction harmonique 
%    par exp(g*x) est integre analytiquement
% sinon: approximation parabolique 
%  
% par defaut parm=4 ,g=0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SECONDE VERSION
%
% function a=retintegre(f,parametres de f,n ,m ,tracer)
% somme de -inf a inf  de f(x,parametres de f)/(sqrt(1-x^2))
%  parametres facultatifs:
%  n ordre de gauss( 20 par defaut)
%  m nombre de segments de gauss( 100 par defaut)
%  tracer=1   trace de controle (0 par defaut)
%% Exemples
%  ld=1.5;w=.35;ep=1;k0=2*pi/ld;a=sqrt(ep)*k0*w/(2*pi);
%  f=inline('retsinc(a*x).^2','x','a'); 
%  k0*retintegre(f,a)/2
%  k0*retintegre(f,a,50,200,1)/2 
%
%  f=inline('retsinc(a*x)./(.01+sqrt(1-x.^2))','x','a'); 
%  retintegre(f,a)
%  k0*retintegre(f,a,[],[],1)
%
% % test:somme de -inf a inf de exp(-pi*(x-.4).^2)=1 
%  f=inline('sqrt(1-x.^2).*exp(-pi*(x-.4).^2)','x');
%  retintegre(f,5,1)-1, retintegre(f,10,5)-1



                         %%%%%%%%%%%%%%%%%%%%%
if isnumeric(x);         % PREMIERE VERSION  %
                         %%%%%%%%%%%%%%%%%%%%%
if nargin<3;g=0;end;
if nargin<4;parm=4;end;
n=length(x);
a=zeros(size(g));

for ii=1:2:n-1;
if ii+2<=n;
x1=x(ii);x2=x(ii+2);xx=[x(ii);x(ii+1);x(ii+2)];yy=[y(ii);y(ii+1);y(ii+2)];
else;  % bornes d'integration si n est pair
if ii==1;x1=x(1);x2=x(2);xx=[x(1);(x(1)+x(2))/2;x(2)];yy=[y(1);(y(1)+y(2))/2;y(2)];else;
x1=x(ii);x2=x(ii+1);xx=[x(ii-1);x(ii);x(ii+1)];yy=[y(ii-1);y(ii);y(ii+1)];end;
end;
[f0,f1]=retfind(abs(g)*max(abs([x1,x2]))<1.e-3);

% g~=0
if ~isempty(f1);
if parm~=0;
aaa=[xx,[1;1;1],-yy.*xx,-yy];
try;[prv1,prv,s]=svd(aaa);aa=s(:,4);catch;aa=[1;1;1;1];end;
if abs(aa(1)*aa(4)-aa(2)*aa(3))<eps*sum(abs(aa).^2);aa(3)=0;end
end;
if parm==0|(abs((x1+x2)*aa(3)/2+aa(4))>=parm*abs((x2-x1)*aa(3)));% lineaire
aa=[xx.^2,xx,[1;1;1]]\yy;
a(f1)=a(f1)+(exp(g(f1)*x2).*(aa(1)*(x2*g(f1)).^2+(aa(2)*g(f1)-2*aa(1))*x2.*g(f1)+aa(3)*g(f1).^2-aa(2)*g(f1)+2*aa(1))...
-exp(g(f1)*x1).*(aa(1)*(x1*g(f1)).^2+(aa(2)*g(f1)-2*aa(1))*x1.*g(f1)+aa(3)*g(f1).^2-aa(2)*g(f1)+2*aa(1)))./(g(f1).^3);
else; % harmonique
z1=-g(f1)*(x2+aa(4)/aa(3));z2=-g(f1)*(x1+aa(4)/aa(3));    
a(f1)=a(f1)+aa(1)/aa(3)*(exp(g(f1)*x2)-exp(g(f1)*x1))./g(f1)...
-(aa(2)*aa(3)-aa(1)*aa(4))/(aa(3)^2)*exp(-g(f1)*aa(4)/aa(3)).*(expint(z1)-expint(z2)+pi*i*round((angle(z2./z1)-angle(z2)+angle(z1))/pi));
end
end;

for jj=f0.';% g=0
yyy=yy.*exp(g(jj)*xx);
if parm~=0;
aaa=[xx,[1;1;1],-yyy.*xx,-yyy];
[prv1,prv,s]=svd(aaa);aa=s(:,4);
if abs(aa(1)*aa(4)-aa(2)*aa(3))<eps*sum(abs(aa).^2);aa(3)=0;end
end;
if parm==0|(abs((x1+x2)*aa(3)/2+aa(4))>=parm*abs((x2-x1)*aa(3)));% lineaire
aa=[xx.^2,xx,[1;1;1]]\yyy;
a(jj)=a(jj)+aa(1)*(x2^3-x1^3)/3+aa(2)*(x2^2-x1^2)/2+aa(3)*(x2-x1);
else; % harmonique
a(jj)=a(jj)+aa(1)/aa(3)*(x2-x1)+(aa(2)*aa(3)-aa(1)*aa(4))/(aa(3)^2)*log((x2*aa(3)+aa(4))/(x1*aa(3)+aa(4)));
end
end
end


                %%%%%%%%%%%%%%%%%%%%%
else;           % SECONDE VERSION   %
                %%%%%%%%%%%%%%%%%%%%%
switch(nargin)
case 1;a=integre(x)
case 2;a=integre(x,y);
case 3;a=integre(x,y,g);
case 4;a=integre(x,y,g,parm);
otherwise;a=integre(x,y,g,parm,varargin{:});
end;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function a=integre(f,varargin);

p=length(argnames(f))-1;
if length(varargin)<p+3;tracer=0;else;tracer=varargin{p+3};end;
if length(varargin)<p+2;m=[];else;m=varargin{p+2};end;if isempty(m);m=100;end;
if length(varargin)<p+1;n=[];else;n=varargin{p+1};end;if isempty(n);n=20;end;
% de 0 a 1  u=cos(x)
[x,w]=retgauss(0,pi/2,n,m);
y=f(cos(x),varargin{1:p})+f(-cos(x),varargin{1:p});

% de 1  a inf  uuu=sqrt(1-uu.^2) uuu=1./xx-1

[xx,ww]=retgauss(0,1,n,m,[.001,.01,.1]);uuu=1./xx-1;uu=sqrt(uuu.^2+1);
yy=(feval(f,uu,varargin{1:p})+feval(f,-uu,varargin{1:p})).*((uuu+1).^2)./uu;
a=sum(w.*y)-i*sum(ww.*yy);

%  trace de controle
if tracer==1;
figure;
subplot(2,2,1);plot(x,real(y),'.-');title('de 0 a 1 : real');grid;
subplot(2,2,2);plot(x,imag(y),'.-');title('de 0 a 1 : imag');grid;
subplot(2,2,3);plot(xx,real(-i*yy),'.-');title('de 1 a inf : real');grid;
subplot(2,2,4);plot(xx,imag(-i*yy),'.-');title('de 1 a inf : imag');grid;
end;


