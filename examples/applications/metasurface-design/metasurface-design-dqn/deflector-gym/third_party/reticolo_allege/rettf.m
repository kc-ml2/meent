function [g,uuu]=rettf(x,f,u,apod,k,sens);
% function [g,uuu]=rettf(x,f,u,apod,k,sens);
%
% si uuu existe : 
% g(uuu)=   tf de x-->f   en u uuu 
% uuu couvrant l'intervalle [min(u)  max(u)] avec un pas 1/(x(end)-x(1)) 
% uuu(1)=min(u)
%
% si uuu n'existe pas: 
% g(u)=   tf de x-->f   en u   (interpolation option spline )
%
% apod apodisation (voir retapod) 
% l'echantillonnage en x et en u ne sont pas necessairement reguliers 
% mais il vaut mieux que l'echantillonnage en x soit regulier avec 2^n+1 points 
% 
% sens 1 :transformation de Fourier  -1 :transformation de Fourier inverse
% k :on complete f par des 0 pour avo1r k valeurs (sur echantilonnage)
% de preference k est une puissance de 2
%
%% exemple
%  x=linspace(-50,50,257);y=retsinc(x);u=[-2,2];[g1,u1]=rettf(x,y,u);
%  u=[.4,.6];[g2,u2]=rettf(x,y,u,0,1024);
%  u=linspace(.4,.6,500);g3=rettf(x,y,u);
%  figure;subplot(2,1,1);plot(u1,real(g1));title('[g1,u1]=rettf(x,y,u)');xlabel('u1');
%  subplot(2,2,3.1);plot(u2,real(g2));title('[g2,u2]=rettf(x,y,u,0,1024)');xlabel('u2');
%  subplot(2,2,4);plot(u,real(g3));title('g3=rettf(x,y,u)');xlabel('u');



if nargin<4;apod=0;end;if nargin<5;k=0;end;if nargin<6;sens=1;end;
n=length(x)-1;sz=size(u);u=u(:);x=x(:);f=f(:);
xx=linspace(x(1),x(end),n+1).';if max(abs(x-xx))>eps*max(abs(x));f=retinterp(x,f,xx,'linear');end;x=xx(1:n);clear xx;% interpolation de f
if apod~=0;f=retapod(f,apod);end;
f=[(f(1)+f(n+1))/2;f(2:n)];% elimination du dernier point

minu=min(u);f=exp(-2i*sens*pi*x*minu).*f;

deltax=x(2)-x(1);

if k>n;f=[f;zeros(k-n,1)];n=k;end; % on complete par des 0


uu=linspace(0,1/deltax,n+1).';uu=uu(1:n);
if sens==1;gg=fft(f)*deltax;else;gg=ifft(f)*deltax*n;end;
n1=ceil(deltax*(max(u)-minu));
uuu=[];g=[];
for ii=1:n1;uuu=[uuu;uu+(ii-1)/deltax];g=[g;gg];end;g=exp(-2i*sens*pi*x(1)*mod(uuu,1/deltax)).*g;uuu=uuu+minu;
if nargout==1;g=interp1(uuu,g,u,'spline');
else f=find((uuu<=max(u))&(uuu>=minu));uuu=uuu(f);g=g(f);   
end;
if sz(1)==1;g=g.';uuu=uuu.';end