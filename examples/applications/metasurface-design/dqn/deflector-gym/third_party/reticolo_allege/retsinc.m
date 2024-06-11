function [y,yy]=retsinc(x,p)
% function y=retsinc(x,p);
%
%  si p=0 ou p absent: sin(pi*x)/(pi*x)
%
%  si p entier >0 y(x)=somme de -.5 a .5 de (p+1)*(t+.5).^p.*exp(-2i*pi*t*x)    dt
%   ( c'est la tf de  (p+1)*rect(t)*(t+.5)^p qui vaut 1 en x=0)
%                 precision relative: 1.e-14 si 0<=p<=15  1.e-12 p>15
%
%  si p =-1 : y(x)=somme de 0 a x de sin(pi*t)/(pi*t)   dt = Si(pi*x)/pi
%
%  si p =-2 : y(x)=somme de 0 a x de(cos(pi*t)-1)/(pi*t)   dt = (Ci(pi*x)- log(pi*x) -gama) /pi
%            yy(x)=somme de 0 a x de(cos(pi*t)-1)/(pi*t)   dt + (log(pi*x)+gama)/pi= Ci(pi*x) /pi
%
% si p est complexe N=round(imag(p)) N entier sin(i*pi*x)/sin(pi*x)%
% See also RETFRESNEL,EXPINT,RETINTEGRE,RETTFSIMPLEX

if isempty(x);y=x;return;end;
if nargin<2;p=0;end;%sin(pi*x)/(pi*x)

sx=size(x);x=x(:);
if imag(p)>0;% sin(m*pi*x)/sin(pi*x);
n=round(imag(p));
s=round(x);x=x-s;s=find(mod((n-1)*s,2)==1);
y=n*retsinc(n*x)./retsinc(x);
y(s)=-y(s);
y=reshape(y,sx);
return
end;

x=pi*x;
y=ones(size(x));

if p==0;% sinc
[xxx,xx0]=retfind(abs(x)>.1);
%y(xxx)=exp(-i*x(xxx)).*(expm1(2i*x(xxx)))./(2i*x(xxx));
%y(xxx)=exp(-i*x(xxx)).*(exp(2i*x(xxx))-1)./(2i*x(xxx));
y(xxx)=sin(x(xxx))./x(xxx);
y(xx0)=polyval([-1/39916800,1/362880,-1/5040,1/120,-1/6,1],x(xx0).^2);
else;
if p>0;  % tf de  (p+1)*rect(t)*(t+.5)^p
if p>=25;ng=50;
%if p>=25;if p<=50;ng=50;else;ng=30;end;
nnn=0;xmax=p/2;
aa=.1;[tt,ww]=retgauss(0,1,ng);ttt=exp((tt-1)./(aa*tt));t=ttt.^(1/p)-.5;
else;
nnn=6+2*p;
nn=(nnn+1):-1:0;factnn=[fliplr(cumprod(1:(nnn+1))),1];
pp=(p+1)./(factnn.*(p+1+nn));
xmax=.5*exp((-30-log(pp(1)))/(nnn+1));
pp=pp.*xmax.^[(nnn+1):-1:0];
end
[xxx,xx0]=retfind(abs(x)>xmax);
if ~isempty(xx0);
if nnn>0;
y(xx0)=exp(i*x(xx0)).*polyval(pp(2:end),-2i*x(xx0)/xmax);
else;
yy=(exp(-2i*x(xx0)*(t-.5))-1)*((ww.*aa.*((tt*aa).^-2).*ttt.*(t+.5))).';
y(xx0)=(((p+1)/p)*yy+1).*exp(-i*x((xx0)));
end;
end;
yy=exp(-i*x(xxx));
y(xxx)=yy.*(exp(2i*x(xxx))-1)./(2i*x(xxx));
for ii=1:p;y(xxx)=(-.5i*(ii+1))*(y(xxx)-yy)./x(xxx);end;% recurrence sur p

else;
switch(p);
case -1; % Sinus integral
[xxx,xx0]=retfind(abs(x)>1.e-1);
y(xxx)=(-.5i/pi)*(expint(i*x(xxx))-expint(-i*x(xxx)) )+.5*sign(real(x(xxx)));
y(xx0)=(x(xx0)-x(xx0).^3/18+x(xx0).^5/600-x(xx0).^7/35280)/pi;
case -2; % cosinus integral -log(x) -gamma =somme de 0 a x  de (cos(pi*t)-1)/(pi*t)
gama=.5772156649015328606;
yy=y;
[xxx,xx0]=retfind(abs(x)>5.e-2);
yy(xxx)=(-.5/pi)*(expint(i*x(xxx))+expint(-i*x(xxx)));
y(xx0)=(-x(xx0).^2/4+x(xx0).^4/96-x(xx0).^6/4320)/pi;
y(xxx)=yy(xxx)-(log(abs(x(xxx)))+gama)/pi;
[f,ff]=retfind(abs(xx0)~=0);
yy(xx0(f))=y(xx0(f))+(log(abs(x(xx0(f))))+gama)/pi;
yy(xx0(ff))=y(xx0(ff))+(log(abs(x(xx0(ff))))+gama)/pi;
end;
end;end;

y=reshape(y,sx);