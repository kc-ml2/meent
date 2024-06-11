function x=retlorentz(z,n,k);
%  function x=retlorentz(z,n,k);
% points d'echantillonnage optimaux l'une courbe de resonance abs(1/(x-z))^2
% z=x0+i*y0 pole  (y0=x0/(2*q))
% x:n points de x0-k*y0 a x0+k*y0 (vecteur ligne)
% par defaut n=31 k=10
%
% See also: RET_INT_LORENTZ

if nargin<3;k=10;end;
if nargin<2;n=31;end;
if n==1;x=real(z);return;end;

xx=[0.1000    0.1887    0.2774    0.3660    0.4547    0.4820    0.5092    0.5365    0.5637    0.5910    0.6182    0.6455    0.6727 ...
0.7000    1.7750    2.8500    3.9250    5.0000    7.5000   10.0000   12.5000   15.0000   21.2500   27.5000   33.7500   40. 100 500];
tt=[0.0581    0.1062    0.1485    0.1836    0.2105    0.2169    0.2223    0.2266    0.2296    0.2309    0.2338    0.2378    0.2425 ...
0.2478    0.5164    0.6729    0.7575    0.8088    0.8738    0.9070    0.9272    0.9407    0.9606    0.9714    0.9783    0.9830 1-eps 1];
xx=[-fliplr(xx),0,xx];tt=[-fliplr(tt),0,tt];

x0=real(z);y0=abs(imag(z));
tmax=retinterp(xx,tt,k);
x=x0+y0*retinterp(tt,xx,linspace(-tmax,tmax,n),'cubic');
