function yy=retinterp(x,y,xx,a);
% function yy=retinterp(x,y,xx,a);
% interpolation (ou extrapolation) avec possibilite de x egaux
%   x--> y  deja calcules  xx-->yy interpoles 
%   x et y sont des VECTEURS de meme longueur
%
%    si x et xx sont REELS
%   si a est une chaine de caracteres : yy=interp1(x,y,xx,a)
%   si a est un scalaire interpolation (ou extrapolation) de degre a (1 par defaut) sur les a+1 points les plus proches
%   si length(a )=2  pade de degre a(1) a(2)
%
%    si x et xx sont COMPLEXES pade de degre a(1) a(2) 
%    par defaut a=[1,1]  si a a un seul element a(1)=a(2)=a
%    si x n'a pas assez d'elements polynome de degre length(x)-1
%
%  extrapolation d'une courbe de convergence
% t_inf=retinterp(m,t)  t(m) extrapole a t(inf) (m nom nul ,en general nb de termes de fourier)
%  (m pas necessairement ordonne ni distinct)
%  approximation polynomiale de t(1/m^2) le degre etant cherche de 0 a 3 jusque au moment ou la precision stagne
%% EXEMPLE 
%  x=rand(1,20);xx=linspace(0,1,1000);fonc=inline('sin(10*x)./((x-.5).^2+.001)','x');
%  y=fonc(x);yy=fonc(xx);
%  a=[3,2];yyy=retinterp(x,y,xx,a);figure;plot(x,y,'o',xx,yy,'--',xx,yyy);legend('donnees','exact','interpolation',3);title(rettexte(a));

% extrapolation d'une courbe de convergence
if nargin==2;
[x,prv]=retelimine(x);y=y(prv);
[x,prv]=sort(x);y=y(prv);
x=x.^-2;
normr=realmax;for ii=0:3;[cc,s,mu]=polyfit(x,y,ii);if s.normr>.95*normr;break;end;c=cc;normr=s.normr;end;% recherche du degre
yy=polyval(c,-mu(1)/mu(2));
%    % test
% figure;
% xxx=linspace(0,x(1),1001);yyy=polyval(c,(xxx-mu(1))/mu(2));
% if ~isreal(y);
% subplot(1,2,1);plot(xxx,real(yyy),x,real(y),'.');title(rettexte(c,ii));
% subplot(1,2,2);plot(xxx,imag(yyy),x,imag(y),'.');
% else;;plot(xxx,yyy,x,y,'.');title(rettexte(c,ii));
% end;
%   %fin test
return;end;

if isempty(xx);yy=xx;return;end;
if length(x)==1;yy=y(ones(size(xx)));return;end;
x=x(:).';y=y(:).';
[x,ii]=retelimine(x,-eps);y=y(ii); % elimination des egaux sans tolerance
m=length(x);if m==1;yy=y(ones(size(xx)));return;end;
if nargin<4;a=1;end;

if ~(isreal(x)&isreal(xx));  % <------------------ fonction de variable complexe
if ischar(a);a=[1,1];end;if length(a)==1;a=[a(1),a(1)];a=round(a);end;
if m<(sum(a)+1);a=[m-1,0];end;
for ii=1:length(xx(:));
[prv,iii]=sort(abs(x-xx(ii)));iii=iii(1:min(end,sum(a)+1));
yy(ii)=pade(x(iii),y(iii),xx(ii),a);
% [prv,aa]=meshgrid([a(1):-1:0,a(2)-1:-1:0],x(iii));aa=aa.^prv;aa(:,a(1)+2:end)=aa(:,a(1)+2:end).*repmat(y(iii).',1,a(2));
% st_warning=warning;warning off;aa=aa\(y(iii).'.*(x(iii).^a(2)).');warning(st_warning);
% P=aa(1:a(1)+1);Q=[1;-aa(a(1)+2:end)];
% yy(ii)=polyval(P,xx(ii))./polyval(Q,xx(ii));
end;

else;               % <---------------------fonction de variable reelle
siz=size(xx);[x,ii]=sort(x);y=y(ii);[xx,ordre]=sort(xx);
if ischar(a);yy(ordre)=interp1(x,y,xx,a);yy=reshape(yy,siz);return;end;%  Matlab
if length(a)==2&a(end)==0;a=a(1);end;
n=prod(siz);xx=xx(:).';yy=zeros(1,n);
% 
ii1=max(find(x<min(xx)));ii2=min(find(x>max(xx)));

if isempty(ii1);ii1=1;end;
if isempty(ii2);ii2=m;end;
fin=0;
ii=ii1-1;while ii<=ii2;
if ii==0;f=find(xx<=x(1));xcentre=x(1);
elseif ii==m;f=find(xx>x(m));xcentre=x(m);
else;f=find((xx>x(ii))&(xx<=x(ii+1)));xcentre=(x(ii)+x(ii+1))/2;end;
if ~isempty(f);% pour gagner du temps
[prv,iii]=sort(abs(x-xcentre));iii=iii(1:min(end,sum(a)+1));
if length(a)==1; % polynome
yy(f)=rettcheb(x(iii),y(iii),xx(f),length(iii));
else % pade
yy(f)=pade(x(iii),y(iii),xx(f),a);
end;
if f(end)==length(xx);fin=1;else;
ii=max(find(x<min(xx(f(end)+1:end))));
end;
else;ii=ii+1;end; % ~isempty(f)
if fin==1;break;end;

end;% while  
yy(ordre)=yy;yy=reshape(yy,siz);
end; % <------------------ fonction de variable reelle ou complexe


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yy=rettcheb(x,y,xx,n);
x0=min([x,xx]);x1=max([x,xx]);
if x0==x1;yy=sum(y)/length(y);return;end;
a=zeros(length(x),n);
for ii=1:n;a(:,ii)=cos((ii-1)*acos(2*(x.'-x0)/(x1-x0)-1));end;
a=a\y.';
yy=cos(acos(2*(xx(:)-x0)/(x1-x0)-1)*[0:n-1])*a;

%yy=cos([0:n-1]*acos(2*(xx-x0)/(x1-x0)-1))*a;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yy=pade(x,y,xx,a);
[prv,aa]=meshgrid([a(1):-1:0,a(2)-1:-1:0],x);aa=aa.^prv;aa(:,a(1)+2:end)=aa(:,a(1)+2:end).*repmat(y.',1,a(2));
st_warning=warning;warning off;aa=aa\(y.'.*(x.^a(2)).');warning(st_warning);
P=aa(1:a(1)+1);Q=[1;-aa(a(1)+2:end)];
yy=polyval(P,xx)./polyval(Q,xx);

