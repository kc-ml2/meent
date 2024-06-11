function [X,E]=retfpcomplexe(X1,X2);
% [X,E]=retfpcomplexe(X1,X2);
% calcul des min et max de (abs(X-X1)*abs(X-X2))^2 quand X decrit le cercle unite
% X1, X2 complexes (tableaux de longueur n )
% X positions des extremums  (tableau complexe de size (n,2)  )
% E valeur correspondante (tableau reel de size (n,2)  )
% E est croissant
% en cas de non existance nan. En ce cas les 2 premiers sont des ninima les 2 autres des maximas [ min nan max nan]
%
%%% EXEMPLE
%  r1=1+.5*rand;r2=1+.5*rand;X1=r1*exp(2i*pi*rand);X2=r2*exp(2i*pi*rand);[X,E]=retfpcomplexe(X1,X2)
%  figure;phi=linspace(-pi,pi,1000);x=exp(i*phi);y=abs((x-X1).*(x-X2)).^2;plot(phi,y,'-k',angle(X(1:2)),E(1:2),'or',angle(X(3:4)),E(3:4),'ob');
X1=X1(:);X2=X2(:);n=length(X1);X=zeros(n,4);
teta1=(angle(X2)+angle(X1))/2;
teta=(angle(X1)-angle(X2))/2;
r1=abs(X1);r2=abs(X2);
b=-.25*cos(teta).*((1+r1.^2)./r1+(1+r2.^2)./r2);
a=-.25*sin(teta).*((1+r1.^2)./r1-(1+r2.^2)./r2);
p=[ones(n,1),2*a,a.^2+b.^2-1,-2*a,-a.^2];

for ii=1:n;
X(ii,:)=roots(p(ii,:)).';	
end;
a=repmat(a,1,4);b=repmat(b,1,4);
X(abs(imag(X))>eps)=nan;
[f,ff]=retfind(abs(a+X)>eps);
X(f)=i*X(f)-b(f).*X(f)./(a(f)+X(f));
X(ff)=i*X(ff);
X=X.*repmat(exp(i*teta1),1,4);
if nargout>2;return;end;
E=zeros(n,4);
for ii=1:4;E(:,ii)=abs((X(:,ii)-X1).*(X(:,ii)-X2)).^2;end;
[E,ii]=sort(E,2);for jj=1:n;X(jj,:)=X(jj,ii(jj,:));end;% mise en ordre
f=find(isnan(E(:,2)));E(f,[2,3])=E(f,[3,2]);% separation min max en cas de non existance

