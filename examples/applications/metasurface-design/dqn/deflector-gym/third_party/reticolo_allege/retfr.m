function [ff,x,d,ld]=retfr(xx,ee,x,d,ld,mf);
% function ff=retfr(xx,ee,x,d,ld,mf);
% diffracton de Fresnel 1D a la distance d, absisse x ,longueur d'onde ld
%   si x,d,ld sont des tableaux de meme dimension on calcule sur (x,d,ld)
%  l'amplitude diffractee ff est alors un vecteur de dimension size(x)= size(d)= size(ld) 
%
%   sinon on calcule sur [x,d,ld]=meshgrid(x,d,ld) (que l'on peut obtenir en sortie)
%   l'amplitude diffractee ff est alors un vecteur de dimension  squeeze([length(d),length(x),length(ld)])
%
%  xx,ee  abscisses et 'epaisseurs optiques(eventuellement complexes) correspondantes (x croissant)
%  en nombre IMPAIR
%  entre xx(2*k+1) et xx(2*k+3) f(x)=nonanexp(2i*pi/ld * e(x))
%  ou e(x)=a*x^2+b*x=c  passe par les 3 points xx(2*k+1:2*k+3)    ee(2*k+1:2*k+3)
%
% si xx(2*k+1)=xx(2*k+3) ou si ee(2*k+1:2*k+3) ne sont pas tous finis,
% le calcul n'est pas fait sur le tronçon 2*k+1:2*k+3
%
%  mf 'precision' de retfresnel (20 par defaut)
%  si imag(mf) ~=0 approximation lineaire au lieu de quadratique 
%
% pour les 'demi ecrans':
%   xx=[xx(1)+i,xx(2),xx(3),....]  ee= .... demi ecran a gauche
%  (on prolonge a gauche a partir de [xx(1),xx(2),xx(3)] et [ee(1),ee(2),ee(3)])
%  xx=[... ,xx(end-3),xx(end-2),xx(end-1),xx(end)*i]  ee=....  demi ecran a droite
%
%% Exemples
% x=linspace(-50,50,1001);ld=2*pi;d=50;figure;
%% lentille convergente limitée de focale f
% f=50;xx=[-25,0,25];ee=-(xx.^2)/(2*f);ff=retfr(xx,ee,x,d,ld);
% subplot(3,1,1);plot(x,abs(ff).^2);title('lentille');
%
%% demi ecran ( transparent pour x>0,opaque pour x<0)
% xx=[0,1,2+i];ee=[0,0,0] ;ff=retfr(xx,ee,x,d,ld);
% subplot(3,1,2);plot(x,abs(ff).^2);title('demi ecran');
%
%% fentes d' young  (sur la partie -18,0,18 on ne calcule pas car ee=0,inf,0)
% xx=[-22,-20,-18,0,18,20,22];ee=[0,0,0,inf,0,0,0];ff=retfr(xx,ee,x,d,ld);
%  subplot(3,1,3);plot(x,abs(ff).^2);title('fentes d''young');


if nargin<6;mf=0;end;
if ~( all(size(x)==size(d)) & all(size(x)==size(ld)));[x,d,ld]=meshgrid(x,d,ld);x=squeeze(x);d=squeeze(d);ld=squeeze(ld);end;

ff=zeros(size(x));
xxx=imag(xx(:));
xx=real(xx(:));ee=ee(:);
% approximation lineaire
if imag(mf)~=0;xx0=xx;ee0=ee;xx=xx(1);ee=ee(1);
for kk=2:length(xx0);xx=[xx;(xx0(kk-1)+xx0(kk))/2;xx0(kk)];ee=[ee;(ee0(kk-1)+ee0(kk))/2;ee0(kk)];end;
xxx=[xxx(1);zeros(length(xx)-2,1);xxx(end)];
end;
mf=real(mf);if mf==0;mf=20;end; % par defaut

[dz,dnz]=retfind(abs(d)<eps);


for kk=1:2:length(xx)-1;
f=zeros(size(x));
if (retcompare(xx(kk),xx(kk+2))>100*eps)&(all(isfinite(ee(kk:kk+2))));%  <--------------------- si bornes identiques on ne calcule pas
a=[xx(kk:kk+2).^2,xx(kk:kk+2).^1,xx(kk:kk+2).^0]\ee(kk:kk+2);% ee=a(1)*x^2+a(2)*x+a(3)
% prolongement sur un demi espace (demi ecrans)
masque=[1,1];if xxx(kk)~=0;masque(1)=0;xx(kk)=-realmax*max(abs(ld(:))/(2*pi));end;if xxx(kk+2)~=0;masque(2)=0;xx(kk+2)=realmax*max(abs(ld(:))/(2*pi));end;% prolongement sur un demi espace (demi ecrans)
ddz=find((x(dz)>=xx(kk))&(x(dz)<=xx(kk+2)));
ff(dz(ddz))=nonanexp((2i*pi./ld(dz(ddz))).*polyval(a,x(dz(ddz)))); % cas d=0
aa=1+2*a(1)*d(dnz);
[uu,u]=retfind(abs(aa)<eps); % uu cas particulier du 'plan de Fourier local '
uu=dnz(uu);u=dnz(u);

%%%%%%%%%%%%% en dehors du 'plan de Fourier local '
if ~isempty(u);
aa=sqrt(1+2*a(1)*d(u));
bb=(d(u)*a(2)-x(u))./aa;

% pour calculer exp(i*pi/2 * z3.^2) et  exp(i*pi/2 * z1.^2)
z1=sqrt(2./(ld(u).*d(u))).*(xx(kk)*aa+bb);zz1=2./(ld(u).*d(u)).*(xx(kk)*aa).*(xx(kk)*aa+2*bb);
z3=sqrt(2./(ld(u).*d(u))).*(xx(kk+2)*aa+bb);zz3=2./(ld(u).*d(u)).*(xx(kk+2)*aa).*(xx(kk+2)*aa+2*bb);

%[y3,yy3,yyy3,parm3]=retfresnel(z3,mf);[y1,yy1,yyy1,parm1]=retfresnel(z1,mf);
mmax=round(2.e5/mf);% pour ne pas saturer la memoire dans retfresnel
mm=length(z3);y3=zeros(size(z3));yy3=zeros(size(z3));yyy3=zeros(size(z3));parm3=zeros(size(z3));
ii=0;while ii<mm;mmm=ii+1:min(mm,ii+mmax);[y3(mmm),yy3(mmm),yyy3(mmm),parm3(mmm)]=retfresnel(z3(mmm),mf);ii=ii+mmax;end;
mm=length(z1);y1=zeros(size(z1));yy1=zeros(size(z1));yyy1=zeros(size(z1));parm1=zeros(size(z1));
ii=0;while ii<mm;mmm=ii+1:min(mm,ii+mmax);[y1(mmm),yy1(mmm),yyy1(mmm),parm1(mmm)]=retfresnel(z1(mmm),mf);ii=ii+mmax;end;

[p,g]=retfind((parm3==1)|(parm1==1));
% grands
if ~isempty(g);
if (masque(1)~=0)&(masque(2)~=0);zz31=(zz3(g)+zz1(g))/2;else zz31=zeros(size(g));end;% pour eviter les overflow dans le calcul des exponentielles
if masque(1)~=0;z1=nonanexp((i*pi/2)*(zz1(g)-zz31))./z1(g);else z1=zeros(size(g));end;
if masque(2)~=0;z3=nonanexp((i*pi/2)*(zz3(g)-zz31))./z3(g);else z3=zeros(size(g));end;

[ggg,gge]=retfind(yyy1(g)~=yyy3(g));
gg=g(ggg);ge=g(gge);
f(u(g))=-i/pi*(z3-z1)-.5i*(yy3(g).*z3-yy1(g).*z1);
if ~isempty(gg); % yyy1 ~= yyy3
gggg=find(zz31(ggg)~=0);f(u(gg(gggg)))=nonanexp((i*pi/2)*zz31(ggg(gggg))+log(f(u(gg(gggg)))));
f(u(gg))=f(u(gg))+(yyy3(gg)-yyy1(gg)).*nonanexp(-i*pi.*bb(gg).^2./(ld(u(gg)).*d(u(gg))));
ggg=gg(find((f(u(gg))~=0)&(isfinite(f(u(gg))))));
ff(u(ggg))=ff(u(ggg))+(sqrt(-i/2)./aa(ggg)).*nonanexp(i*pi./(ld(u(ggg)).*d(u(ggg))).*(x(u(ggg)).^2+2.*d(u(ggg)).*(a(3)+d(u(ggg))))).*f(u(ggg));
end;
if ~isempty(ge); % yyy1 = yyy3
gggg=find(zz31(gge)~=0);f(u(ge(gggg)))=nonanexp((i*pi/2)*zz31(gge(gggg))+log(f(u(ge(gggg))))); 
gee=(find((f(u(ge))~=0)&(isfinite(f(u(ge))))));ge=ge(gee);gge=gge(gee);
ff(u(ge))=ff(u(ge))+(sqrt(-i/2)./aa(ge)).*nonanexp(i*pi./(ld(u(ge)).*d(u(ge))).*(x(u(ge)).^2+2.*d(u(ge)).*(a(3)+d(u(ge))))).*f(u(ge));
end;
end;
% petits
ff(u(p))=ff(u(p))+(sqrt(-i/2)./aa(p)).*(y3(p)-y1(p)).*nonanexp(i*pi./(ld(u(p)).*d(u(p))).*(x(u(p)).^2+2.*d(u(p)).*(a(3)+d(u(p)))-bb(p).^2));

end;
%%%%%%%%%%%%% dans le 'plan de Fourier local '
aa=d(uu)*a(2)-x(uu);
ff(uu)=ff(uu)+(xx(kk+2)-xx(kk))*sqrt(-i./(ld(uu).*d(uu))).*nonanexp(i*pi./(ld(uu).*d(uu)).*(x(uu).^2+2.*d(uu).*(a(3)+d(uu))+(xx(kk)+xx(kk+2))*aa)).*...
retsinc(( xx(kk+2)-xx(kk))*aa./(ld(uu).*d(uu)));   

end;        %  <---------------------
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
function xx=nonanexp(x);xx=exp(x);xx(~isfinite(xx))=realmax; % evite les nan

