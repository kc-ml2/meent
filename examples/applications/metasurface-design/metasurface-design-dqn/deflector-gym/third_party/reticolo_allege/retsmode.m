function [gama,kmax,vm,erm,e,o,yy,wyy,vmc]=retsmode(pol,n,h,ordre,sym,h_tracer,k0);
n=retcolonne(n,1);h=retcolonne(h,1);h_tracer=retcolonne(h_tracer,1);
if nargin<7;k0=1;end;
if nargin<6;h_tracer=[];end;
tab=[h(:),n(2:end)];

if ordre==0;
for kk=1:20;
khi_int_2=.2*k0*randn^2*pi/h(end);
[khi_int_2,niter,er,erfonc,test]=retcadilhac(@calq,struct('niter',50,'tol',1.e-10,'tolf',1.e-10),khi_int_2,pol,k0,n,tab,ordre,sym,-1);
if all(test);break;end;
end;
[q,gama]=calq(khi_int_2,pol,k0,n,tab,ordre,sym,-1);
	
else;
for kk=1:20;
talfa=randn;
[talfa,niter,er,erfonc,test]=retcadilhac(@calq,struct('niter',50,'tol',1.e-10,'tolf',1.e-10),talfa,pol,k0,n,tab,ordre,sym,1);
if all(test);break;end;
end;
[q,gama]=calq(talfa,pol,k0,n,tab,ordre,sym,1);
end;

gama/k0
nn=[n,fliplr(n)];hh=[h,fliplr(h)];if ~isempty(h_tracer);h_tracer=[h_tracer(1),h_tracer];end;
[varargout{1:nargout}]=retmode(pol,nn,hh,gama,[],h_tracer,nan,k0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [q,gama]=calq(x,pol,k0,n,tab,ordre,sym,sens);
if sens==1;khi_int=(ordre*pi+atan(x))/tab(end,1);gama=retsqrt(k0^2*n(end)^2-khi_int.^2,-1);
else;gama=retsqrt(k0^2*n(end)^2-x,-1);end;
init={pol,gama,k0};sh=retb(init,n(1),1); sb=retb(init,n(end),-1); 
s=reteval(retss(sh,retcouche(init,tab),sb));
rb=s(2,1,:);q=rb-sym;
