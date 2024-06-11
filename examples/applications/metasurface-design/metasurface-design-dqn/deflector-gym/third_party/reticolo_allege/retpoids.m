function y=retpoids(x,varargin)
%  y=retpoids(x,c,a,d,cas);
% fonction periodique,de periode d,(sauf sur la derniere dimension ) de centre c, et largeur a
% si real(cas)=0 f=exp(-x^(2*imag(cas)+2)) 
% si real(cas)=1 f=4*(exp(-x.^(2*imag(cas)+2))-exp(-2*x.^(2*imag(cas)+2))) ( contours)
% si real(cas)=3 f=(3-2*(b+1-y(f))/b).*((b+1-y(f))/b).^2;(b=imag(k))
% par defaut cas=0 (gaussienne)
%
% possibilite de tester: retpoids(poids,centre,h,d);
%  centre :coordonnees du centre(1D,ou 2D) :on trace une periode autour de ce centre
%   en 1 D :h hauteur du maillage trace sur tout le maillage
%   en 2 D : si imag(h)=0  tracé sur une coupe Oxy  à z= centre(3)
%            si imag(h)=1  tracé sur une coupe Oxz  à y= centre(2)
%            si imag(h)=-1 tracé sur une coupe Oyz  à x= centre(1)
% acceleration:
% poids=retpoids(poids,h,d); le poids retourne est alors toujours>0(abs)

if ~isnumeric(x);
if nargin>3;testpoids(x,varargin{:});else y=fastpoids(x,varargin{:});end;
return;end;
[c,a,d]=deal(varargin{1:3});
if nargin>4;cas=varargin{4};else;cas=0;end;
y=((x(:,end)-c(end))/a(end)).^2;
for ii=1:size(x,2)-1;
y=y+(d(ii)/(pi*a(ii))*sin(pi*(x(:,ii)-c(ii))/d(ii))).^2;
end;
y=sqrt(y);
b=imag(cas);puiss=2*b+2;
switch real(cas);
case 0;y=exp(-y.^puiss);
case 1;y=y*(log(2)^(1/puiss));y=4*(exp(-y.^puiss)-exp(-2*y.^puiss));% couronne
case 3;y(y<1)=1;y(y>1+imag(cas))=0;f=find((y>1)&y<(1+b));y(f)=(3-2*(b+1-y(f))/b).*((b+1-y(f))/b).^2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function testpoids(poids,centre,h,d);%[poids,centre,h,d]=deal(varargin{:});
if length(d)==2;% 2D
switch(imag(h));
case 0;  %  tracé sur une coupe Oxy  à z = centre(3)
x=linspace(0,d(1),91);
y=linspace(0,d(2),101);
[X,Y,Z]=ndgrid(mod(x,d(1)),mod(y,d(2)),centre(3));
P=feval(poids,[X(:),Y(:),Z(:)]);P=reshape(P,size(X));
figure('color','w');retcolor(x,y,P.',2);axis equal;axis tight

case -1; %  tracé sur une coupe Oyz  à x = centre(1)
z=linspace(0,real(h),91);
y=linspace(0,d(2),101);
[Y,Z,X]=ndgrid(mod(y,d(2)),z,mod(centre(1),d(1)));
P=feval(poids,[X(:),Y(:),Z(:)]);P=reshape(P,size(Y));
	
figure('color','w');retcolor(y,z,P.',2);axis equal;axis tight

case 1;  %  tracé sur une coupe Oxz  à y = centre(2)
z=linspace(0,real(h),91);
x=linspace(0,d(1),101);
[X,Z,Y]=ndgrid(mod(x,d(1)),z,mod(centre(2),d(2)));
P=feval(poids,[X(:),Y(:),Z(:)]);P=reshape(P,size(Y));
	
figure('color','w');retcolor(x,z,P.',2);axis equal;axis tight
	
end;


else;% 1D 
x=linspace(-d/2,d/2,501)+centre;
y=linspace(0,h,101);
[X,Y]=ndgrid(mod(x,d),y);
P=feval(poids,[X(:),Y(:)]);P=reshape(P,size(X));
figure('color','w');retcolor(x,y,P.',2);axis equal;axis tight

end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Poids=fastpoids(poids,h,d);
if length(d)==2;% 2D 
N=200000;a=(N/(h*d(1)*d(2)))^(1/3);
x=linspace(0,d(1),3+ceil(d(1)*a));
y=linspace(0,d(2),3+ceil(d(2)*a));
z=linspace(0,h,3+ceil(h*a));
[X,Y,Z]=meshgrid(x,y,z);
P=reshape(poids([X(:),Y(:),Z(:)]),size(X));
Poids=@(x) (abs(interp3(X,Y,Z,P,mod(x(:,1),d(1)),mod(x(:,2),d(2)),x(:,3),'*linear')));

	
%Poids=poids;
else;% 1D 
N=200000;
x=linspace(0,d,3+ceil(sqrt(N*d/h)));
y=linspace(0,h,3+ceil(sqrt(N*h/d)));
[X,Y]=meshgrid(x,y);
P=reshape(poids([X(:),Y(:)]),size(X));
Poids=@(x) (abs(interp2(X,Y,P,mod(x(:,1),d),x(:,2),'*linear')));
end;
