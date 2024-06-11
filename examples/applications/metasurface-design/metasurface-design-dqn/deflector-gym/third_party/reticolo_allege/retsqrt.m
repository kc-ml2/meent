function y=retsqrt(x,k,tet);
%  function y=retsqrt(x,k,tet);
%  determination de sqrt(x)
%
%  k=0 ou abs: DETERMINATION DES ONDES PLANES   PRIORITEE A L'ATTENUATION   (coupure de Petit) 
%     imag(y)>0 et si imag(y)==0,real(y)>=0  
%     exp(i y *z) onde bornee si z>0 ou  dirigee vers les z>0
%
%  k=-1 DETERMINATION DES ONDES PLANES   exp( iy *z ) est une onde dirigee vers les  z>0
%  tet :angle de la coupure (rad) pi/2 par defaut (coupure de Maystre )
%
%  k=1 DETERMINATION DES MODES   exp( y *z ) est une onde dirigee vers les  z<0
%    ( si yy=iy      exp(i yy *z) est une onde dirigee vers les  z>0  )
%  tet :angle de la coupure (rad) pi/2 par defaut (coupure de Maystre )
%  pour changer tet  retsqrt(0,0,tet)  tet reste alors  change pour la suite
%  retsqrt retourne la valeur actuelle de tet



persistent teta e1 e2;if isempty(teta);teta=pi/2;e1=-i;e2=(1-i)/rettsqrt(2);end;
if nargin==3;teta=tet;if teta~=pi/2;e1=exp(-i*teta);e2=exp(-.5i*teta);else;e1=-i;e2=(1-i)/rettsqrt(2);end;end;
if nargin<1;y=teta;return;end;
if nargin<2;k=0;end;
switch(k);
case 0;   % coupure de Petit
y=i*conj(rettsqrt(-conj(x)));
case 1;  % coupure a teta determination des MODES
if teta==pi;y=-i*rettsqrt(-x);else;y=conj(rettsqrt(e1*conj(x)))*e2;end;
case -1;  % coupure a teta determination des ONDES PLANES
if teta==pi;y=rettsqrt(x);else;y=i*conj(rettsqrt(-e1*conj(x)))*e2;end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=rettsqrt(x);% c'est la racine mathematique
if isempty(x);return;end;
x=sqrt(x);f=find((real(x)<0)|(real(x)==0&imag(x)<0));x(f)=-x(f);