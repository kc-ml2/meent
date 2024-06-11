function d=retbidouille(d,k,parm);
% function d=retbidouille(d,k); modification des valeurs propres
% k:origine  1:retcouche  2:retb  3:retc  4:retfp 6:retfpm
%if exist('k')~=1;k=0;end;

if nargin<3;parm=1;end;
alpha=parm*100*eps;
ii=find(abs(d)<sqrt(alpha));if isempty(ii);return;end;% pour accelerer...
iii=find(imag(d(ii).^2)>0);d(ii(iii))=retsqrt(real(d(ii(iii)).^2),1);
d(ii)=retsqrt(d(ii).^2-i*alpha*sqrt(1-(abs(d(ii)).^2/alpha).^2),1);
