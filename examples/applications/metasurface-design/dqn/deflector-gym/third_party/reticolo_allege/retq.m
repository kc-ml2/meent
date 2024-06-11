function [q,ldz,hr]=retq(ld,h,hr);
%  function [q,ldz,hr]=retq(ld,h,hr);
% calcul de q , ldz complexe ,hr reel,a partir des poles pour la variable complexe h en 2 ou 3 valeurs de ld(eventuellement complexes)
% si hr est precise:c'est la valeur desiree pour hr  sinon on prend la moyenne des h

if nargin<3;hr=mean(real(h));end;

ldm=mean(ld);
n=length(h);
[a,s,mu]=polyfit(h,ld-ldm,n-1);
%ldz=ldm+polyval(a,(real(hr)-mu(1))/mu(2));
ldz=ldm+polyval(a,real(hr),s,mu);

q=real(ldz)/(2*imag(ldz));
