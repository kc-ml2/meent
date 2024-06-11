function f=retf(x,ff,alpha,cas)
%  f=retf(x,ff,alpha)
% calcul des coefficients de fourier d'une fonction constante par morceaux
% points de discontiniute:x valeurs à gauche ff
% ff peut etre forme de plusieurs lignes (plusieurs tf a la fois)
% f=somme_de_0_à_1 ff*exp(-i alpha x )
% si size(ff)=[p,m] , length(alpha)=n :  size(f) = [p,n]
% si x est un cellarray [x,w]=deal(x{1}(:),x{2}(:))
% on calcule l'integrale de fourier par une methode de Gauss

if nargin<4;cas=1;end;if iscell(x);cas=3;end;
switch cas;
case 1 % ancienne version 
x=x(:).';alpha=alpha(:).';        
f=([ff(:,2:end),ff(:,1)]-ff)*exp(-i*x'*alpha);% tf de la derivee au sens des distributions
[tt,t]=retfind(alpha==0);if ~isempty(t);f(:,t)=-i*f(:,t)*diag(1./alpha(t));end;% on divise par i*alpha
if ~isempty(tt);f(:,tt)=ff*(x-[x(end)-1,x(1:end-1)]).';end; % en 0 somme

case 2 % nouvelle version plus precise 
x=x(:);
x=[x(end)-1;x];alpha=alpha(:).';
f=zeros(size(ff,1),length(alpha));
for ii=1:length(x)-1;
f=f+ff(:,ii)*((x(ii+1)-x(ii))*exp(-i*alpha*(x(ii+1)+x(ii))/2).*retsinc(alpha*(x(ii+1)-x(ii))/(2*pi)));
end;

case 3 % fonction avec poids 
[x,w]=deal(x{1}(:),x{2}(:));
alpha=alpha(:).';
f=ff*(retdiag(w)*exp(-i*x*alpha));

end
%f=f*retdiag(retchamp([7.25,size(f,2)])); 






