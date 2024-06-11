function [e,a]=retreal(e);
% function [ee,a]=retreal(e);
% transformation d'un tableau complexe en tableau
% reel le plus voisin par multiplication par un complexe a de module 1
% de plus a est tel que real(a)>=0
% ee=e*a
% on minimise la distance à la droite passant par l' origine dans le plan complexe
f=find(isfinite(real(e))&isfinite(imag(e)));
if isempty(f);a=1;return;end;
xx=sum(real(e(f)).^2);
yy=sum(imag(e(f)).^2);
if (xx+yy)==0;a=1;return;end;
xy=sum(real(e(f)).*imag(e(f)));
if xx>yy; % a tel que imag(e)=a*real(e);
a=roots([xy,xx-yy,-xy]);
[prv,ii]=min((a.^2*xx-2*a*xy+yy)./(1+a.^2));
a=1/(1+i*a(ii));
else; %  a tel que real(e)=a*imag(e);
a=roots([xy,yy-xx,-xy]);
[prv,ii]=min((a.^2*yy-2*a*xy+xx)./(1+a.^2));
a=1/(a(ii)+i);
end;
a=a/norm(a);if real(a)<0;a=-a;end;% normalisation et choix de la determination
e(f)=e(f)*a;