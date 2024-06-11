function [ii,w]=retcoupe(x,x0);
%function [ii,w]=retcoupe(x,x0);
% recherche du point le plus voisin de x0 dans le tableau x
% w poids pour avoir une meilleure approche  sum(x.*w)=x0;

[prv,ii]=sort(abs(x-x0));
w=zeros(size(x));
xx=x(ii(2))-x(ii(1));
if xx~=0;
w(ii(1))=(x(ii(2))-x0)/xx;
w(ii(2))=(x0-x(ii(1)))/xx;
else w(ii(1))=1;end
ii=ii(1);    