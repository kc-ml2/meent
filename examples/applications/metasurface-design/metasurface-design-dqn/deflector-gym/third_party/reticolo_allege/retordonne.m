function [xx,eep]=retordonne(x,ep,d);
%  function [xx,eep]=retordonne(x,ep,d);
%  mise en forme de x ep utilises par retu
%  d pas
%  x,ep discontinuitees de x (croissant )  valeurs de ep a gauche  
%  xx eep:idem mais xx est divise par d et ordonnee   xx(0)>0  xx(end)=1

m=length(x);
masque=ones(size(x)); % on ne garde que les x qui sont strictement plus grands que les precedents en imposant leur ep
for ii=1:m-1;f=find(x(ii+1:m)<=x(ii));x(f+ii)=x(ii);masque(f+ii)=0;end;
f=find(masque==1);x=x(f);ep=ep(:,f);

f=[1;1+find(diff(x(:))>10*eps*d)];x=x(f);ep=ep(:,f);% elimination des tres voisins TEST
if abs(x(end)-x(1)-d)<10*eps*d;x=x(2:end);ep=ep(:,2:end);end;

% elimination des frontieres inutiles
[x,ii]=sort(mod(x/d,1));ep=ep(:,ii);
xx=1;eep=ep(:,1);
for jj=length(x):-1:1;
if x(jj)>0 & ~all(ep(:,jj)==eep(:,1));
xx=[x(jj),xx];eep=[ep(:,jj),eep];
end;
end;

