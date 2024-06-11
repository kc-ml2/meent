function cao=retcao(xc,lc,d,n);
%calcul de cao:changement de coordonnees
%xc:point ou la derivee du changement de coordonnees est nulle
%lc:largeur de la partie modifiee
g=imag(lc);xc=xc/d;lc=real(lc)/d;

if g==0;  %changement de coordonnees reel
cao=-lc/4*exp(-[-n+1:n-1]*(2*pi*i*xc)).*(retsinc(lc*[-n+1:n-1]-1)+2*retsinc(lc*[-n+1:n-1])+retsinc(lc*[-n+1:n-1]+1));
else;     % changement de coordonnees complexe
g=(1-1/(1+i*g))/4;    
cao=-lc/4*exp(-[-n+1:n-1]*(2*pi*i*xc)).*(-g*retsinc(lc*[-n+1:n-1]-2)+retsinc(lc*[-n+1:n-1]-1)+2*(1+g)*retsinc(lc*[-n+1:n-1])+retsinc(lc*[-n+1:n-1]+1)-g*retsinc(lc*[-n+1:n-1]+2));
end
cao(n)=cao(n)+1;
