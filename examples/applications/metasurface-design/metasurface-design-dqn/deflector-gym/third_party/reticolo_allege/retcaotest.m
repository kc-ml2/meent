function cao=retcao(xc,lc,d,n);
%calcul de cao:changement de coordonnees
%xc:point ou la derivee du changement de variable est nulle
%lc:largeur de la partie modifiee
gg=imag(lc);g=gg;xc=xc/d;lc=real(lc)/d;
if g==0;  %changement de coordonnees reel
cao=-lc/4*exp(-[-n+1:n-1]*(2*pi*i*xc)).*(retsinc(lc*[-n+1:n-1]-1)+2*retsinc(lc*[-n+1:n-1])+retsinc(lc*[-n+1:n-1]+1));
else;     % changement de coordonnees complexe
g=(1-1/(1+i*gg))/4;  
%g=i*imag(g);  %<------------ modif
cao=-lc/4*exp(-[-n+1:n-1]*(2*pi*i*xc)).*(-g*retsinc(lc*[-n+1:n-1]-2)+retsinc(lc*[-n+1:n-1]-1)+2*(1+g)*retsinc(lc*[-n+1:n-1])+retsinc(lc*[-n+1:n-1]+1)-g*retsinc(lc*[-n+1:n-1]+2));
    
end

cao(n)=cao(n)+1;


return

cao=-lc/4*exp(-[-n+1:n-1]*(2*pi*i*xc)).*(retsinc(lc*[-n+1:n-1]-1)+2*retsinc(lc*[-n+1:n-1])+retsinc(lc*[-n+1:n-1]+1));
cao(n)=cao(n)+1;


nn=2*1024;
x=xc+linspace(-1,1,nn+1)/2;x=x(1:end-1);
f=find(abs(x-xc)<lc/2);
cc=cos((x(f)-xc)*pi/lc).^2;
%dd=min(1,2*(1-abs( (x(f)-xc)/(lc/2) )));
dd=1-abs( (x(f)-xc)/(lc/2) );cc=dd;
%dd=ones(size(f));
%dd=exp(1)*exp(-1./(1-(2*(x(f)-xc)/lc).^4));



yy=ones(size(x));
yy(f)=(1-cc).*(1+(1/(1+i*gg)-1)*dd);
zz=fftshift(fft(yy,nn))/nn;zz=zz(nn/2-n+2:nn/2+n);cao=zz;return


cao=rettoeplitz(cao);
cao=cao*inv(rettoeplitz(zz));

return

yy(yy==0)=inf;zz=fftshift(fft(1./yy,nn))/nn;
zz=zz(nn/2-n+2:nn/2+n);
cao=inv(rettoeplitz(zz));



%xc=xc/d;lc=lc/d;
%cao=-lc/4*exp(-[-n+1:n-1]*(2*pi*i*xc)).*(retsinc(lc*[-n+1:n-1]-1)+2*retsinc(lc*[-n+1:n-1])+retsinc(lc*[-n+1:n-1]+1));
%cao(n)=cao(n)+1;
