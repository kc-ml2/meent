function s=reti(init,x,ep,h,teta);
% calcul de la matrice s associee au passage dans un milieu periodique invariant 
% dans une direction inclinee d'un angle teta(degres) sur y
% de parametres dielectriques constants par morceaux
% x:points de discontinuite en y=0 
% ep valeurs de [epsilon;mux;1/muy]
% h hauteur de la couche

n=init{1};d=init{end}.d;beta=init{2};
if iscell(x);x{1}=x{1}/d;x{2}=x{2}/d;else;x=x/d;end;
[fmu,fmmu,fep]=retc1(x,ep,beta);
amu=inv(rettoeplitz(fmu));ammu=rettoeplitz(fmmu);
ct=cos(teta*pi/180);st=sin(teta*pi/180);tt=st/ct;
mu1=amu.*(ct*ct)+ammu.*(st*st);mu2=inv(amu.*(st*st)+ammu.*(ct*ct));mu0=(ammu-amu).*(st*ct);
m=[[(i*tt).*diag(beta)-i.*(mu2*mu0)*diag(beta),mu2];[diag(beta)*(mu1-mu0*mu2*mu0)*diag(beta)-rettoeplitz(fep),-i.*diag(beta)*(mu0*mu2)+(i*tt).*diag(beta)]];
[p,d]=reteig(m);%diagonalisation
dd=diag(d);[y,io]=sort(real(dd)+1.e-6*imag(dd));p=p(:,io);dd=dd(io); %classement en real(vp) decroissantes 
pp=inv(p);
pp=retgs(pp);
p=retgs(diag(exp(-(i*h*tt).*[beta,beta]))*p);
s={diag(exp([dd(1:n);-dd(n+1:2*n)].*h));n;n;1};
s=retss(p,s,pp);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fmu,fmmu,fep]=retc1(x,ep,beta);n=length(beta);alpha=[-n+1:n-1]*(2*pi);
fmu=retf(x,ep(2,:),alpha);fmmu=retf(x,ep(3,:),alpha);fep=retf(x,ep(1,:),alpha);
