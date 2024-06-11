function x=retx(xx,x0,x1,n);
% function x=retx(xx,x0,x1,n); generation de n nouvelles valeur de xx entre x0 et x1
% si n est absent 1 seule valeur

if nargin<4;n=1;end;
if x0>x1;[x1,x0]=retpermute(x0,x1);end;
f=find((xx>x0)&(xx<x1));xx=sort([xx(f),x0,x1]);
x=[];for ii=1:n;
[prv,ii]=max(xx(2:end)-xx(1:end-1));
a=rand;b=rand;xnv=(a*xx(ii+1)+b*xx(ii))/(a+b); 
x=[x,xnv];xx=sort([xx,xnv]);
end;
