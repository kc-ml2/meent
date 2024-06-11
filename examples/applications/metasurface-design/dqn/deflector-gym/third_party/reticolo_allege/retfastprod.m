function a=retfastprod(a1,a2);
%n a=retfastprod(a1,a2);
% produit de 2 matrices sparse a1 tres allongee,a2 tres haute,dont une partie est pleine l'autre creuse
% au retour a est pleine
prv=sum((a1~=0),1)+sum((a2~=0),2).';
[f,ff]=retfind(prv>.5*(size(a1,1)+size(a2,2)));
ff=ff(prv(ff)>0);
a=full(a1(:,ff)*a2(ff,:))+full(a1(:,f))*full(a2(f,:));
