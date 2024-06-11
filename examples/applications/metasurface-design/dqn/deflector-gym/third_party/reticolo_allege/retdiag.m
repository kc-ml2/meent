function dd=retdiag(d);
% function dd=retdiag(d);d=sparse(diag(d));marche meme si d est vide ou un vecteur ligne
if numel(d)<=4;if isempty(d);dd=[];return;end;dd=sparse(diag(d));return;end;
n=size(d);
if (n(1)==1)|(n(2)==1); % diagonale -> matrice carree creuse
%m=max(n);dd=spdiags(d(:),0,m,m);
m=max(n);dd=1:m;dd=sparse(dd,dd,d(:),m,m,m);% plus rapide que spdiags
%if n(2)>1;dd=spdiags(d.',0,n(2),n(2));else;dd=spdiags(d,0,n(1),n(1));end;
else                   %  matrice->diagonale
dd=diag(d);    
end;
