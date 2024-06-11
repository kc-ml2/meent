function [v,d]=retds(s,h);
% function [v,d]=retds(s,h);
% diagonalisation matrice T associee a S (sans passer par T)
%  v:vecteurs propres(colonnes)  d:matrice diagonale des valeurs propres associees
% si h est precise: objet symetrique de hauteur h  diagonalisation directe
%
% remarque: s peut avoir ete stocké sur fichier par retio(s,1)

s=retio(s);


if nargin<2; % pas symetrique diagonalisation generalisee  matrice T
if size(s,1)<6; %matrices s
n1=s{2};n2=s{3};
if nargout<2;% seulement valeurs propres
v=reteig([s{1}(:,1:n1),[zeros(n1,n2);-eye(n2)]],[[eye(n1);zeros(n2,n1)],-s{1}(:,n1+1:n1+n2)]);
else;        % diagonalisation
[v,d]=reteig([s{1}(:,1:n1),[zeros(n1,n2);-eye(n2)]],[[eye(n1);zeros(n2,n1)],-s{1}(:,n1+1:n1+n2)]);
end;
else %matrices g
[v,d]=reteig(s{1},s{2});    
end;    

else; %symetrique diagonalisation matrice S 
if size(s,1)>=6;sog=0;s=retgs(s);else;sog=1;end;
n=s{2};
if nargout<2;% seulement valeurs propres
v=reteig(s{1}(1:n,1:n));v=acosh(1./v);v=diag(exp([v;-v]));
else;       % diagonalisation
[q,d]=reteig(s{1}(1:n,1:n));
d=acosh(1./diag(d));
p=-s{1}(n+1:2*n,1:n)*q*diag(1./tanh(d));
d=diag(exp([d;-d]));
v=[[q;p],[q;-p]];
end;
end;    