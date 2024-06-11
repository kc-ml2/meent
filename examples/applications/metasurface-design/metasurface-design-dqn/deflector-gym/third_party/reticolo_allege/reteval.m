function ss=reteval(s,n,m);
% function ss=reteval(s,n,m);
%  evaluation d'elements d'une matrice S
%  mais s peut etre une matrice G ou etre un fichier temporaire ecrit avec retio
%  utilisation conseillee pour la generalitee des programmes
%
%  n m:    vecteurs   eventuellement absents:
%
%  ss=reteval(s,n,m)  equivaut a ss=ss{1}(n,m)
%  ss=reteval(s,0,m)  equivaut a ss=ss{1}(:,m)
%  ss=reteval(s,n,0)  equivaut a ss=ss{1}(n,:)
%
%  ss=reteval(s,0,0)  equivaut a ss=ss{1}(:,:)
%  ss=reteval(s,n)    equivaut a ss=ss{1}(n) matrices a 1 dimension
%  ss=reteval(s,0)    equivaut a ss=ss{1}(:) matrices a 1 dimension
%
%  ss=reteval(s)      equivaut a ss=ss{1} toute la matrice

ss=retgs(retio(s));if iscell(ss);ss=full(ss{1});end;
if nargin<2;return;end;
if (nargin<3);if(all(n>0));ss=ss(n);else ss=ss(:);end;return;end;

if all(n>0);
if all(m>0);ss=ss(n,m);else;ss=ss(n,:);end;
else;
if all(m>0);ss=ss(:,m);end;
end;
