function  a=retfullsparse(a,taillemin,densite);
% function  a=retfullsparse(a,taillemin,densite);
% transformation de a en sparse ou full suivant sa densite (1/20 par defaut)
% a peut etre un cell_array l'action porte alors sur les tableaux numeriques
% seuls les elements de longueur>taillemin (100 par defaut) sont traites
%
% See also:RETSPARSE

if nargin<3;densite=1/20;end;
if nargin<2;taillemin=100;end;
if isempty(a);return;end;
if iscell(a);
for ii=1:length(a(:));a{ii}=retfullsparse(a{ii},taillemin,densite);end;
else;
% if  ~isnumeric(a);return;end;
try;
if length(a(:))<taillemin;return;end
if nnz(a)<(densite*numel(a));a=sparse(a);else;a=full(a);end;
end;
end;
