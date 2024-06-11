function a=retsparse(a,k);
%function a=retsparse(a,k);
% transforme les tableaux d'un cell array en sparse si nnz<k*numel(a) k=.2 par defaut
%
% See also:RETFULLSPARSE

if nargin<2;k=.2;end;
if iscell(a);
for ii=1:numel(a);a{ii}=retsparse(a{ii});end;
else;
try;if nnz(a)<k*numel(a);a=sparse(a);end;end;	% plus rapide que:  if isnumeric(a)&(size(size(a),2)<=2)&(nnz(a)<k*numel(a));a=sparse(a);end;
end;
