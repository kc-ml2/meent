function a=retsqueeze(a)
% idem à squeeze mais applique le traitement aux tableaux de dimension2
%
% %% Exemple
% retsqueeze([1,2]),squeeze([1,2],retsqueeze([1;2]),squeeze([1;2])
% 
%   See also SQUEEZE SHIFTDIM 

sz=size(a);
sz(sz==1)=[];
if length(sz)==1;a=reshape(a,sz,1);else;a=reshape(a,sz);end;
