function retdir(texte,ext);
%  retdir(texte,extension);
% impression des fichiers contenant une chaine de caractere sauf les fichiers ne commençant pas par 'ret' (fichiers temporaires)
% par defaut extension='*.mat'
if nargin<2;ext='*.mat';end;
a=dir(ext);
for ii=1:length(a);
if ~isempty(strfind(a(ii).name,texte))&(sum(strfind(a(ii).name,'ret')==1)==0);disp(a(ii).name),
end;end