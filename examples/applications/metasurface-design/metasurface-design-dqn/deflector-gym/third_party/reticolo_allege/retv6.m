function retv6(fich4472154fdhjtruf5srt)
% function retv6(fich)
% transforme un fichier en version v6
% si fich n'est pas precise,tous les fichiers du repertoir sont transformes
% sauf les fichiers temporaires crees par retio 
% See also: RETEFFACE

if nargin>0;try;ret4472154fdhjtruf5srt=who('-file',fich4472154fdhjtruf5srt);load(fich4472154fdhjtruf5srt);save(fich4472154fdhjtruf5srt,ret4472154fdhjtruf5srt{:},'-v6');end;return;end;
a=dir('*.mat');
for ii=1:size(a,1);
if ~(all(a(ii).name(1:3)=='ret')&(~isempty(str2num(a(ii).name(8:end-4)))));retv6(a(ii).name);end;    
end;
