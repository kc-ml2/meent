function retwhos(w,nom);
% retwhos
% construction d'une structure à champs 'Donnees'
% contenant toutes les variables de petite taille,les chaines de caracteres
% ainsi que l'heure et le nom du programme
% (utile pour stocker les parametres de calcul)
% 


if nargin==0;
evalin('caller','ret_heure=retheure;ret_PGM=dbstack;ret_PGM=ret_PGM(end).file;retwhos(whos);');	
return;
end;
if nargin<2;nom='Donnees';end;
prv='';
for ii=1:length(w);if~isempty(strfind(w(ii).class,'char'));prv=[prv,',''',w(ii).name,''',',w(ii).name];end;end;
f=find([w(:).bytes]<=32);w=w(f);
for ii=1:length(w);if~isempty(strfind(w(ii).class,'double')) & isempty(strfind(w(ii).name,'ans'));prv=[prv,',''',w(ii).name,''',',w(ii).name];end;end;
prv=prv(2:end);
evalin('caller',[nom,'=struct(',prv,');'])
