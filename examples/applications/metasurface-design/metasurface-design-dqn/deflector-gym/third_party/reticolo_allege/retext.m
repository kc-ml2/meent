function ext=retext(a);
% ext=retext(fich);
% extensions du fichier 'fich'
% ext est un cell_array contenant toutes les extensions du fichier donne
a=dir([a,'.*']);
ext=cell(size(a));
for ii=1:length(ext);
m=strfind(a(ii).name,'.');
ext{ii}=a(ii).name(m(end)+1:end);
end;