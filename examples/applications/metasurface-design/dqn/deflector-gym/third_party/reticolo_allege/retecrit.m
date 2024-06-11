function [nom,t]=retecrit(a,nom,t);
% function nom=retecrit(a,nom);
% ecriture d'une structure sous forme 'developpee'

    
if nargin<2;nom=[];end;if isempty(nom);nom=inputname(1);end;

if nargout<2;  % ecriture
if isstruct(a);% structure
sz=size(a);m=prod(sz);
if m>1;
v=cell(size(sz));
for ii=1:m;
[v{:}]=ind2sub(sz,ii);nom1=[nom,'('];for jj=1:length(v)-1;nom1=[nom1,num2str(v{jj}),','];end;nom1=[nom1,num2str(v{end}),')'];       
nom1=retecrit(a(ii),nom1);
end;
return
end;
s=fieldnames(a);
for ii=1:length(s);af=getfield(a,s{ii});
if isstruct(af);nom1=[nom,'.',s{ii}];nom1=retecrit(af,nom1);
else;
if (size(af,1)>1)|(~isnumeric(af));    
disp([nom,'.',s{ii},'=']);disp(af);
else;
disp([nom,'.',s{ii},'=',num2str(af)]);
end; 
end;end;
else;% non structures
if (size(a,1)>1)|(~isnumeric(a));    
disp([nom,'=']);disp(a);
else;
disp([nom,'=',num2str(a)]);
end; 
end;
    



else; % creation d'une ligne de texte t
if nargin<3;t='';end;
if isstruct(a); % structure
sz=size(a);m=prod(sz);
if m>1;
v=cell(size(sz));
for ii=1:m;
[v{:}]=ind2sub(sz,ii);nom1=[nom,'('];for jj=1:length(v)-1;nom1=[nom1,num2str(v{jj}),','];end;nom1=[nom1,num2str(v{end}),')'];       
[nom1,t]=retecrit(a(ii),nom1,t);
end;
t=t(1:end-2);
return
end;
s=fieldnames(a);
for ii=1:length(s);af=getfield(a,s{ii});
if isstruct(af);nom1=[nom,'.',s{ii}];[nom1,t]=retecrit(af,nom1,t);
else;
t=[t,nom,'.',s{ii},'=',num2str(af(:).'),'  '];
end;
end;
else;   % non structures
t=[t,nom,'=',num2str(a(:).'),'  '];
end;

end;
