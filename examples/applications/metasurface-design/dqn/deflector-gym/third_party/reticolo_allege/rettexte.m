function [t,l,T]=rettexte(varargin);
% function [t,l,T]=rettexte(a,b,...); t='a=1  b=2.5'  l=longueur de t
% ou si pas de sortie impression des variables en colonne
% les entrees, en nombre quelconque, peuvent etre des structures  elles sont alors ecrites sous forme 'developpee'
% soit en ligne (si variables en sortie) soit en colonne (si pas de variable en sortie)
% T contient sous forme de cell_array de texte la forme developpee en
% colonnes et peut par exemple etre ecrite dans une figure avec text
% 
% % Exemples
% a=struct('ch1',1,'ch2',1:3,'ch3',rand(2,2));x=78;tab=rand(2,3);
% disp(' ');
% disp(' ecrit en colonne sous forme developpee');
% rettexte(x,a,pi,tab)
% disp(' ');
% disp(' t contient tout en ligne');
% t=rettexte(x,pi,tab)
% disp(' ');
% disp('disp(rettexte...) :    tout en ligne');
% disp(rettexte(x,pi,tab));
% disp(' ');
% figure;title(' ecriture sous forme developpee dans une figure');
% [t,l,T]=rettexte(x,a,pi,tab);
% text(.1,0.5,T);


if nargout==0;for ii=1:nargin;retecrit(varargin{ii},inputname(ii));end;return;end;
t='';T={};for ii=1:nargin;[prv,tt,TT]=retecrit(varargin{ii},inputname(ii));t=[t,tt];T=[T,TT];if ii<nargin;t=[t,'  '];end;end
l=size(t,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [nom,t,T]=retecrit(a,nom,t,T);
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
disp([nom,'.',s{ii},'=',num2str(full(af))]);
end; 
end;end;
else;% non structures
if (size(a,1)>1)|(~isnumeric(a));    
if isempty(nom);disp([' ']);else;disp([nom,'=']);end;disp(a);
else;
if isempty(nom);disp(['',num2str(full(a))]);else;disp([nom,'=',num2str(full(a))]);end;
end; 
end;
    



else; % creation d'une ligne de texte t
if nargin<3;t='';T={};end;
if isstruct(a); % structure
sz=size(a);m=prod(sz);
if m>1;
v=cell(size(sz));
for ii=1:m;
[v{:}]=ind2sub(sz,ii);nom1=[nom,'('];for jj=1:length(v)-1;nom1=[nom1,num2str(v{jj}),','];end;nom1=[nom1,num2str(v{end}),')'];       
[nom1,t,T]=retecrit(a(ii),nom1,t,T);
end;
t=t(1:end-2);
return
end;
s=fieldnames(a);
for ii=1:length(s);af=getfield(a,s{ii});
if isstruct(af);nom1=[nom,'.',s{ii}];[nom1,t,T]=retecrit(af,nom1,t,T);
else;
t=[t,nom,'.',s{ii},'=',num2str(af(:).'),'  '];
if length(find(size(af)>1))>1;T=[T,{[nom,'.',s{ii},'='],num2str(af)}];else;T=[T,{[nom,'.',s{ii},'=',num2str(af(:).')]}];end;
end;
end;
else;   % non structures
if isempty(nom);t=[t,nom,' ',num2str(a(:).'),'  '];T=[T,{num2str(a)}];
else;t=[t,nom,'=',num2str(a(:).'),'  '];
if length(find(size(a)>1))>1;T=[T,{[nom,'=']},{num2str(a)}];else;T=[T,{[nom,'=',num2str(a(:).')]}];end;
end;
end;

end;
