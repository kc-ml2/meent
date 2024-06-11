function s=retrenverse(s,sens);
% function s=retrenverse(ss,sens);
% retournement matrice ss 
%
% si sens=0 ou absent  matrice S des modes
% ss(nn3+nn4,nn1+nn2) devient s(n3+n4,n1+n2) 
%                                  
%   Dh     Ib       Db    Ih               
%      =SS             =S              
%   Db     Ih       Dh    Ib               
%                                  
%
%  si sens=0 des modes aux modes   ss:des modes(en bas) aux modes(en haut)    devient s des modes(en bas) aux modes(en haut)
%  si sens=1 des champs aux champs ss:du champ(en bas) au champ(en haut)    devient s du champ(en bas) au champ(en haut)  
%  si sens=2 des champs aux modes  ss:du champ(en bas) aux modes(en haut)    devient s des modes(en bas) au champ(en haut) (Sh devient Sb)
%  si sens=3 des modes aux champs  ss des modes(en bas) au champ(en haut)  devient s du champ(en bas) aux modes(en haut)   (Sb devient Sh)
%
% remarque: ss peut avoir ete stocké sur fichier par retio(ss,1) mais au retour s ne l'est pas
% See also:RETTR


s=retio(s);
if isempty(s);s=[];return;end; % matrices vides
if nargin<2; sens=0;end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(s,2)==2;s=retsp(s,-1); %  0 D
switch sens;
case 0;% entre modes s
s={s{2,2},s{2,1};s{1,2},s{1,1}};
case 1; %entre champs 
s={s{1,1},-s{1,2};-s{2,1},s{2,2}};
case 2; % des champs aux modes  s:du champ aux modes    devient s des modes au champ
s={s{1,2},s{1,1};-s{2,2},-s{2,1}};
case 3; % des modes aux champs  s des modes au champ  devient s du champ aux modes
s={s{2,1},-s{2,2};s{1,1},-s{1,2}};
end;  
   
    
return;end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(s,1)<6; %matrices s
nn1=s{2};nn2=size(s{1},2)-nn1;nn3=s{3};nn4=size(s{1},1)-nn3;
switch sens;
case 0;% entre modes s
s={s{1}([[nn3+1:nn3+nn4],[1:nn3]],[[nn1+1:nn1+nn2],[1:nn1]]);nn2;nn4;1}; 
case 1; %entre champs    
s=inv(s{1});s([1:nn1],[nn1+1:2*nn1])=-s([1:nn1],[nn1+1:2*nn1]);s([nn1+1:2*nn1],[1:nn1])=-s([nn1+1:2*nn1],[1:nn1]);
s={s;nn1;nn1;1};
case 2; % des champs aux modes  s:du champ aux modes    devient s des modes au champ
s={[-s{1}(:,1:nn1),[eye(nn3);zeros(nn4,nn3)]]\[s{1}(:,nn1+1:nn1+nn2),[zeros(nn3,nn4);eye(nn4)]];nn2;nn1;1};    
case 3; % des modes aux champs  s des modes au champ  devient s du champ aux modes
s={[[zeros(nn3,nn4);eye(nn4)],s{1}(:,nn1+1:nn1+nn2)]\[[-eye(nn3);zeros(nn4,nn3)],s{1}(:,1:nn1)];nn3;nn4;1};    
end;  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else % matrices g
n1=s{3};n2=s{4};n3=s{5};n4=s{6};
switch sens;
case 0; % entre modes s
s={[s{2}(:,n3+1:n3+n4),s{2}(:,1:n3)];[s{1}(:,n1+1:n1+n2),s{1}(:,1:n1)];n4;n3;n2;n1}; 
case 1; %entre champs    
s={[s{2}(:,1:n3),-s{2}(:,n3+1:n3+n4)];[s{1}(:,1:n1),-s{1}(:,n1+1:n1+n2)];n3;n4;n1;n2};    
case 2; % des champs aux modes  s:du champ aux modes    devient s des modes au champ
s={[s{2}(:,n3+1:n3+n4),s{2}(:,1:n3)];[s{1}(:,1:n1),-s{1}(:,n1+1:n1+n2)];n4;n3;n1;n2};    
case 3; % des modes aux champs  s des modes au champ  devient s du champ aux modes
s={[s{2}(:,1:n3),-s{2}(:,n3+1:n3+n4)];[s{1}(:,n1+1:n1+n2),s{1}(:,1:n1)];n3;n4;n2;n1};    
end;    
    
    
    
end;    