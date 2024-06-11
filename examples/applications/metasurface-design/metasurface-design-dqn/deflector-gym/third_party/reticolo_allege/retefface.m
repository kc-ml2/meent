function retefface(parm,ok)
% function retefface(parm,ok);
%   parm=0 (ou absent) efface tous les fichiers temporaires apres demande d'acceptation
%   parm=-1  efface tous les fichiers temporaires
%   parm=1  efface touts les figures(aprés acceptation)
%   parm=i  efface tous les fichiers temporaires du repertoire et des sous repertoires 
%   si ok=1 on efface sans demande d'acceptation

if nargin==0;parm=0;end;if nargin<2;ok=0;end;

switch(parm);
    
case i;% purge des fichiers temporaires du repertoire et des sous repertoires
retefface;
a=dir;aa=a([a.isdir]);CD=cd;disp(CD);
k=length(aa);
if k==2;return;end;
for ii=3:k;
cd(aa(ii).name);try;retefface(i);end;cd(CD);
end;
return
    
case {0,-1};type='.mat';% efface tous les fichiers temporaires
nn=[];
a=dir('*.mat');
for ii=1:size(a,1);
if all(a(ii).name(1:3)=='ret')&(~isempty(str2num(a(ii).name(8:end-4))));nn=[nn,ii];end;    
end;

case 1;type='.fig';% toutes les figures
a=dir('*.fig');
nn=1:size(a,1);
end; % switch(parm)
if isempty(nn);return;end;

nnn=length(nn);
if (parm>=0)&(~ok);
ii=1;while ii<=nnn;disp(rettexte(a(nn(ii:min(ii+7,nnn))).name));ii=ii+8;end;
k=input(['voulez vous effacer ces fichiers',type,'  ?  1-->oui  0-->non  ']);
if k==1;for ii=1:nnn;delete(a(nn(ii)).name);end;end;
else
for ii=1:nnn;delete(a(nn(ii)).name);end;
end;    

