function xdisc=retdisc(init,w);
% function xdisc=retdisc(init,w);
%  points de discontinuite d'un maillage de texture ou de tronçon w
%  discontinuitees en xdisc en 1 D  xdisc est un vecteur
%  en 2 D xdisc={xdisc{1} xdisx{2}}est un cellarray contenant les discontinuites en x et en y 
%  init  peut etre remplace par d le pas du reseau
% 
% EXEMPLE d'utilisation en 2D avec ret gauss
% disc=retdisc(init,w); % points de discontinuite
% [x,px]=retgauss(-d(1)/2,d(1)/2,10,2,disc{1},d(1));[y,py]=retgauss(-d(2)/2,d(2)/2,20,1,disc{1},d(2));


% si on entre d a la place de init ...
if iscell(init);
d=init{end}.d; 
else;d=init;end;    
    

if length(d)==2; %  2 D
if isempty(w);xdisc=cell(1,2);return;end;
if iscell(w{1});% maillage de tronçon;
xdisc=cell(1,2);for ii=1:length(w);prv=retdisc(d,w{ii}{2});xdisc{1}=[xdisc{1},prv{1}];xdisc{2}=[xdisc{2},prv{2}];end;
xdisc{1}=retelimine(xdisc{1});xdisc{2}=retelimine(xdisc{2});
return;end;

xdisc={w{1}*d(1),w{2}*d(2)};
if  all(all(w{3}(end,:,:)==w{3}(1,:,:)));xdisc{1}=xdisc{1}(1:end-1);end;
if  all(all(w{3}(:,end,:)==w{3}(:,1,:)));xdisc{2}=xdisc{2}(1:end-1);end;


else;  % 1 D    
if isempty(w);xdisc=[];return;end;
if iscell(w{1});% maillage de tronçon;
xdisc=[];for ii=1:length(w);xdisc=[xdisc,retdisc(d,w{ii}{2})];end;
xdisc=retelimine(xdisc);
return;end;

xdisc=w{1};
if  all(w{2}(end,:)==w{2}(1,:));xdisc=xdisc(1:end-1);end;
end;    
    
    
    
    
    
    
    
    
    












