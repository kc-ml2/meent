function [a,vm,dimgs]=retio(a,k,varargin);
%  function [a,vm,dimgs]=retio(a,k);
%
%  gestion des fichiers temporaires ou permanents
%  si a est un numero(ou un nom)  de fichier:
%     k=0 ou [] ou absent lecture du contenu du fichier dans a
%     k=-2 lecture du contenu du fichier dans a et effacage du fichier mais pas des 'satellites'
%     k=-1  effacage du fichier et des 'satellites' retourne []
%  si a est une variable  (pas texte):
%     k='fich' creation d'un fichier 'fich123.mat' et ecriture de a quelque soit sa taille    a='fich123'
%          (dans ce cas 'fich' ne doit pas etre de la forme '12')
%     k=1 et fich n'existe pas
%          creation d'un fichier,ecriture de a, au retour a contient le numero du fichier
%          (si a=[]  on ne cree pas de fichier et on retourne []))
%          seuls les objets de taille>vmax sont stockés (taille=nombre de complexes)
%          par defaut vmax=5000
%             pour changer vmax: faire retio(a,k+i*vmax)  changement valable jusqu'a nouvel ordre
%             pour reprendre la valeur par defaut de vmax: faire retio(a,k-i) 
%     k=0 ou k=[] retour sans modifier a
%
% en fait:pour permettre l'utilisation en parallele de plusieurs processeurs travaillant 
% dans le meme repertoire,les noms de fichiers sont precedes d'un prefixe
% aleatoire de 4 caracteres en base 36 (un risque de conflit sur 36^4)
% 
%  formes speciales:
%..............................................................................................
%  retio:efface tous les fichiers crees dont le numero est contenu dans '0.mat' 
%  t=retio retourne les numeros des fichiers crees
%  retio(a,3):retourne les numeros des fichiers associes au cell array a
%  retio(t,-4):efface tous les fichiers sauf ceux contenus dans le tableau de numeros t et les satellites
%      on peut ainsi effacer les fichiers crees depuis un point du programme
%      en preservant  certains fichiers
%
%          |-------------|
%          |  EXEMPLES   |
%          |-------------|
%
%-----------------------------------------------------------------------------------|
%            |  a='15'               |     a= {q,t,..'12'}                          |
%            |                       |     ou a=3.14                                |
%------------------------------------------------------------------------------------
%  k=0 ou [] |  a= lecture           |    a inchange                                |
%  ou abs    |  de '15.mat'          |                                              |
%------------------------------------------------------------------------------------
%            | a=retio(a)            | lecture des satellites dans a                |
%            | puis a=retio(a,'fich')| a={q,t,..retio('12')}                        |
%  k='fich'  |                       | ecriture de a dans 'fich123.mat', a='fich123'|                                |
%            |                       |  si 3 entrees ecriture danc 'fich.mat'       |
%------------------------------------------------------------------------------------
%    k=1     | a='15'                | ecriture de a                                |
%    +i vmax |                       | dans '16.mat' si taille(a)>vmax              |
%            |                       | a='16'                                       |
%------------------------------------------------------------------------------------
%    k=-1    | efface '15.mat'       | efface les satellites de a  ('12.mat' etc.)  |
%            | et les satellites     |  a=[]                                        |
%            | a=[]                  |                                              |
%------------------------------------------------------------------------------------
%    k=-2    | a=lecture de '15.mat'|                                               |
%            | efface '15.mat' mais |  a inchange                                   |
%            | pas ses satellites   |                                               |
%------------------------------------------------------------------------------------
%    k=3     | a=tableau des numeros| a=tableau des numeros des fichiers satellites |
%            | des fichiers utilises|  du cell array a                              |
%            |  par   '15.mat'      |                                               |
%            | et ses satellites    |                                               |
%------------------------------------------------------------------------------------
%    k=4     |a=size du cell array  | a=size(a)                                     |
%            | ou du tableau stocké |                                               |
%[a,vm,dimgs]|  dans '15,mat'       |                                               |
% =retio(a,4)|                      |                                               |
%            |vm=taille du fichier  |vm=taille du cell array a                      |
%            | stocke dans '15.mat' |           (nombre de complexes)               |
%            | sans les satellites  |                                               |
%            |                      |                                               |
%            |dimgs: volume et      |dimgs:volume et repetition                     |
%            |   repetion  des      | des  matrices S ou G                          |
%            |   matrices S  ou G   |  tableau de dim 2                             |
%------------------------------------------------------------------------------------
%    k=-4   retio(t,-4):efface tous les fichiers sauf ceux contenus dans le tableau |
%             de numeros t                                                          |
%------------------------------------------------------------------------------------
%  retio:  efface tous les fichiers provisoires                                     | 
%------------------------------------------------------------------------------------
%  t=retio retourne les numeros des fichiers crees                                  |
%------------------------------------------------------------------------------------
% [prv,vmax]=retio([],100i) met vmax a 100                                          |
% [prv,vmax]=retio([],-100i) remet vmax a la valeur par defaut                      |
% [prv,vmax]=retio([],inf*i); empeche toute ecriture                      |
%------------------------------------------------------------------------------------
%  k=2                 retour sans action                                           |
%------------------------------------------------------------------------------------
%
% Exemples les plus courants:
%
% a=retio(a,1)  ecriture de a (a est remplace par le nom de son fichier)  
% a=retio(a)  lecture de a sans effacer le fichier
% a=retio(a,-2)  lecture de a en effaçant le fichier
% retio(a,-1)  effaçage du fichier (ou clear a)
% retio effaçage de tous les fichiers 
%
% OPTIONS DES FICHIERS: toujours v6 ( sinon risque du 'Bing_Bugg' )
%     Caught std::excoption Exception message is:
%     bad allocation
% pour forcer la version v7,faire avant: [v,vv,v6]=retversion('v7'),pour forcer la version v7.3, faire avant: [v,vv,v6]=retversion('v7.3')
%  remarque: v7.3 permet d'enregistrer de trés gros fichiers (>2GB) mais si on repete l'ecriture risque de bugg
%           v7 est tres long car compresse les fichiers
% 
% %%%%%%%%%%%%%  test_retio.m  %%%%%%%%%%%%%%%%
% %   ENGLISH VERSION BY HAITAO
% clear;
% 
% % % 1. vc=retio(v,1) saves variable v into a disk file, and returns a random character vc different from the file name if v is of large size,
% % %      and returns v without saving it into disk if v is of small size.
% % % 2.   The returned vc can be treated as an input argument representing variable v for some variables of reticolo functions.
% % % 3.   Then v=retio(vc,-2) returns the variable v from the disk file and delete this file.
% % % 4.   Or then v=retio(vc) returns the variable v from the disk file without deleting this file.
% % a1=retio(rand(10),1);b1=retio(a1,-2);size(sqrt(b1)) %the returned a1=rand(10)
% % a1=retio(rand(1000),1);b1=retio(a1);size(sqrt(b1))
% 
% % % 1. vc=retio(v,'c') saves variable v into a disk file, and returns the full file name beginning with 'c'.
% % % 2.   The returned vc can be treated as an input argument representing variable v for some variables of reticolo functions.
% % % 3.   Then v=retio(vc,-2) returns the variable from the disk file named by vc and delete this file.
% % % 4.   Or then v=retio(vc) returns variable v from disk file named by vc without deleting this file.
% % %      This point can be used to save v as a data file named by vc, and load v from the file after Matlab is restarted.
% fich='testio';a2=retio(rand(10),fich);b2=retio(a2,-2);size(sqrt(b2))
% % fich='testio';a2=retio(rand(1000),fich);b2=retio(a2,-2);size(sqrt(b2))
% % fich='testio';a2=retio(rand(1000),fich);b2=retio(a2);size(sqrt(b2))
% save test_retio
% See also:RETEFFACE,RETSAVE,MFILENAME,RETVERSION

if (nargin==1)&(~ischar(a))&(nargout==1);return;end;% a tester 
persistent vmax z v6;vm=vmax;if isempty(v6);[prv,vers,v6]=retversion;end;
if (nargin==1)&(~ischar(a));return;end;
cdir=fullfile(cd,' ');cdir=cdir(1:end-1);

if isempty(z)&(nargin>0);rand('state',sum(100*clock));while length(z)~=4;z=dec2base(round(rand*1679615),36);end;end;cdir=[cdir,'ret',z];


% cdir,z
if (nargin==0)&(nargout==1); % on retourne les numeros de fichiers sans les effacer
try;tt=[];load([cdir,'0.mat'],'tt');catch;tt=[];end;a=tt;return;end;

if nargin==0;tt=retio;efface(cdir,tt,v6,tt);retcouche;return;end; % on efface tous les fichiers et la bibliotheque de retcouche

if nargin<2;k=[];end;if isempty(k);k=0;end;% par defaut lecture
if ischar(k);fich=k;k=1;cdir=cdir(1:end-7);else fich=[];end;% fichiers permanents pas de prefixe
if k==2;return;end;
if ischar(a);if isempty(str2num(a));cdir=cdir(1:end-7);end;end;% fichiers permanents pas de prefixe


vmax0=5000;if isempty(vmax);vmax=vmax0;end; %valeur par defaut de vmax
if imag(k)>0;vmax=imag(k);end;if imag(k)<0;vmax=vmax0;end;k=real(k);vm=vmax; % modification de vmax

if k==3;% retourne les numeros des fichiers satellites du cell array a ( a peut etre un numero de fichier)
if ischar(a);try;c=[];load([cdir,a,'.mat'],'c');a=c;catch;a=[];end;else;a=retfich(cdir,a);end;    
return;end;

if k==-4; % on efface TOUS les fichiers SAUF ceux associes aux numeros a(tableau de numeros) et leurs satellites
tt=retio;if isempty(tt);return;end;
ttt=tt;
for ii=a;f=num2str(ii);   % elimination dans ttt des numeros 'proteges'
try;c=[];load([cdir,f,'.mat'],'c');for iii=c;ttt=ttt(find(iii~=ttt));if isempty(ttt);return;end;end;end;
end;% boucle sur ii
efface(cdir,ttt,v6,tt);
return;end; 


               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ischar(a);  %     a est un numero (ou un nom) de fichier     %
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ischar(fich);a=retio(retio(a),fich,varargin{:});return;end; % a est lu puis stocké en un fichier permanent

if k==0;  % lecture
try;load([cdir,a,'.mat'],'a');catch;return;end;
end;


if k==-1;% efface a et ses satellites
if isempty(str2num(a));try;delete([cdir,a,'.mat']);end;a=[];return;end;% fichiers permanents      
c=[];try;load([cdir,a,'.mat'],'c');catch;return;end;   
efface(cdir,c,v6);a=[];% efface les fichiers dont le numero est contenus dans c
return;end;

if k==-2;%lit a et efface le fichier mais pas ses satellites
b=str2num(a);aa=a;a=retio(a);if isempty(b);try;delete([cdir,aa,'.mat']);end;else;efface(cdir,b,v6);end;
return;end;

if k==4;% a=size de l'objet stocké ,vm= taille du fichier de a sans ses satellites, dimgs,dimensions et repetion pour les matrices S
try;v=[];s=[];load([cdir,a,'.mat'],'v','s','dimgs');a=s;vm=v;catch;a=0;return;end;  
return;end;
 

               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else           %   a n'est pas un numero(ou un nom) de fichier  %  
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (k==-2)|(k==0);return;end; % retour sans modifier a
if k==-1;efface(cdir,retfich(cdir,a),v6);a=[];return;end;
if k==1; % ecriture creation d'un nouveau nom
v=taille(a);s=size(a);dimgs=[];if (iscell(a)&s(2)==1);if s(1)==4;dimgs=[numel(a{1}),a{4}];end;if s(1)==6;dimgs=[numel(a{1})+numel(a{2}),a{6}];end;end;
% dimgs=volume et repetition des matrices S ou G 
if ischar(fich); % fichier permanent
a=lit(cdir,a);c=[]; % lecture des satellites
if nargin<3;
ii=round(1.e7*rand)+1;while exist([cdir,fich,int2str(ii),'.mat'],'file')~=0;ii=ii+1;end;
b=[fich,int2str(ii)];
else b=fich;end;
save([cdir,b,'.mat'],'c','v','s','dimgs','a',v6);a=b;return;
end;  % fin fichier permanent

if v<vmax;return;end;
tt=retio;if isempty(tt);tt=1;else tt=[tt,max(tt)+1];end;    
c=retelimine([tt(end),retfich(cdir,a)]);% numeros de fichiers dans a
try;    % bug ?
b=int2str(tt(end));save([cdir,b,'.mat'],'c','v','s','dimgs','a',v6);
catch
    
tt=[tt,max(tt)+1];  
c=retelimine([tt(end),retfich(cdir,a)]);% numeros de fichiers dans a
b=int2str(tt(end));
%heure=retheure;sza=size(a);save retprv c v s dimgs sza v6 cdir b heure
c,v,s,dimgs,size(a),v6,cdir,b,
lasterr
pause(.1);
save([cdir,b,'.mat'],'c','v','s','dimgs','a',v6);
end


try;    % bug ?
save([cdir,'0.mat'],'tt',v6);a=b;
catch;
lasterr
pause(.1);
save([cdir,'0.mat'],'tt',v6);a=b;
end;    
    
    
return;end;

if k==4;% taille de a sans ses satellites,size(a),volume et repetition des matrices S ou G 
if nargout>=2;vm=taille(a);dimgs=[];s=size(a);if (iscell(a)&s(2)==1);if s(1)==4;dimgs=[numel(a{1}),a{4}];end;if s(1)==6;dimgs=[prod(size(a{1}))+prod(size(a{2})),a{6}];end;end;end;
a=size(a);return;end;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function t=retfich(cdir,a)
% t:numeros des fichiers dans le cell array a
t=[];
if iscell(a);for ii=1:numel(a);t=[t,retfich(cdir,a{ii})];end;
else
if ischar(a);
c=[];try;load([cdir,a,'.mat'],'c');end;   
t=[t,str2num(a),c];
end;
end;  
t=retelimine(t);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function efface(cdir,c,v6,t) % efface tous les fichiers de numeros c(tableau) et met a jour '0.mat'
% t contient la liste de tous les fichiers(s'il n'est pas donne on le determine en lisant 0.mat)
if isempty(c);return;end;
if nargin<4;try;tt=[];load([cdir,'0.mat'],'tt');catch;return;end;else;tt=t;end;
for ii=c;bb=[num2str(ii),'.mat'];
try;delete([cdir,bb]);end;
if ~isempty(tt);tt=tt(ii~=tt);end;
end;% boucle sur ii
if isempty(tt);delete([cdir,'0.mat']);else save([cdir,'0.mat'],'tt',v6);end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v=taille(a)  % taille du cell array a ,ou de la structure = nombre de nb complexes 
v=whos('a');v=floor(v.bytes/16);
% v=0;
% if iscell(a);v=0;for ii=1:prod(size(a));v=v+taille(a{ii});end;
% else;if ~ischar(a);if issparse(a);v=v+nnz(a);else,v=v+prod(size(a));end;end;end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=lit(cdir,a)  % lecture des satellites de a
if isempty(retfich(cdir,a)); return;end;
if iscell(a);for ii=1:numel(a);a{ii}=lit(cdir,a{ii});end;end;
a=retio(a);a=lit(cdir,a);
