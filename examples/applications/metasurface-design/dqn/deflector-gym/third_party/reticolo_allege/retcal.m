function cal=retcal(var,fich,cal,varargin);
% function cal=retcal(var,fich,cal,varargin);
% gestion des variables, empilement ,tri, stockage ,reprise du calcul, ...
% var={'var1','a',..} cell_array de chaînes de caractères :nom des variables à stocker.
% fich: nom du fichier où on stocke les variables
% 
% --> initialisation: retcal(var,fich,cal,donnee_1,donnee_2,...); si cal=1 on ecrase le fichier 
% sinon on reprend le calcul (dans ce cas si le fichier n'existe pas il est cree)
% (il est conseille de n'utiliser cal=1 que pour écraser le fichier et reprendre tout le calcul)
% Dans cette étape d'initialisation on crée dans l'espace de travail des variables de nom: vvar1,aa 
% en doublant le premier caractère   (attention aux incompatibilités si par exemple on a comme variables a et aa)
% ces variables sont initialisées par [] puis stockées sur le fichier (en version v6).
% divers paramètres peuvent être stockes à cette étape (donnee_1,donnee_2,...) et seront conserves dans la suite
% on stocke l'heure (variable ret_heure)
% 
%-->  test pour savoir si le calcul a déjà été fait: retcal({'a','b',..})
% crée dans l'espace de travail une variable de nom ret_cal 
% qui vaut 1 si le calcul n'a pas encore été fait pour les valeurs des parametres a,b,c
% (ce qui permet très facilement de compléter le calcul )
% autres formes possible si une seule variable varie: retcal('a') ou  retcal(a)
% ( si cal==0 ret_cal=0 ce qui permet de ne faire que le trace)
% 
%-->  stockage des variables apres le calcul 
% retcal(var,fich);
% les variables du fichier (vvar1, aa ..),sont complétées: vvar1=[vvar1,var1(:)],aa=[aa,a(:)]
% puis triées par rapport à vvar1(1,:),avant d'être stockées sur le fichier
%   (si certaines variables ont plusieurs dimensions elles sont mises en colonnes)
%   si les dimensions changent on complete par des nan 
%   pour les structures,les champs doivent rester les mëmes tout au cours du calcul
%   si au cours d'un nouveau calcul on desir stocker de nouvelles variables,
% les anciennes valeurs non stockees de ces variables sont remplacees par nan
% (attention:ces nouvelles variables ne peuvent pas etre des cell_array ni des structures à champ)
% on stocke automatiquement le temps de calcul ret_cpu (variable rret_cpu)
% Les fichiers temporaires crees par retio dans la boucle sont effacés en preservant ceux crees avant 
%
%--> concatenation de calculs
% retcal(var_dir,fich1,fich2,fich3 ... );
% fich1 et fich2,.. crees avec le meme retcal par exemple sur des machines differentes
% var_dir variable directrice (chaine de caracteres sans doubler la premiere lettre)
% le premier fichier est completé et trié  TRAVAILLER AVEC DES COPIES
%
%--> élimination de points
% retcal(var_dir,numeros_a_eliminer,fich);
% var_dir variable directrice (chaine de caracteres sans doubler la premiere lettre)
% le fichier est modifié TRAVAILLER AVEC UNE COPIE
%
%% % EXEMPLE GENERAL
% figure
% donnee_1=5;donnee_2=20;
% var={'a','b','c','d'};fich='fichier';
% cal=1;retcal(var,fich,cal,donnee_1,donnee_2);% initialisation
% for a=2:2:6;for b=2:2:4
% retcal({'a','b'});
% 	if ret_cal; % calcul deja fait ?
% 	c={0};d=rand(2,2);rand(100);
% 	retcal(var,fich);
% 	end;
%   if ret_trace;plot(aa,dd(1,:));drawnow;end;% trace
% end;end;
% clear aa bb cc dd;
%% on peut ici arrêter Matlab pour le redémarrer plus tard
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% var={'a','b','c','d','e'};% on ajoute la variable e
% cal=2;retcal(var,fich,cal); % on veut maintenant completer le calcul
% for a=1:6;for b=1:4;  
% retcal({'a','b'});
% 	if ret_cal; % calcul deja fait ?
% 	c={1};e=[0,2];d=rand(1,2);rand(200);% calcul
% 	retcal(var,fich);
% 	end;
%   if ret_trace;plot(aa,dd(1,:));drawnow;end;% trace
% end;end;
% clear aa bb cc dd;
% load(fich)
% rettexte(aa,bb,cc,dd,ee,rret_cpu,donnee_1,donnee_2);
%% on peut ici arrêter Matlab pour le redémarrer plus tard et tracer les courbes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure
% var={'a','b','c','d'};fich='fichier';
% cal=0;retcal(var,fich,cal,donnee_1,donnee_2);% initialisation
% for a=2:2:8;for b=2:2:4
% retcal({'a','b'});
% 	if ret_cal; % calcul deja fait ?
% 	c={0};d=rand(2,2);rand(100);
% 	retcal(var,fich);
% 	end;
%   if ret_trace;plot(aa,dd(1,:));drawnow;end;% trace
% end;end;
%
% delete ([fich,'.mat']); % pour effacer le fichier crée pour l'exemple
%
%
%% % EXEMPLE SIMPLE
% donnee_1=5;donnee_2=20;
% var={'ld','n'};fich='fichier';
% cal=1;retcal(var,fich,cal,donnee_1,donnee_2);% initialisation
% for ld=linspace(.9,1,10);retcal(ld);if ret_cal; % calcul deja fait ?
% n=retindice(ld,1);
% retcal(var,fich);
% end;end
% lld,nn
%% Si on veut ensuite ajouter des valeurs, on peut repasser le programme avec cal=2.
%% Les points deja calcules ne sont par recalculés.Les resultats sont triés par lld croissants
%
% delete ([fich,'.mat']);% pour effacer le fichier crée pour l'exemple

% ANCIENNE FORME calcul deja fait ?
% function cal=retcal(x,xx,tol);cal=1 si le scalaire x n'appartient pas au tableau xx (a tol pres,relatif)
% permet de tester si le calcul a déjà été fait if retcal(x,xx,tol); calcul..end;
% x peut etre un cell_array  pour tester plusieurs variables (xx doit alors etre un cell_array de meme dimension)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (nargin>=3)& ischar(cal);
if ~ischar(fich);elimination(var,fich,cal,varargin{:});return; % elimination de points
else;concatene(var,fich,cal,varargin{:});return;end;           % concatenation de fichiers
end;

if (nargout>0)|(nargin==1);% calcul deja fait ?
switch nargin;   
case 3;cal=calcul(var,fich,cal);
case 2;cal=calcul(var,fich);
case 1;if ~iscell(var);if ~ischar(var);var=inputname(1);end;var={var};end;
prv='{'; pprv='{';
for ii=1:length(var);pprv=[pprv,var{ii}(1),var{ii},',']; prv=[prv,var{ii},','];end;
prv=[prv(1:end-1),'}'];pprv=[pprv(1:end-1),'}'];
evalin('caller',['ret_cal=(cal~=0)&retcal(',prv,',',pprv,');']);
evalin('caller',['ret_trace=(cal==0)|ret_cal;']);
evalin('caller','ret_preserve=retio;');
%evalin('caller','disp(''CE PROGRAMME PEUT ETRE ARRETE A TOUT MOMENT: TOUTES LES VARIABLES SONT ENREGISTREES'');');
end;    
return;end; % calcul deja fait ?   
% function cal=store(var,fich,cal,varargin);

var=[var,{'ret_cpu'}];% on ajoute la variable ret_cpu
if nargin>2;% initialisation  si cal=1 on ecrase le fichier sinon on tente de le lire  
evalin('caller','ret_cal=0;');
if (cal~=1)&(exist([fich,'.mat'])~=2);cal=1;end;
if cal==1;
for ii=1:length(var);evalin('caller',[var{ii}(1),var{ii},'=[];']);end;% initiation a []
ret_prv=['save ',fich,' ret_heure '];% on stocke l'heure et les variables de varargin
for ii=4:nargin;ret_prv=[ret_prv,' ',inputname(ii)];end;
evalin('caller',['ret_cal=1;ret_heure=retheure;',ret_prv]);
end ;% cal=1
evalin('caller',['load ',fich,';ret_cpu=cputime;']);

else; % empilement,tri par rapport à la premiere variable, et ecriture sur fichier
evalin('caller','retio(ret_preserve,-4);');% on efface les fichiers temporaires crees
evalin('caller','ret_cpu=cputime-ret_cpu;');% temps de calcul
for ii=1:length(var);prv=[var{ii}(1),var{ii}];
evalin('caller',['if exist(''',prv,''',''var'')~=1;',prv,'=nan*ones(length(',var{ii},'(:)),size(',var{1}(1),var{1},',2)-1);end;']);


%evalin('caller',['ret_prv=size(',prv,');',prv,'=reshape(',prv,',prod(ret_prv(1:end-1)),ret_prv(end));']);%nv
%evalin('caller',['ret_prv=size(',var{ii},');',var{ii},'=reshape(',var{ii},',prod(ret_prv(1:end-1)),ret_prv(end));']);%nv


evalin('caller',['ret_prv=[size(',var{ii},'(:));size(',var{ii}(1),var{ii},')];']);    
evalin('caller',['if ret_prv(1,1)>ret_prv(2,1);',var{ii}(1),var{ii},'=[',var{ii}(1),var{ii},';nan*ones(ret_prv(1,1)-ret_prv(2,1),ret_prv(2,2))];end;']);    
evalin('caller',['if ret_prv(2,1)>ret_prv(1,1);',var{ii},'=[',var{ii},'(:);nan*ones(ret_prv(2,1)-ret_prv(1,1),ret_prv(1,2))];end;']);    
evalin('caller',[var{ii}(1),var{ii},'=[',var{ii}(1),var{ii},',',var{ii},'(:)];']);





end;% ii
evalin('caller',['[ret_prv,ret_ii]=sort(',var{1}(1),var{1},'(1,:));']);
for ii=1:length(var);ret_prv=[var{ii}(1),var{ii}];
evalin('caller',[ret_prv,'=',ret_prv,'(:,ret_ii);']);
%evalin('caller',[ret_prv,'=squeeze(reshape(',ret_prv,',[size(',var{ii},'),size(',ret_prv,',2)]));']);%nv

end;
end;% initialisation ?

%  ecriture sur fichier
if retversion>6;v6=''',''-v6';else v6=[];end;

ret_prv=['save(''',fich];
for ii=1:length(var);ret_prv=[ret_prv,''',''',var{ii}(1),var{ii}];end;
evalin('caller',[ret_prv,v6,''',''-append'');ret_cpu=cputime;'],'');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cal=calcul(x,xx,tol)
% function cal=retcal(x,xx,tol);1 si le scalaire x n'appartient pas au tableau xx(a tol pres)
%  permet de tester si le calcul a deja ete fait if retcal(x,xx,tol); calcul..end;
% x peut etre un cell_array  pour tester plusieurs variables (xx doit alors etre un cell_array de meme dimension)

if ~iscell(x);x={x};xx={xx};end;
if length(x{1})>1;[x,xx]=deal(xx,x);end;
if isempty(xx{1});cal=1;return;end;
if nargin<3;tol=10*eps;end;
f=[1:length(xx{1})].';
for ii=1:length(x);f=f(abs(x{ii}-xx{ii}(f))<=.5*tol*(abs(x{ii})+abs(xx{ii}(f))));end;
cal=isempty(f);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function concatene(vardir_z656u214t6h5d,fich1_z656u214t6h5d,fich2_z656u214t6h5d,varargin);
% fich1 et fich2 crees avec le meme retcal par exemple sur des machines differentes
% var_dir variable directrice (chaine de caracteres sans doubler la premiere lettre)
if length(varargin)>=1;concatene(vardir_z656u214t6h5d,fich1_z656u214t6h5d,fich2_z656u214t6h5d);concatene(vardir_z656u214t6h5d,fich1_z656u214t6h5d,varargin{:});end;

load(fich1_z656u214t6h5d);
az656u214t6h5d=whos;
z_z656u214t6h5d=zeros(size(az656u214t6h5d)); % recherche des variables a concatener, mise en tête de la variable directrice 
for k_z656u214t6h5d=1:length(az656u214t6h5d)%
z_z656u214t6h5d(k_z656u214t6h5d)=strcmp(az656u214t6h5d(k_z656u214t6h5d).name,'fich1_z656u214t6h5d')|strcmp(az656u214t6h5d(k_z656u214t6h5d).name,'fich2_z656u214t6h5d')|strcmp(az656u214t6h5d(k_z656u214t6h5d).name,'vardir_z656u214t6h5d')|~ismember(2,strfind(az656u214t6h5d(k_z656u214t6h5d).name,az656u214t6h5d(k_z656u214t6h5d).name(1)));
if strcmp(az656u214t6h5d(k_z656u214t6h5d).name,vardir_z656u214t6h5d([1,1:end]));K_z656u214t6h5d=k_z656u214t6h5d;end;
end;
f_z656u214t6h5d=find(~z_z656u214t6h5d);K_z656u214t6h5d=find(f_z656u214t6h5d==K_z656u214t6h5d);f_z656u214t6h5d=[f_z656u214t6h5d(K_z656u214t6h5d);f_z656u214t6h5d(1:K_z656u214t6h5d-1);f_z656u214t6h5d(K_z656u214t6h5d+1:end)];
az656u214t6h5d=az656u214t6h5d(f_z656u214t6h5d);
for k_z656u214t6h5d=1:length(az656u214t6h5d);eval(['z656u214t6h5d',az656u214t6h5d(k_z656u214t6h5d).name,'=',az656u214t6h5d(k_z656u214t6h5d).name,';']);end;% on stocke les variables lues
load(fich2_z656u214t6h5d);

% concatenation et tri
eval([az656u214t6h5d(1).name,'=[z656u214t6h5d',az656u214t6h5d(1).name,',',az656u214t6h5d(1).name,'];[prv_z656u214t6h5d,L_z656u214t6h5d]=retelimine(',az656u214t6h5d(1).name,',0);',az656u214t6h5d(1).name,'=',az656u214t6h5d(1).name,'(:,L_z656u214t6h5d);'])
for k_z656u214t6h5d=2:length(az656u214t6h5d);
eval([az656u214t6h5d(k_z656u214t6h5d).name,'=[z656u214t6h5d',az656u214t6h5d(k_z656u214t6h5d).name,',',az656u214t6h5d(k_z656u214t6h5d).name,'];',az656u214t6h5d(k_z656u214t6h5d).name,'=',az656u214t6h5d(k_z656u214t6h5d).name,'(:,L_z656u214t6h5d);'])
end
nom_z656u214t6h5d='';
for k_z656u214t6h5d=1:length(az656u214t6h5d);
nom_z656u214t6h5d=[nom_z656u214t6h5d,'''',az656u214t6h5d(k_z656u214t6h5d).name,''','];	
end;
[prv_z656u214t6h5d,vers_z656u214t6h5d,v6_z656u214t6h5d]=retversion;
nom_z656u214t6h5d=[nom_z656u214t6h5d,'''-append'',''',v6_z656u214t6h5d,''');'];
eval(['save(fich1_z656u214t6h5d,',nom_z656u214t6h5d]);	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function elimination(vardir_z656u214t6h5d,num_z656u214t6h5d,fich1_z656u214t6h5d,varargin);
% fich1 et fich2 crees avec le même retcal par exemple sur des machines differentes
% var_dir variable directrice (chaine de caracteres sans doubler la premiere lettre)


load(fich1_z656u214t6h5d);
az656u214t6h5d=whos;
z_z656u214t6h5d=zeros(size(az656u214t6h5d)); % recherche des variables a trier, mise en tête de la variable directrice 
for k_z656u214t6h5d=1:length(az656u214t6h5d)%
z_z656u214t6h5d(k_z656u214t6h5d)=strcmp(az656u214t6h5d(k_z656u214t6h5d).name,'fich1_z656u214t6h5d')|strcmp(az656u214t6h5d(k_z656u214t6h5d).name,'fich2_z656u214t6h5d')|strcmp(az656u214t6h5d(k_z656u214t6h5d).name,'vardir_z656u214t6h5d')|~ismember(2,strfind(az656u214t6h5d(k_z656u214t6h5d).name,az656u214t6h5d(k_z656u214t6h5d).name(1)));
if strcmp(az656u214t6h5d(k_z656u214t6h5d).name,vardir_z656u214t6h5d([1,1:end]));K_z656u214t6h5d=k_z656u214t6h5d;end;
end;
f_z656u214t6h5d=find(~z_z656u214t6h5d);K_z656u214t6h5d=find(f_z656u214t6h5d==K_z656u214t6h5d);f_z656u214t6h5d=[f_z656u214t6h5d(K_z656u214t6h5d);f_z656u214t6h5d(1:K_z656u214t6h5d-1);f_z656u214t6h5d(K_z656u214t6h5d+1:end)];
az656u214t6h5d=az656u214t6h5d(f_z656u214t6h5d);
for k_z656u214t6h5d=1:length(az656u214t6h5d);eval(['z656u214t6h5d',az656u214t6h5d(k_z656u214t6h5d).name,'=',az656u214t6h5d(k_z656u214t6h5d).name,';']);end;% on stocke les variables lues

% elimination
eval(['[L0_z656u214t6h5d,L_z656u214t6h5d]=retfind(ismember([1:length(',az656u214t6h5d(1).name,')],num_z656u214t6h5d));']);
if (input(['elimination de  ',int2str(L0_z656u214t6h5d),'? (repondre 1 si OK)  '])~=1);return;end;

for k_z656u214t6h5d=1:length(az656u214t6h5d);
eval([az656u214t6h5d(k_z656u214t6h5d).name,'=',az656u214t6h5d(k_z656u214t6h5d).name,'(:,L_z656u214t6h5d);'])
end
nom_z656u214t6h5d='';
for k_z656u214t6h5d=1:length(az656u214t6h5d);
nom_z656u214t6h5d=[nom_z656u214t6h5d,'''',az656u214t6h5d(k_z656u214t6h5d).name,''','];	
end;
[prv_z656u214t6h5d,vers_z656u214t6h5d,v6_z656u214t6h5d]=retversion;
nom_z656u214t6h5d=[nom_z656u214t6h5d,'''-append'',''',v6_z656u214t6h5d,''');'];
eval(['save(fich1_z656u214t6h5d,',nom_z656u214t6h5d]);	
