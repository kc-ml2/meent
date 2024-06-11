function st=rettr(init,s,t);
% function St=rettr(init,S,t);
%  transformation de la matrice S  d' un tronçon,
%  en u St apres une translation globale d'un vecteur t (scalaire en 1D)  
%  (matrices pour les champs de type S ou G )
% ATTENTION  à ce que la translation conserve bien les symetries
%
% remarque: si S a éte stoquée sur fichier par retio(S,1) , St ne l'est pas
%
% avec seulement 2 arguments: At=rettr(A,t) translation d'objets de type
%   init 
%   descripteur de texture droites ou inclinees
%
% See also: RETROTATION

if nargin==2;st=rettr_general(init,s);return;end; % autre que matrices S
t=t(:).';
s=retio(s);
st=s;
n=init{1};beta=init{2};
if init{end}.dim==2; %2D
si=init{8};    
v=repmat(exp(i*t*beta),1,2);vv=1./v;
if ~isempty(si);v=sparse([[si{2}*diag(v)*si{1},si{2}*diag(v)*si{3}];[si{4}*diag(v)*si{1},si{4}*diag(v)*si{3}]]);vv=sparse([[si{2}*diag(vv)*si{1},si{2}*diag(vv)*si{3}];[si{4}*diag(vv)*si{1},si{4}*diag(vv)*si{3}]]);else v=sparse(diag([v,v]));vv=sparse(diag([vv,vv]));end
else               %1D
si=init{3};    
v=exp(i*t*beta);vv=1./v;
if ~isempty(si);v=sparse(si{2}*diag(v)*si{1});v=[[v,spalloc(n,n,0)];[spalloc(n,n,0),v]];vv=sparse(si{2}*diag(vv)*si{1});vv=sparse([[vv,zeros(n,n)];[zeros(n,n),vv]]);else v=sparse(diag([v,v]));vv=sparse(diag([vv,vv]));end
end    
if size(s,1)==4;  %  matrice S 
st{1}=vv*st{1}*v;
else              %  matrice  G
st{1}=st{1}*v;
st{2}=st{2}*v;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=rettr_general(a,t);t=t(:).';if all(t==0);return;end;
a=retio(a);
switch a{end}.genre;

case 0;	% init on modifie cao  ------------------------------------------
if a{end}.dim==1;  % 1D
if ~isempty(a{end}.cao);
cao=a{end}.cao;cao(1)=cao(1)+t;a{end}.cao=cao;
a{5}=retcao(cao(1),cao(2),a{end}.d,a{end}.nfourier);
end;	
else;              % 2D
if ~isempty(a{end}.cao);
a{end}.cao(1)=mod(a{end}.cao(1)+t(1),a{end}.d(1));a{end}.cao(2)=mod(a{end}.cao(2)+t(2),a{end}.d(2));
cao=a{end}.parm.cao;cao(1:2)=cao(1:2)+t;a{end}.parm.cao=cao;
a{10}={retcao(cao(1),cao(3),a{end}.d(1),a{6}),retcao(cao(2),cao(4),a{end}.d(2),a{7})};
end;               % 1 D  2 D ?	
end;

case 2;	% descripteurs de texture  -------------------------------------------------------------
switch a{end}.type;
case 5;	% textures inclinees
if a{end}.dim==1;  % 1D
n=a{4};beta=a{9};prv1=retdiag(exp(i*t*beta));prv2=retdiag(exp(-i*t*beta));pprv2=retdiag(exp(-i*t*beta(1:n)));
a{1}=produit(prv2,a{1});a{2}=produit(a{2},prv1);
if ~isempty(a{6});a{6}=produit(pprv2,a{6},prv1);end;
if ~isempty(a{7});a{7}=produit(pprv2,a{7},prv1);end;
a{8}=retrotation(t+i,a{8},a{end}.d);
else;              % 2D
d=a{13};beta=a{14};beta_sym=a{9};
prv1=retdiag(exp(i*t*beta_sym));prv2=retdiag(exp(-i*t*beta_sym));
pprv1=retdiag(exp(i*t*beta));pprv2=retdiag(exp(-i*t*beta));
Prv1=retdiag(exp(i*t*[beta,beta,beta,beta]));Prv2=retdiag(exp(-i*t*[beta,beta,beta]));
a{1}=produit(prv2,a{1});a{2}=produit(a{2},prv1);
[a{12},prv,sym,cao,permute]=retrotation(t+i,a{12},d,[],[]);permute=permute{(3+a{end}.sens)/2};%sens=1  y  sens=-1  x

if ~isempty(a{6});for ii=1:numel(a{6});for jj=1:numel(a{6}{ii});a{6}{ii}{jj}=produit(pprv2,a{6}{ii}{jj},pprv1);end;end;a{6}=a{6}(permute);end;
if ~isempty(a{7});for ii=1:numel(a{7});for jj=1:numel(a{7}{ii});a{7}{ii}{jj}=produit(pprv2,a{7}{ii}{jj},pprv1);end;end;a{7}=a{7}(permute);end;
if ~isempty(a{10});a{10}=Prv2*a{10}*Prv1;end;
if ~isempty(a{11});a{11}=Prv2*a{11}*Prv1;end;
end;               % 1 D  2 D ?

case 1;	% textures dielectriques
if a{end}.dim==1;  % 1D seulement si pas de symetrie
% a={p,pp,q,qq,d,a3,a1,w,pas,beta,struct('dim',1,'genre',2,'type',1,'sog',sog)};
%   1  2  3 4  5 6  7  8  9   10
d=a{9};beta=a{10};prv1=retdiag(exp(i*t*beta));prv2=retdiag(exp(-i*t*beta));
a{1}=produit(prv2,a{1});a{2}=produit(a{2},prv1);a{3}=produit(prv2,a{3});a{4}=produit(a{4},prv1);
a{8}=retrotation(t+i,a{8},d);
if ~isempty(a{6});a{6}=produit(prv2,a{6},prv1);end;
if ~isempty(a{7});a{7}=produit(prv2,a{7},prv1);end;

else;              % 2D
%	a={p,pp,q,qq,d,ez,hz,fex,fhx,fey,fhy,w,pas,beta,beta_sym,struct('dim',2,'genre',2,'type',1,'sog',sog)};
%      1  2 3  4 5  6  7  8   9   10 11  12 13  14    15
d=a{13};beta=a{14};n=size(a{1},1);beta_sym_E=a{15}(:,1:n);beta_sym_H=a{15}(:,n+1:2*n);
prv1E=retdiag(exp(i*t*beta_sym_E));prv2E=retdiag(exp(-i*t*beta_sym_E));
prv1H=retdiag(exp(i*t*beta_sym_H));prv2H=retdiag(exp(-i*t*beta_sym_H));
pprv1=retdiag(exp(i*t*beta));pprv2=retdiag(exp(-i*t*beta));
[a{12},prv,sym,cao,permute]=retrotation(t+i,a{12},d,[],[]);

a{1}=produit(prv2E,a{1});a{2}=produit(a{2},prv1E);a{3}=produit(prv2H,a{3});a{4}=produit(a{4},prv1H);
if ~isempty(a{6});a{6}=produit(pprv2,a{6},prv1H);end;
if ~isempty(a{7});a{7}=produit(pprv2,a{7},prv1E);end;
for ii=1:numel(a{8});a{8}{ii}=produit(pprv2,a{8}{ii},pprv1);end;if ii>0;a{8}=a{8}(permute{2});end;
for ii=1:numel(a{9});a{9}{ii}=produit(pprv2,a{9}{ii},pprv1);end;if ii>0;a{9}=a{9}(permute{2});end;
for ii=1:numel(a{10});a{10}{ii}=produit(pprv2,a{10}{ii},pprv1);end;if ii>0;a{10}=a{10}(permute{1});end;
for ii=1:numel(a{11});a{11}{ii}=produit(pprv2,a{11}{ii},pprv1);end;if ii>0;a{11}=a{11}(permute{1});end;
end;               % 1 D  2 D ?



end;  % type
end;	% genre  -------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=produit(varargin);% produit avec lecture et ecriture sur fichier
n=length(varargin);
a=varargin{n};
io=ischar(a);if io;a=retio(a);end;
for ii=n-1:-1:1;
if ischar(varargin{ii});io=1;a=retio(varargin{ii})*a;else;a=varargin{ii}*a;end;
end;
if io;a=retio(a,1);end;


