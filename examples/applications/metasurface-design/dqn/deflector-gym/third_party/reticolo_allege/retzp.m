function [Z,P]=retzp(f,parm,domaine,nz,np,varargin);
% function [Z,P]=retzp(f,parm,domaine,nz,np,variables_de_f);
%
% recherche des zeros et des poles d'une fonction construite a partir de f :
%
% parm: parm de retcadilhac
%   par defaut:parm=struct('fonc','','nout',1,'nin',1,'niter',30,'nitermin',0,'tol',eps,'tolf',inf,...
%    'bornes',[-inf,inf,-inf,inf],'parm',0);
%    nout:numero de la variable en entree 
%    nin:numero de la variable en sortie  
%    fonc: on cherche les zeros ou les poles de [... ,zz,..]= f(..,z,...) 
%                                                     %            %
%                                                   nout          nin
%    niter: nombre max d'iterations
%    nitermin: nombre min d'iterations
%    tol:  critere d'arret sur la variable
%    tolf:  critere d'arret sur la fonction
%    bornes=[min(real),max(real),min(imag),max(imag)]:on arrete quand la variable sort de bornes
%    parm: 0 fonction analytique  1 fonction non analytique
%
%  domaine:domaine de depart de la recherche domaine =[min(real),max(real),min(imag),max(imag)]
%  si domaine=[min(real),max(real)],point de depart reel
%  nz,np :nombre d'essai des zeros et des poles
%  variables_de_f:toutes les variables de f
% ( y compris celle pour laquelle on cherche les P et les Z qui peut avoir n'importe quelle valeur)
%
% [Z,P] zeros et poles obtenus tries par partie reele croissante
%  (les valeurs obtenues ne sont pas necessairement dans domaine mais a l'interieur de parm.bornes)
% PRINCIPE
%  recherche alternative des zeros ed des poles a partir d'un point de depart alleatoire dans 'domaine'.
%  A chaque etape la fonction est multipliee par la fraction rationelle 
%  formee par les zeros et poles deja obtenus ,de sorte que l'on ne trouve jamais 
%  2 fois le meme pole (ou zeros) sauf dans le cas de multiplictee
%
%% EXEMPLES
% [Z,P]=retzp(@tan,[],[-2,2],2,3),   % 2 zeros et 3 poles de tan
%
%%  exemple plus complique
% f=inline('polyval(poly(zz),z)./polyval(poly(pp),z)','zz','pp','z');
% zz=[1,1.001+.00005i];pp=[2+.00001i,.2+.00002i];
% [Z,P]=retzp(f,struct('fonc','z=1./z','nin',3,'tol',1.e-6,'tolf',1.e-9,'bornes',[-2,10,-1,1]),...
%     [-1,5],5,5,zz,pp,0)
%
% See also: RETCADILHAC 
 
 
 
 if nargin<6;warargin=[];end;
 if length(domaine)==2;domaine=[domaine(:);0;0];end;

pparm=struct('fonc','','nout',1,'nin',1);
   
fonc=retoptimget(parm,'fonc',pparm);fonc=[fonc,';'];
nout=retoptimget(parm,'nout',pparm);
nin=retoptimget(parm,'nin',pparm);
parm.fonc='';parm.nout=1;parm.nin=1;
pparm=parm;pparm.fonc='z=1/z';
varargin=[{f,fonc,nin,nout},varargin];

Z=[];P=[];
for ii=1:max(nz,np);
if length(Z)<nz;% recherche d'un Z   
[zz,itermax,erz,erfonc,test]=retcadilhac(@ff,parm,zrand(domaine),Z,P,varargin{:});
% disp(rettexte('Z',ii,zz,itermax,erz,erfonc,test));
if all(test);Z=[Z,zz];end;
end;
if length(P)<np;% recherche d'un pole   
[zz,itermax,erz,erfonc,test]=retcadilhac(@ff,pparm,zrand(domaine),Z,P,varargin{:});
% disp(rettexte('pole',ii,zz,itermax,erz,erfonc,test));
if all(test);P=[P,zz];end;
end;
[prv,ii]=sort(real(Z));Z=Z(ii);[prv,ii]=sort(real(P));P=P(ii);

% 
% x=linspace(domaine(1),domaine(2),100);y=ff(x,Z,P,varargin{:});
% plot(x,abs(y))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function z=trans(z,fonc);eval(fonc);


function zz=ff(z,Z,P,f,fonc,nin,nout,varargin);
varargout=cell(1,nout);
[varargout{:}]=feval(f,varargin{1:nin-1},z,varargin{nin+1:end});zz=trans(varargout{nout},fonc);
zz=zz.*(polyval(poly(P),z)./polyval(poly(Z),z));

function z=zrand(domaine);% point alleatoire dans domaine
z=domaine(1)+rand*(domaine(2)-domaine(1))+i*(domaine(4)-domaine(3));
