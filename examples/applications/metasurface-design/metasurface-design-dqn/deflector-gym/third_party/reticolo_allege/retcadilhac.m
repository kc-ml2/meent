function varargout=retcadilhac(varargin);
% function [z0,z,zz,er,test]=retcadilhac(z,zz,parm);
%
% FORME GENERALE
%
% z,zz :vecteurs colonnes (ou ligne)
%
%  er:estimation de l'erreur sur z0
%  test: 1 si approximation homographique valable  2 si approximation lineaire  0 sinon
%
% recherche des zeros complexes d'une fonction complexe zz(z)=0
% si parm=0 ou est absent la fonction est analytique (approximation harmonique)
% si parm=1 la fonction n'est par analytique (2 fonctions reelles de 2 variables reelles)
%
% si length(z)=3:
% on a deja calcule zz(z(1)) zz(z(2)) zz(z(3)) 
% on calcule la nouvelle approximation z0 et on ordonne z par 'qualitee' decroissante (z(1) est le meilleur)
%
% si length(z)=1:   z0=1.0001*z (ou z0=.0001si z=0)
% si length(z)=2:   z0 est obtenu par interpolation lineaire  z est ordonne;
% si length(z)=4:   on remplace z(3) par z(4) et on calcule z0
% si length(z)>length(zz)    z0 est la valeur suivante de z (valeurs  de depart)
%
%  exemple d'utilisation:  dep1 dep2 dep3 valeurs de depart  
%  er=inf;iter=0;z0=dep1;     z=[];( ou z=dep2; ou z=[dep2;dep3];)             zz=[];
%  while(er>eps)&(iter<30);zz0=fonc(z0);iter=iter+1;z=[z;z0];zz=[zz;zz0];[z0,z,zz,er]=retcadilhac(z,zz);end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      FORME SIMPLIFIEE ( fonction)
% function [z0,itermax,erz,erfonc,test]=retcadilhac(@fun,parm,varargin);
%       on cherche le zero de  zz
%   [... ,zz,..]= fonc(fun(..,z,...))
%       z donne la ou les (au max 3) valeurs de depart
%
% parm:structure de parametres
% par defaut:
%   parm=struct('fonc','','nout',1,'nin',1,'niter',30,'nitermin',0,'tol',eps,'tolf',inf,...
%    'bornes',[-inf,inf,-inf,inf],'precis',0,'tol_precis',6,'parm',0);% par defaut
%    nout:numero de la variable en entree 
%    nin:numero de la variable en sortie  
%    fonc: on cherche le zero de [... ,zz,..]= fonc(fun(..,z,...) )
%                                       %                  %
%                                     nout                nin
%    fonc est une fonction ( ou encore de la forme 'z=z-2' chaine de caracteres  (archaique)..)
%    niter: nombre max d'iterations
%    nitermin: nombre min d'iterations
%    tol:  critere d'arret sur la variable
%    tolf:  critere d'arret sur la fonction
%    bornes=[min(real),max(real),min(imag),max(imag)]:on arrete quand la variable sort de bornes
%    parm: 0 fonction analytique  1 fonction non analytique
%    precis: poursuite du calcul en precision etendue( la fonction fun doit naturellement etre prevue pour accepter un argument de classe retprecis )
%            0  non, i precision standard  , entier nombre de chiffres decimaux 
%    tol_precis diminution de tol et tolf de 10^-tolprecis
%    test:tableau logique:
%  [er<=tol ,tolf>=abs(zz0), iter<niter, (real(z0)>bornes(1))&(real(z0)<bornes(2))&(imag(z0)>bornes(3))&(imag(z0)<bornes(4))  ];
%  EXEMPLES
%  % fichier zz.m:   function [zs,zc,zt]=zz(a,b,x,c);zs=sin(a*x)-b; zc=cos(a*x)-b; zt=tan(a*x)-b; 
%
%    a=1;b=.5;z=.1;c=1;x=0;
%%   [a0,niter,er,erfonc,test]=retcadilhac(@zz,[],a,b,x,c);%a0 tel que;sin(a0*x)==b
%   xc=retcadilhac(@zz,struct('nout',2,'nin',3),a,b,x,c);% xc tel que cos(a*xc)==b
%   [xt,niter,er,erfonc,test]=retcadilhac(@zz,struct('nout',3,'nin',3,'tol',1.e-9,'niter',20),a,b,x,c);% xt tel que tan(a*xt)==b
%   xxt=retcadilhac(@zz,struct('nout',3,'nin',3,'fonc','z=1/z'),a,b,x,c);% xt tel que tan(a*xxt)-b =inf 
% 
%%    trois exemples equivalents pour calculer acos(.8):
%      fc=@(x) (x-.8); retcadilhac(@cos,struct('fonc',fc),0)  
%      fun=@(t) cos(t);fc=inline('x-.8','x'); retcadilhac(fun,struct('fonc',fc),0) 
%      retcadilhac(@cos,struct('fonc','z=z-.8'),0)  
%   [z0,itermax,erz,erfonc,test]=retcadilhac(inline('(z-2)/((z+3)*(1+1.e-3*z))'),struct('tol',1.e-10,'precis',i,'tol_precis',30),4) 
%
% See also: RETZP

% ancienne version
%  parm:tableau 
%   parm(1):meme sens que plus haut(analytique ou non)
%      si parm(1) a une partie imaginaire:partie entiere de cette partie imaginaire= nb max d'iterations(defaut 30)
%                                 partie decimale de cette partie imaginaire= tolerance sur z(defaut eps)
%   parm(2):numero de la variable z en entree dans varargin (par defaut 1) 
%   parm(3):numero de la variable zz  en sortie (par defaut 1)
%   parm: peut aussi etre un cell array de dimension 2
%       parm{1} a le sens de ci-dessus
%       parm{2} est une chaine de caracteres du type 'z=1-1/z' qui permet de modifier la fonction que l'on veut annuler
%                     on resoud alors 0=1-1/zz (par exemple) la variable est obligatoirement z
%
%


if  ~isnumeric(varargin{1});[varargout{1:nargout}]=retcadilhac_nv(varargin{:});else;[varargout{1:nargout}]=retcadilhac_ancien(varargin{:});end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z0,z,zz,er,test]=retcadilhac_nv(zz,parm,varargin);
                                               %%%%%%%%%%%%%%%%%%%%%%%%
                                               %   forme simplifiee   %
                                               %%%%%%%%%%%%%%%%%%%%%%%%
nin=1;nout=1;  % par defaut

if isstruct(parm);% ************************
pparm=struct('fonc','','nout',1,'nin',1,'niter',30,'nitermin',0,'tol',eps,'tolf',inf,'bornes',[-inf,inf,-inf,inf],'precis',0,'tol_precis',6,'parm',0);% par defaut
   
fonc=retoptimget(parm,'fonc',pparm);if ischar(fonc);fonc=[fonc,';'];end;
nout=retoptimget(parm,'nout',pparm);
nin=retoptimget(parm,'nin',pparm);
niter=retoptimget(parm,'niter',pparm);
nitermin=retoptimget(parm,'nitermin',pparm);
tol=retoptimget(parm,'tol',pparm);
tolf=retoptimget(parm,'tolf',pparm);
bornes=retoptimget(parm,'bornes',pparm);
precis=retoptimget(parm,'precis',pparm);
tol_precis=retoptimget(parm,'tol_precis',pparm);
parm=retoptimget(parm,'parm',pparm); % a mettre en dernier car ecrase parm ...

else;  % ************************ ancienne version
precis=0;
nitermin=0;
if iscell(parm);fonc=[parm{2},';'];parm=parm{1};else fonc='';end;
if ~isempty(parm);if length(parm)>=3;nout=parm(3);end;if length(parm)>=2;nin=parm(2);end;parm=parm(1);else parm=0;end;
if imag(parm)==0;niter=30;tol=eps;else;niter=round(imag(parm));tol=imag(parm)-niter;parm=real(parm);end
tolf=inf;
bornes=[-inf,inf,-inf,inf];
end;  % ************************


varargout=cell(1,nout);
z=varargin{nin}(:);
er=inf;iter=0;z0=z(1);zzz=[];z=z(2:end);zz0=inf;
while (er>tol)|(tolf<abs(zz0))|iter<nitermin; %-----------------------
st_warning=warning;warning off;
[varargout{:}]=feval(zz,varargin{1:nin-1},z0,varargin{nin+1:end});zz0=trans(varargout{nout},fonc);
iter=iter+1;z=[z;z0];zzz=[zzz;zz0];[z0,z,zzz,er]=retcadilhac_ancien(z,zzz,parm);
warning(st_warning);
if (iter>=niter)| (real(z0)<=bornes(1))|(real(z0)>=bornes(2))|(imag(z0)<=bornes(3))|(imag(z0)>=bornes(4));break;end;
end;                                         %-----------------------
retcadilhac_ancien;% on efface nbz
if isempty(zz0);test2=0;else;test2=tolf>=abs(zz0(1));end;
test=[er<=tol,test2,iter<niter,(real(z0)>bornes(1))& (real(z0)<bornes(2))& (imag(z0)>bornes(3))& (imag(z0)<bornes(4))];
if test(3)==0;z0=z(1);end;% en cas de non convergence on prend la meilleure valeur calculee et non l'estimee

if precis~=0; % continuation du calcul en precision etendue
nitermin=nitermin+iter-3;tol=tol/(10^tol_precis);tolf=tolf/(10^tol_precis);zz0=inf;er=inf;
if isreal(precis);z=retprecis(z,real(precis));zzz=retprecis(zzz,real(precis));else;z=retprecis(z);zzz=retprecis(zzz);end;
for ii=1:1;[varargout{:}]=feval(zz,varargin{1:nin-1},z(ii),varargin{nin+1:end});zzz(ii)=trans(varargout{nout},fonc);end;
[z0,z,zzz]=cad_precision(z,zzz);
while (er>tol)|(tolf<abs(zz0))|(iter<nitermin);
[varargout{:}]=feval(zz,varargin{1:nin-1},z0,varargin{nin+1:end});zz0=trans(varargout{nout},fonc);
iter=iter+1;z=[z;z0];zzz=[zzz;zz0];
[z0,z,zzz,er]=cad_precision(z,zzz);
Z0=double(z0);if (iter>=niter)| (real(Z0)<=bornes(1))|(real(Z0)>=bornes(2))|(imag(Z0)<=bornes(3))|(imag(Z0)>=bornes(4));break;end;
end;
test=[er<=tol,tolf>=abs(zz0),iter<niter,(real(Z0)>bornes(1))& (real(Z0)<bornes(2))& (imag(Z0)>bornes(3))& (imag(Z0)<bornes(4))];
if test(3)==0;z0=z(1);end;% en cas de non convergence on prend la meilleure valeur calculee et non l'estimee
z=iter;zz=er;if isempty(zzz);er=inf;else;er=zzz(1);end;
itermax=iter;er=er(1);erfonc=abs(zz0);
end;% fin du calcul en precision etendue


z=iter;zz=er;if isempty(zzz);er=inf;else;er=zzz(1);end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%   forme generale   %
%%%%%%%%%%%%%%%%%%%%%%
function [z0,z,zz,er]=retcadilhac_ancien(z,zz,parm);
persistent nbz
if isempty(nbz);nbz=0;end;if nargin==0;nbz=0;return;end; 
nbz=[nbz,length(z)];
if length(z)>length(zz);z0=z(1);z=z(2:end);er=inf;return;end;% valeurs initiales
tr=size(z,2)>1;if tr;z=z.';zz=zz.';end;

if nargin<3;parm=0;end;

if length(z)==1;% valeur initiale
er=inf;
if parm==0; if z==0;z0=.0001;else;z0=1.0001*z;end; else;if z==0;z0=.0001*(1+i);else;z0=(1.0001+.0001i)*z;end;end;
return;end;

if length(z)==2;
if parm==0;[z0,z,zz,er]=cad(z,zz,4,tr);else;z0=(1.001+.001i)*z(2);er=inf;if tr;z=z.';zz=zz.';end;end;
return;end;

if length(z)==3 &  nbz(end-1)==2 ;
[prv,jj]=sort(abs(zz));
if jj(1)~=3;[z0,z,zz,er,test]=cad(z(jj(1:2)),zz(jj(1:2)),6,tr);return;end;
end;

if length(z)==4 ; % z(4) remplace z(3)
	
% 	[prv,jj]=sort(abs(zz));jj,z,abs(zz)
% z=z([4,1,2]);zz=zz([4,1,2]);	
if nbz(end-1)==4 ;
[prv,jj]=sort(abs(zz));
if jj(1)~=4;[z0,z,zz,er,test]=cad(z(jj(1:3)),zz(jj(1:3)),6,tr);nbz(end)=3;return;else;z=z([4,1,2]);zz=zz([4,1,2]);end;
else;z=z([4,1,2]);zz=zz([4,1,2]);end;
end;    

if parm==1;[z0,z,zz,er,test]=cad(z,zz,5,tr);end;  % fonction non analytique   
if parm~=1;[z0,z,zz,er,test]=cad(z,zz,1,tr);end;
   
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z0,z,zz,er,test]=cad(z,zz,met,tr);
test=0;
[prv,jj]=sort(abs(zz));z=z(jj);zz=zz(jj);% mise en ordre 
	
if met==1;   % approximation homographique..
zm=sum(z)/length(z);a=[ones(3,1),zz,zz.*(z-zm)];
if rcond(a)>eps;u=a\(z-zm);z0=u(1)+zm;test=1;else;met=4;end;
%else;[p,s,mu]=polyfit(zz,z,2);z0=polyval(p,0,s,mu);test=1;end;

end;

%if met==2;     % approximation zz=polynome degre 2 pour memoire...
% a=[z.^2,2*z,ones(3,1)];
% if rcond(a)<1.e-10;met=4;else;
% a=a\zz;
% if a(1)==0;met=4;else;
% if a(2)==0&a(3)==0;z0=0;else;
% z0=sqrt(a(2)*a(2)-a(1)*a(3));
% if abs(a(2)-z0)>abs(a(2)+z0);z0=-z0;end;
% zz0=-(a(2)+z0)/a(1);z0=-a(3)/(a(2)+z0);
% if min(abs(z-zz0))<min(abs(z-z0));z0=zz0;end;    
% end;end;end;
%end;

if met==5;     % fonction non analytique
a=[real(z),imag(z),[1;1;1]];
if rcond(a)<eps;met=4;else;test=2;
a=a\zz;z0=(imag(a(2)*conj(a(3)))+i*imag(conj(a(1))*a(3)))/imag(a(1)*conj(a(2)));
end;
end;

if met==4;   % approximation lineaire..
% a=[ones(size(zz)),zz];    
% if ~all(isfinite(a(:)))|rank(a)<2;met=6;
% else;u=a\z;z0=u(1);if abs(z0-z(1))>3*abs(z(1)-z(end));met=6;end;end;  % si ca deborde trop on tire au hasard
[p,prv,mu]=polyfit(z,zz,1);z0=mu(1)-mu(2)*p(2)/p(1);
if ~isfinite(z0) | (length(z)>2&(abs(z0-z(1))>3*abs(z(1)-z(end))));met=6;end;  % si ca deborde trop on tire au hasard sauf pour le premier tir
%if (abs(z0-z(1))>3*abs(z(1)-z(end)));met=6;end;  % si ca deborde trop on tire au hasard sauf pour le premier tir
end;

if met==6;    % tiré au hasard ( avec petite probabilite de sortir du triangle )
a=rand(size(z))-.1;z0=sum(z.*a)/sum(a);
end;

er=abs(z-z0);if (er(1)<er(2)&(met<3));er=er(1)*max(er(1)/er(2),abs(zz(1)/zz(2)));else er=er(1);end;if isnan(er);er=inf;end;% estimation de  er 
if tr;z=z.';zz=zz.';end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function z=trans(z,fonc);
if ischar(fonc);eval(fonc);else;z=feval(fonc,z);end;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z0,z,zz,er]=cad_precision(z,zz);
[prv,jj]=sort(abs(zz));jj=jj(1:min(3,end));z=z(jj);zz=zz(jj); % mise en ordre
switch length(z);
case 2;
a=[ones(size(zz)),zz];    
case 3;
a=[ones(size(zz)),zz,zz.*z];
end;
%a=displayn(a);for ii=1:numel(a);a{ii}{4}=n+2;end;a=retprecis(a);
u=a\z;z0=u(1);
%z0=displayn(z0);z0{1}{4}=n;z0=retprecis(z0);
er=double(abs(z0-z(1)));