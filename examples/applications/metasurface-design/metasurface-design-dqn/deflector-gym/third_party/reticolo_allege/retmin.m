
function varargout=retmin(fun,parm,varargin);

%  recherche du minimum d' une fonction
% avec fminsearch , fminbnd (1 D) ou retminsearch (fminsearch ameliore par newton )
%**********************************************************************************
% function [ x_nin(1),   x_nin(end) ,fval,exitflag,output]
%    =retmin(fun, options, toutes les variables de fun );
%
%    [y1,  y(nout),    ym] =  fun(   x1,  x_nin(1),  x3,   x_nin(end)   ,xn  ) 
%             %                             %                 %
%           nout                          nin(1)            nin(2)
%
%      EN ENTREE
%   options=struct('nin',1,'nout',1,'fonc','','tolx',1.e-4,'tolf',1.e-4...                         %  (par defaut)
%     [],'maxfunevals',[],'maxiter',[],'opt_fmin',[],'methode',1,...                               %  (par defaut)
%     'fac','quad',1,'bornes',[],'trans',[],1,'tr',0,'nv_figure',1,'nessai',1000,'temp',.2);       %  (par defaut)
%                                   
%
%     nin: tableau des numeros des x1,  xn a faire varier
%     nout:  numero dela variable en sortie a minimiser
%             on cherche le minimum de   fonc ( y(nout) )
%     fonc est une chaine de caracteres de la forme  'z=-z'  ( par exemple pour calculer un maximum )
%     si fonc n'est pas vide, on cherche le minimum de   fonc ( y(nout) )
%     tolx:  critere d'arret sur la variable (tolx peut etre ecrit: ,'Tolx','TolX')
%     tolf:  critere d'arret sur la fonction (tolf peut etre ecrit:'TolFun','Tolfun' )
%     maxfunevals : nombre max d'evaluations de la fonction (maxfunevals peut etre ecrit:,'MaxFunEvals','MaxFunevals','Maxfunevals' )
%     maxiter:  nombre max d'iterations (maxiter peut etre ecrit: )
%     opt_fmin: options à transmettre directement à fminsearch ou fminbnd
%     methode:  1 fminsearch  2 fminbnd (1 d bornes obligatoire)  3
%     retminsearch 0 recuit
%     fac:  facteur 'loupe' :on fait varier x0+fac*dx ( pour fminsearch)
%     quad:  utilisation de l'approximation quadratique ( pour retminsearch)
%     bornes: bornes;vecteur de n lignes et 2 colones donnant le min et le max des variables( pour retminsearch)
%     trans: bornes fictives par l'intermediaire d'une transformation en sin 
%         meme format que bornes  prioritaire sur bornes
%     tr:  trace de l'evolution de la recherche
%     nv_figure: si on veut une nouvelle figure pour ce trace
%     nessai, temp pour le recuit
%
%      EN SORTIE
%     x_nin(1),   x_nin(end)  resultat de la minimisation
%     fval  ,exitflag , output: sorties de fminsearch 
%        pour retminsearch output est une structure contenant les valeurs de f(x) calculees dans le calcul
%
%               PRINCIPE DE RETMINSEARCH ET PERFECTIONNEMENTS PAR RAPPORT A FMINSEARCH
%   methode du simplex  depart d'un point ou d'un simplexe (n+1 points)
%   quand (n+1)*(n+2)/2 points ont ete calcules,
%  on approche la fonction par une forme de degre 2 pour accelerer la convergence a la fin
%  possibilité d'utiliser des contraintes sous forme de bornes,
%  soit directement soit avec un changement de variable 
%
%% EXEMPLES
%   x=retmin(@sin,[],1.51)  % min de sin(x)
%   x0=.1;y0=.2;z=retmin(inline('1+x^2-y^2;'),[],x0,y0)  % min de f(x,y0)
%   x0=.1;y0=.2;[x,y]=retmin(inline('1+x^2+y^2;'),[],x0,y0)  % min de f(x,y)
%
%   x0=.1;y0=.2;[z,min,test,sortie]=retmin(inline('1+x^2-y^2+x*y;'),struct('methode',1,'nin',2,'fonc','z=-z'),x0,y0) 
%                                             % max de 1+x0^2-y^2+x0*y   fonction de y avec fminsearch
%   x0=.1;y0=.2;[z,min,test,sortie]=retmin(inline('1+x^2-y^2+x*y;'),struct('methode',2,'nin',2,'fonc','z=-z','bornes',[-1,1]),x0,y0)  
%                                             % max de 1+x0^2-y^2+x0*y   fonction de y avec fminsearch
%   x0=.1;y0=.2;[z,min,test,sortie]=retmin(inline('1+x^2+2*y^2+x+y;'),struct('methode',3,'quad',0,'nin',[1,2]),x0,y0)%                                            % max de 1+x^2+2*y^2+x*y, retminsearch avec utilisation des approximations quadratiques
%                                            % min de 1+x^2+2*y^2+x+y, retminsearch sans utilisation des approximations quadratiques
%   x0=.1;y0=.2;[z,min,test,sortie]=retmin(inline('1+x^2+2*y^2+x+y;'),struct('methode',3,'quad',1,'nin',[1,2]),x0,y0)%                                            % max de 1+x^2+2*y^2+x*y, retminsearch avec utilisation des approximations quadratiques
%                                            % min de 1+x^2+2*y^2+x+y, retminsearch avec utilisation des approximations quadratiques
%   x0=.1;y0=.2;[z,min,test,sortie]=retmin(inline('1+x^2+2*y^2+x*y;'),struct('methode',3,'quad',0,'nin',[1,2]),x0,y0) 
%                                            % min de 1+x^2+2*y^2+x*y, retminsearch sans utilisation des approximations quadratiques
%   x0=.1;y0=.2;[z,min,test,sortie]=retmin(inline('1+x^2-2*y^2+x*y;'),struct('methode',3,'nin',[1,2],'trans',[-1,1;-2,3]),x0,y0) 
%          % min de 1+x^2+2*y^2+x*y sur le domaine defini par trans, 
%   x0=.1;y0=.2;[z,min,test,sortie]=retmin(inline('1+x^2-2*y^2+x*y;'),struct('methode',3,'quad',0,'nin',[1,2],'trans',[-1,1;-2,3]),x0,y0) 
%         % min de 1+x^2+2*y^2+x*y sur le domaine defini par trans, retminsearch sans utilisation des approximations quadratiques
% avec le recuit simule(en test)
%*****************
% function [  x_nin(1),    x_nin(end) ,          fopt,test,temp,funeval]=retmin(fun,parm,varargin);
%
%     parm={-nin,nout,'z=fonc(z)',tolx,maxfunevals,dx}
%            par defaut parm=[]  nin=-1 nout=1 fonc:'' tolx=[] maxfuneval=[] maxiter=[]
%
% 
% [x_nin(1),    x_nin(end) ,fopt,test,temp,funeval]=retmin(@sin,{-1},1.51)  % min de sin(x)




%   ancienne version
% avec fminsearch ou fminbnd
%****************************
% function [  x_nin(1),    x_nin(end)    ,fval,exitflag,output]=retmin(fun,options,varargin);
%
%
%    [y1,  y(nout),    ym] =  fun(   x1,  x_nin(1),  x3,   x_nin(end)   ,xn  ) 
%
%
%     nin: tableau des numeros des x1,  xn a faire varier
%     on cherche le minimum de   y(nout) eventuellement transforme par fonc 
%
%     parm={nin,nout,'z=fonc(z)',tolx,maxfunevals,maxiter,fac}
%            par defaut parm=[]  nin=1 nout=1 fonc:'' tols=[] maxfuneval=[] maxiter=[] fac=1
%            si imag(nout)~=0 trace (nv figure si imag(nout)>0)   
%      fval,exitflag,output: sorties de fminsearch 
%      fac:si 1 seul terme:fminsearch  fac represente un facteur 'loupe' :on fait varier x0+fac*dx
%      fac:si 2 termes: fminbnd les bornes sont fac.  nin doit alors etre de longueur 1(une seule variable reelle)



varargout=cell(1,nargout);
defaultopt=struct('nin',1,'nout',1,'fonc','','tolx',1.e-4,'tolf',1.e-4,'maxfunevals',[],'maxiter',[],'dx',0,...
'opt_fmin',[],'methode',1,'fac',1,'quad',1,'bornes',[],'trans',[],'tr',0,'nv_figure',1,'nessai',1000,'temp',.2);
if nargin<2;parm=[];end;
if isempty(parm);parm=defaultopt;end;
if iscell(parm);  % <----- compatibilite avec une ancienne version
defaultopt.nin=parm{1};if defaultopt.nin(1)<0;defaultopt.methode=0;defaultopt.nin=abs(defaultopt.nin);end;
if length(parm)>=2;defaultopt.nout=parm{2};end;
if length(parm)>=3;defaultopt.fonc=[parm{3},';'];end;
if length(parm)>=4;defaultopt.tolx=parm{4};end;
if length(parm)>=5;defaultopt.maxfunevals=parm{5};end;
if length(parm)>=6;defaultopt.maxiter=parm{6};end;
if length(parm)>=7;defaultopt.fac=parm{7};end;
if length(defaultopt.fac)>1;defaultopt.methode=2;parm.bornes=fac;end;% fminbound
if isempty(defaultopt.nin);defaultopt.nin=1;end;
if isempty(defaultopt.nout);defaultopt.nout=1;end;
defaultopt.tr=imag(defaultopt.nout);nout=real(defaultopt.nout);
if defaultopt.tr==1;defaultopt.nv_figure=1;end;defaultopt.tr=(defaultopt.tr~=0);
end;            % <----- parm est maintenant une structure a champs

nin=retoptimget(parm,'nin',defaultopt);
nout=retoptimget(parm,'nout',defaultopt);
fonc=retoptimget(parm,'fonc',defaultopt);fonc=[fonc,';'];
tolx=retoptimget(parm,{'tolx','Tolx','TolX'},defaultopt);
tolf=retoptimget(parm,{'tolf','TolFun','Tolfun'},defaultopt);
maxfunevals=retoptimget(parm,{'maxfunevals','MaxFunEvals','MaxFunevals','Maxfunevals'},defaultopt);
maxiter=retoptimget(parm,{'maxiter','MaxIter','Maxiter'},defaultopt);
opt_fmin=retoptimget(parm,'opt_fmin',defaultopt);
methode=retoptimget(parm,'methode',defaultopt);
fac=retoptimget(parm,'fac',defaultopt);
quad=retoptimget(parm,'quad',defaultopt);
bornes=retoptimget(parm,'bornes',defaultopt);
trans=retoptimget(parm,'trans',defaultopt);
tr=retoptimget(parm,'tr',defaultopt);
nv_figure=retoptimget(parm,'nv_figure',defaultopt);
dx=retoptimget(parm,'dx',defaultopt);if dx==0;dx=maxiter;end;% compatibilite avec une ancienne version
nessai=retoptimget(parm,'nessai',defaultopt);
temp=retoptimget(parm,'temp',defaultopt);

if tr&nv_figure;retplot;end;
vgout=cell(1,nout);

% x=varargin(nin);lx=length(x);
% sz=cell(size(x));xx=[];for ii=1:lx;xx=[xx,x{ii}];sz{ii}=size(x{ii});end
% lxx=length(xx);
% 

% construction de la variable de depart en concatenant les variables de nin
x=varargin(nin);
sz=cell(size(x));lx=length(x);sz=cell(size(x));
if iscell(x{1});% retminsearch avec simplexe initial
xx=cell(size(x{1}));
for jj=1:length(xx);
for ii=1:lx;xx{jj}=[xx{jj},(x{ii}{jj}(:)).'];if jj==1;sz{ii}=size(x{ii}{jj});end;end;
end;lxx=length(xx{1});
else;
xx=[];for ii=1:lx;xx=[xx,(x{ii}(:)).'];sz{ii}=size(x{ii});end
lxx=length(xx);
end;


switch(methode);

case 0; %recuit
if ~isempty(dx);if length(dx)<length(xx);ddx=xx;ddx(:)=dx(end);ddx(1:length(dx))=dx;dx=ddx;else dx=dx(1:length(xx));end;else dx=abs(xx)/10;end;    
if isempty(tolx);tolx=1.e-3;end;
if isempty(maxfunevals);maxfunevals=1.e5;end;
[xx,varargout{lx+1:end}]=frecuit(@cala,xx,dx,nessai,temp,tolx,maxfunevals,tr,xx,1,0,fun,fonc,nin,sz,vgout,varargin{:});
case 1; %fminsearch 
if isempty(opt_fmin);opt_fmin=struct('TolX',tolx,'Tolf',tolf,'MaxFunEvals',maxfunevals,'MaxIter',maxiter);end;
xx0=xx;  
[xx,varargout{lx+1:end}]=fminsearch(@cala,xx,opt_fmin,xx0,fac,tr,fun,fonc,nin,sz,vgout,varargin{:});
xx=xx0*(1-fac)+xx*fac;
case 2; %fminbnd 
if ~isempty(trans);bornes=trans;end;   
if isempty(opt_fmin);opt_fmin=struct('TolX',tolx,'Tolf',tolf,'MaxFunEvals',maxfunevals,'MaxIter',maxiter);end;
[xx,varargout{lx+1:end}]=fminbnd(@cala,bornes(1),bornes(2),opt_fmin,xx,1,tr,fun,fonc,nin,sz,vgout,varargin{:});
case 3; %retminsearch 
if isempty(opt_fmin);opt_fmin=struct('tolx',tolx,'tolf',tolf,'maxfunevals',maxfunevals,'maxiter',maxiter,'quad',quad,'bornes',bornes,'trans',trans);end;
xx0=0;fac=1;  
[xx,varargout{lx+1:end}]=retminsearch(@cala,xx,opt_fmin,xx0,fac,tr,fun,fonc,nin,sz,vgout,varargin{:});
end;

n=0;for ii=1:length(x);nn=prod(sz{ii});varargout{ii}=reshape(xx(n+1:n+nn),sz{ii});n=n+nn;end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function z=trans(z,fonc);eval(fonc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=cala(xx,xx0,fac,tr,fun,fonc,nin,sz,varargout,varargin);
if fac~=1;xx=xx0*(1-fac)+xx*fac;end;
x=cell(size(sz));n=0;for ii=1:length(x);nn=prod(sz{ii});x{ii}=reshape(xx(n+1:n+nn),sz{ii});n=n+nn;end;
for ii=1:length(nin);varargin{nin(ii)}=x{ii};end;
[varargout{:}]=feval(fun,varargin{:});a=trans(varargout{end},fonc);if tr retplot(a);end;



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %    RETMINSEARCH
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,fval,exitflag,output]=retminsearch(funfcn,x,options,varargin);
%figure;hold on;
if ~iscell(x);x={x};end;
sx=size(x{1});n=prod(sx);
defaultopt=struct('maxiter',200*n,'maxfunevals',200*n,'tolx',1e-4,'tolf',1e-4,'quad',0,'bornes',[],'trans',[]);
if nargin<3;options=[];end;
tolx=retoptimget(options,'tolx',defaultopt,'fast');
tolf=retoptimget(options,'tolf',defaultopt,'fast');
maxfunevals=retoptimget(options,'maxfunevals',defaultopt,'fast');
maxiter=retoptimget(options,'maxiter',defaultopt,'fast');
quad=retoptimget(options,'quad',defaultopt,'fast');
bornes=retoptimget(options,'bornes',defaultopt,'fast');
trans=retoptimget(options,'trans',defaultopt,'fast');
if ~isempty(trans);bornes=[];end;% priorite a trans
parm=struct('tolx',tolx,'tolf',tolf,'maxfunevals',maxfunevals,'maxiter',maxiter,'quad',quad,'bornes',bornes,'trans',trans);

% Initialisation
rho=1;chi=2;psi=0.5;sigma=0.5;onesn=ones(1,n);two2np1=2:n+1;one2n=1:n;
x_store=zeros(n,0);f_store=zeros(0,1);;xx_store=zeros(n,0);;ff_store=[];func_eval=1;itercount=0;its=0;itq=0;

%  simplex initial
v=zeros(n,n+1);fv=zeros(1,n+1);
if length(x)==1;x=x{1};% construction
if ~isempty(trans);x=transforme(x(:),trans,1);end;
xin=x(:);

v(:,1)=xin;% Place input guess in the simplex! (credit L.Pfeffer at Stanford)
usual_delta=0.05;             % 5 percent deltas for non-zero terms
zero_term_delta=0.00025;      % Even smaller delta for zero elements of x
for jj=1:n;y=xin;
if y(jj)~=0;y(jj)=(1+usual_delta)*y(jj);else;y(jj)=zero_term_delta;end;
if ~isempty(bornes);% le simplex ne doit pas depasser des bornes
if y(jj)<=bornes(jj,1);y(jj)=bornes(jj,1)+usual_delta*(bornes(jj,2)-bornes(jj,1));end
if y(jj)>=bornes(jj,2);y(jj)=bornes(jj,2)-usual_delta*(bornes(jj,2)-bornes(jj,1));end
end;
v(:,jj+1)=y;
end
else;%simplex donne en entree dans x
if ~isempty(trans);for ii=1:length(x);x{ii}=transforme(x{ii}(:),trans,1);end;end;
for jj=1:n+1;v(:,jj)=x{jj}(:);end;
end;
% calcul de f aux sommets du simplex
for jj=1:n+1;[fv(1,jj),x_store,f_store,xx_store,ff_store,func_eval]=fonceval(funfcn,v(:,jj),[],sx,bornes,trans,x_store,f_store,xx_store,ff_store,func_eval,varargin{:});end;
[fv,jj]=sort(fv);v=v(:,jj);

% Main algorithm
% Iterate until the diameter of the simplex is less than tolx
%   AND the function values differ from the min by less than tolf,
%   or the max function evaluations are exceeded.
while (func_eval<maxfunevals)&(itercount<maxiter);
% critere d'arret

if (max(abs(ff_store(1)-ff_store(two2np1)))<=tolf)&(max(max(abs(xx_store(:,2:n+1)-xx_store(:,ones(1,n)))))<= tolx);break;end;
%if (max(abs(fv(1)-fv(two2np1)))<=tolf)&(max(max(abs(xx_store(:,2:n+1)-xx_store(:,ones(1,n)))))<= tolx);break;end;

%if(abs(ff_store(1)-ff_store(2))<=tolf)&(norm(xx_store(:,2)-xx_store(:,1))<=tolx);break;end;
% approximation quadratique
if quad;[xq,fnv]=calxnv(x_store,f_store,bornes,trans);else;xq=[];end;fq=inf;
if ~isempty(xq);
[fq,x_store,f_store,xx_store,ff_store,func_eval]=fonceval(funfcn,xq,[],sx,bornes,trans,x_store,f_store,xx_store,ff_store,func_eval,varargin{:});
if abs(fnv-fq)>.5*mean(abs(f_store));fq=inf;end; % on elimine si quadratique est trop mauvais
end;


if fq<fv(end);
fv=[fq,fv(1:end-1)];v=[xq,v(:,1:end-1)];[fv,ii]=sort(fv);v=v(:,ii);itq=itq+1;  % on garde l'approximation quadratique

else;shrink=0;its=its+1;% <------on reprend le simplex  
    % Compute the reflection point
    % xbar = average of the n (NOT n+1) best points
    xbar=sum(v(:,one2n),2)/n;xr=(1+rho)*xbar-rho*v(:,end);
    [fxr,x_store,f_store,xx_store,ff_store,func_eval]=fonceval(funfcn,xr,[],sx,bornes,trans,x_store,f_store,xx_store,ff_store,func_eval,varargin{:});
    if fxr<fv(:,1);
        % Calculate the expansion point
        xe=(1+rho*chi)*xbar-rho*chi*v(:,end);
        [fxe,x_store,f_store,xx_store,ff_store,func_eval]=fonceval(funfcn,xe,[xr,v(:,end)],sx,bornes,trans,x_store,f_store,xx_store,ff_store,func_eval,varargin{:});
        if fxe<fxr;v(:,end)=xe;fv(:,end)=fxe;
        else;v(:,end)=xr;fv(:,end)=fxr;end;
    else % fv(:,1) <= fxr
        if fxr<fv(:,n);v(:,end)=xr;fv(:,end)=fxr;
        else % fxr >= fv(:,n)
            % Perform contraction
            if fxr < fv(:,end)
                xc=(1+psi*rho)*xbar-psi*rho*v(:,end);
                [fxc,x_store,f_store,xx_store,ff_store,func_eval]=fonceval(funfcn,xc,[xr,v(:,end)],sx,bornes,trans,x_store,f_store,xx_store,ff_store,func_eval,varargin{:});
                if fxc<=fxr;v(:,end)=xc;fv(:,end)=fxc;
                else;shrink=1; end
            else
                xcc=(1-psi)*xbar+psi*v(:,end);
                [fxcc,x_store,f_store,xx_store,ff_store,func_eval]=fonceval(funfcn,xcc,[xr,v(:,end)],sx,bornes,trans,x_store,f_store,xx_store,ff_store,func_eval,varargin{:});
                if fxcc < fv(:,end);v(:,end)=xcc;fv(:,end)=fxcc;
                else;shrink=1;end;
            end
            if shrink;%'shrink'
                for jj=two2np1;vv=v(:,jj);
                   v(:,jj)=v(:,1)+sigma*(v(:,jj)-v(:,1));
                  [fv(:,jj),x_store,f_store,xx_store,ff_store,func_eval]=fonceval(funfcn,v(:,jj),[vv,v(:,1)],sx,bornes,trans,x_store,f_store,xx_store,ff_store,func_eval,varargin{:});
                end
            end
        end
    end
    [fv,jj]=sort(fv);v=v(:,jj);
   
end;% <------on reprend le simplex
itercount=itercount+1;
%fill3(v(1,:),v(2,:),itercount*ones(size(v(1,:))),'y');
end   % while
if ~isempty(trans);xx_store=transforme(xx_store,trans,-1);end;
x=reshape(xx_store(:,1),sx);
fval=ff_store(1);
output=struct('x',xx_store,'f',ff_store);
commentaire='OK';if itercount>=maxiter;commentaire='nombre maximal d''iterations atteint';end;
if func_eval>=maxfunevals;commentaire='nombre maximal d''évaluations de la fonction atteint';end;
exitflag=struct('test',commentaire,'iter',itercount,'func_eval',func_eval,'iterations_du_simplex',its,'iterations_quadratiques',itq,'parm',parm);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xnv,fnv]=calxnv(x,f,bornes,trans);% approximation quadratique
n=size(x,1);mm=size(x,2);

m=((n+1)*(n+2))/2;
if mm<m;xnv=[];fnv=[];return;end; % pas assez de points
mmm=min(2*m,mm);
xx=x;ff=f;
cd=0;kkk=0;kkkmax=10;
while (cd<eps)&(kkk<kkkmax);kkk=kkk+1;
if kkk>1;perm=randperm(mmm);x=xx(:,perm);f=ff(perm);else;x=xx;f=ff;end;
x=x(:,1:m);f=f(1:m);mm=m;
aa=zeros(mm,m);
k=0;for ii=1:n;for jj=ii:n;k=k+1;
if ii==jj;aa(:,k)=(x(ii,:).^2).';else;aa(:,k)=(2*x(ii,:).*x(jj,:)).';end;
end;end;
aa(:,k+1:k+n)=x.';
aa(:,k+n+1)=ones(mm,1);
cd=rcond(aa);
end

if cd<eps;xnv=[];fnv=[];return;end;% mal conditionne
aa=aa\(f.');
a=zeros(n);b=zeros(n,1);
k=0;
for ii=1:n;for jj=ii:n;k=k+1;
a(ii,jj)=aa(k);a(jj,ii)=aa(k);
end;end;
b=aa(k+1:k+n);c=aa(end);
if ~all(isfinite(a));xnv=[];fnv=[];return;end;
vp=eig(a);if ~all(vp>0);
if isempty(bornes);xnv=[];fnv=[];return;
else;xnv=realmax*ones(n,1);fnv=realmax;end;% le max est sur les bords
else;
xnv=-.5*(a\b);fnv=xnv.'*a*xnv+b.'*xnv+c;
end;

% solution strictement dans les bornes ?
if ~isempty(bornes);
if ~all((xnv>bornes(:,1))&(xnv<bornes(:,2)));xnv=[];fnv=[];end;% ok
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f,x_store,f_store,xx_store,ff_store,func_eval]=fonceval(funfcn,x,xdegenere,sx,bornes,trans,x_store,f_store,xx_store,ff_store,func_eval,varargin);


if ~isempty(bornes);if ~all((x>=bornes(:,1))&(x<=bornes(:,2)));f=realmax;return;end;end;
ii=find(all(xx_store==repmat(x,1,size(xx_store,2))));if isempty(xx_store);ii=[];end;
if ~isempty(ii);f=ff_store(ii(1));% deja calcule
else;
if ~isempty(trans);xt=transforme(x,trans,-1);else;xt=x;end;
f=funfcn(reshape(xt,sx),varargin{:});func_eval=func_eval+1;
end;
xx_store=[xx_store,x(:)];ff_store=[ff_store,f];[ff_store,ii]=sort(ff_store);xx_store=xx_store(:,ii);% tabulation et mise en ordre

if ~isempty(xdegenere)&~isempty(x_store);
degenere=[];m=size(x_store,2);
for ii=1:size(xdegenere,2);
degenere=[degenere,find(all(x_store==repmat(xdegenere(:,ii),1,m)))];end;
[fpire,pire]=max(f_store(degenere));pire=degenere(pire);
if f<fpire;x_store(:,pire)=x(:);f_store(pire)=f;end;
else;
x_store=[x_store,x(:)];f_store=[f_store,f];
end;
[f_store,ii]=sort(f_store);x_store=x_store(:,ii);% tabulation et mise en ordre
%x_store=[x_store,x(:)];[f_store,ii]=sort([f_store,f]);x_store=x_store(:,ii);% tabulation et mise en ordre

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x=transforme(x,trans,sens);
if sens==1;
for ii=1:size(x,1);
x(ii,:)=asin(2*(x(ii,:)-(trans(ii,1)+trans(ii,2))/2)/(trans(ii,2)-trans(ii,1)));    
end;
else;
for ii=1:size(x,1);
x(ii,:)=((trans(ii,2)-trans(ii,1))/2)*sin(x(ii,:))+(trans(ii,1)+trans(ii,2))/2;
end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       RECUIT  a tester
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xopt,fopt,test,temp,funeval]=frecuit(func,x,dx,nessai,temp,tolx,maxfunevals,tr,varargin)
fopt=feval(func,x,varargin{:});
ff=fopt;xx=x;xopt=x;test=1;
%func=fcnchk(func,length(varargin));
%varargout=cell(1,nargout-6);var=varargout;
funeval=1;
while max(abs(dx))>tolx;
oui=0;mieux=0;
for essai=1:nessai;
f=inf;while ~isfinite(f); %  on elimine les valeurs f=nan
x=xx+dx.*(rand(size(dx))-.5); % nouveau choix de x
f=feval(func,x,varargin{:});
funeval=funeval+1;if funeval>maxfunevals test=1;return;end;
end;
if f<fopt;fopt=f;xopt=x;end;
if exp((fopt-f)/((abs(f)+abs(fopt))*temp))>rand;
if f<ff;mieux=mieux+1;end;oui=oui+1;ff=f;xx=x;
if tr retplot(f,rettexte(temp,'dxmax=',max(dx)));end;
end;
end;

% reevaluation de dx et temp
if oui<(nessai/5);dx=dx*.7;end;
if oui>(nessai/1.5);dx=dx/.7;end;
if (mieux<.5*(1-1/sqrt(nessai))*oui)&(oui>0);temp=temp*.7;end;
if (mieux>.9*oui)&(oui>0);temp=temp/.7;end;
disp(rettexte(oui,mieux,temp,'dxmax=',max(dx)))
xx=xopt;ff=fopt;
end;
