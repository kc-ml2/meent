function [ii,jj]=retassocie(x,xx,tol)
%  function [ii,jj]=retassocie(x,xx,tol);
%
% association de 2 ensembles de n vecteurs de dimension dim tres voisins 
%  x(1:n,1:dim) ,xx(1:n,1:dim) voisins mais dans un ordre divers
%  xx=x(ii,:);x=xx(jj,:); à mieux que tol ( precision relative globale, par defaut tol=1.e-8)
%  principe: x et xx sont projetes sur une droite aleatoire puis tries
% ( en cas d'echec on recommence sur les points non associes)
% Attention:si les 2 familles des vecteurs different de plus de tol risque de bouclage
%
% si tol est complexe association sans tolerence
% on cherche la meilleure correspondance entre x et xx
% 
%  See also: RETELIMINE,RETCOMPARE,UNIQUE
%
%% Exemple et test du cas d'echec 
%  for kk=1:10;n=10000;m=100;x=rand(n,m)+i*rand(n,m);k=randperm(n);xx=x(k,:).*(1+.5e-8*randn(size(x)));
%  tic;[ii,jj]=retassocie(x,xx);toc;retcompare(xx,x(ii,:)),retcompare(x,xx(jj,:)),end;

if isempty(x)|isempty(xx);ii=[];jj=[];return;end;
if nargin<3;tol=1.e-8;end;
if ~isreal(tol);[ii,jj]=retassocie_sans_tol(x,xx);return;end;
if ~isreal(x)|~isreal(xx);[ii,jj]=retassocie([real(x),imag(x)],[real(xx),imag(xx)],tol);return;end;% plus rapide ..
ermax=tol*mean(sqrt(sum(abs(x.^2),2)));
dim=size(x,2);
% essai;
tab=randn(dim,1);
[prv,k]=sort(x*tab);[prv,l]=sort(k);
[prv,kk]=sort(xx*tab);[prv,ll]=sort(kk);
ii=k(ll);jj=kk(l);
[f,ff]=retfind( sqrt(sum(abs(xx-x(ii,:)).^2,2))<ermax);
if isempty(ff);return;end;
% en cas d'echec (tres rare) on recommence avec une tolerance plus forte (pour eviter de boucler)
[iii,jjj]=retassocie(x(ii(ff),:),xx(ff,:),tol*1.5);
ii(ff)=ii(ff(iii));jj(ii(ff))=jj(ii(ff(jjj)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ii,jj]=retassocie_sans_tol(x,xx);
[n,dim]=size(x);a=zeros(n);
for ii=1:dim;a=a+abs(repmat(x(:,ii).',length(xx),1)-repmat(xx(:,ii),1,length(x))).^2;end;
prv=min(a,[],1);[prv,iii]=sort(prv);a=a(:,iii);% tri des colonnes
prv=min(a,[],2);[prv,jjj]=sort(prv);a=a(jjj,:);% tri des lignes
[prv,kk]=sort(iii);jj=jjj(kk);
[prv,kk]=sort(jjj);ii=iii(kk);





