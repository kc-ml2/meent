function [b,k,kk]=retelimine(a,tol);
%  [b,k,kk]=retelimine(a,tol);
%%%%%%%%%%%%%%%%%%%%%%%% SCALAIRES %%%%%%%%%%   tol absent ou reel
% elimination dans  un vecteur a (ligne ou colonne) des elements egaux à mieux que tol
% ( precision relative par defaut tol=1.e-8)
%  si tol<=0 egalite stricte ( par exemple pour des entiers)
% si a est un vecteur colonne, b est un vecteur colonne  sinon   b est un vecteur ligne
% b=a(k)  a 'voisin' de b(kk)
%
%%%%%%%%%%%%%%%%%%%%%%%% VECTEURS LIGNE (par exemple points dans le plan) %%%%%%%%%%%%  imag(tol) ~0 :
% elimination dans un ensemble a de n vecteurs de longueur dim , a(1:n,1:dim)     
% des elements egaux à mieux que real(tol) ( precision relative 'globale')
%                                           ermax=tol*mean(sqrt(sum(abs(a.^2),2)))
%  
%  b=a(k,:)  a 'voisin' de b(kk,:)
%
% EXEMPLE
%  a=rand(1000,2);tol=.1+i;[b,k,kk]=retelimine(a,tol);figure;plot(a(:,1),a(:,2),'.g',b(:,1),b(:,2),'or');retcompare(a,b(kk,:))
% See also: RETCOMPARE,RETASSOCIE,UNIQUE

if isempty(a);b=a;k=[];kk=[];return;end;
if nargin<2;tol=1.e-8;end;
vect=imag(tol)~=0;
tol=real(tol);if tol<0;tol=0;end;
%tol=real(tol);if tol==0;tol=1.e-8;end;if tol<0;tol=0;end;
if ~iscell(a);                                % <<<<<<<<<<<<<<<<<<<<<<
                          %%%%%%%%%%%%%%%%
if vect;                  % VECTEURS     %
                          %%%%%%%%%%%%%%%%
ermax=tol*mean(sqrt(sum(abs(a.^2),2)));
if ermax>0;						  
b=ermax*round(a/ermax);
[b,k,kk]=unique(b,'rows');
%bb=ermax*ceil(a(k,:)/ermax);% inutile
% [bb,l,ll]=unique(b,'rows');
% kk=ll(kk);k=k(l);
b=a(k,:);
else;
[b,k,kk]=unique(a,'rows');	
end;	
                          %%%%%%%%%%%%%%%%
else;                     % SCALAIRES    %
                          %%%%%%%%%%%%%%%%
if isreal(a); % ********************* reels

	
[b,ii]=sort(a(:)); % version plus rapide (à tester)
switch (length(a));
case 0;b=a;k=[];kk=[];	
case 1;b=a;k=1;kk=1;	
otherwise;	
  %prv=abs(diff(b))>tol*(abs(b(1:end-1))+abs(b(2:end)));% version fausse   % modif 18/08/2009
ermax=tol*mean(abs(b));
if ermax>0 prv=diff(round(b/ermax))~=0;else;prv=diff(b)~=0;end;
                                                                             % fin modif 18/08/2009

f=[1;1+find(prv)];
k=ii(f);b=a(k);
if nargout>2;
kk=[1;1+cumsum(prv)];
[prv,jj]=sort(ii);
kk=kk(jj);	
else kk=[];end;
end;
[b,k,kk]=retelimine(a(:),tol+i);

else;  % ********************* complexes ou se ramene à comparer les vecteurs [real,imag] 
[b,k,kk]=retelimine([real(a(:)),imag(a(:))],tol+i);
b=b(:,1)+i*b(:,2);
end;
if size(a,1)==1;b=retcolonne(b,1);k=retcolonne(k,1);kk=retcolonne(kk,1);else;b=retcolonne(b);k=retcolonne(k);kk=retcolonne(kk);end;
end

                           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                           % forme generale valable aussi pour les cell-array  %
                           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else;                                % <<<<<<<<<<<<<<<<<<<<<<
sz=size(a);n=prod(sz);
kk=zeros(sz);
k=zeros(n,1);
n0=0;

for ii=1:n;
if kk(ii)==0;
n0=n0+1;
k(n0)=ii;
if iscell(a);f=ii;for jj=ii+1:n;if abs(retcompare(a{ii},a{jj}))<tol;f=[jj;f(:)];end;end;
else;f=find(abs(a(ii)-a(ii+1:end))<=.5*tol*(abs(a(ii))+abs(a(ii+1:end))))+ii;f=[ii;f(:)];end;
kk(f)=n0;
end;
end;
k=k(1:n0);
b=a(k);b=b(:);
if sz(2)>1;b=b.';k=k';end;

end;                                % <<<<<<<<<<<<<<<<<<<<<<
