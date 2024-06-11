function [x,xx]=retgs(x,k);
% function x=retgs(x,k);
%  transformation de matrices S G ou T en:
%  matrice S si k=0 (ou non specifié)
%  matrice G si k=1
%  matrice T si k=2
%
%  la matrice x peut avoir ete stoquée sur fichiers par retio
%   au retour x n'est pas mise sur fichier( sauf si x l'est dejà et que son type n'est pas modifié)
%
% Attention: il s'agit de matrices S G au 'sens reticolo' c a d des cell-array 


if nargin<2;k=0;end;
siz=retio(x,4); % size de x

if siz(1)==siz(2);% x est une matrice T ou 0D
if iscell(x);  % 0D
xx=x;sz=size(x{1,1});
if (k==0)|(k==1);% --> S
% xx=x;xx{1,2}=0;	xx{2,2}=-1;	
% x{1,1}=1;x{2,1}=0;x{1,2}=-x{1,2};x{2,2}=-x{2,2};	
% xx=retss(retsp(x,-1),xx);	

xx{2,1}=-x{2,1}./x{2,2};
xx{1,1}=x{1,1}+xx{2,1}.*x{1,2};
xx{2,2}=1./x{2,2};
xx{1,2}=xx{2,2}.*x{1,2};
end;   %   --> on reste en T
if nargout>1;x=[];return;end
x=zeros(2,2,sz(1),sz(2));
for ii=1:2;for jj=1:2;x(ii,jj,:,:)=reshape(xx{ii,jj},1,1,sz(1),sz(2));end;end % matrice: 2 2 nk nbeta  
return;
end;  % fin 0D

if k==2;return;else;x=retio(x);end;% T->T    
if k==0;        % T->S
n=length(x)/2;x={full([[speye(n);zeros(n,n)],-x(:,n+1:2*n)]\[x(:,1:n),[zeros(n,n);-speye(n)]]);n;n;1};
return;end;
if k==1;        % T->G
n=fix(length(x)/2);x={x;eye(2*n);n;n;n;n};
return;end;

else    
if siz(1)==4;% x est une matrice S
if k==0;return;else;x=retio(x);end;% S->S 
nrep=x{4};% x est a elever a la puissance nrep
if k==1;        % S->G
n1=x{2};n2=size(x{1},2)-n1;n3=x{3};n4=size(x{1},1)-n3;
x={[x{1}(:,1:n1),[zeros(n3,n4);-eye(n4)]];[[eye(n3);zeros(n4,n3)],-x{1}(:,n1+1:n1+n2)];n1;n4;n3;n2};
x=retsp(x,nrep);return;end;
if k==2;        % S->T
n=x{2};x=full([[speye(n);zeros(n,n)],-x{1}(:,n+1:2*n)]\[x{1}(:,1:n),[zeros(n,n);-speye(n)]]);x=x^nrep;
return;end;

else;    
if siz(1)==6;% x est une matrice G
if k==1;return;else;x=retio(x);end;% G->G    
if k==0;        % G->S
n1=x{3};n2=x{4};n3=x{5};n4=x{6};
x={full([x{2}(:,1:n3),-x{1}(:,n1+1:n1+n2)])\full([x{1}(:,1:n1),-x{2}(:,n3+1:n3+n4)]);n1;n3;1};
return;end;
if k==2;        % G->T
x=full(x{2})\full(x{1});    
return;end;
end;
end;    
end;
