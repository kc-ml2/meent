function a=retblkinv(a,decoupe,N,M,n_bloc);
% b=retblkinv(a,decoupe,N,M,n_bloc);
% M*a*N  blocs de longueur decoupe
% M=inv(N);
% b=inv(a);
% si n_bloc est precisé on ne calcule que la contribution du bloc numero n_bloc 
% prv=inv(a);

if nargin<2;a=inv(a);return;end;
if nargin<5;n_bloc=0;end;

n=length(decoupe);
n2=cumsum(decoupe);n1=1+[0,n2(1:n-1)];
if n_bloc==0;
b=M*a*N;
a=0;
for n_bloc=1:n;a=a+N(:,n1(n_bloc):n2(n_bloc))*inv(b(n1(n_bloc):n2(n_bloc),n(n_bloc):n2(n_bloc)))*M(n1(n_bloc):n2(n_bloc),:);end;
else;
a=N(:,n1(n_bloc):n2(n_bloc))*inv(M(n1(n_bloc):n2(n_bloc),:)*a*N(:,n1(n_bloc):n2(n_bloc)))*M(n1(n_bloc):n2(n_bloc),:);	
end;	


