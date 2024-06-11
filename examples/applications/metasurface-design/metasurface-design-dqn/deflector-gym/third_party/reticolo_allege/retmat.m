function y=retmat(x,n);
% function y=retmat(x,n);replication diagonale 'abs(n) fois'
%
%    n<0            n>0
%
%  1 0 4 0        1 4 0 0
%  0 1 0 4        2 5 0 0
%  2 0 5 0        3 6 0 0
%  0 2 0 5        0 0 1 4
%  3 0 6 0        0 0 2 5
%  0 3 0 6        0 0 3 6
%
% See also: BLKDIAG

if n==0;y=[];return;end;
nx=size(x,1);ny=size(x,2);
if n<0;n=-n;
ii=repmat(1:nx*n,1,ny);jj=reshape(1:ny*n,n,ny);jj=repmat(jj,nx,1);x=repmat(x(:).',n,1);
y=sparse(ii,jj(:),x(:),nx*n,ny*n,n*nx*ny);

else;  %n>0
ii=repmat(reshape([1:nx*n],nx,n),ny,1);jj=repmat(1:n*ny,nx,1);x=repmat(x(:),n,1);  
y=sparse(ii(:),jj(:),x(:),n*nx,n*ny,n*nx*ny);   
end;
