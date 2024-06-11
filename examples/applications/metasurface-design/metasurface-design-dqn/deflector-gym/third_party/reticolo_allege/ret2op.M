function epm=ret22op(e,o,x,y,z,wx,wy,wz,u,v,w,kk);
% developpement en ondes planes dans un milieu homogene isotrope
% e o z w sont calcules par retchamp x wx  y,wy par retgauss
% [x,wx]=retgauss(...);[y,wy]=retgauss(...);
% [e,z,w,o]=retchamp(init,a,sh,sb,inc,{x,y},tab,[],[1:6],1,1,[1:6]);
% u,v,w:constantes de propagation où on veut le developpement (vecteurs)
% au moins un des vecteurs x,y,z doit etre de longueur 1
if nargin<12;kk=0;end;
if length(x)==1;wx=1;u=0;k=1;end;
if length(y)==1;wy=1;v=0;k=2;end;
if length(z)==1;wz=1;w=0;k=3;end;

mu=o(1,1,1,1);ep=o(1,1,1,end);

epm=zeros(length(w),length(u),length(v),12);
uux=exp(-i*u(:)*(x(:)).')*diag(wx);vvy=exp(-i*v(:)*(y(:)).')*diag(wy);wwz=exp(-i*w(:)*(z(:)).')*diag(wz);

nx=length(x);ny=length(y);nz=length(z);nu=length(u);nv=length(v);nw=length(w);
e=wwz*reshape(e,nz,nx*ny*6);e=reshape(e,nw,nx,ny,6);
e=permute(e,[2,1,3,4]);e=uux*reshape(e,nx,nw*ny*6);e=reshape(e,nu,nw,ny,6);
e=permute(e,[3,1,2,4]);e=vvy*reshape(e,ny,nu*nw*6);e=reshape(e,nv,nu,nw,6);%e=sparse(retmat(vvy,-nu*nw))*reshape(e,nw*nu*ny,6);
e=permute(e,[3,2,1,4])/(4*pi*pi);
if kk==0;epm=e;return;end;

% separation :ondes sortantes,ondes entrantes
i3=eye(3);z3=zeros(3);
for iu=1:length(u);;for iv=1:length(v);for iw=1:length(w);
uvw=sqrt(ep*mu-v(iv)^2-w(iw)^2-u(iu)^2);
if k==1;wp=[[0,-w(iw),v(iv)];[ w(iw),0, -uvw];[ -v(iv), uvw,0]];wm=[[0,-w(iw),v(iv)];[w(iw),0,   uvw];[-v(iv), -uvw,0]];end;
if k==2;wp=[[0,-w(iw), uvw ];[w(iw),0,-u(iu)];[-uvw , u(iu),0]];wm=[[0,-w(iw), -uvw];[w(iw),0,-u(iu)];[ uvw,  u(iu),0]];end;
if k==3;wp=[[0,-uvw , v(iv)]; [uvw ,0,-u(iu)];[-v(iv),u(iu),0]];wm=[[0, uvw,  v(iv)];[-uvw, 0,-u(iu)];[-v(iv),u(iu),0]];end;
epm(iw,iu,iv,:)=[[i3,z3,i3,z3];[z3,i3,z3,i3];[wp,-mu*i3,z3,z3];[ep*i3,wp,z3,z3];[z3,z3,wm,-mu*i3];[z3,z3,ep*i3,wm]]\[squeeze(e(iw,iu,iv,:));zeros(12,1)];
end;end;end;
