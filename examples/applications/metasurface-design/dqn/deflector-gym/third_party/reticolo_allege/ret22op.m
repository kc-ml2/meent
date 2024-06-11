function epm=ret22op(e,o,x,y,z,wx,wy,wz,u,v,kk);
% developpement en ondes planes dans un milieu homogene isotrope
%%%%%%%%%%%%%%%%%
%     2 D       %
%%%%%%%%%%%%%%%%%
% epm=retop(e,o,x,y,z,wx,wy,wz,u,v,kk);
% e o z w sont calcules par retchamp x wx  y,wy par retgauss
% [x,wx]=retgauss(...);[y,wy]=retgauss(...);
% [e,z,w,o]=retchamp(init,a,sh,sb,inc,{x,y},tab,[],[1:6],1,1,[1:6]);
% u,v:constantes de propagation où on veut le developpement (vecteurs)
%   dans l'ordre des x y z  
%   un des vecteurs x,y,z doit etre de longueur les 2 autres de longueur >1
%  on a donc 3 cas;
%     z=z0   u associe à x    v associe à y
%     y=y0   u associe à x                       v associe à z 
%     x=x0                    u associe à y       v associe à z 
%   kk: 0 ou absent: epm(length(u),length(v),6) composantes de Fourier de Ex,Ey,Ez,Hx,Hy,Hz
%       1  epm(length(u),length(v),12) composantes de Fourier de Ex+,Ey+,Ez+,Hx+,Hy+,Hz+,Ex-,Ey-,Ez-,Hx-,Hy-,Hz-
%%%%%%%%%%%%%%%%%
%     1 D       %
%%%%%%%%%%%%%%%%%
% epm=retop(e,o,x,y,wx,wy,u,kk);
% e o y w sont calcules par retchamp x wx par retgauss
% [x,wx]=retgauss(...);
% [e,y,w,o]=retchamp(init,a,sh,sb,inc,x,tab,[],[1:6],1,1,[1:6]);
% u:constantes de propagation où on veut le developpement (vecteur)
%   un des vecteurs x,y doit etre de longueur les 2 autres de longueur >1
%  on a donc 2 cas;
%     y=y0   u associe à x
%     x=x0                    u associe à y 
%   kk: 0 ou absent: epm(length(u),3) composantes de Fourier de E,Hx,Hy
%       1  epm(length(u),6) composantes de Fourier de E,Hx+,Hy+,E-,Hx-,Hy-
% dans les 2 cas o peut etre remplace par [mu,ep] du milieu

if nargin<9;
   
%%%%%%%%%%%%%%%%%
%     1 D       %
%%%%%%%%%%%%%%%%%
% epm=retop(e,o,x,y,wx,wy,u,kk);
if nargin<8;kk=0;else kk=wz;end;u=wy;wy=wx;wx=z;
if length(x)==1;wx=1;v=u;u=0;k=1;end;
if length(y)==1;wy=1;v=0;k=2;end;
mu=o(1,1,1);ep=o(1,1,end);
uux=exp(-i*u(:)*(x(:)).')*diag(wx);vvy=exp(-i*v(:)*(y(:)).')*diag(wy);
nx=length(x);ny=length(y);nu=length(u);nv=length(v);
e=e(:,:,1:3);
e=vvy*reshape(e,ny,nx*3);e=reshape(e,nv,nx,3);
e=permute(e,[2,1,3]);e=uux*reshape(e,nx,nv*3);e=reshape(e,nu,nv,3);
e=permute(e,[2,1,3])/(2*pi);
if kk==0;epm=e;return;end;
% separation :ondes sortantes,ondes entrantes
epm=zeros(length(v),length(u),6);
i3=eye(3);z3=zeros(3);
for iu=1:length(u);for iv=1:length(v);
uv=sqrt(ep*mu-v(iv)^2-u(iu)^2);
if k==1;up=uv;um=-uv;vp=v(iv);vm=v(iv);end;
if k==2;up=u(iu);um=u(iu);vp=uv;vm=-uv;end;
epm(iv,iu,:)=[[i3,i3];...
        [[[-vp,mu,0];[up,0,mu];[ep,-vp,up]],z3];...
          [z3,[[-vm,mu,0];[um,0,mu];[ep,-vm,um]]]]...
    \[squeeze(e(iv,iu,:));zeros(6,1)];
end;end;
else;
%%%%%%%%%%%%%%%%%
%     2 D       %
%%%%%%%%%%%%%%%%%

if nargin<11;kk=0;end;
if length(x)==1;wx=1;w=v;v=u;u=0;k=1;end;
if length(y)==1;wy=1;w=v;v=0;k=2;end;
if length(z)==1;wz=1;w=0;k=3;end;

mu=o(1,1,1,1);ep=o(1,1,1,end);

uux=exp(-i*u(:)*(x(:)).')*diag(wx);vvy=exp(-i*v(:)*(y(:)).')*diag(wy);wwz=exp(-i*w(:)*(z(:)).')*diag(wz);

nx=length(x);ny=length(y);nz=length(z);nu=length(u);nv=length(v);nw=length(w);
e=wwz*reshape(e,nz,nx*ny*6);e=reshape(e,nw,nx,ny,6);
e=permute(e,[2,1,3,4]);e=uux*reshape(e,nx,nw*ny*6);e=reshape(e,nu,nw,ny,6);
e=permute(e,[3,1,2,4]);e=vvy*reshape(e,ny,nu*nw*6);e=reshape(e,nv,nu,nw,6);%e=sparse(retmat(vvy,-nu*nw))*reshape(e,nw*nu*ny,6);
e=permute(e,[3,2,1,4])/(4*pi*pi);
if kk==0;epm=e;return;end;

% separation :ondes sortantes,ondes entrantes
epm=zeros(length(w),length(u),length(v),12);
i3=eye(3);z3=zeros(3);
for iu=1:length(u);;for iv=1:length(v);for iw=1:length(w);
uvw=sqrt(ep*mu-v(iv)^2-w(iw)^2-u(iu)^2);
if k==1;wp=[[0,-w(iw),v(iv)];[ w(iw),0, -uvw];[ -v(iv), uvw,0]];wm=[[0,-w(iw),v(iv)];[w(iw),0,   uvw];[-v(iv), -uvw,0]];end;
if k==2;wp=[[0,-w(iw), uvw ];[w(iw),0,-u(iu)];[-uvw , u(iu),0]];wm=[[0,-w(iw), -uvw];[w(iw),0,-u(iu)];[ uvw,  u(iu),0]];end;
if k==3;wp=[[0,-uvw , v(iv)]; [uvw ,0,-u(iu)];[-v(iv),u(iu),0]];wm=[[0, uvw,  v(iv)];[-uvw, 0,-u(iu)];[-v(iv),u(iu),0]];end;

epm(iw,iu,iv,:)=[[i3,z3,i3,z3];[z3,i3,z3,i3];...
     [wp([6,7,2]),zeros(1,9)];[zeros(1,3),wp([6,7,2]),zeros(1,6)];[zeros(1,6),wm([6,7,2]),zeros(1,3)];[zeros(1,9),wm([6,7,2])];...
        [wp,-mu*i3,z3,z3];[ep*i3,wp,z3,z3];[z3,z3,wm,-mu*i3];[z3,z3,ep*i3,wm]]...
    \[squeeze(e(iw,iu,iv,:));zeros(16,1)];
end;end;end;
end;

