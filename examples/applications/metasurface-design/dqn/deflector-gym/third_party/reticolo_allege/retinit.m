
function [init,n,beta]=retinit(d,ordre,beta0,sym,xl,varargin);
% [init,n,beta]=retinit(d,ordre,beta0,sym,cao);
%
% INITIALISATIONS                         1D          ou           2D
%
% d pas du reseau                          d                    [dx,dy]
% ordre:                                [n1,n2]               [nx1,nx2,ny1,ny2]  ordres de fourier
% ( si n1(ou nx1) a une partie imaginaire on passe en formalisme G sinon on est en S) 
%
% beta0=k0*n0*sin(teta0)                 beta0                 [beta0x,beta0y] 
%                                                    ou [beta0x,beta0y,delt] si on donne delt choix du plan d'incidence
%                                                dans le cas degenere de l'onde plane normale           
%
% sym:symetries                         [s,x0]                  [sx,sy,x0,y0]
%                       s=sym(1):symetrie du champ(E ou H)      [sx,sy]=sym(1:2): symetries de EX en x et en y 
%                     x0= sym(2):centre de symetrie             [x0,y0]=sym(3:4): origines des axes de symetrie 
%
% plus precisement en 2D on a les symetries des champs s=symetrie  a=antisymetrie 0=non symetrique
% (par exemple 'sa' veut dire fonction symetrique en x  ansymetrique en y )
% Attention: si y est axe de symétrie les fonctions sont symetriques ou antisymetriques en x
%
%                                                Ex    Ey    Ez   Hx   Hy   Hz
%                         sym=[ 1, 1,x0,y0]      ss    aa    as   aa   ss   sa
%                         sym=[-1, 1,x0,y0]      as    sa    ss   sa   as   aa
%                         sym=[ 1,-1,x0,y0]      sa    as    aa   as   sa   ss 
%                         sym=[-1,-1,x0,y0]      aa    ss    sa   ss   aa   as 
%
%                                                Ex    Ey    Ez   Hx   Hy   Hz
%                         sym=[ 1, 0,x0,y0]      s0    a0    a0   a0   s0   s0
%                         sym=[-1, 0,x0,y0]      a0    s0    s0   s0   a0   a0
%                         sym=[ 0, 1,x0,y0]      0s    0a    0s   0a   0s   0a 
%                         sym=[ 0,-1,x0,y0]      0a    0s    0a   0s   0a   0s 
%
% et en 1D                                  Ez    Hx    Hy 
%                         sym=[1,x0]         s     s    a  
%                         sym=[-1,x0]        a     a    s  
%
%     Fourier  = si{1}  * base symetrique de [Ex;Ey]       base symetrique de [Ex;Ey] = si{2} Fourier
%     Fourier  = si{3}  * base symetrique de [Hx;Hy]       base symetrique de [Hx;Hy] = si{4} Fourier
% nota:  les elements de si de 5 a 8 ne sont utilisés que par retbrillouin
%
% cao:transformee de coordonnees        [xinf,l+i*c]             [xinf,yinf,lx+i*cx,ly+i*cy]
%          image de l'infini             xinf                      xinf,yinf
%  largeur totale de la transformation     l                       lx,ly  
%  partie imaginaire                       c                       cx,cy 
%
%
% ce programme calcule un cell array  init
%   init=                          {n,beta,si,d,cao,parm}          {n,beta,ordre,alx,aly,nx,ny,si,d,cao,beta_sym,parm}
% parm=struct('dim',1,'d',d,'beta0',beta0,'sym',sym,'cao',xl,'sog',sog,'nsym',n,'nfourier',size(beta,2),'genre',0);
% parm=struct('dim',2,'d',d,'beta0',beta0,'delt',delt,'sym',sym,'cao',xl,'sog',sog,'nsym',nsym,'nfourier',n,'genre',0)
%        genre: 0 RCWA,  1 cylindres Polpov ,  2 cylindrique radial                          
%   n:nombre de composantes(compte tenu des symetries (retcouche diagonalise une matrice n*n)
%   beta: constantes de propagation du developpement de Rayleigh (en x ou en x et y)
%   beta_sym: pour les translations dans la base symetrique 
%   (  1 D  init{end}.dim==1            2 D  init{end}.dim==2 )
% 
% See also: RETHELP_POPOV




d=d(:).';
if nargin<4;sym=[];si=[];end;sym=sym(:).';
if nargin<5;xl=[];end;xl=xl(:).';
if nargin<3;beta0=[];end;beta0=beta0(:).';

if ~isreal(ordre);sog=0;ordre=real(ordre);else sog=1;end; % s ou g (partie imaginaire a ordre)

if length(ordre)==1; % <<<<<<<<<<<<<<<<<  POPOV cylindres
cao=xl(:).';if isempty(cao);cao=[inf,1,inf];end;if length(cao)<3;cao=[cao,2*cao(1)];end;cao=[cao,cao(1),1];
% cao=[ r0_num , pml_complexe , r1_num , r0_phys , Pml_reelle_raccord  ] pour pml reelles
if isempty(varargin);Pml={[],1};else;Pml=varargin{1};end;if isempty(Pml);Pml={[],1};end;
%if ~isempty(Pml);% pml reelles
r_phys=0;r_num=0;Pml{1}=Pml{1}(:).';Pml{2}=Pml{2}(:).';
for ii=length(Pml{1}):-1:1;r_phys=[r_phys,Pml{1}(ii)];r_num=[r_num,r_num(end)+(r_phys(end)-r_phys(end-1))/Pml{2}(ii+1)];end;

if isfinite(cao(1));r_phys=[r_phys,cao(1)];else;r_phys=[r_phys,2*r_phys(end)];end;
r_num=[r_num,r_num(end)+(r_phys(end)-r_phys(end-1))/Pml{2}(1)];% prolongement jusqu'à r0 avec la pente Pml{2}(1)

if cao(2)~=1;% pml complexe
rr0=r_num(end);rr1=rr0+cao(3)-cao(1);
r_num=[r_num,rr1];r_phys=[r_phys,cao(2)*rr1];
end;
[r_num,ii]=retelimine(r_num);r_phys=r_phys(ii);
if all(r_num==0);Pml=[];else;
ppml=diff(r_phys)./diff(r_num);
rr_num=[r_num(2:end-1)*(1-5*eps);r_num(2:end-1)*(1+5*eps)];rr_num=[r_num(1);rr_num(:);r_num(end)+realmax/100];
ppml=[ppml;ppml];
Pml={[r_num(:),r_phys(:)],[rr_num(:),ppml(:)]};
% r_phys=retinterp_popov(r_num,Pml,1);
% r_num=retinterp_popov(r_phys,Pml,2);% pour les traces
% pml=retinterp_popov(r_num,Pml,2);

% modification de cao passage en r numerique 
if cao(2)~=1;cao(1)=rr0;cao(3)=rr1;cao(5)=Pml{2}(end,2);end;
%d=retinterp(Pml{1}(:,2),Pml{1}(:,1),d);
end;

Kmax=ordre;
%ng1=min(25,ceil(d*Kmax));
%ng1=min(50,ceil(d*Kmax));
ng1=min(50,ceil(d*Kmax));
ng2=ceil(2*d*Kmax/ng1);
[k,wk,k_disc]=retgauss(0,Kmax,ng1,ng2);%k=k+k(1)/2;wk(:)=k(2)-k(1);%wk(1)=wk(1)/2;wk(end)=wk(end)/2;

%[kk,wkk,kk_disc]=retgauss(Kmax,inf+i,5);k=[k,kk];wk=[wk,wkk];k_disc=retelimine([k_disc,kk_disc,1.5*kk(end)-.5*kk(end-1)]);
%[kk,wkk,kk_disc]=retgauss(Kmax,1.5*Kmax,1,10);k=[k,kk];wk=[wk,wkk];k_disc=retelimine([k_disc,kk_disc,1.5*kk(end)-.5*kk(end-1)]);

N=length(k);n=2*N;
% calcul de Psip et Psim
L=beta0;I=speye(N);
if L==0;[Psip,Psim]=deal(speye(N));end;
if L>0;
Psip=retio(I-2*L*calpsi(L-1,-L,1,k,wk,k_disc),1);	
Psim=retio(I-2*L*calpsi(-L-1,L,-1,k,wk,k_disc),1);	
end;
if L<0; 
Psip=retio(I+2*L*calpsi(L-1,-L,-1,k,wk,k_disc),1);	
Psim=retio(I+2*L*calpsi(-L-1,L,1,k,wk,k_disc),1);	
end;
parm=struct('dim',2,'d',d,'cao',cao,'sog',sog,'delt',0,'sym',real(sym(1)),'nhanckel',N,'genre',1);
init={n,L,k,wk,cao,d,{Psip,Psim},[],[],Pml,parm};
return
end;              % <<<<<<<<<<<<<<<<<  POPOV cylindres FIN


if length(d)==2;[init,n,beta]=ret2init(sog,d,[ordre(1):ordre(2)],[ordre(3):ordre(4)],beta0,sym,xl);  % 2D
else;  %1D 
n1=ordre(1);n2=ordre(end);
beta=(2*pi/d)*[n1:n2];n=1+n2-n1;
if ~isempty(xl)&(real(xl(2))~=0);cao=retcao(xl(1),xl(2),d,n);else cao=[];end;
if isempty(beta0);beta0=0;end;
beta=beta+beta0;    

if isempty(sym);
si=[];
else;
if sym(1)==0;si=[];
else; %utilisation des symetries

r=spalloc(n,n,4*n);ns=0;na=0;
for ip=0:n2;im=n1+n2-ip;
kp=ip-n1+1;km=im-n1+1;
if ip~=im;
r(kp,km)=1;r(km,km)=1;na=na+1;    
r(kp,kp)=1;r(km,kp)=-1;ns=ns+1;    
else
r(kp,kp)=1;ns=ns+1;    
end    
end
Cho=exp(-i*beta*sym(2));
rr=r.';rr=rr*retdiag(1./sum(abs(rr).^2,1));
r=retdiag(Cho)*r;rr=rr*retdiag(1./Cho);

if sym(1)==1;f=1:ns;end;     %  champ  symetrique
if sym(1)==-1;f=ns+1:n;end;  %  champ  antisymetrique
si={r(:,f),rr(f,:)};
n=length(f);  %modification de n

end
end
parm=struct('dim',1,'d',d,'beta0',beta0,'sym',sym,'cao',xl,'sog',sog,'nsym',n,'nfourier',size(beta,2),'L',[],'genre',0);
init={n,beta,si,d,cao,parm};
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [init,n,beta]=ret2init(sog,d,ordrex,ordrey,beta0,sym,xl)
% initialisations
nx=size(ordrex,2);ny=size(ordrey,2);n=nx*ny;
alx=2*pi*ordrex;aly=2*pi*ordrey;
ordre=[repmat(ordrex,1,ny);ordrey(reshape(repmat(1:ny,nx,1),1,nx*ny))];
beta=ordre.*repmat((2*pi)./d.',1,n);

if nargin<6;sym=[];si=[];end;
if nargin<7;xl=[];end;
if nargin<5;beta0=[];end;
if isempty(beta0);beta0=[0,0];end;
if length(beta0)==4;L=beta0(4);beta0=beta0(1:3);genre=2;else;genre=0;L=[];end;% cylindrique radial	
if length(beta0)==3;delt=beta0(3);beta0=beta0(1:2);else delt=0;end;% incidence normale degeneree
parm=struct('dim',2,'d',d,'beta0',beta0,'delt',delt,'sym',sym,'cao',xl,'sog',sog,'nsym',n,'nfourier',n,'L',L,'parm',struct('sym',sym,'cao',xl),'genre',genre);
beta=beta+repmat(beta0.',1,n);
if ~isempty(sym)&sym(1)==0&sym(2)==0;sym=[];parm.sym=[];end;
if ~isempty(xl)&((real(xl(3))~=0)|(real(xl(4))~=0));cao={retcao(xl(1),xl(3),d(1),nx),retcao(xl(2),xl(4),d(2),ny)};else cao=[];parm.cao=[];end

%sym(1:2):symetries de EX en x et en y sym(3:4):origines des axes de symetrie 
if isempty(sym); %non utilisation des symetries
n=2*n; %modification de n
si=[];
else; %utilisation des symetries
if sym(1)==0&sym(2)==0;si=[];n=2*n;init={n,beta,ordre,alx,aly,nx,ny,si,d,cao,parm};return;end

nx1=ordrex(1);nx2=ordrex(end);ny1=ordrey(1);ny2=ordrey(end);

if (sym(1)~=0)&(sym(2)~=0); %symetrie en x et en y
rrr=spalloc(n,n,4*n);ff=zeros(2,n);
%k=@(ii,jj) (ii-nx1+1+nx*(jj-ny1));
k=inline('ii-nx1+1+nx*(jj-ny1)','ii','jj','nx','nx1','ny1');
kk=0;
for ip=ceil((nx1+nx2)/2):nx2;
for jp=ceil((ny1+ny2)/2):ny2;
im=nx1+nx2-ip;jm=ny1+ny2-jp;    
kpp=k(ip,jp,nx,nx1,ny1);kpm=k(ip,jm,nx,nx1,ny1);kmp=k(im,jp,nx,nx1,ny1);kmm=k(im,jm,nx,nx1,ny1); 
kk=kk+1;rrr(kpp,kk)=1;rrr(kpm,kk)=1;rrr(kmp,kk)=1;rrr(kmm,kk)=1;ff(:,kk)=[1;1];    
if jp~=jm; kk=kk+1;rrr(kpp,kk)=1;rrr(kpm,kk)=-1;rrr(kmp,kk)=1;rrr(kmm,kk)=-1;ff(:,kk)=[1;-1];end;    
if ip~=im;kk=kk+1;rrr(kpp,kk)=1;rrr(kpm,kk)=1;rrr(kmp,kk)=-1;rrr(kmm,kk)=-1;ff(:,kk)=[-1;1];end;    
if ip~=im&jp~=jm;kk=kk+1;rrr(kpp,kk)=1;rrr(kpm,kk)=-1;rrr(kmp,kk)=-1;rrr(kmm,kk)=1;ff(:,kk)=[-1;-1];end;    
end
end
fss=find(ff(1,:)==1&ff(2,:)==1);fsa=find(ff(1,:)==1&ff(2,:)==-1);fas=find(ff(1,:)==-1&ff(2,:)==1);faa=find(ff(1,:)==-1&ff(2,:)==-1);
nss=length(fss);nsa=length(fsa);nas=length(fas);naa=length(faa);
r=[[rrr(:,fss);spalloc(n,nss,0)],[spalloc(n,naa,0);rrr(:,faa)],[rrr(:,fsa);spalloc(n,nsa,0)],[spalloc(n,nas,0);rrr(:,fas)],[rrr(:,faa);spalloc(n,naa,0)],[spalloc(n,nss,0);rrr(:,fss)],[rrr(:,fas);spalloc(n,nas,0)],[spalloc(n,nsa,0);rrr(:,fsa)]];
r1=[[rrr(:,faa);spalloc(n,naa,0)],[spalloc(n,nss,0);rrr(:,fss)],[rrr(:,fas);spalloc(n,nas,0)],[spalloc(n,nsa,0);rrr(:,fsa)],[rrr(:,fss);spalloc(n,nss,0)],[spalloc(n,naa,0);rrr(:,faa)],[rrr(:,fsa);spalloc(n,nsa,0)],[spalloc(n,nas,0);rrr(:,fas)]];
n1=nss+naa;n2=nsa+nas;
f1=1:n1;f2=n1+1:n1+n2;f3=n1+n2+1:2*n1+n2;f4=2*n1+n2+1:2*n1+2*n2;
decoupe=[nss,nsa,naa,nas];
                                                                                     %   EX    EY    HX    HY     EZ  HZ
if isequal(sym(1:2),[ 1, 1]);f=f1;fz=f4;nez=nas;nhz=nsa;symetries=[1,3,4,3,1,2];end; %   ss    aa    aa    ss     as  sa
if isequal(sym(1:2),[-1, 1]);f=f4;fz=f1;nez=nss;nhz=naa;symetries=[4,2,1,2,4,3];end; %   as    sa    sa    as     ss  aa
if isequal(sym(1:2),[ 1,-1]);f=f2;fz=f3;nez=naa;nhz=nss;symetries=[2,4,3,4,2,1];end; %   sa    as    as    sa     aa  ss 
if isequal(sym(1:2),[-1,-1]);f=f3;fz=f2;nez=nsa;nhz=nas;symetries=[3,1,2,1,3,4];end; %   aa    ss    ss    aa     sa  as 
end;

if (sym(1)~=0)&(sym(2)==0); %symetrie en x seulement
rrr=spalloc(n,n,3*n);ff=zeros(2,n);
%kx=@(ii,jj) (ii-nx1+1+nx*(jj-1));
kx=inline('ii-nx1+1+nx*(jj-1)','ii','jj','nx','nx1');
kk=0;
for j0=1:ny;
for ip=ceil((nx1+nx2)/2):nx2;im=nx1+nx2-ip;
kp=kx(ip,j0,nx,nx1);km=kx(im,j0,nx,nx1);     
kk=kk+1;rrr(kp,kk)=1;rrr(km,kk)=1;ff(kk)=1;    
if ip~=im;kk=kk+1;rrr(kp,kk)=1;rrr(km,kk)=-1;ff(kk)=-1;end;    
end
end
fs=find(ff==1);fa=find(ff==-1);
ns=length(fs);na=length(fa);n1=na+ns;
r=[[rrr(:,fs);spalloc(n,ns,0)],[spalloc(n,na,0);rrr(:,fa)],[rrr(:,fa);spalloc(n,na,0)],[spalloc(n,ns,0);rrr(:,fs)]];
r1=[[rrr(:,fa);spalloc(n,na,0)],[spalloc(n,ns,0);rrr(:,fs)],[rrr(:,fs);spalloc(n,ns,0)],[spalloc(n,na,0);rrr(:,fa)]];
decoupe=[ns,na];
                                                                                %   EX   EY  EZ   HX   HY    HZ
if sym(1)==1;f=1:n1;fz=n1+1:2*n;nez=na;nhz=ns;symetries=[1,2,2,2,1,1];end;      %   s    a    a   a    s     s
if sym(1)==-1;f=n1+1:2*n;fz=1:n1;nez=ns;nhz=na;symetries=[2,1,1,1,2,2];end;     %   a    s    s   s    a      a
end;    

if (sym(1)==0)&(sym(2)~=0); %symetrie en y seulement
rrr=spalloc(n,n,3*n);ff=zeros(2,n);
ky='i0+nx*(jj-ny1)';

kk=0;
for i0=1:nx;
for jp=ceil((ny1+ny2)/2):ny2;jm=ny1+ny2-jp;
% kp=ky(i0,jp);    
% km=ky(i0,jm);
jj=jp;kp=eval(ky);    
jj=jm;km=eval(ky);    

kk=kk+1;rrr(kp,kk)=1;rrr(km,kk)=1;ff(kk)=1;    
if jp~=jm;kk=kk+1;rrr(kp,kk)=1;rrr(km,kk)=-1;ff(kk)=-1;end;    
end
end
fs=find(ff==1);fa=find(ff==-1);
ns=length(fs);na=length(fa);n1=ns+na;
r=[[rrr(:,fs);spalloc(n,ns,0)],[spalloc(n,na,0);rrr(:,fa)],[rrr(:,fa);spalloc(n,na,0)],[spalloc(n,ns,0);rrr(:,fs)]];
r1=[[rrr(:,fa);spalloc(n,na,0)],[spalloc(n,ns,0);rrr(:,fs)],[rrr(:,fs);spalloc(n,ns,0)],[spalloc(n,na,0);rrr(:,fa)]];
decoupe=[ns,na];
                                                                                 %   EX   EY   EZ   HX   HY  HZ
if sym(2)==1;f=1:n1;fz=1:n1;nez=ns;nhz=na;symetries=[1,2,1,2,1,2];end;           %   s    a    s    a    s    a
if sym(2)==-1;f=n1+1:2*n;fz=n1+1:2*n;nez=na;nhz=ns;symetries=[2,1,2,1,2,1];end;  %   a    s    a    s    a    s
end;

fsymli=find(all(r(n+1:2*n,:)==0,1));% pour symetries de ret2li

Cho=repmat(exp(-i*(beta(1,:)*sym(3)+beta(2,:)*sym(4))),1,2);
rr=r.';rr=rr*retdiag(1./sum(abs(rr).^2,1));
rr1=r1.';rr1=rr1*retdiag(1./sum(abs(rr1),1));
r=retdiag(Cho)*r;rr=rr*retdiag(1./Cho);
r1=retdiag(Cho)*r1;rr1=rr1*retdiag(1./Cho);


si={r(:,f),rr(f,:),r1(:,f),rr1(f,:),r(1:n,fz(1:nez)),rr(fz(1:nez),1:n),r(n+1:2*n,fz(nez+1:nez+nhz)),rr(fz(nez+1:nez+nhz),n+1:2*n),r(1:n,fsymli),rr(fsymli,1:n),decoupe,symetries};
n=length(f);  %modification de n dans le cas d'utilisation des symetries
% nota:  les elements de si de 5 à 8 ne sont utilisés que par retbrillouin
end
parm.nsym=n;

if ~isempty(si);           % utilisation des proprietes de symetrie
beta_sym=full([diag(blkdiag(si{2},si{4})*retdiag(repmat(beta(1,:),1,4))*blkdiag(si{1},si{3})).';...
diag(blkdiag(si{2},si{4})*retdiag(repmat(beta(2,:),1,4))*blkdiag(si{1},si{3})).']);
% beta_sym=full([diag([si{2},sparse(size(si{4}));sparse(size(si{2})),si{4}]*retdiag(repmat(beta(1,:),1,4))*[si{1},sparse(size(si{3}));sparse(size(si{1})),si{3}]).';...
% diag([si{2},sparse(size(si{4}));sparse(size(si{2})),si{4}]*retdiag(repmat(beta(2,:),1,4))*[si{1},sparse(size(si{3}));sparse(size(si{1})),si{3}]).']);
else;beta_sym=[beta,beta,beta,beta];% pour les translations
end;	
init={n,beta,ordre,alx,aly,nx,ny,si,d,cao,beta_sym,parm};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Psi=calpsi(an,am,sens,k,wk,k_disc);% cylindres Popov
Psi=((k.'.^an)*k.^am)*retdiag(wk);
if sens==1;Psi=triu(Psi);else;Psi=tril(Psi);end;

for jj=1:length(k_disc)-1; % jj
k1=k_disc(jj);k2=k_disc(jj+1);
f=find((k>k1)&(k<k2));
n0=length(f);
T=cos(acos((2*k(f)-k1-k2)/(k2-k1)).'*[0:n0-1]);
[kk,wkk]=retgauss(k1,k2,10,20,k(f));
TT=cos(acos((2*kk-k1-k2)/(k2-k1)).'*[0:n0-1]);
A=repmat((kk.^am).*wkk,n0,1);
if sens==1;for ii=1:n0;A(ii,kk<k(f(ii)))=0;end;
else;for ii=1:n0;A(ii,kk>k(f(ii)))=0;end;end;
Psi(f,f)=retdiag(k(f).^an)*((A*TT)/T);
end;                % jj
% 

% for jj=1:length(k_disc)-1; % jj
% k1=k_disc(jj);k2=k_disc(jj+1);
% f=find((k>k1)&(k<k2));
% n0=length(f);
% lmin=1;lmax=length(k_disc)-1;
% if sens==1;lmin=jj;else;lmax=jj;end;
% for ll=lmin:lmax;
% g=find((k>k_disc(ll))&(k<k_disc(ll+1)));
% m0=length(g);
% T=cos(acos((2*k(f)-k1-k2)/(k2-k1)).'*[0:n0-1]);
% [kk,wkk]=retgauss(k1,k2,10,20,k(f));
% TT=cos(acos((2*kk-k1-k2)/(k2-k1)).'*[0:n0-1]);
% A=repmat((kk.^am).*wkk,m0,1);
% if sens==1;for ii=1:m0;A(ii,kk<k(g(ii)))=0;end;
% else;for ii=1:m0;A(ii,kk>k(g(ii)))=0;end;end;
% Psi(g,f)=retdiag(k(g).^an)*((A*TT)/T);
% end;                % jj
% end;                % jj



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Psi=calpsi1(an,am,sens,k,wk,k_disc);% cylindres Popov
n=length(k);
Psi=zeros(n,n);
if sens==1;
for ii=1:n;
Psi(ii,ii:n-1)=Psi(ii,ii:n-1)+calP(am,k(ii:n-1),k(ii+1:n));
Psi(ii,ii+1:n)=Psi(ii,ii+1:n)+calP(am,k(ii+1:n),k(ii:n-1));
end;
else;
for ii=1:n;
Psi(ii,2:ii)=Psi(ii,2:ii)+calP(am,k(2:ii),k(1:ii-1));
Psi(ii,1:ii-1)=Psi(ii,1:ii-1)+calP(am,k(1:ii-1),k(2:ii));
end;
end
Psi=retdiag(k.^an)*Psi;
%%%%%%%%%%%%%%%%%%%%%
function P=calP(m,k1,k2);
P=sign(k2-k1).*(k2.*calPP(m,k1,k2)-calPP(m+1,k1,k2))./(k2-k1);
%%%%%%%%%%%%%%%%%%%%%
function PP=calPP(m,k1,k2);
if m==-1;PP=log(k2./k1);else;PP=(k2.^(m+1)-k1.^(m+1))/(m+1);end;






