function [s,s_elf]=retc(a,h,f,init);
% s=retc(a,h);
% calcul de la matrice S( ou G) associee au passage dans un milieu periodique invariant en y
% de parametres dielectriques constants par morceaux decrit par le descripteur de texture a initialisee par retcouche
% (a peut eventuellement avoir ete stocké sur disque par retio)
%
%
%  MATRICE S
%  on remplit un cell array :S={ss,n1,n2,m}
%   la matrice S est en fait ss=S{1}
%  dim n2 -->      X            x     <--- dim n1  
%                      =S{1}}
%                  y            Y
%  dans le cas pathologique (lambda/4n)   s={retc(a,h/m);m} sera automatiguement eleve a la puissance m par retss
%  ici n1=n2=n mais ce ne sera plus vrai si on tronque les matrices
%
%
%  MATRICE G
%  on remplit un cell array :G={G1,G2,n1,n2,n3,n4}
%  dim n1 --->           x              X    <----- dim n3
%                 G1*         =  G2 *
%  dim n2 --->           y              Y    <----- dim n4
%
%  ici n1=n2=n3=n4=n mais ce ne sera plus vrai si on tronque les matrices
%
%
%  pour les metaux infiniment conducteurs G obligatoire
%
% TRONCATURE
% s=retc(a,h,f,init)
%  Si f est indiqué en entrée  :troncature aux modes de numero correspondants
%  il est alors obligatoire de donner init
%   on a alors un resultat approché.La matrice s est toujours dans la base de Fourier
%  Attention dans le cas des modes de bloch non symetriques, f indique les numeros 
%  allant dans les deux sens et doit être de longueur paire
%  si f=0 pas de troncature
% 
%   ELEMENTS FINIS
%..............................................
%  [s,s_elf]=retc(init,a,ah,ab);
%  a descripteur du tronçon ah ab descripteurs des milieux en haut et en bas
%  s est en fait une matrice G
%  a ah ab peuvent etre mis sur fichier par retio


if nargin>=4 ;h=retio(h);if iscell(h);   % elements finis
[s,s_elf]=retelf_c(a,h,f,init);%retc(init,a_elf,ah,ab)
return;end;end;

a=retio(a);

if a{end}.type==7;s=retc_cylindrique_radial(a,h,f);return;end;% cylindrique_radial	


%<--------------- troncature aux modes f
%if (nargin>2)&(size(a,2)>7|length(a)<=6);
%if (nargin>2)&(a{end}.type~=2);
if (nargin>3)&(a{end}.type~=2);
sh=retb(init,a,1,0);sh=rettronc(sh,f,f,1);
sb=retb(init,a,-1,0);sb=rettronc(sb,f,f,-1);    
d=a{5};if (length(f)==1)&(f==0);f=1:length(d);end;
dd=exp(-d(f(:))*h);nn=length(f);
if a{end}.type==3;  % modes de bloch non symetriques
nn=nn/2;
else; % dielectriques
dd=[dd;dd];   
end;    
if a{end}.sog==1;   % matrices S
s=retss(sb,{diag(dd);nn;nn;1},sh);    
else;               % matrices G
s=retss(sb,{diag([dd(1:nn);ones(nn,1)]);diag([ones(nn,1);dd(nn+1:2*nn)]);nn;nn;nn;nn},sh);    
end;    
return;
end;
%<---------------  fin troncature

if isempty(a{1});s=[];return;end;% matrices vides 
switch a{end}.type;% <*******************************************************************  
case 5;  % <******************   textures inclinees   ***********************
[p,pp,tt,n,dd,beta]=deal(a{[1:5,9]});
[f,ff]=retfind(real(dd>0));d1=dd;d1(f)=0;d2=-dd;d2(ff)=0;
s={retdiag(exp(d1*h))*pp;retdiag(exp(d2*h))*pp*retdiag(exp((i*h*tt)*beta));n;n;n;n};
if a{end}.sog;s=retgs(s);end;% matrices S

case 3;  % <******************   modes de bloch non symetriques  ****** *****************
%d=a{5};n=length(d);s=retss(a{1},{diag(exp([-d;-d]*h));n;n;1},a{2});
d=a{5};n=length(d)/2;
if a{end}.sog==1;   % matrices S
s={diag(exp(-d*h));n;n;1}; 
else;          % matrices G
s={diag([exp(-d(1:n)*h);ones(n,1)]);diag([ones(n,1);exp(-d(n+1:2*n)*h)]);n;n;n;n}; 
end;
s=retss(a{1},s,a{2});

case 2;  % <****************** metaux **************************************************
g1=a{1};g2=a{2};d=a{3};m=a{4};n=a{5};
% propagation sur la hauteur h
dd=exp(-d*h);
gh=g2;gh(:,1:m)=gh(:,1:m)*diag(dd);
gb=g2;gb(:,m+1:2*m)=gb(:,m+1:2*m)*diag(dd);
s=retss({gh;g1;m;m;n;n},{g1;gb;n;n;m;m});

case {1,6};  %<******************   dielectriques   ***************************************
sog=a{end}.sog;

if sog==1;  %< ..........matrices S
p=a{1};pp=a{2};q=a{3};qq=a{4};d=a{5};sog=a{end}.sog;clear a;
d=retbidouille(d,3);
m=0;test=1;
while(test==1);m=m+1;hh=h/m;[t,c,test]=retcc7(hh*d);end; %cas pathologique lambda/(4*n)
n=length(d);
t=hh*t;
if n>2000;q=retio(q,1);p=retio(p,1);qq=retio(qq,1);pp=retio(pp,1);% pour economiser la memoire
s=[retio(q)*retdiag(c)*retio(qq),retio(q)*retdiag(t)*retio(pp);-retio(p)*retdiag(t.*d.*d)*retio(qq),retio(p)*retdiag(c)*retio(pp)];
q=retio(q,-1);p=retio(p,-1);qq=retio(qq,-1);pp=retio(pp,-1);
else;
s=[q*retdiag(c)*qq,q*retdiag(t)*pp;-p*retdiag(t.*d.*d)*qq,p*retdiag(c)*pp];
end;
s={s;n;n;m};

else;     %< ..........matrices G
pp=a{2};qq=a{4};d=a{5};clear a;
d=retbidouille(d,3);
n=length(d);
dd=-d*h;

if n==1  %++ pour accelerer le cas 0 D...
if real(dd)>.5;   %cas tres exeptionnel
a2=exp(-dd);
s={[[d*a2*qq,a2*pp];[d*qq,-pp]];[[d*qq,pp];[a2*d*qq,-a2*pp]];n;n;n;n};return;
else;             % cas normal
if abs(real(dd))<.5; % d petits
b1=cosh(dd);b2=-h*retsinc(dd/(i*pi));
s={[[qq,0];[0,pp]];[[b1*qq,b2*pp];[(d^2)*b2*qq,b1*pp]];n;n;n;n};return;
else;
ddd=exp(dd);    
s={[[d*qq,pp];[d*ddd*qq,-ddd*pp]];[[d*ddd*qq,ddd*pp];[d*qq,-pp]];n;n;n;n};return;
end;end;
end;   % ++fin cas 0 D

% cas normal
a1=d;a2=ones(n,1);a3=zeros(size(dd));a4=a3;b1=a3;b2=a3;
b3=d;b4=-a2;
[f1,f3]=retfind(real(dd)>.5); % f1:cas tres exeptionnel 
b2(f3)=exp(dd(f3));a3(f3)=b2(f3).*d(f3);a4(f3)=-b2(f3);b1(f3)=a3(f3);
if ~isempty(f1);
a2(f1)=exp(-dd(f1));a1(f1)=a2(f1).*d(f1);a3(f1)=d(f1);a4(f1)=-1;
b1(f1)=d(f1);b2(f1)=1;b3(f1)=a1(f1);b4(f1)=-a2(f1);
end;
f2=find(abs(real(dd))<.5); % d petits
if ~isempty(f2);
a1(f2)=1;a2(f2)=0;a3(f2)=0;a4(f2)=1;
b1(f2)=cosh(dd(f2));
b2(f2)=-h*retsinc(dd(f2)/(i*pi));
b3(f2)=(d(f2).^2).*b2(f2);b4(f2)=b1(f2);
end;
s={[[retdiag(a1)*qq,retdiag(a2)*pp];[retdiag(a3)*qq,retdiag(a4)*pp]];[[retdiag(b1)*qq,retdiag(b2)*pp];[retdiag(b3)*qq,retdiag(b4)*pp]];n;n;n;n};
end   %< ............................. S ou G

end;% <*********************************************************************************  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [t,c,test]=retcc7(x);
% t=(1-exp(-2*x))/(x*(1+exp(-2*x))=tanh(x)/x  c=2*exp(-x)/(1+exp(-2*x))=1/cosh(x)
%  dans le cas pathologique ou 1+exp(-2x) s'annule test=1  sinon test=0
f=find(real(x<0));x(f)=-x(f);% fonctions paires
t=zeros(size(x));c=t;test=0;
dd=exp(-2*x);if ~isempty(find(abs(1+dd)<1.e-6));test=1;return;end
c=exp(-x);prv=abs(c);c(prv<1.e-80*max(prv))=0;% modif 8-2010 pour eviter l'augmentation du temps de calcul du aux nombres tres petits
c=2*c./(1+dd);
f=find(abs(x)>1);ff=find((abs(x)<=1)&(abs(x)>1.e-10));fff=find(abs(x)<=1.e-10);
t(f)=(1-dd(f))./x(f);t(ff)=exp(-x(ff)).*2.*sinh(x(ff))./x(ff);t(fff)=2-2*x(fff);
t=t./(1+dd);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%                  ELEMENTS FINIS    1  D                %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [s,s_elf]=retelf_c(init,a_elf,ah,ab);
if init{end}.dim==2;[s,s_elf]=retelf_c_2D(init,a_elf,ah,ab);return;end;    % 2 D

%a_elf={a,aa,mh,mb,m,me,Hx,Hy,Bx,grad_grad,x,maille,indices,ep};
ah=retio(ah);ab=retio(ab);si=init{3};n=init{1};d=init{end}.d;bet=init{2};
aa=a_elf{2};mh=a_elf{3};mb=a_elf{4};m=a_elf{5};x=a_elf{11};clear a_elf;

%%%%%%%%%%%%%%%%%%%%%%%%%%
%   LIAISON A FOURIER    %
%%%%%%%%%%%%%%%%%%%%%%%%%%
nfourier=init{end}.nfourier;fh=eye(nfourier);
Eb=zeros(nfourier,mb);Hb=zeros(mb,nfourier);Eh=zeros(nfourier,mh);Hh=zeros(mh,nfourier);
% xb=[x(m-mh,1)-d;x(m-mb-mh+1:m-mh,1);x(m-mb-mh+1,1)+d]; % periodisation
% xh=[x(m,1)-d;x(m-mh+1:m,1);x(m-mh+1,1)+d]; % periodisation
xb=[x(m,1)-d;x(m-mb+1:m,1);x(m-mb+1,1)+d]; % periodisation
xh=[x(m-mb,1)-d;x(m-mb-mh+1:m-mb,1);x(m-mb-mh+1,1)+d]; % periodisation
xcb=(xb(2:end)+xb(1:end-1))/2;xch=(xh(2:end)+xh(1:end-1))/2;
muyb=rettestobjet(init,ab,-1,0,xcb,3);muyh=rettestobjet(init,ah,-1,0,xch,3);

for ii=1:mb;
xc=xcb(ii);l=xb(ii+1)-xb(ii);
Eb(:,ii)=Eb(:,ii)+abs(l)/(2*d)*exp(-i*bet.'*xc).*retsinc(l*bet.'/(2*pi),1);    
Hb(ii,:)=Hb(ii,:)+(abs(l)/2)*exp(i*bet*xc).*retsinc(-l*bet/(2*pi),1)/muyb(ii);    
xc=(xb(ii+1)+xb(ii+2))/2;l=xb(ii+1)-xb(ii+2);
Eb(:,ii)=Eb(:,ii)+abs(l)/(2*d)*exp(-i*bet.'*xc).*retsinc(l*bet.'/(2*pi),1);
Hb(ii,:)=Hb(ii,:)+(abs(l)/2)*exp(i*bet*xc).*retsinc(-l*bet/(2*pi),1)/muyb(ii+1); 
end;

for ii=1:mh;
xc=xch(ii);l=xh(ii+1)-xh(ii);
Eh(:,ii)=Eh(:,ii)+abs(l)/(2*d)*exp(-i*bet.'*xc).*retsinc(l*bet.'/(2*pi),1);    
Hh(ii,:)=Hh(ii,:)+(abs(l)/2)*exp(i*bet*xc).*retsinc(-l*bet/(2*pi),1)/muyh(ii);    
xc=(xh(ii+1)+xh(ii+2))/2;l=xh(ii+1)-xh(ii+2);
Eh(:,ii)=Eh(:,ii)+abs(l)/(2*d)*exp(-i*bet.'*xc).*retsinc(l*bet.'/(2*pi),1);
Hh(ii,:)=Hh(ii,:)+(abs(l)/2)*exp(i*bet*xc).*retsinc(-l*bet/(2*pi),1)/muyh(ii+1);    
end;


Hh=Hh*ah{7};Hb=Hb*ab{7};
if ~isempty(si);Eh=si{2}*Eh;Eb=si{2}*Eb;end;
% matrice G
s={[[zeros(n),-Eh*aa(1:mh,mh+1:mh+mb)*Hb];[eye(n),Eb*aa(mh+1:mh+mb,mh+1:mh+mb)*Hb]];[[eye(n),-Eh*aa(1:mh,1:mh)*Hh];[zeros(n),Eb*aa(mh+1:mh+mb,1:mh)*Hh]];n;n;n;n};
if init{end}.sog==1;s=retgs(s);end;% passage en S
s_elf={s,Hh,Hb,Eh,Eb};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%                  ELEMENTS FINIS    2  D                %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [s,s_elf]=retelf_c_2D(init,a_elf,ah,ab);
%a_elf={a,aa,maillage,Ah1.',Ah2,Ab1.',Ab2,Bh1.',Bh2,Bb1.',Bb2,Ch1.',Ch2,Cb1.',Cb2,si_elf,ssi_elf,si_bord,ssi_bord,alphax,alphay,EH,ep,domaines_h,domaines_b,struct('dim',1,'type',3,'sog',sog)};
%       1 2     3      4     5   6     7   8     9   10   11   12    13  14    15  16      17     18       19      20     21    22 23     24         25          end
a_elf=retio(a_elf);ah=retio(ah);ab=retio(ab);si=init{8};n=init{1};d=init{end}.d;bet=init{2};
aa=retio(a_elf{2});maillage=a_elf{3};
Ah1=a_elf{4};Ah2=a_elf{5};Ab1=a_elf{6};Ab2=a_elf{7};
Bh1=a_elf{8};Bh2=a_elf{9};Bb1=a_elf{10};Bb2=a_elf{11};
Ch1=a_elf{12};Ch2=a_elf{13};Cb1=a_elf{14};Cb2=a_elf{15};
domaines_h=a_elf{24};domaines_b=a_elf{25};pol=a_elf{end}.pol;
clear a_elf;

%%%%%%%%%%%%%%%%%%%%%%%%%%
%   LIAISON A FOURIER    %
%%%%%%%%%%%%%%%%%%%%%%%%%%

nfourier=init{end}.nfourier;
mh=maillage.nb_sym.nh_sym; % degres de liberte  en haut compte tenu des symetries
mb=maillage.nb_sym.nb_sym;  % degres de liberte  en bas compte tenu des symetries
[Hh,Eh]=calHE(maillage.haut,maillage.x,Ah1,Ah2,Bh1,Bh2,Ch1,Ch2,ah,d,nfourier,bet,domaines_h,pol);
[Hb,Eb]=calHE(maillage.bas,maillage.x,Ab1,Ab2,Bb1,Bb2,Cb1,Cb2,ab,d,nfourier,bet,domaines_b,pol);

%figure;subplot(2,2,1);retcolor(abs(Eh));subplot(2,2,2);retcolor(abs(Eb));subplot(2,2,3);retcolor(abs(Hh));subplot(2,2,4);retcolor(abs(Hb));stop

if pol==0;
if ~isempty(si);Eh=si{2}*Eh;Eb=si{2}*Eb;Hh=Hh*si{3};Hb=Hb*si{3};end;
s={[[zeros(n),-Eh*aa(1:mh,mh+1:mh+mb)*Hb];[eye(n),Eb*aa(mh+1:mh+mb,mh+1:mh+mb)*Hb]];[[eye(n),-Eh*aa(1:mh,1:mh)*Hh];[zeros(n),Eb*aa(mh+1:mh+mb,1:mh)*Hh]];n;n;n;n};
else;
if ~isempty(si);Eh=si{4}*Eh;Eb=si{4}*Eb;Hh=Hh*si{1};Hb=Hb*si{1};end;
s={[[-Eh*aa(1:mh,mh+1:mh+mb)*Hb,zeros(n)];[Eb*aa(mh+1:mh+mb,mh+1:mh+mb)*Hb,eye(n)]];[[-Eh*aa(1:mh,1:mh)*Hh,eye(n)];[Eb*aa(mh+1:mh+mb,1:mh)*Hh,zeros(n)]];n;n;n;n};
end;
if init{end}.sog==1;s=retgs(s);end;% passage en S
s_elf={s,Hh,Hb,Eh,Eb};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [H,E]=calHE(maillage,X,A1,A2,B1,B2,C1,C2,a,d,nfourier,bet,domaines,pol);bet=bet/(2*pi);
me=maillage.me;
if pol==0;fhx=a{9};fhy=a{11};else;fhx=a{8};fhy=a{10};end; % si pol=2, H signifie en fait E, et E signifie  H
xx=a{12}{1};yy=a{12}{2};uu=a{12}{3};clear a;
x=[X(maillage.noeuds(:,1),1),X(maillage.noeuds(:,2),1),X(maillage.noeuds(:,3),1)].';
y=[X(maillage.noeuds(:,1),2),X(maillage.noeuds(:,2),2),X(maillage.noeuds(:,3),2)].';

[tf_triangles_x2,tf_triangles_y2,du_tf_triangles2,dv_tf_triangles2]=cal_tftriangles(x,y,-bet(1,:),-bet(2,:),domaines);
centres=[sum(x,1)/3;sum(y,1)/3];% centres des triangles
%calcul de eps et mu et tranches
ee=zeros(6,me);
xxx=centres(1,:)/d(1);yyy=centres(2,:)/d(2);
mx=length(xx);my=length(yy);xx=[0,xx];yy=[0,yy];
for ii=1:mx;for jj=1:my;
f=find((xxx>=xx(ii))&(xxx<=xx(ii+1))&(yyy>=yy(jj))&(yyy<=yy(jj+1)));
for kk=1:6;ee(kk,f)=uu(ii,jj,kk);end;
end;end;
if pol==0;kpol=0;else;kpol=3;end;
Hx_tf=zeros(me,nfourier);Hx_du_tf=zeros(me,nfourier);
for ii=1:my; % Hx
f=find((yyy>=yy(ii))&(yyy<=yy(ii+1)));
if ~isempty(f);fff=retio(fhx{ii});mm=length(f);
Hx_tf(f,:)=Hx_tf(f,:)+spdiags(1./ee(1+kpol,f).',0,mm,mm)*tf_triangles_y2(f,:)*fff;
Hx_du_tf(f,:)=Hx_du_tf(f,:)+spdiags(1./ee(1+kpol,f).',0,mm,mm)*du_tf_triangles2(f,:)*fff;
end;end;
Hx_tf=B1*Hx_tf;clear B1 du_tf_triangles_x2 du_tf_triangles_y2;
Hx_du_tf=A1*Hx_du_tf;

Hy_tf=zeros(me,nfourier);Hy_dv_tf=zeros(me,nfourier);

for ii=1:mx; % Hy
f=find((xxx>=xx(ii))&(xxx<=xx(ii+1)));mm=length(f);
if ~isempty(f);fff=retio(fhy{ii});
Hy_tf(f,:)=Hy_tf(f,:)+spdiags(1./ee(2+kpol,f).',0,mm,mm)*tf_triangles_x2(f,:)*fff;
Hy_dv_tf(f,:)=Hy_dv_tf(f,:)+spdiags(1./ee(2+kpol,f).',0,mm,mm)*dv_tf_triangles2(f,:)*fff;
end;end;

Hy_tf=C1*Hy_tf;clear C1 tf_triangles2 du_tf_triangles2;
Hy_dv_tf=A1*Hy_dv_tf;clear A1;
H=[-(1/(2i*pi))*Hx_du_tf+Hx_tf,-(1/(2i*pi))*Hy_dv_tf+Hy_tf];
clear Hy_tf Hx_du_tf Hy_dv_tf

[tf_triangles_x1,tf_triangles_y1,du_tf_triangles1,dv_tf_triangles1]=cal_tftriangles(x,y,bet(1,:),bet(2,:),domaines);

C2=tf_triangles_x1.'*C2;B2=-tf_triangles_y1.'*B2;clear tf_triangles1;
B2=B2+(1/(2i*pi))*du_tf_triangles1.'*A2;
C2=C2-(1/(2i*pi))*dv_tf_triangles1.'*A2;clear A2;
E=[C2;B2]/prod(d);

if pol==0;H=i*H;else;E=-i*E;end;  % reticolo rcwa 2 D n'est pas ( encore !) clone 
% H va de fourier a elf  E de elf a fourier d'ou le i et le -i
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [tf_triangles_x,tf_triangles_y,du_tf_triangles,dv_tf_triangles]=cal_tftriangles(x,y,ux,uy,domaines);
tf_triangles_x=zeros(length(x),length(ux));tf_triangles_y=zeros(length(x),length(ux));du_tf_triangles=zeros(length(x),length(ux));dv_tf_triangles=zeros(length(x),length(ux));

for k=1:length(domaines);lx=domaines(k).lx;ly=domaines(k).ly;if lx==0;lx=1;end;if ly==0;ly=1;end;
i0=(length(domaines(k).bx)-1)/2;j0=(length(domaines(k).by)-1)/2;
f=domaines(k).f;
for ii=-i0:i0;[tf,gtf]=rettfsimplex(x(:,f),y(:,f),ux+ii/lx,uy);
tf_triangles_x(f,:)=tf_triangles_x(f,:)+domaines(k).bx(ii+i0+1)*squeeze(tf);
dv_tf_triangles(f,:)=dv_tf_triangles(f,:)+domaines(k).bx(ii+i0+1)*squeeze(gtf(:,1,:,2));
end;
for jj=-j0:j0;[tf,gtf]=rettfsimplex(x(:,f),y(:,f),ux,uy+jj/ly);
tf_triangles_y(f,:)=tf_triangles_y(f,:)+domaines(k).by(jj+j0+1)*squeeze(tf);
du_tf_triangles(f,:)=du_tf_triangles(f,:)+domaines(k).by(jj+j0+1)*squeeze(gtf(:,1,:,1));
end;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%                  cylindrique radial                    %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function s=retc_cylindrique_radial(a,deltar,r1);
a=retio(a);if r1==0;s=rets1(a{end}.nfourier,a{end}.sog);return;end;
r2=r1+deltar;
parm=a{end};L=parm.L;[Q_HE,Q_HH,Q_EH,Q_EE,D,P_E,P_H]=deal(a{1:7});clear a;
ne=size(Q_EE,1);nh=size(Q_HE,1);n=ne+nh;

if deltar>eps;%%%%%%%%%%%%%%%%%%%
if ne==0;H1_E=zeros(0,2,3);H2_E=zeros(0,2,3);else;
H1_E=retbessel('h',[L-1,L,L+1],1,retcolonne(1i*D(1:ne)*[r1,r2]),1);H1_E=reshape(H1_E,ne,2,3);
H2_E=retbessel('j',[L-1,L,L+1],retcolonne(1i*D(1:ne)*[r1,r2]),1);H2_E=reshape(H2_E,ne,2,3);
end;
if nh==0;H1_H=zeros(0,2,3);H2_H=zeros(0,2,3);else;
H1_H=retbessel('h',[L-1,L,L+1],1,retcolonne(1i*D(ne+1:ne+nh)*[r1,r2]),1);H1_H=reshape(H1_H,nh,2,3);
H2_H=retbessel('j',[L-1,L,L+1],retcolonne(1i*D(ne+1:ne+nh)*[r1,r2]),1);H2_H=reshape(H2_H,nh,2,3);
end;


% premier indice vp
% second indice r1 ou r2
% troisieme indice [L-1,L,L+1]

M=cell(1,2);
for ii=1:2;
M{ii}=[[zeros(nh,ne),P_H*retdiag(H1_H(:,ii,2)),zeros(nh,ne),P_H*retdiag(H2_H(:,ii,2))];...
[Q_EE*retdiag(H1_E(:,ii,1)-H1_E(:,ii,3)),Q_EH*retdiag(H1_H(:,ii,1)+H1_H(:,ii,3)),Q_EE*retdiag(H2_E(:,ii,1)-H2_E(:,ii,3)),Q_EH*retdiag(H2_H(:,ii,1)+H2_H(:,ii,3))];...
[P_E*retdiag(H1_E(:,ii,2)),zeros(ne,nh),P_E*retdiag(H2_E(:,ii,2)),zeros(ne,nh)];...
[Q_HE*retdiag(H1_E(:,ii,1)+H1_E(:,ii,3)),Q_HH*retdiag(H1_H(:,ii,1)-H1_H(:,ii,3)),Q_HE*retdiag(H2_E(:,ii,1)+H2_E(:,ii,3)),Q_HH*retdiag(H2_H(:,ii,1)-H2_H(:,ii,3))]];
end;
% 
% G1={speye(2*n);M{1}*retdiag([ones(n,1);exp(-D*(r2-r1))]);n;n;n;n};
% G2={M{2}*retdiag([exp(-D*(r2-r1));ones(n,1)]);speye(2*n);n;n;n;n};


if parm.sog;
M{1}(:,n+1:2*n)=M{1}(:,n+1:2*n)*retdiag(exp(-abs(real(D))*(r2-r1)));	
M{2}(:,1:n)=M{2}(:,1:n)*retdiag(exp(-D*(r2-r1)));
%s={[M{2}(1:n,:);M{1}(n+1:2*n,:)]/([M{1}(1:n,:);M{2}(n+1:2*n,:)]);n;n;1};
s={retslash([M{2}(1:n,:);M{1}(n+1:2*n,:)],[M{1}(1:n,:);M{2}(n+1:2*n,:)]);n;n;1};

%s=retss(retgs(G2),retgs(G1));
else;
G1={speye(2*n);M{1}*retdiag([ones(n,1);exp(-abs(real(D))*(r2-r1))]);n;n;n;n};
G2={M{2}*retdiag([exp(-D*(r2-r1));ones(n,1)]);speye(2*n);n;n;n;n};
s=retss(G2,G1);
end;
else;        %%%%%%%%%%%%%%%%%%%
if parm.sog;s={eye(2*n);n;n;1};else;s={eye(2*n);eye(2*n);n;n;n;n};end;
end;         %%%%%%%%%%%%%%%%%%%


