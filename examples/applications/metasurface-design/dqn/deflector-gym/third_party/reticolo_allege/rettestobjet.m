function [ee,xy]=rettestobjet(init,w,parm,nn,xy,calo,met);
% calcul,tracé de eps et mu definis par w maillage de texture ou de tronçon
%
% 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	%            objet defini par un maillage de texture u               %
% 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [ee,xy]=rettestobjet(init (ou d),u (ou a),parm,nn,xy,calo,met);
%  remarque:u peut etre remplace en entree par le descripteur de texture a de la couche 
%..........................................................................................
%  en 1D;
%  nn nombre de points en x   (par defaut 100 ) et eventuellement offset(par defaut 0 )
%  ee: valeur des eps mux muy aux points x  (size(ee)=[length(x),3])
%..........................................................................................
%  en 2D;
%  nn nombre de points en xy={x,y}  (par defaut [101,101] )  et eventuellement offset (par defaut [0,0] )
%  ee: valeur des mu et eps au maillage x,y  (size(ee)=[length(x),length(y),6])
%..........................................................................................
% si xy est en entree: calcul en ces points ( alors nn ne sert pas)
%
% si on veut seulement le tracé de l'objet (pas de variable en sortie), le programme détermine les valeurs de x et y 
% en fonction des discontinuités de l'objet. nn et xy ne servent qu'à definir les limites du tracé 
% parm pilote le tracé
% real(parm) : 0 couleur   1 gris    -1:pas de trace  (par defaut 0)
% imag(parm) 0: tracé de eps et mu  (réel et imag :3 courbes en 1D , 12 images en 2 D) 
%            positive: tracé  de Re(n) 
%            negative: tracé  de Re(n) et Im(n) 
%    0  eps et mu en couleurs          1 eps et mu en gris               -1 pas de tracé    
%    i  Re(n)  en couleurs             1+i Re(n) en gris                 -1+i pas de tracé    
%   -i  Re(n) et Im(n) en couleurs     1-i Re(n) et Im(n)  en gris       -1-i pas de tracé    
%
%     calo   tableau des composantes de o desirees  dans ee   
%                                 en 2D (1 a 6):mux,muy,muz,epx,epy,epz        
%                                 en 1D (1 a 3):eps,mux,muy  ( en fait  eps,muy,mux du 2D )      
%            ou bien si calo=i:     o=indice*k0   -i indice                  defaut:  i
%                           (l'option -i n'est pas active en 1D: on a indice*k0)
% met: pour les metaux
%     pour les metaux electriques:mu=1 ,si met==0 eps=2i*ceil(5*rand)  sinon eps=met*i
%     pour les metaux magnetiques:eps=1 ,si met==0 mu=2i*ceil(5*rand)  sinon mu=met*i
%  par defaut met=0
%
% 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	%         objet complexe defini par un maillage de tronçon:          %
% 	%         w=retautomatique(...                                       %
% 	%         ou w={a,tab} a et tab étant les parametres de retchamp     %
% 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  hand=rettestobjet(init(ou d), w ou {a,tab} ,parm,nn,xy);
%        dans ce cas pas de calcul de eps et mu mais seulement tracé 
% nn xy  même fonction que dans le cas du maillage de tronçon,
% 	le programme détermine les valeurs de x et y en fonction des discontinuités de l'objet. 
% 	nn et xy ne servent qu'à definir les limites du tracé 
%
%  en 1 D:    parm parametre de retcolor
%
%  en 2 D:    parm(1)=0 (par defaut) figure 3D  , sinon coupes ( obsolete voir ci dessous les nouvelles options des coupes)
%  parm(2)  -1 un subplot par indice, 
%           -2 une figure par indice,
%		   -1.5 une figure 'fermée' avec tous les indices en excluantles indices donnés par  parm(3,..,end) 
%                             ( souvent l'air sinon on a un parallepipede)
% pour les coupes:si parm est un cell array
%    parm{1}: x ,parm{2}: y, parm{3}: z,des coupes ( definies comme dans slice )
% il est possible d'avoir plusieurs coupes.Si une seule coupe, view est determiné pour avoir une figure 2D 
%
% hand permet de changer les couleurs des indices (set(hand(2),'facecolor','b'))
%
%
%% EXEMPLES 1 D
%% Maillage de texture
% d=1.5;u=retu(d,{[-.5,.5],[1,.1+5i],0,2*pi/1.5});u=retautomatique(d,u,[-.7,.7],[5+5i,1]);
% rettestobjet(d,u,i,[2,-d/2]);
%%                 ^
%%    tracé de reel(indice)
%% Maillage de tronçon
%  [d,sym,cao,w]=retw1(struct('pol',2,'sens',-1,'nrangs',3));
%    rettestobjet(d,w,[],[101,-d/2]);%   trace recentre en x
%
%% EXEMPLES 2 D
%% Maillage de texture
% d=[1,.8];u=retu(d,{[-d(2)/2;-.17],[1,1.44],[0,0,.5,.34,3.48,1],2*pi/1.5});
% rettestobjet(d,u,i,[2,2,-d/2]);
%%                 ^
%%    trace de reel(indice)
%% Maillage de tronçon
%  [d,sym,cao,w]=retw1(struct('sens',-1,'h',.2,'nb',1.5,'nrangs',3));
%    rettestobjet(d,w,[0,-1.5,1 ],[2,2,-d(1)/2,-d(2)/2]);%   tracé recentre en x et y 
% %                       ^
% %                  tous les indices sur une meme figure,fermée,sans representer l'air (option recommandée)
%    rettestobjet(d,w,[0,-1],[2,2,-d/2]);      
% %                     ^
% %                 tous les indices sur une meme figure
%    rettestobjet(d,w,{[],0,[]},[2,2,-d/2]);%   
% %                       ^
% %                   coupe y=0
%    hand=rettestobjet(d,w,[],[],{[0,3*d(1)],[-d(2)/2,d(2)/2]});set(hand(2),'facecolor','b')% tracé sur 3 periodes en x changement de couleur de l'indice 2
% %                         ^
% %        une seule figure avec tous les indices sur le meme tracé
%    rettestobjet(d,w,[0,-2],[2,2,-d/2]);
% %                     ^
% %                 1 figure par indice
% %
% %
%
% % See also: SLICE RETAUTOMATIQUE RETCOLOR RETCHAMP

if ~iscell(init);init=retinit(init,[0,0,0,0]);end; % si on entre d a la place de init ...

w=retio(w);

if nargin<7;met=[];end;if isempty(met);met=0;end;
if nargin<6;calo=i;end;if isempty(calo);calo=i;end;
if nargin<3;parm=0;end;
if nargin<4;nn=[];end;
if nargin<5;xy=[];end;

if iscell(w{1});ee=ret2testobjet(init,w,parm,nn,xy);xy=[];return;end;% Maillage de tronçon..    
font=retfont;font{2}='tex';

if init{end}.dim==2;  %%%%%%%%%%%%%      2D    %%%%%%%%%%%%%%%%%%%%%%%%
if isempty(nn)|nn==0;nn=[101,101];end;

d=init{end}.d;    
if size(w,2)>5; % le parametre w est en fait a
switch w{end}.type
case 2;w=w{7};% metaux
case 6;w=w{7};% cylindres
otherwise;w=w{12}; % dielectriques
end;
end;

if size(w,2)>3;% calcul de w pour les metaux
epz=ret2ep(0);epun=ret2ep(1);if w{4}==0;epinf=[1;1;1;10i;10i;10i];else;epinf=[10i;10i;10i;1;1;1];end;
if isempty(w{1});w{1}=[0,0];w{2}=[1,1];w{3}=epinf.';end;% metal massif
ww=retu(epinf);
w{2}(:,1)=min(d(1),w{2}(:,1));w{2}(:,2)=min(d(2),w{2}(:,2));
for ii=1:size(w{1},1);
ww=retu(init,[w{1}(ii,1)-w{2}(ii,1)/2,w{1}(ii,1)+w{2}(ii,1)/2],[epz,w{3}(ii,:).'],[w{1}(ii,2)-w{2}(ii,2)/2,w{1}(ii,2)+w{2}(ii,2)/2],[epz,epun],ww);
end;    
w=ww;  % fin calcul de w pour les metaux    
end;

xx=w{1};yy=w{2};uu=w{3};
mx=nn(1);my=nn(2);if length(nn)==4;x0=nn(3);y0=nn(4);else x0=0;y0=0;end;

if nargin<5;xy=[];end;if~isempty(xy);x=xy{1};y=xy{2};else;x=x0+linspace(0,d(1),mx);y=y0+linspace(0,d(2),my);xy={x,y};end;


if nargout==0;tol=1.e-10;% juste tracé: on redetermine x et y fonction des discontinuites
xmin=min(xy{1});xmax=max(xy{1});ymin=min(xy{2});ymax=max(xy{2});
xxx=[0,mod(xmin,d(1)),mod(xmax,d(1))];yyy=[0,mod(ymin,d(2)),mod(ymax,d(2))];
xxx=[xxx,xx*d(1)];yyy=[yyy,yy*d(2)];xxx=retelimine(xxx,1.e-10);yyy=retelimine(yyy,1.e-10);
x=[];for ii=floor(xmin/d(1)):ceil(xmax/d(1))-1;x=[x,xxx+ii*d(1)];end;x=retelimine(x,1.e-10);x=sort([xmin,xmax,x+tol*d(1),x-tol*d(1)]);x=x(x>=xmin & x<=xmax);
y=[];for ii=floor(ymin/d(2)):ceil(ymax/d(1))-1;y=[y,yyy+ii*d(2)];end;y=retelimine(y,1.e-10);y=sort([ymin,ymax,y+tol*d(2),y-tol*d(2)]);y=y(y>=ymin & y<=ymax);
met=nan;
end;



if init{end}.genre~=1;% <<< fourier cylindrique radial pas Popov;
ee=zeros(length(x),length(y),6);
xxx=mod(x./d(1),1);yyy=mod(y./d(2),1);

indx=zeros(size(xxx));indy=zeros(size(yyy));
for ii=1:length(xx);if ii==1 x0=0;else x0=xx(ii-1);end;indx((xxx>=x0)&(xxx<=xx(ii)))=ii;end;   
for jj=1:length(yy);if jj==1 y0=0;else y0=yy(jj-1);end;indy((yyy>=y0)&(yyy<=yy(jj)))=jj;end; 
ee=uu(indx,indy,:);

% for ii=1:length(xx);if ii==1 x0=0;else x0=xx(ii-1);end;
% fx=find((xxx>=x0)&(xxx<=xx(ii)));    
% for jj=1:length(yy);if jj==1 y0=0;else y0=yy(jj-1);end;
% fy=find((yyy>=y0)&(yyy<=yy(jj)));
% %for kk=1:6;ee(fx,fy,kk)=uu(ii,jj,kk);end;
% ee(fx,fy,1:6)=uu(ii(ones(size(fx))),jj(ones(size(fy))),1:6);
% end;end;% ii jj

else;              % <<< bessel Popov
uu=w{2};Pml=init{10};	
[X,Y]=ndgrid(x,y);R=sqrt(X.^2+Y.^2);R=R(:);
if ~isempty(Pml);% Pml reelles
f=find(R>0);
R(f)=retinterp_popov(R(f),Pml,2);% R numerique
end;

r=[inf,xx,0];
ee=zeros(length(x)*length(y),6);
for ii=1:length(r)-1;jj=find((R>=r(ii+1)) & (R<r(ii)));ee(jj,:)=repmat(uu(:,ii).',length(jj),1);end;
ee=reshape(ee,[length(x),length(y),6]);
end;               % <<<

f=find(ee==10i);if ~isempty(f);if met==0;ee(f)=.2*ceil(5*rand(size(f))).*ee(f);else;ee(f)=met*i;end;end;
if real(parm)~=-1;figure('color','w');
text={'\mu_x';'\mu_y';'\mu_z';'\epsilon_x';'\epsilon_y';'\epsilon_z'};
if imag(parm)==0;% <<<<<<
kkk=0;
for kk=1:6;
kkk=kkk+1;retsubplot(3,4,kkk);retcolor(x,y,real(ee(:,:,kk)).',real(parm));title(['\Ree ',text{kk}],font{:},'fontsize',12);set(gca,font{3:end});
if kkk==9;xlabel('x',font{:});ylabel('y',font{:});end;if kkk<8;set(gca,'xtick',[]);end;
kkk=kkk+1;retsubplot(3,4,kkk);retcolor(x,y,imag(ee(:,:,kk)).',real(parm));title(['\Imm ',text{kk}],font{:},'fontsize',12);set(gca,font{3:end});
if kkk==9;xlabel('x',font{:});ylabel('y',font{:});end;if kkk<8;set(gca,'xtick',[]);end;
end
else;              % <<<<<<
if imag(parm)>0;
retcolor(x,y,real(sqrt(ee(:,:,4)./ee(:,:,1))).',real(parm));title('Re(n)',font{:});set(gca,font{3:end});
%retcolor(x,y,real(sqrt(prod(sqrt(ee(:,:,[1,2,4,5])),3))).',real(parm));title('real indice');
%retcolor(x,y,real(sqrt(ee(:,:,3).*ee(:,:,6))).',real(parm));title('real indice');
end
if imag(parm)<0;
% retsubplot(2,2,3);retcolor(x,y,real(sqrt(prod(sqrt(ee(:,:,[1,2,4,5])),3))).',real(parm));title('real indice');
% retsubplot(2,2,4);retcolor(x,y,imag(sqrt(prod(sqrt(ee(:,:,[1,2,4,5])),3))).',real(parm));title('imag indice');
retsubplot(2,2,1);retcolor(x,y,real(sqrt(ee(:,:,4)./ee(:,:,1))).',real(parm));title('\Ree(n)',font{:});set(gca,font{3:end});
retsubplot(2,2,2);retcolor(x,y,imag(sqrt(ee(:,:,4)./ee(:,:,1))).',real(parm));title('\Imm(n)',font{:});set(gca,font{3:end});
end
end;              % <<<<<<
drawnow;
end;

switch(calo(1));
case i;ee=sqrt(prod( sqrt(ee(:,:,[1,2,4,5])),3));
case -i;ee=sqrt(ee(:,:,4)./ee(:,:,1));
otherwise;ee=ee(:,:,calo);
end

else;  %%%%%%%%%%%%%      1D    %%%%%%%%%%%%%%%%%%%%%%%%
d=init{end}.d;

if size(w,2)>5; % le parametre w est en fait a
if w{end}.type==2;w=w{7};% metaux
else;    % dielectriques
w=w{8};    
end;
end;

if size(w,2)>3; % calcul de w pour les metaux
if w{4}==0;epinf=[10i;1;1];else;epinf=[1;10i;-.1i];end;
%epinf=retep(-i);
w{2}=min(d,w{2});
if isempty(w{1});w{1}=0;w{2}=1;w{3}=epinf.';end;% metal massif
ww=[];eep=[];
for ii=1:size(w{1});
ww=[ww,w{1}(ii)-w{2}(ii)/2,w{1}(ii)+w{2}(ii)/2];
eep=[eep,epinf,w{3}(ii,:).'];
end;
w=retu(init,ww,eep);
end;  % fin calcul de w pour les metaux    

xx=w{1};ep=w{2};
if isempty(xy);
if isempty(nn)|nn==0;nn=100;end;
mx=nn(1);if length(nn)==2;x0=nn(2);else;x0=0;end;
xy=x0+linspace(0,d,mx);
end;

if parm~=-1;figure('color','w');  % <<<<< parm~=-1 tracé: on calcule x pour le tracé
n_periodes=2+floor(xy(2)-xy(1))/d;
X=0;EE=ep(:,1);
for ii=1:length(xx);X=[X,X(end),xx(ii)];EE=[EE,ep(:,ii),ep(:,ii)];end;%x=[x,x(end)];ee=[ee,ep(:,1)];
x=X;ee=EE;for ii=1:n_periodes-1;x=[x,X+ii*d];ee=[ee,EE];end;  % np periodes    
ee(3,:)=1./ee(3,:);
text={'k0*\epsilon';'k_0*\mu_x';'k_0*\mu_y'};
x0=xy(1);x1=xy(end);xx0=mod(x0,d);nd=round((x0-xx0)/d);  % offset
x=x+nd*d;% offset

if imag(parm)==0;
kkk=0;
for kk=1:3;
kkk=kkk+1;retsubplot(3,1,kkk);plot(x,real(ee(kk,:)),'-r',x,imag(ee(kk,:)),'--b','linewidth',2);box off;title(text{kk},font{:},'fontsize',14);set(gca,font{3:end});xlim([x0,x1]);
if kk==1;set(legend('Re','Im'),font{:});end;
end
end
if imag(parm)>0;
plot(x,real(sqrt(ee(1,:).*ee(3,:))),'-k','linewidth',2);box off;title('\Ree(k0*n)',font{:});set(gca,font{3:end});xlim([x0,x1]);
end
if imag(parm)<0;
retsubplot(2,1,1);plot(x,real(sqrt(ee(1,:).*ee(3,:))),'-k','linewidth',2);box off;title('\Ree(k0*n)',font{:});set(gca,font{3:end});xlim([x0,x1]);
retsubplot(2,1,2);plot(x,imag(sqrt(ee(1,:).*ee(3,:))),'-k','linewidth',2);box off;title('\Imm(k0*n)',font{:});set(gca,font{3:end});xlim([x0,x1]);
end;
end;  %else                              % <<<<<<< parm=-1 pas tracé

if nargout>0; % nargout>0;
k3=size(ep,1);	
xxx=mod(xy,d);
ee=zeros(length(xy),1);ee(:,1)=1;
for ii=2:length(xx);ee((xxx<=xx(ii))&(xxx>=xx(ii-1)),1)=ii;end;
%for ii=1:size(ee,1);ee(ii,:)=[ep(1,ee(ii,1)),ep(2,ee(ii,1)),1/ep(3,ee(ii,1))];end
if k3==3;ee=[ep(1,ee(:,1)).',ep(2,ee(:,1)).',1./ep(3,ee(:,1)).'];else;ee=[ep(1,ee(:,1)).',ep(2,ee(:,1)).',1./ep(3,ee(:,1)).',ep(4,ee(:,1)).'];end


end;        % nargout>0;
%end;                               % <<<<<<< parm=-1 ?
if nargout==0;return;end;
f=find(ee==10i);if met==0;ee(f)=.2*ceil(5*rand(size(f))).*ee(f);else;ee(f)=met*i;end;

if all(calo==i)|all(calo==-i);ee=sqrt(ee(:,1).*ee(:,3));else;ee=ee(:,calo);end;;% elimine les pml		

end;  %%%%%%%%%%%%%      1D   2 D ?  %%%%%%%%%%%%%%%%%%%%%%%%

%if nargout==0;ee=[];xy=[];end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maillage de tronçon défini par w=retautomatique(...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h=ret2testobjet(init,w,parm,nn,xy);h=[];
if length(w{1})>2;  % <<<<<<<<<<<<<<<objet défini par par tab et init. On construit le maillage de tronçon
a=w{1};tab=w{2};
if init{end}.dim==2;k=12;else;k=8;end;% pour trouver le maillage de tronçon
ww={};	
for ii=1:size(tab,1);	
if tab(ii,1)>0;ww=[ww,{{tab(ii,1),a{tab(ii,2)}{k}}}];end;	
end;	
h=ret2testobjet(init,ww,parm,nn,xy);return;
end;                % <<<<<<<<<<<<<<



if iscell(parm);Parm=parm;parm=[1,-1];else Parm=[];end;% pour les coupes
tol=1.e-10;
font=retfont;font{2}='tex';

if init{end}.dim==2; % +++++++++++2D
d=init{end}.d;
if isempty(nn);nn=[2,2];end;
%mx=nn(1);my=nn(2);if length(nn)==4;x0=nn(3);y0=nn(4);else x0=0;y0=0;end;
if length(nn)==4;x0=nn(3);y0=nn(4);else x0=0;y0=0;end;
if isempty(xy);x=x0+[eps,d(1)-eps];y=y0+[eps,d(2)-eps];xy={x,y};end;
% determination de xy
xmin=min(xy{1});xmax=max(xy{1});ymin=min(xy{2});ymax=max(xy{2});
xx=[0,mod(xmin,d(1)),mod(xmax,d(1))];yy=[0,mod(ymin,d(2)),mod(ymax,d(2))];
for ii=1:length(w);xx=[xx,w{ii}{2}{1}*d(1)];yy=[yy,w{ii}{2}{2}*d(2)];end;xx=retelimine(xx,1.e-10);yy=retelimine(yy,1.e-10);
xxx=[];for ii=floor(xmin/d(1)):ceil(xmax/d(1))-1;xxx=[xxx,xx+ii*d(1)];end;xxx=retelimine(xxx,1.e-10);xxx=sort([xmin,xmax,xxx+tol*d(1),xxx-tol*d(1)]);xxx=xxx(xxx>=xmin & xxx<=xmax);
yyy=[];for ii=floor(ymin/d(2)):ceil(ymax/d(1))-1;yyy=[yyy,yy+ii*d(2)];end;yyy=retelimine(yyy,1.e-10);yyy=sort([ymin,ymax,yyy+tol*d(2),yyy-tol*d(2)]);yyy=yyy(yyy>=ymin & yyy<=ymax);
xy={xxx,yyy};
% calcul de o 
o=zeros(2*length(w)+2,length(xy{1}),length(xy{2}));	
z=0;hh=0;k=0;
oo=reshape(rettestobjet(d,w{end}{2},-1,0,xy,-i),1,length(xy{1}),length(xy{2}));
k=k+1;o(k,:,:)=oo;
for ii=length(w):-1:1;
z=[z,[hh+tol*w{ii}{1},hh+w{ii}{1}*(1-tol)]];hh=hh+w{ii}{1};
oo=reshape(rettestobjet(d,w{ii}{2},-1,0,xy,-i),1,length(xy{1}),length(xy{2}));
k=k+1;o(k,:,:)=oo;
k=k+1;o(k,:,:)=oo;
end;
z=[z,hh];
oo=reshape(rettestobjet(d,w{1}{2},-1,0,xy,-i),1,length(xy{1}),length(xy{2}));
k=k+1;o(k,:,:)=oo;
else;             % +++++++++++++++ 1 D
	
d=init{end}.d;
if isempty(nn);nn=101;end;
mx=nn(1);if length(nn)==2;x0=nn(2);else;x0=0;end;
if isempty(xy);xy=x0+linspace(0,d,mx);end;
% determination de xy
xmin=min(xy);xmax=max(xy);
xx=[0,mod(xmin,d),mod(xmax,d)];
for ii=1:length(w);xx=[xx,w{ii}{2}{1}];end;xx=retelimine(xx,1.e-10);
xy=[];for ii=floor(xmin/d):ceil(xmax/d)-1;xy=[xy,xx+ii*d(1)];end;xy=retelimine(xy,1.e-10);xy=sort([xmin,xmax,xy+tol*d,xy-tol*d]);xy=xy(xy>=xmin & xy<=xmax);

% calcul de o 
o=zeros(2*length(w)+2,length(xy));	
z=0;hh=0;k=0;
oo=rettestobjet(d,w{end}{2},-1,0,xy,-i).';
k=k+1;o(k,:)=oo;
for ii=length(w):-1:1;
z=[z,[hh+tol*w{ii}{1},hh+w{ii}{1}*(1-tol)]];hh=hh+w{ii}{1};
oo=rettestobjet(d,w{ii}{2},-1,0,xy,-i).';
k=k+1;o(k,:)=oo;
k=k+1;o(k,:)=oo;
end;
z=[z,hh];
oo=rettestobjet(d,w{1}{2},-1,0,xy,-i).';
k=k+1;o(k,:)=oo;
end;  % +++++++++++++++ 1 D ou 2D ?  


figure('color','w');
if isempty(parm);parm=[0,1.5];end;
if length(d)==1;% 1 D
retcolor(xy,z,real(o),parm);axis equal;axis tight;xlabel('X');ylabel('Y');title('\Ree(n)',font{:});set(gca,font{3:end});
return
end;

% 2 D
x=xy{1};y=xy{2};
[X,Z,Y]=meshgrid(x,z,y);ax=[min(X(:)),max(X(:)),min(Z(:)),max(Z(:)),min(Y(:)),max(Y(:))];% axis equal
ll=max(ax([2,4,6])-ax([1,3,5]))/2;ax=([(ax(1)+ax(2))/2-ll,(ax(1)+ax(2))/2+ll,(ax(3)+ax(4))/2-ll,(ax(3)+ax(4))/2+ll,(ax(3)+ax(6))/2-ll,(ax(5)+ax(6))/2+ll,]);

co=retelimine(o);
if length(parm>2);for ii=3:length(parm);co=co(abs(co-parm(ii))>eps);end;end;% elimination de certains indices

nco=length(co);
coo=fix(linspace(15,55,nco));

if parm(1)==0;% <<<<<<<<<<<< figure 3 D
colormap(jet);cmap=colormap;% whitebg([.4,.4,.4]);
rota=real(parm(2));
%stereo=imag(parm(end));rota=real(parm(end));if (rota==-1)|(rota>0);stereo=0;end;
% for k=1:1;
% retsubplot(1,1,k);
h=zeros(1,nco);
for ii=1:nco;
if rota==-1;retsubplot(ceil(sqrt(nco)),ceil(sqrt(nco)),ii);title(['n=',num2str(co(ii))],'fontweigh','bold');end;    
if rota==-2;if ii>1;figure('color','w');end;title(['n=',num2str(co(ii))],'fontweigh','bold');end;    
oo=abs(o-co(ii))<1.e-6;
if rota<0;oo(1,:,:)=0;oo(end,:,:)=0;oo(:,1,:)=0;oo(:,end,:)=0;oo(:,:,1)=0;oo(:,:,end)=0;end;% fermeture

h(ii)=retisosurface(X,Z,Y,oo);
%h(ii)=patch(isosurface(X,Z,Y,oo,.5));
set(h(ii),'AlphaDataMapping','none','facecolor',[cmap(coo(ii),:)],'edgecolor','none','facelighting','flat','ambientstrength',.3,'diffusestrength',.9,'backfacelighting','reverselit','specularexponent',5,'specularcolorreflectance',.5);
axis(ax);
xlabel('x');ylabel('z');zlabel('y');
%camproj('perspective');% deplace les titres ??
if ~(abs(rota)==1.5 & ii>1)
light('position',[100,50,200],'color',[1,1,1]);
light('position',[-100,-50,200],'color',[1,1,1]);
view(15,10);axis equal;axis tight;
end;
end; % ii=1:nco
axis equal;axis tight;
% end;
% if rota>0;
% retsubplot(1,1,1);jj=0;for ii=0:180/rota:180-180/rota;view(22+ii,10+jj);drawnow;end;
% ii=0;for jj=0:180/rota:180-180/rota;view(22+ii,10+jj);drawnow;end;end;


%colordef white;
else % <<<<<<<<<<<< coupes
if isempty(Parm);
slice(X,Z,Y,o,[],z,[]);alpha(.5);
else,
slice(X,Z,Y,real(o),Parm{1},Parm{3},Parm{2});
nxyz=[length(Parm{1}),length(Parm{2}),length(Parm{3})];
if all(nxyz==[1,0,0]);view([90,0]);end;  % 1 coupe perpendiculaire à x
if all(nxyz==[0,1,0]);view([90,90]);end; % 1 coupe perpendiculaire à y
if all(nxyz==[0,0,1]);view([0,0]);end;   % 1 coupe perpendiculaire à z
if sum(nxyz)>1;alpha(.5);end;
end;
shading flat;colorbar;xlabel('x');ylabel('z');zlabel('y');title('n');axis equal;axis tight;
axis([min(X(:)),max(X(:)),min(Z(:)),max(Z(:)),min(Y(:)),max(Y(:))]);

end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h=retisosurface(X,Y,Z,oo);
s=size(oo);nz=s(1);nx=s(2);ny=s(3);

t=zeros((nx-1)*(ny-1)*(nz-1),8);
[Ix,Iy,Iz]=ndgrid(1:nx-1,1:ny-1,1:nz-1);
t(:,1)=retsub2ind(s,Iz(:),Ix(:),Iy(:));
t(:,2)=t(:,1)+nz;        %t(:,2)=sub2ind(s,Iz(:),Ix(:)+1,Iy(:));%% gain de temps
t(:,3)=t(:,1)+nz+nz*nx;  % t(:,3)=sub2ind(s,Iz(:),Ix(:)+1,Iy(:)+1);%% gain de temps
t(:,4)=t(:,1)+nz*nx;     % t(:,4)=sub2ind(s,Iz(:),Ix(:),Iy(:)+1);%% gain de temps
t(:,5)=t(:,1)+1;         % t(:,5)=sub2ind(s,Iz(:)+1,Ix(:),Iy(:));%% gain de temps
t(:,6)=t(:,1)+1+nz;      % t(:,6)=sub2ind(s,Iz(:)+1,Ix(:)+1,Iy(:));%% gain de temps
t(:,7)=t(:,1)+1+nz+nz*nx;% t(:,7)=sub2ind(s,Iz(:)+1,Ix(:)+1,Iy(:)+1);%% gain de temps
t(:,8)=t(:,1)+1+nz*nx;   % t(:,8)=sub2ind(s,Iz(:)+1,Ix(:),Iy(:)+1);%% gain de temps

nb_int=sum(oo(t),2);
particulier=find(nb_int<8 & nb_int>4  );
t_particulier=zeros(length(particulier),4,6);% carres particuliers
kk={[1,2,3,4],[5,6,7,8],[1,2,6,5],[4,3,7,8],[6,2,3,7],[5,1,4,8]};
for k=1:6;t_particulier(:,:,k)=t(particulier,kk{k});end;
t_particulier=reshape(permute(t_particulier,[1,3,2]),[],4);
t_particulier=t_particulier((sum(oo(t_particulier),2)==4),:);

frontiere=find(nb_int==4);
tt=t(frontiere,:);
ooo=(oo(tt)==1);
ttt=reshape(tt(find(ooo)),[],4);
ttt=[ttt;t_particulier];

ttt=ttt(:,[1,2,3,4,1]);,
for k=1:6;f=find(ooo(:,kk{k}(1))&ooo(:,kk{k}(2))&ooo(:,kk{k}(3))&ooo(:,kk{k}(4)));ttt(f,:)=tt(f,[kk{k},kk{k}(1)]);end;
clear t tt oo ooo 
h=patch(X(ttt).',Y(ttt).',Z(ttt).','g');

