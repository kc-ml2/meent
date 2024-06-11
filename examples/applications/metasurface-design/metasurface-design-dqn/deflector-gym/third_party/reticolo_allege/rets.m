function s=rets(init,p,a,polinc,poldif,apod,tf);
% function s=rets(init,p,a,polinc,poldif,apod,tf);
%
%  calcul de la matrice s associée au point source en p
%  s calcule la discontinuitée du champ due à la source
%
%   en 1D :EZ,HX,HY                <-- 3 composantes du vecteur source  ( en H// c'est HZ,-EX,-EY ) 
%     d*  dirac(M-P) dans  rot(E):  HX,HY,0
%     d*  dirac(M-P) dans  rot(H):  0,0,EZ
%
%
%   en 2D :EX,EY,EZ,HX,HY,HZ       <-- 6 composantes du vecteur source
%     d(1)*d(2)*  dirac(M-P) dans  rot(E):  HX,HY,HZ
%     d(1)*d(2)*  dirac(M-P) dans  rot(H):  EX,EY,EZ
%                 ( la multiplication par d pro vient d'une premiere version où d n'etait pas disponible 
%                  dans le programme qui calculait les sources)
%    ces matrices s'utilisent comme les autres matrices S dans les produits 
%
%   polinc,poldif:composantes a garder parmi les composantes incidentes (source) et diffractees (champ)
%  (rettronc(s,polinc,poldif,0)) par defaut pas de troncature si poldif est absent poldif=polinc
%    si polinc a une partie imaginaire: clonage de la source et du champ sur la source et division par prod(d) 
%                                                     Si l'amplitude de la source est un vecteur colonne S,et qu(il n'y ait pas de modes incidents
%                                                     l'energie emise est alors .5*imag(S'*s*S);
%
%    si on a plusieurs sources les composantes incidentes(ou diffractees) correspondantes se concatennent 
%    en commencant par la source la plus a gauche dans les produits
%   remarque:  a peut avoir été stockée sur fichier par retio(a,1)
% apod(1): apodisation des s   apod(2): apodisation des champs
% par defaut apod=[7.25 7.25]  la 'meilleure' apodisation
% %
% Remarque:en 2D il est possible de definir des sources plus generales que des distributions de Dirac 
% avec la fonction tf
%  tf(u) = transformée de Fourier de l'amplitude de la source
% % Exemple: faisceau gaussien en y de waist wy:  source=rets(init,p,[],0,@(bet) exp(-(pi*wy*bet(:,2)).^2));
%% Exemple: problème reseau 2D
%  faisceau gaussien cylindrique en y de waist wy: source=rets(init,p,a,pols+i,[],0,@(bet) exp(-(pi*wy*bet(:,2)).^2));
%% On peut aussi moduler les sources pour obtenir un faisceau gaussien dans les 2 dimensions 
%%   inc=exp(i*beta0*(xs-centre)).*exp(-((xs-centre)/wx).^2); 
% See also: RETHELP_POPOV RETINC

if nargin<6;apod=[7.25 7.25];end;if length(apod)==1;apod=[apod,apod];end;
if nargin<4;polinc=0;end;
if nargin<5;poldif=polinc;end;
if (~isreal(polinc))|(~isreal(poldif));clonage=1;polinc=real(polinc);poldif=real(poldif);else;clonage=0;end;
a=retio(a);
if init{end}.genre==1;s=rets_popov(init,a,polinc,poldif,apod,clonage);return;end;% cylindres Popov
if isfield(init{end},'L')&~isempty(init{end}.L);s=rets_cylindrique_radial(init,p,a,polinc,poldif,apod,clonage);return;end;% cylindrique_radial

n=init{1};bet=init{2}.';m=size(bet,1);z=zeros(m,1);

w0=exp(i*bet*(p.'));

if init{end}.dim==2; %2D
if size(apod,1)==1;apod=[apod;apod];end;	
mx=init{6};my=init{7};% apodisation eventuelle du champ calculé
w0=w0.*reshape(retapod(ones(mx,1),apod(2,2))*retapod(ones(my,1),apod(1,2)).',mx*my,1);w0=w0.';

d=init{end}.d;si=init{8};   
if clonage;clonesource=[-i,-i,-i,1,1,1]/prod(d);clonechamp=[1,1,1,i,i,i];end;
% calcul de w
w=zeros(6,2*n);
xx=a{12}{1};yy=a{12}{2};uu=a{12}{3};pas=a{13};
%calcul de eps , mu ,ii et jj position de la source
xxx=mod(real(p(1))/pas(1),1);yyy=mod(real(p(2))/pas(2),1);
for iii=1:size(xx,2);if iii==1 x0=0;else x0=xx(iii-1);end;
fx=find((xxx>=x0)&(xxx<=xx(iii)));    
for jjj=1:size(yy,2);if jjj==1 y0=0;else y0=yy(jjj-1);end;
fy=find((yyy>=y0)&(yyy<=yy(jjj)));
if ~isempty(fx)&~isempty(fy) ee=uu(iii,jjj,:);ii=iii;jj=jjj;end;
end;
end;
epz=ee(6);muz=ee(3);


%champs continus
hz=retio(a{7});

if ~isempty(hz);  %calcul propre des champs HZ et EZ 
w(6,1:n)=w0*hz;clear hz;%HZ
w(3,n+1:2*n)=w0*retio(a{6});%EZ
else              %calcul approche des champs HZ et EZ 
if ~isempty(si);  %symetries
w(6,1:n)=[-bet(:,2).'.*w0,bet(:,1).'.*w0]*si{1}/muz;%HZ
w(3,n+1:2*n)=[bet(:,2).'.*w0,-bet(:,1).'.*w0]*si{3}/epz;%EZ
else
w(6,1:n)=[-bet(:,2).'.*w0,bet(:,1).'.*w0]/muz;%HZ
w(3,n+1:2*n)=[bet(:,2).'.*w0,-bet(:,1).'.*w0]/epz;%EZ
end
end
fex=retio(a{8});if~isempty(fex);fex=retio(fex{min(jj,end)});end;fey=retio(a{10});if~isempty(fey);fey=retio(fey{min(ii,end)});end;
fhx=retio(a{9});if~isempty(fhx);fhx=retio(fhx{min(jj,end)});end;fhy=retio(a{11});if~isempty(fhy);fhy=retio(fhy{min(ii,end)});end;

if ~isempty(si);  %symetries
if ~isempty(fex);  %calcul propre des champs discontinus 
w(1:2,1:n)=[[w0*fex/ee(4),spalloc(1,m,0)]*si{1};[spalloc(1,m,0),w0*fey/ee(5)]*si{1}];         %  EX EY  
w(4:5,n+1:2*n)=[[w0*fhx/ee(1),spalloc(1,m,0)]*si{3};[spalloc(1,m,0),w0*fhy/ee(2)]*si{3}];     %  HX HY
else; %calcul approche des champs discontinus
w(1:2,1:n)=[[w0,spalloc(1,m,0)]*si{1};[spalloc(1,m,0),w0]*si{1}];                     %  EX EY
w(4:5,n+1:2*n)=[[w0,spalloc(1,m,0)]*si{3};[spalloc(1,m,0),w0]*si{3}];                 %  HX HY
end    
else %pas de symetrie
if ~isempty(fex);  %calcul propre des champs discontinus 
w(1:2,1:2*m)=[[w0*fex/ee(4),spalloc(1,m,0)];[spalloc(1,m,0),w0*fey/ee(5)]];         %  EX EY  
w(4:5,2*m+1:4*m)=[[w0*fhx/ee(1),spalloc(1,m,0)];[spalloc(1,m,0),w0*fhy/ee(2)]];     %  HX HY
else; %calcul approche des champs discontinus
w(1:2,1:2*m)=[[w0,spalloc(1,m,0)];[spalloc(1,m,0),w0]];                             %  EX EY
w(4:5,2*m+1:4*m)=[[w0,spalloc(1,m,0)];[spalloc(1,m,0),w0]];                         %  HX HY
end    
end
clear fex fey fhx fhy;



% calcul de u
aa=exp(-i*bet*(p.'));
if nargin>6;aa=aa.*tf(bet/(2*pi));	end;

% apodisation 
aa=aa.*reshape(retapod(ones(mx,1),apod(2,1))*retapod(ones(my,1),apod(1,1)).',mx*my,1);w0=w0.';


%aa=retapod(aa,apod(1));% BUG


ax=i*bet(:,1).*aa;ay=i*bet(:,2).*aa;
u=[[z,z,-i*ax/epz,z,aa,z];[z,z,-i*ay/epz,-aa,z,z];[z,aa,z,z,z,i*ax/muz];[-aa,z,z,z,z,i*ay/muz]];
if ~isempty(si);u=[si{2}*u(1:2*m,:);si{4}*u(2*m+1:4*m,:)];end;

%%%%%%%%%%%%%%%% 1D
else; 
w0=retapod(w0,apod(2));w0=w0.';% apodisation eventuelle du champ calcule
    
d=init{end}.d;si=init{3};a3=a{6};a1=a{7};wob=a{8};
if clonage;clonesource=[-i,1,1]/d;clonechamp=[1,i,i];end;

k=find(mod(real(p),d)<wob{1});if isempty(k);k=1;end;muy=1/wob{2}(3,k(1));mux=wob{2}(2,k(1));
% attention mux muy correspondent a la notation 1D (en fait c'est muy mux du 2D)
aa=exp(-i*bet*p);if nargin>6;aa=aa.*tf(bet/(2*pi));end;aa=retapod(aa,apod(1));ax=i*bet.*aa;
%calcul de W

   %  E                     HX                    HY
if ~isempty(a3);  %  calcul propre de H
if isempty(si);
w=[[w0,zeros(1,n)];[zeros(1,n),-i*w0*a1/muy];[w0*a3,zeros(1,n)]];
else;
w=[[w0*si{1},zeros(1,n)];[zeros(1,n),-i*w0*a1/muy];[w0*a3,zeros(1,n)]];
end;
else
if isempty(si);
w=[[w0,zeros(1,n)];[zeros(1,n),-i*w0];[-w0.*bet.'/mux,zeros(1,n)]];
else
w=[[w0*si{1},zeros(1,n)];[zeros(1,n),-i*w0*si{1}];[(-w0.*bet.'/mux)*si{1},zeros(1,n)]];
end 
    
end;

% calcul de u
u=[[z,aa,z];[-i*aa,z,-ax/mux]];
if ~isempty(si);u=[si{2}*u(1:m,:);si{2}*u(m+1:2*m,:)];end;

end


% clonage
if clonage;u=u*diag(clonesource);w=diag(clonechamp)*w;end;

% construction de la matrice S ou G associee a la source
sog=init{end}.sog;pp=size(u,2);
if sog==1;  %  matrice S 
%s={[[eye(n),zeros(n,n),u(1:n,:)];[w,zeros(pp,pp)];[zeros(n,n),eye(n),-u(n+1:2*n,:)]];n;n+pp;1};    
s={[[speye(n),spalloc(n,n,0),u(1:n,:)];[w,w*diag([.5*ones(1,n),-.5*ones(1,n)])*u];[spalloc(n,n,0),speye(n),-u(n+1:2*n,:)]];n;n+pp;1};    
else;       % matrice G
%s={[eye(2*n);w];[[eye(n),zeros(n,n+pp),-u(1:n,:)];[zeros(n,n+pp),eye(n),-u(n+1:2*n,:)];[zeros(pp,n),eye(pp),zeros(pp,n+pp)]];n;n;n+pp;n+pp};    
s={[speye(2*n);[w(:,1:n),spalloc(pp,n,0)]];...
[[speye(n),spalloc(n,n+pp,0),-u(1:n,:)];[spalloc(n,n+pp,0),speye(n),-u(n+1:2*n,:)];[spalloc(pp,n,0),speye(pp),-w(:,n+1:2*n),w*retdiag([-.5*ones(n,1);.5*ones(n,1)])*u]];...
n;n;n+pp;n+pp};    
end;    

if ~all(polinc==0)|~all(poldif==0)|isempty(polinc)|isempty(poldif);s=rettronc(s,polinc,poldif,0);end;  % troncature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function s=rets_popov(init,a,polinc,poldif,apod,clonage);

% 
% if nargin<5;apod=[7.125,7.125];end;
% if nargin<3;polinc=0;end;
% if nargin<4;poldif=polinc;end;
% if (~isreal(polinc))|(~isreal(poldif));clonage=1;polinc=real(polinc);poldif=real(poldif);else;clonage=0;end;


n=init{1};N=init{end}.nhanckel;L=init{2};k=init{3};wk=init{4};ep=a{7}{2};
[Ez,epErp,epErm,Hz,muHrp,muHrm,Psim,Psip]=deal(a{6}{:});Ez=Ez*diag(k);Hz=Hz*diag(k);


zero=sparse(N,N);un=speye(N);kw=k.*wk;
Ep=ep(4,end);Mu=ep(1,end);% au centre
if apod(2)~=0;n_apod=200;apod_C=interp1(-n_apod:n_apod,retchamp([apod(2),2*n_apod+1]),k*(n_apod-1)/max(k));kw=kw.*apod_C;end; % Modif 2010 apodisation des champs modif 2010
if apod(1)~=0;n_apod=200;apod_S=interp1(-n_apod:n_apod,retchamp([apod(1),2*n_apod+1]),k*(n_apod-1)/max(k));end; % Modif 2010 apodisation des champs modif 2010
%if apod(1)~=0;apod_S=retchamp([apod(1),2*N-1]);apod_S=apod_S(N:2*N-1);end;
%if apod(2)~=0;apod_C=retchamp([apod(2),2*N-1]);apod_C=apod_C(N:2*N-1);end;% Modif 2010 

switch sign(L);% pour abs(L)>1 les sources sont 'bidon' et peuvent etre utilisées pour chercher des pöles
case -1;
	switch init{end}.sym;
% 	case 0;% test ????????????????
% 		%JLp1*ep+JLm1*em;         % E_teta
% 		%(i./Ep).*fac_r.*(JLp1*epErp*(ep+Psim*em));% E_r
%  	c=[(i/Ep)*kw*epErp*[un,Psim,zero,zero]+i*kw*[un,zero,zero,zero];% E_r+i*E_teta
% 	(i/Mu)*kw*muHrp*[zero,zero,un,Psim]+i*kw*[un,zero,zero,zero]];  % H_r+i*H_teta
% 	case 1;
% 	c=[(i/Ep)*kw*epErp*[un,Psim,zero,zero];%(i*FF./Ep).*(JLp1*epErp*(ep+Psim*em)     % E_x
% 	kw*[zero,zero,un,zero]];           %e(:,5)=FF.*(JLp1*hp+JLm1*hm);                % H_y
% 	case -1;
% 	c=[kw*[un,zero,zero,zero];%  F.*(JLp1*ep+JLm1*em);                               % E_y
% 	(i/Mu)*kw*muHrp*[zero,zero,un,Psim]]; %i*(F./Mu).*(fac*(JLp1*muHrp*(hp+Psim*hm)) % H_x
% 	end;

	case 0;
 	c=[2i*kw*[un,zero,zero,zero];             % E_r+i*E_teta
	2i*kw*[un,zero,zero,zero]];               % H_r+i*H_teta
	case 1;
	c=[i*kw*[un,zero,zero,zero];              % E_x
	kw*[zero,zero,un,zero]];                  % H_r
	case -1;
	c=[kw*[un,zero,zero,zero];                % E_r
	i*kw*[zero,zero,un,zero]];                % H_x
	end;

case 1;	
	switch init{end}.sym;
% 	case 0;% test ????????????????
% 		%JLp1*ep+JLm1*em;        % E_teta
% 		%(-i./Ep).*(JLm1*epErm*(em+Psip*ep));% E_r
%     c=[(-i/Ep)*kw*epErm*[Psip,un,zero,zero]-i*kw*[zero,un,zero,zero]; % E_r-i*E_teta
% 	(-i/Mu)*kw*muHrm*[zero,zero,Psip,un]-i*kw*[zero,zero,zero,un] ];  % H_r-i*H_teta
% 	case 1;
% 	c=[(-i/Ep)*kw*epErm*[Psip,un,zero,zero];%-i*(FF./Ep).*(JLm1*epErm*(em+Psip*ep)); % E_x
% 	kw*[zero,zero,zero,un]];           % FF.*(JLp1*hp+JLm1*hm);                      % H_y
% 	case -1;
% 	c=[kw*[zero,un,zero,zero];%  F.*(JLp1*ep+JLm1*em);                               % E_y
% 	(-i/Mu)*kw*muHrm*[zero,zero,Psip,un]]; %  -i*(F./Mu).*(JLm1*muHrm*(hm+Psip*hp)); % H_x
% 	end;
% 
	case 0;
    c=[-2i*kw*[zero,un,zero,zero];           % E_r-i*E_teta
	-2i*kw*[zero,zero,zero,un] ];            % H_r-i*H_teta
	case 1;
	c=[-i*kw*[zero,un,zero,zero];            % E_r
	kw*[zero,zero,zero,un]];                 % H_teta
	case -1;
	c=[kw*[zero,un,zero,zero];               % E_teta
	-i*kw*[zero,zero,zero,un]];              % H_r
	end;

case 0;
c=[kw*Ez*[zero,zero,un,-un];...%  FF.*(JL*Ez*(hp-hm));  % E_z
kw*Hz*[un,-un,zero,zero]];     %  F.*(JL*Hz*(ep-em)); % H_z

end;
s1=zeros(2*N,2);s2=zeros(2*N,2);
if L==0; %<<<<<<<<<< L=0
s1(:,1)=(-i/(4*pi*Ep))*[k,k].';	
s2(:,2)=(-i/(4*pi*Mu))*[k,k].';
else;
	
if L>0; %<<<<<<<<<< L>0
switch init{end}.sym
case 1;s1(N+1:2*N,2)=i/(2*pi);s2(N+1:2*N,1)=-1/(2*pi);
case -1;s1(N+1:2*N,2)=-1/(2*pi);s2(N+1:2*N,1)=i/(2*pi);
otherwise s1(N+1:2*N,2)=-1/(2*pi);s2(N+1:2*N,1)=-1/(2*pi);
end;
end;

if L<0; %<<<<<<<<<< L<0
switch init{end}.sym
case 1;s1(1:N,2)=-i/(2*pi);s2(1:N,1)=-1/(2*pi);
case -1;s1(1:N,2)=-1/(2*pi);s2(1:N,1)=-i/(2*pi);
otherwise s1(1:N,2)=-1/(2*pi);s2(1:N,1)=-1/(2*pi);
end;
end;
end;

if apod(1)~=0;s1=retdiag([apod_S,apod_S])*s1;s2=retdiag([apod_S,apod_S])*s2;end;% apodisation des sources
% if apod(2)~=0;c=c*retdiag([apod_C,apod_C,apod_C,apod_C]);end; % Modif 2010 l'apodisation du champ duit etre faite avant

if clonage==0;% declonage
c(2,:)=-i*c(2,:);	
s1(:,1)=i*s1(:,1);s2(:,1)=i*s2(:,1);	
end


if init{end}.sog==1;  %  matrice S 
cc=(c(:,1:2*N)*s1-c(:,2*N+1:4*N)*s2)/2;
s={[[speye(n),sparse(n,n),s1];[c,cc];[sparse(n,n),speye(n),-s2]];n;n+2;1};    
else;             % matrice G
cc=(c(:,1:2*N)*s1+c(:,2*N+1:4*N)*s2)/2;
s={[speye(2*n);c];[[speye(n),sparse(n,n+2),-s1];[sparse(n,n+2),eye(n),-s2];[sparse(2,n),speye(2),sparse(2,n),cc]];n;n;n+2;n+2};    
end;    
%if ~all(polinc==0)|~all(poldif==0);s=rettronc(s,polinc,poldif,0);end;  % troncature
if ~all(polinc==0)|~all(poldif==0)|isempty(polinc)|isempty(poldif);s=rettronc(s,polinc,poldif,0);end;  % troncature


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%                  cylindrique radial                    %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function s=rets_cylindrique_radial(init,p,a,polinc,poldif,apod,clonage);
L=init{end}.L;
a=retio(a);

[Q_HE,Q_HH,Q_EH,Q_EE,D,P_E,P_H,N_EE,N_EH,N_HE,N_HH,u,O_E,O_H,Iep,Imu,Ieep,Immu,K]=deal(a{1:19});clear a;
ne=size(Q_EE,1);nh=size(Q_HH,1);r=p(2);p=p(1);
% H1_E=retbessel('h',[L-1,L,L+1],1,retcolonne(1i*D(1:ne)*r),1);
% H1_H=retbessel('h',[L-1,L,L+1],1,retcolonne(1i*D(ne+1:ne+nh)*r),1);
% 
% H2_H=retbessel('j',[L-1,L,L+1],retcolonne(1i*D(ne+1:ne+nh)*r),1);
% H2_E=retbessel('j',[L-1,L,L+1],retcolonne(1i*D(1:ne)*r),1);
si=init{8};
if ~isempty(si);si=si(5:8);end;


bet=init{2}(1,:);
F=exp(1i*bet*p);aa=exp(-1i*bet*p).';

% calcul de epz et muz pour calcul propre de Hz et Ez
z=mod(p/init{end}.d(1),1);
for ii=1:length(u{1});if ii==1 z0=0;else z0=u{1}(ii-1);end;
fz=find((z>=z0)&(z<=u{1}(ii)));
if ~isempty(fz);eepz=1/u{3}(ii,1,4);eep=1/u{3}(ii,1,5);mmuz=1/u{3}(ii,1,1);mmu=1/u{3}(ii,1,2);break;end;% attention c'est bien la composante x et y
end;
 
mx=init{6};
aa=aa.*retchamp([apod(1),mx]).'; % apodisation eventuelle de la source
F=F.*retchamp([apod(2),mx]);    % apodisation eventuelle du champ calculé

% SE=[[sparse(nh,2),-1i*calsym(K*Iep*aa,si,2,nan),sparse(nh,1),calsym(-aa,si,2,nan),sparse(nh,1)];...
%    [sparse(ne,2),(-1i*L/r)*calsym(Iep*aa,si,0,nan),calsym(aa,si,0,nan),sparse(ne,2)]]/init{end}.d(1);
% SH=[[sparse(ne,1),calsym(-aa,si,0,nan),sparse(ne,3),-1i*calsym(K*Imu*aa,si,0,nan)];...
%    [calsym(aa,si,2,nan),sparse(nh,4),(-1i*L/r)*calsym(Imu*aa,si,2,nan)]]/init{end}.d(1);
% 
SE=[[sparse(nh,2),-1i*calsym(eep*bet.'.*aa,si,2,nan),sparse(nh,1),calsym(-aa,si,2,nan),sparse(nh,1)];...
   [sparse(ne,2),(-1i*L/r)*calsym(eep*aa,si,0,nan),calsym(aa,si,0,nan),sparse(ne,2)]]/init{end}.d(1);
SH=[[sparse(ne,1),calsym(-aa,si,0,nan),sparse(ne,3),-1i*calsym(mmu*bet.'.*aa,si,0,nan)];...
   [calsym(aa,si,2,nan),sparse(nh,4),(-1i*L/r)*calsym(mmu*aa,si,2,nan)]]/init{end}.d(1);

CE=[[eepz*F*calsym(Ieep,si,nan,2);sparse(4,nh);(1i*L/r)*F*calsym(Imu,si,nan,2)],...
   [sparse(1,ne);F*calsym(speye(length(F)),si,nan,0);sparse(3,ne);-1i*F*calsym(Imu*K,si,nan,0)]];
CH=[[sparse(2,ne);(1i*L/r)*F*calsym(Iep,si,nan,0);mmuz*F*calsym(Immu,si,nan,0);sparse(2,ne)],...
   [sparse(2,nh);-1i*F*calsym(Iep*K,si,nan,2);sparse(1,nh);F*calsym(speye(length(F)),si,nan,2);sparse(1,nh)]];
if ~clonage;SE(:,1:3)=1i*SE(:,1:3);SH(:,1:3)=1i*SH(:,1:3);CE(4:6,:)=-1i*CE(4:6,:);CH(4:6,:)=-1i*CH(4:6,:);end;
s={[[speye(ne+nh),sparse(ne+nh,ne+nh),SE];[CE,CH,.5*(CE*SE-CH*SH)];[sparse(ne+nh,ne+nh),speye(ne+nh),-SH]];ne+nh;ne+nh+6;1};
if init{end}.sog==0;s=retgs(s,1);end;% matrices G

if ~all(polinc==0)|~all(poldif==0)|isempty(polinc)|isempty(poldif);s=rettronc(s,polinc,poldif,0);end;  % troncature

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a=calsym(a,si,n1,n2);
if isempty(si);return;end;
if isnan(n2);a=si{2+n1}*a;return;end;
if isnan(n1);a=a*si{1+n2};return;end;
a=si{2+n1}*a*si{1+n2};








