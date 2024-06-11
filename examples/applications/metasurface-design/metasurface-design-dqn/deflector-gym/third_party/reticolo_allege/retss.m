function s=retss(varargin);
% function s=retss(s1,s2...,sm,k)   
% produit multiple de matrices S (ou G) 
%  (le produit est fait dans le meme sens qu'un produit de matrices)
% k:choix de l'algorithme (par defaut k=3 ou valeur entree par retss(k) )
% retss sans argument retourne la valeur de k
%
% si ss={s} a un seul element ou ss={s,[]) ou ss={[],s}  on retourne s (utile pour initialiser une boucle) 
%
%
%   matrices S
%  0: generale  full,  1: generale sparse,  2: demi inversion ,    3 ou 4: inversion  ,5:choix entre 3 ou 4 
%  si S{4}=m ~=1 on effectue les produits avec une autre matrice pour eviter l'anomalie lambda/(4n)
%  pour les tres grosses matrices et l'option 3 ou 4   ecriture sur fichiers qui sont ensuite effaces
%    on peut forcer celle option en prenant k=30 ou k=40
%    on empecher celle option en prenant k=300 ou k=400  
%
%  matrices G
%  0: QR  2: LU    3: LU Sparse 1:Pivot(a eviter car tres long..)
%    0,2 et 1 traitent les eventuelles degenerescences (avec svd  peut etre long ...)
%  pour les tres grosses matrices et l'option 3    ecriture sur fichiers qui sont ensuite effaces
%    on peut forcer celle option en prenant k=30
%    on empecher celle option en prenant k=300 ou k=400  
%
%  dans le cas de troncature on commence par le cote ou se trouve la plus petite matrice
%
%  certaines des matrices de ss peuvent avoir ete stockées sur fichiers par retio(s,1)
%   au retour s n'est pas mise sur fichier 
%  ( sauf si 1 seul terme qui est deja sur fichier ou si k est introduit en entree avec un partie imaginaire ==1 ou -1
%    si k est introduit en entree avec un partie imaginaire ==1  choix de methode: reak(k)
%    si k est introduit en entree avec un partie imaginaire ==-1 on ne modifie pas le choix de methode...)
%
% (l'ancienne version   s=retss(ss,k) ss: cell array  {1,m} (ss={s1,s2,...sm} reste valable mais deconseillee)
% 
% exemples 
%  s=retss(s1,s2,s3)  s=s1*s2*s3
%  s=retss(s1)  s=s1
%  s=retss(s1,[])  s=s1
%  s=retss([],s1)  s=s1
%  retss(4)  --> on utilise maintenant l'option 4
%  s=retss(s1,s2,s3,3)--> on fait ce produit avec l'option 3
%                 et on conserve l'option 4 pour la suite 
%  s=retss(s1,s2,s3,3+i)--> on fait ce produit avec l'option 3 et s est mise sur fichier(car abs(imag(parm))==1)
%  s=retss(s1,s2,s3,-i) --> on fait ce produit avec l'option 4 et s est mise sur fichier
%  s=retss(s1,s2,s3,-2i)--> on fait ce produit avec l'option 4 et s n'est pas mise sur fichier(car abs(imag(parm))==1)
%
%   retss   --> retourne l'option utilisee  ([] veut dire l'option 3)

persistent k0;
if nargin<1;s=k0;return;end;
io=0;
%if ~isempty(varargin{1})&isnumeric(varargin{1})&(~ all(size(varargin{1})==[1,1]));% produit vectorialisé des matrices T du 0D
%if ~isempty(varargin{1})&iscell(varargin{1})&(size(varargin{1},1)==2);% produit vectorialisé des matrices T du 0D
if ~isempty(varargin{1})&iscell(varargin{1})&(size(varargin{1},2)>1)&(nargin>1);% produit vectorialisé des matrices T du 0D
%s=varargin{1};for ii=2:nargin;s=retss0(s,varargin{ii});end; 
if iscell(varargin{end});
s=varargin{1};for ii=2:nargin;s=retss0(s,varargin{ii});end;% matrices T
else;s=varargin{1};for ii=2:nargin-1;s=retss00(s,varargin{ii});end;% matrices S
end;    
return;end;

if isnumeric(varargin{1})&(   all(size(varargin{1})==[1,1])      );k0=varargin{1};return;end;
if isnumeric(varargin{end})&~isempty(varargin{end});m=nargin-1;
if abs(imag(varargin{end}))==1;io=1;else io=0;end;% s mis sur fichier
if imag(varargin{end})>=0;k=real(varargin{end});else;k=k0;end;
else;m=nargin;k=k0;end;

if isempty(k);k=3;end;% par defaut
%if imag(k)~=0;io=1;k=real(k);else io=0;end;% s mis sur fichier

ancien=0;% compatibilite avec ancienne version 
if iscell(varargin{1});if iscell(varargin{1}{1})|ischar(varargin{1}{1})|(isempty(varargin{1}{1})&length(varargin{1})<3);ancien=1;end;end;
if iscell(varargin{m});if iscell(varargin{m}{1})|ischar(varargin{m}{1})|(isempty(varargin{m}{1})&length(varargin{m})<3);ancien=1;end;end;
if ancien==1;m=length(varargin{1});varargin=varargin{1};end;
if m==1;s=varargin{1};return;end;

if m==2;if isempty(varargin{1});s=varargin{2};return;end;if isempty(varargin{2});s=varargin{1};return;end;end;


% size , volume de ss{1} et ss{m} et repetition pour les matrices S
[siz,v,n1]=retio(varargin{1},4);[siz,v,nm]=retio(varargin{m},4);
parms=spparms;sparms=parms;if length(sparms)>8;sparms(11)=0;spparms(sparms);end;% parametre pour les matrices creuses


if siz(1)<6; % matrices s (avec correction de la 'singularitee ld/4')
if m<3;vmax=max(n1(1),nm(1));else [sz,v,vmax]=retio(varargin{2},4);vmax=vmax(1);end;
if vmax>1.e7;if k==3;k=30;end;if k==4;k=40;end;end;% <test
if k>10;preserves=retio;end; % economie en memoire pour les tres grosses matrices    
 
if n1(1)<nm(1);% produit de gauche a droite
[sz,v,n2]=retio(varargin{2},4);
s=varargin{2};for iii=1:n1(2);s=retss1(varargin{1},s,k);end;for iii=2:n2(2);s=retss1(s,varargin{2},k);end;
for ii=3:m;[sz,v,n1]=retio(varargin{ii},4);for iii=1:n1(2);s=retss1(s,varargin{ii},k);end;end    
else;    % produit de droite a gauche
[sz,v,n2]=retio(varargin{m-1},4);
s=varargin{m-1};for iii=1:nm(2);s=retss1(s,varargin{m},k);end;for iii=2:n2(2);s=retss1(varargin{m-1},s,k);end;
for ii=m-2:-1:1;[sz,v,n1]=retio(varargin{ii},4);for iii=1:n1(2);s=retss1(varargin{ii},s,k);end;end    
end;


else % matrices g
if m<3;vmax=max(n1(1),nm(1));else [sz,v,vmax]=retio(varargin{2},4);vmax=vmax(1);end;
if vmax>1.e6;if k==3;k=30;end;if k==4;k=40;end;end;
if k>10;preserves=retio;end; % economie en memoire pour les tres grosses matrices

if n1(1)<nm(1);% produit de gauche a droite
s=varargin{1};for ii=2:m;s=retgg1(s,varargin{ii},k);end;
else;   %  produit de droite a gauche
s=varargin{m};for ii=m-1:-1:1;s=retgg1(varargin{ii},s,k);end;
end;    
end;

% economie en memoire pour les tres grosses matrices (si io=1 on laisse sur fichier )
if k>10;if ischar(s);if io==1;preserves=[preserves,str2num(s)];else;s=retio(s,-2);end;end;retio(preserves,-4);end; 
s=retio(s,io);
spparms(parms);% retour du parametre pour les matrices creuses
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function s=retss1(sss,ss,k);if isempty(sss);s=ss;return;end;if isempty(ss);s=sss;return;end;
%produit matrice s (n3+n4,n1+n2) =matrice sss(nnn3+nnn4,nnn1+nnn2) par matrice ss(nnn3+nnn4,nnn1+nnn2)
sss=retio(retfullsparse(sss));ss=retio(retfullsparse(ss));if isempty(sss);s=ss;return;end;if isempty(ss);s=sss;return;end;

nn1=ss{2};nn2=size(ss{1},2)-nn1;nn3=ss{3};nn4=size(ss{1},1)-nn3;
nnn1=sss{2};nnn2=size(sss{1},2)-nnn1;nnn3=sss{3};nnn4=size(sss{1},1)-nnn3;

if (nnn1~=nn3)|(nnn4~=nn2);[sss,ss]=retcrs(sss,ss);  %croissance des matrices avant le produit
nn1=ss{2};nn2=size(ss{1},2)-nn1;nn3=ss{3};nn4=size(ss{1},1)-nn3;
nnn1=sss{2};nnn2=size(sss{1},2)-nnn1;nnn3=sss{3};nnn4=size(sss{1},1)-nnn3;
end;
i1=[1:nn3];i2=[nn3+1:nn3+nn4];j1=[1:nn1];j2=[nn1+1:nn1+nn2];
ii1=[1:nnn3];ii2=[nnn3+1:nnn3+nnn4];jj1=[1:nnn1];jj2=[nnn1+1:nnn1+nnn2];
mm=nnn1+nn4;mmm=nnn3+nn2;

switch abs(k);
    
case 0
s=[[zeros(mm,nnn3),[zeros(nnn1,nn4);eye(nn4)],[eye(nnn1);zeros(nn4,nnn1)],-ss{1}(:,j2)];...
[[eye(nnn3);zeros(nn2,nnn3)],zeros(mmm,nn4),-sss{1}(:,jj1),[zeros(nnn3,nn2);eye(nn2)]]]...
\[[ss{1}(:,j1),zeros(mm,nnn2)];[zeros(mmm,nn1),sss{1}(:,jj2)]];
s={s(1:nnn3+nn4,:);nn1;nnn3;1};
   
case 1
s=[[spalloc(mm,nnn3,0),[spalloc(nnn1,nn4,0);speye(nn4)],[speye(nnn1);spalloc(nn4,nnn1,0)],-ss{1}(:,j2)];...
[[speye(nnn3);spalloc(nn2,nnn3,0)],spalloc(mmm,nn4,0),-sss{1}(:,jj1),[spalloc(nnn3,nn2,0);speye(nn2)]]]...
\[[ss{1}(:,j1),spalloc(mm,nnn2,0)];[spalloc(mmm,nn1,0),sss{1}(:,jj2)]];
s={full(s(1:nnn3+nn4,:));nn1;nnn3;1};
case 2
s=[[eye(nnn1);-sss{1}(ii2,jj1)],[-ss{1}(i1,j2);eye(nn2)]]\[[ss{1}(i1,j1);zeros(nn2,nn1)],[zeros(nnn1,nnn2);sss{1}(ii2,jj2)]];
s=[sss{1}(ii1,jj1)*s(1:nnn1,:);ss{1}(i2,j2)*s(nnn1+1:nnn1+nn2,:)];
s={[[zeros(nnn3,nn1);ss{1}(i2,j1)],[sss{1}(ii1,jj2);zeros(nn4,nnn2)]]+s;nn1;nnn3;1};
   
case 30 %economique en memoire
clear i1 i2 j1 j2 ii1 ii2 jj1 jj2;
ss11=retio(ss{1}(1:nn3,1:nn1),1);ss12=retio(ss{1}(1:nn3,nn1+1:nn1+nn2),1);ss21=retio(ss{1}(nn3+1:nn3+nn4,1:nn1),1);ss22=retio(ss{1}(nn3+1:nn3+nn4,nn1+1:nn1+nn2),1); clear ss;   
sss11=retio(sss{1}(1:nnn3,1:nnn1),1);sss12=retio(sss{1}(1:nnn3,nnn1+1:nnn1+nnn2),1);sss21=retio(sss{1}(nnn3+1:nnn3+nnn4,1:nnn1),1);sss22=retio(sss{1}(nnn3+1:nnn3+nnn4,nnn1+1:nnn1+nnn2),1);clear sss;    
s=inv(eye(nnn1)-retio(ss12)*retio(sss21));
s=[s*retio(ss11,-2),s*(retio(ss12,-2)*retio(sss22))];
s=[s;retio(sss21,-2)*s];
s(nnn1+1:nnn1+nn2,nn1+1:nn1+nnn2)=s(nnn1+1:nnn1+nn2,nn1+1:nn1+nnn2)+retio(sss22,-2);
s1=retio(s(1:nnn1,:),1);s=s(nnn1+1:nnn1+nn2,:);
s=retio(retio(ss22,-2)*s,1);
s=[retio(sss11,-2)*retio(s1,-2);retio(s,-2)];
%s=[retio(sss11,-2)*s(1:nnn1,:);retio(ss22,-2)*s(nnn1+1:nnn1+nn2,:)];
s(nnn3+1:end,1:nn1)=s(nnn3+1:end,1:nn1)+retio(ss21,-2);
s(1:nnn3,nn1+1:end)=s(1:nnn3,nn1+1:end)+retio(sss12,-2);s=retio({s;nn1;nnn3;1},1);

case 40 %economique en memoire
clear i1 i2 j1 j2 ii1 ii2 jj1 jj2;
ss11=retio(ss{1}(1:nn3,1:nn1),1);ss12=retio(ss{1}(1:nn3,nn1+1:nn1+nn2),1);ss21=retio(ss{1}(nn3+1:nn3+nn4,1:nn1),1);ss22=retio(ss{1}(nn3+1:nn3+nn4,nn1+1:nn1+nn2),1); clear ss;   
sss11=retio(sss{1}(1:nnn3,1:nnn1),1);sss12=retio(sss{1}(1:nnn3,nnn1+1:nnn1+nnn2),1);sss21=retio(sss{1}(nnn3+1:nnn3+nnn4,1:nnn1),1);sss22=retio(sss{1}(nnn3+1:nnn3+nnn4,nnn1+1:nnn1+nnn2),1);clear sss;    
s=inv(eye(nn2)-retio(sss21)*retio(ss12));
s=[s*(retio(sss21,-2)*retio(ss11)),s*retio(sss22,-2)];
s=[retio(ss12,-2)*s;s];
s(1:nnn1,1:nn1)=s(1:nnn1,1:nn1)+retio(ss11,-2);
s1=retio(s(1:nnn1,:),1);s=s(nnn1+1:nnn1+nn2,:);
s=retio(retio(ss22,-2)*s,1);
s=[retio(sss11,-2)*retio(s1,-2);retio(s,-2)];
%s=[retio(sss11,-2)*s(1:nnn1,:);retio(ss22,-2)*s(nnn1+1:nnn1+nn2,:)];
s(nnn3+1:end,1:nn1)=s(nnn3+1:end,1:nn1)+retio(ss21,-2);
s(1:nnn3,nn1+1:end)=s(1:nnn3,nn1+1:end)+retio(sss12,-2);s=retio({s;nn1;nnn3;1},1);

case {3,300}
s=inv(eye(nnn1)-ss{1}(i1,j2)*sss{1}(ii2,jj1));
s=[s*ss{1}(i1,j1),s*(ss{1}(i1,j2)*sss{1}(ii2,jj2))];
s=[s;sss{1}(ii2,jj1)*s];
s(nnn1+1:nnn1+nn2,nn1+1:nn1+nnn2)=s(nnn1+1:nnn1+nn2,nn1+1:nn1+nnn2)+sss{1}(ii2,jj2);  
s=[sss{1}(ii1,jj1)*s(1:nnn1,:);ss{1}(i2,j2)*s(nnn1+1:nnn1+nn2,:)];

ss=ss{1}(i2,j1);sss=sss{1}(ii1,jj2);
s(nnn3+1:end,1:nn1)=s(nnn3+1:end,1:nn1)+ss;s(1:nnn3,nn1+1:end)=s(1:nnn3,nn1+1:end)+sss;s={s;nn1;nnn3;1};
%s={[[zeros(nnn3,nn1);ss{1}(i2,j1)],[sss{1}(ii1,jj2);zeros(nn4,nnn2)]]+s;nn1;nnn3;1};
   
case {4,400}
s=inv(eye(nn2)-sss{1}(ii2,jj1)*ss{1}(i1,j2));
s=[s*(sss{1}(ii2,jj1)*ss{1}(i1,j1)),s*sss{1}(ii2,jj2)];
s=[ss{1}(i1,j2)*s;s];
s(1:nnn1,1:nn1)=s(1:nnn1,1:nn1)+ss{1}(i1,j1);
s=[sss{1}(ii1,jj1)*s(1:nnn1,:);ss{1}(i2,j2)*s(nnn1+1:nnn1+nn2,:)];
ss=ss{1}(i2,j1);sss=sss{1}(ii1,jj2);s(nnn3+1:end,1:nn1)=s(nnn3+1:end,1:nn1)+ss;s(1:nnn3,nn1+1:end)=s(1:nnn3,nn1+1:end)+sss;s={s;nn1;nnn3;1};
%s={[[zeros(nnn3,nn1);ss{1}(i2,j1)],[sss{1}(ii1,jj2);zeros(nn4,nnn2)]]+s;nn1;nnn3;1};

case 5;
try;choix=rcond(eye(nn2)-sss{1}(ii2,jj1)*ss{1}(i1,j2))>rcond(eye(nnn1)-ss{1}(i1,j2)*sss{1}(ii2,jj1));catch;choix=1;end;  
if choix;  
s=inv(eye(nnn1)-ss{1}(i1,j2)*sss{1}(ii2,jj1));
s=[s*ss{1}(i1,j1),s*(ss{1}(i1,j2)*sss{1}(ii2,jj2))];
s=[s;sss{1}(ii2,jj1)*s];
s(nnn1+1:nnn1+nn2,nn1+1:nn1+nnn2)=s(nnn1+1:nnn1+nn2,nn1+1:nn1+nnn2)+sss{1}(ii2,jj2);  
else
s=inv(eye(nn2)-sss{1}(ii2,jj1)*ss{1}(i1,j2));
s=[s*(sss{1}(ii2,jj1)*ss{1}(i1,j1)),s*sss{1}(ii2,jj2)];
s=[ss{1}(i1,j2)*s;s];
s(1:nnn1,1:nn1)=s(1:nnn1,1:nn1)+ss{1}(i1,j1);
end
s=[sss{1}(ii1,jj1)*s(1:nnn1,:);ss{1}(i2,j2)*s(nnn1+1:nnn1+nn2,:)];
ss=ss{1}(i2,j1);sss=sss{1}(ii1,jj2);s(nnn3+1:end,1:nn1)=s(nnn3+1:end,1:nn1)+ss;s(1:nnn3,nn1+1:end)=s(1:nnn3,nn1+1:end)+sss;s={s;nn1;nnn3;1};

%s={[[zeros(nnn3,nn1);ss{1}(i2,j1)],[sss{1}(ii1,jj2);zeros(nn4,nnn2)]]+s;nn1;nnn3;1};
case 6
prv=sss{1}(ii1,jj1)/(eye(nnn1)-ss{1}(i1,j2)*sss{1}(ii2,jj1));
s=[prv*ss{1}(i1,j1),prv*ss{1}(i1,j2)*sss{1}(ii2,jj2)+sss{1}(ii1,jj2)];
prv=ss{1}(i2,j2)/(eye(nn2)-sss{1}(ii2,jj1)*ss{1}(i1,j2));
s=[s;[prv*sss{1}(ii2,jj1)*ss{1}(i1,j1)+ss{1}(i2,j1),prv*sss{1}(ii2,jj2)]];clear prv;
s={s;nn1;nnn3;1};
%s={[[zeros(nnn3,nn1);ss{1}(i2,j1)],[sss{1}(ii1,jj2);zeros(nn4,nnn2)]]+s;nn1;nnn3;1};


end;  % k

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function g2=retgg1(g2,g1,k);
g1=retfullsparse(retio(g1));g2=retfullsparse(retio(g2));if isempty(g2);g2=g1;return;end;if isempty(g1);return;end;
%g1=retio(retfullsparse(g1));g2=retio(retfullsparse(g2));if isempty(g2);g2=g1;return;end;if isempty(g1);return;end;
%function g2=retgg1(g2,g1,k);
%g2=g2*g1

if (g2{3}~=g1{5})|(g2{4}~=g1{6});[g2,g1]=retcrs(g2,g1);end;  %croissance des matrices avant le produit

n1=size(g1{2},2);
n2=size(g1{2},1);
n3=size(g2{1},1);
m1=size(g1{1},2);
m2=size(g2{2},2);

switch k;
    
case 0;  %methode QR
[q,r]=qr([full(g1{2});full(g2{1})]);
[f,test]=degenere(r,n1);
g2={(q(1:n2,f))'*g1{1};-(q(n2+1:n2+n3,f))'*g2{2};g1{3};g1{4};g2{5};g2{6}};

    
case 4;  %methode QR ne traite pas les degenerescences
[q,r]=qr([full(g1{2});full(g2{1})]);q=q(1:n2+n3,n1+1:end);
g2={(q(1:n2,:))'*g1{1};-(q(n2+1:n2+n3,:))'*g2{2};g1{3};g1{4};g2{5};g2{6}};
test=0;% ne traite pas les degenerescences

case 40;  %methode QR  economique en memoire
g11=retio(full(g1{1}),1);g12=retio(full(g1{2}),1);g13=g1{3};g14=g1{4};clear g1;
g21=retio(full(g2{1}),1);g22=retio(full(g2{2}),1);g25=g2{5};g26=g2{6};clear g2;
[q,r]=qr([retio(g12);retio(g21)]);q=q(1:n2+n3,n1+1:end);
g2={(q(1:n2,:))'*retio(g11);-(q(n2+1:n2+n3,:))'*retio(g22);g13;g14;g25;g26};
g2=retio(g2,1);
test=0;% ne traite pas les degenerescences

case {3,300}; % methode LU  la plus rapide
g2{3}=g1{3};g2{4}=g1{4};
g1=[[g1{2};g2{1}],[-g1{1};zeros(n3,m1)],[zeros(n2,m2);-g2{2}]];g2{1}=[];g2{2}=[];
g1=lu(full(g1));
g1=g1(n1+1:end,n1+1:end);
g1=triu(g1);% for ii=1:n2+n3-n1;g1(ii+1:n2+n3-n1,ii)=0;end;
g2{1}=g1(:,1:m1);g2{2}=-g1(:,m1+1:end);
test=0;% ne traite pas les degenerescences


case 2; % methode LU 
g2{3}=g1{3};g2{4}=g1{4};
[l,g1]=lu([[g1{2};g2{1}],[-g1{1};zeros(n3,m1)],[zeros(n2,m2);-g2{2}]]);clear l ;
[f,test]=degenere(g1,n1);
g2{1}=g1(f,n1+1:n1+m1);g2{2}=-g1(f,n1+m1+1:end);

case 30;  % methode LU sparse  economique en memoire
g11=retio(g1{1},1);g12=retio(g1{2},1);g13=g1{3};g14=g1{4};clear g1;
g21=retio(g2{1},1);g22=retio(g2{2},1);g25=g2{5};g26=g2{6};clear g2;
b=sparse([retio(g12,-2);retio(g21,-2)]);
[b,u,p]=lu(b);clear u;
bb=retio(b(n1+1:n2+n3,:),1);b=b(1:n1,:);b=-(full(b)'\retio(bb,-2)')';
b=sparse([b,speye(n2+n3-n1)])*p;clear p;
b1=retio(b(:,1:n2),1);b2=retio(b(:,n2+1:n2+n3),1);clear b;
g2={full(retio(b1,-2))*full(retio(g11,-2));-full(retio(b2,-2))*full(retio(g22,-2));g13;g14;g25;g26};
g2=retio(g2,1);
test=0;% ne traite pas les degenerescences

case 5;  % methode LU sparse p q à partir de Matlab 7
b=[sparse(g1{2});sparse(g2{1})];g1{2}=[];g2{1}=[];
if ~isempty(b);[b,u,p,q]=lu(b,1);clear u q;else;p=speye(size(b,1));end;
b=full(b');
bb=b(:,n1+1:n2+n3);
b=b(:,1:n1,:);
b=-(b\bb)';clear bb;
b=[b,eye(n2+n3-n1)]*p;clear p;
g2={full(retfullsparse(b(:,1:n2))*g1{1});full(-retfullsparse(b(:,n2+1:n2+n3))*g2{2});g1{3};g1{4};g2{5};g2{6}};
test=0;% ne traite pas les degenerescences


case 50;  % methode LU sparse p q economique en memoire à partir de Matlab 7
g11=retio(g1{1},1);g12=retio(g1{2},1);g13=g1{3};g14=g1{4};clear g1;
g21=retio(g2{1},1);g22=retio(g2{2},1);g25=g2{5};g26=g2{6};clear g2;
b=[sparse(retio(g12));sparse(retio(g21))];
if ~isempty(b);[b,u,p,q]=lu(b,1);clear u q;else;p=speye(size(b,1));end;
b=full(b');
b=-(b(:,1:n1,:)\b(:,n1+1:n2+n3))';
b=[b,eye(n2+n3-n1)]*p;clear p;
g2={full(retfullsparse(b(:,1:n2))*retio(g11));full(-retfullsparse(b(:,n2+1:n2+n3))*retio(g22));g13;g14;g25;g26};
test=0;% ne traite pas les degenerescences


case 33;  % methode LU sparse  economique en memoire (ancienne version)
g11=retio(g1{1},1);g12=retio(g1{2},1);g13=g1{3};g14=g1{4};clear g1;
g21=retio(g2{1},1);g22=retio(g2{2},1);g25=g2{5};g26=g2{6};clear g2;
b=sparse([[retio(g12);retio(g21)],spalloc(n2+n3,n2+n3-n1,0)]);
[b,u]=lu(b);clear u ;
b=(b'\[spalloc(n1,n2+n3-n1,0);speye(n2+n3-n1)])';
b1=retio(b(:,1:n2),1);b2=retio(b(:,n2+1:n2+n3),1);clear b;
g2={full(retio(b1))*full(retio(g11));-full(retio(b2))*full(retio(g22));g13;g14;g25;g26};
g2=retio(g2,1);
test=0;% ne traite pas les degenerescences

case 6;  % methode LU sparse( ancienne version)
g2{3}=g1{3};g2{4}=g1{4};
[l,b]=lu([[g1{2};g2{1}],spalloc(n2+n3,n2+n3-n1,0)]);g1{2}=[];g2{1}=[];
b=(l'\[spalloc(n1,n2+n3-n1,0);speye(n2+n3-n1)])';
g2{1}=full(b(:,1:n2))*full(g1{1});g2{2}=-full(b(:,n2+1:n2+n3))*full(g2{2});
test=0;% ne traite pas les degenerescences


case 1; % methode du pivot (longue)
k1=n2+n3;k2=n1+m1+m2;
a=[g1{2};g2{1}];
gg1=[g1{1};zeros(n3,m1)];gg2=[zeros(n2,m2);-g2{2}];
g2={g1{3};g1{4};g2{5};g2{6}};clear g1;
[y,ii]=max(abs(a),[],1);[amax,jj]=max(y);ii=ii(jj);% recherche du pivot    
while amax>0;
kk=[1:ii-1,ii+1:size(a,1)];    
g21=gg1(ii,:);g22=gg2(ii,:);aa=a(ii,[1:jj-1,jj+1:end]);b=-a(kk,jj)/a(ii,jj);
a=a(kk,[1:jj-1,jj+1:end]);gg1=gg1(kk,:);gg2=gg2(kk,:);
% elimination
a=a+b*aa;
gg1=gg1+b*g21;
gg2=gg2+b*g22;
if isempty(a);break;
else;[y,ii]=max(abs(a),[],1);[amax,jj]=max(y);ii=ii(jj);% recherche du pivot
%else;prv=abs(a);prv=retdiag(1./max(prv,[],2))*prv;[y,ii]=max(abs(prv),[],1);[amax,jj]=max(y);ii=ii(jj);% recherche du pivot
end;
end;
test=size(gg1,1)>n2+n3-n1&n2+n3-n1>0;
g2={gg1;gg2;g2{1};g2{2};g2{3};g2{4}};
end; % k

if test;g2=regenere(g2,n2+n3-n1);end;

%%%%%%%%%%%%%%%%%%%%%%%%
function [f,test]=degenere(g,n)% recherche des degenerescences
if n>0;f=find(all(g(1:n,1:n)==zeros(n),2));else;f=[];end;test=~isempty(f);
f=[f',n+1:size(g,1)];
%%%%%%%%%%%%%%%%%%%%%%%%
function g=regenere(g,m);% traitement des degenerescences
m1=size(g{1},2);m2=size(g{2},2);
try;
[u,s,gg]=svd(full([g{1},-g{2}]));g{1}=(gg(1:m1,1:m))';g{2}=-(gg(m1+1:m1+m2,1:m))';% tri des m 'meilleures' lignes 
%[l,u]=lu(full([g{1},-g{2}]));g{1}=u(1:m,1:m1);g{2}=-u(1:m,m1+1:m1+m2);% tri des m 'meilleures' lignes 
end;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function s=retss0(sss,ss);% produit vectorialisé des matrices T du 0D
% s=zeros(size(sss));
% for ii=1:2;for jj=1:2;s(:,:,ii,jj)=sss(:,:,ii,1).*ss(:,:,1,jj)+sss(:,:,ii,2).*ss(:,:,2,jj);end;end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s=retss0(sss,ss);% produit vectorialisé des matrices T du 0D
[n,p]=size(sss);m=size(ss,2);
s=cell(n,m);
for ii=1:n;for jj=1:m;s{ii,jj}=sss{ii,1}.*ss{1,jj};for kk=2:p;s{ii,jj}=s{ii,jj}+sss{ii,kk}.*ss{kk,jj};end;end;end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s=retss00(sss,ss);% produit vectorialisé des matrices S du 0D
s=cell(2,2);
% s{1,2}=1./(1-ss{1,2}.*sss{2,1});
% s{2,1}=ss{1,1}.*s{1,2};ss{1,1}=[];
% s{1,2}=sss{2,2}.*s{1,2};sss{2,2}=[];
% s{1,1}=sss{1,1}.*s{2,1};
% s{2,2}=ss{2,2}.*s{1,2};
% s{2,1}=sss{2,1}.*ss{2,2}.*s{2,1}+ss{2,1};
% s{1,2}=sss{1,1}.*ss{1,2}.*s{1,2}+sss{1,2};

s{1,2}=1./(1-ss{1,2}.*sss{2,1});
ss{1,1}=ss{1,1}.*s{1,2};
sss{2,2}=sss{2,2}.*s{1,2};
s{1,1}=sss{1,1}.*ss{1,1};
s{2,2}=ss{2,2}.*sss{2,2};

s{2,1}=sss{2,1}.*ss{2,2}.*ss{1,1}+ss{2,1};
s{1,2}=sss{1,1}.*ss{1,2}.*sss{2,2}+sss{1,2};

















