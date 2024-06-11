function s=rettronc(ss,f1,f2,f3,f4);
% function s=rettronc(ss,f1,f2,f3,f4);
%
% TRONCATURE matrice ss 
% ss(n3+n4,n1+n2): matrice complete 
% f1,f2,f3,f4:numeros  a conserver parmi les numeros [1:n1] [1:n2] [1:n3] [1:n4]
% si f4 n'existe pas: f1 f2:numeros a conserver dans Incident et Diffracte  f3:sens(1:haut  -1:bas  0:source)
% si f1 ou f2 etc..  <0 ce sont les numeros a eliminer
%
%  MATRICE S
%   cell array :S={ss,n1,n3,m}
%  dim n3 -->      X            x     <--- dim n1  
%                      =S{1}
%  dim n4--->      y            Y      <---dim n2
% 
%
%  MATRICE G
%   cell array :G={G1,G2,n1,n4,n3,n2}
%  dim n1 --->           x              X    <----- dim n3
%                 G1*         =  G2 *
%  dim n4--->           y              Y    <----- dim n2
%
% forme plus simple:
% troncature en haut: s=rettronc(ss,Incidents gardes,Diffractes gardes,1) si on garde tout mettre 0
% troncature en bas: s=rettronc(ss,Incidents gardes,Diffractes gardes,-1) si on garde tout mettre 0
% troncature des sources: s=rettronc(ss,Incidents gardes,Diffractes gardes,0)  a faire avant tout produit
%
% remarque: ss peut avoir ete stocké sur fichier par retio(ss,1)
%

f1=reshape(f1,1,length(f1));
f2=reshape(f2,1,length(f2));
if nargin>=5;f3=reshape(f3,1,length(f3));f4=reshape(f4,1,length(f4));end;

ss=retio(ss);
if size(ss,1)<6;n1=ss{2};n2=size(ss{1},2)-n1;n3=ss{3};n4=size(ss{1},1)-n3; %matrices S
else;n1=ss{3};n4=ss{4};n3=ss{5};n2=ss{6};end;                              %matrices G


% determination de f1,f2,f3 f4 dans le cas des donnees incompletes
if nargin<5; % f1 f2:numeros a conserver (ou eliminer) dans Incident et Diffracte  f3:sens
g1=f1;g2=f2;sens=f3;
if sens==1; if g1==0;g1=1:n2;end;   if g2==0;g2=1:n3;end;   f1=1:n1;f2=g1;f3=g2;f4=1:n4;end;    %troncature en haut f1 f2 : Ih et Dh
if sens==-1;if g1==0;g1=1:n1;end;   if g2==0;g2=1:n4;end;   f1=g1;f2=1:n2;f3=1:n3;f4=g2;end;       %troncature en bas  f1 f2:Ib  Db  
if sens==0; if g1==0;g1=1:n2-n4;end;if g2==0;g2=1:n3-n1;end;g1=retcomp(g1,n2-n4);g2=retcomp(g2,n3-n1);f1=1:n1;f2=[1:n4,n4+g1];f3=[1:n1,n1+g2];f4=1:n4;end;%troncature des sources  f1 f2:Is  Ds  
end;
f1=retcomp(f1,n1);f2=retcomp(f2,n2);f3=retcomp(f3,n3);f4=retcomp(f4,n4);% si <0,complementaire

% troncature
if size(ss,1)<6; %matrices S
s=ss{1}([f3,f4+n3],[f1,f2+n1]);
s={s;length(f1);length(f3);1};

else;  %matrices G
ie=n1-length(f1); % nombre d'incidents a eliminer en entree (x)
is=n2-length(f2); % nombre d'incidents a eliminer en sortie (Y)
% on ecrit qu'ils sont nuls
ff1=[f1,f4+n1];nn1=length(ff1); % conserves en entree (x,y)
ff2=[f3,f2+n3];nn2=length(ff2); % conserves en sortie (X Y)
kk1=[1:n1+n4];kk1(ff1)=0;kk1=find(kk1~=0);mm1=length(kk1); % a eliminer en entree (x,y)
kk2=[1:n3+n2];kk2(ff2)=0;kk2=find(kk2~=0);mm2=length(kk2); % a eliminer en sortie (X Y)
sprov={[ss{1}(:,ff1),-ss{2}(:,ff2)];[-ss{1}(:,kk1),ss{2}(:,kk2)];nn1;nn2;mm1;mm2};
sproj={[[eye(ie),zeros(ie,mm1+mm2-ie)];[zeros(is,mm1+mm2-is),eye(is)]];zeros(ie+is,0);mm1;mm2;0;0};
s0=retss(sproj,sprov);
s={s0{1}(:,1:nn1);-s0{1}(:,nn1+1:nn1+nn2);length(f1);length(f4);length(f3);length(f2)};
end    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f=retcomp(f,n);% complementaire
if isempty(f);return;end;
if f(1)>0;f=1+mod(f-1,n);return;end;% mod pour les textures inclinees ...
if f(1)==0;f=1:n;return;end;
ff=1:n;ff(1+mod(-f-1,n))=0;f=ff(ff~=0);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function f=retcomp(f,n);% complementaire
% if isempty(f);return;end;
% if f(1)>0;return;end;
% if f(1)==0;f=1:n;return;end;
% ff=1:n;ff(-f)=0;f=ff(ff~=0);




